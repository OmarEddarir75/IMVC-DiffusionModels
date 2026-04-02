import os
import csv
import json
import random
import itertools
from pathlib import Path

import math
import torch
import logging
import datetime
import numpy as np


from baseModels import (Autoencoder, AttentionLayer, Unet, NoiseScheduler, MissingViewImputer, ViewClusterHead,)

def build_optimizer(autoencoders, attention_layer, unets, view_head, lr):
    params = itertools.chain(
        *[ae.parameters() for ae in autoencoders],
        attention_layer.parameters(),
        *[unet.parameters() for unet in unets],
        view_head.parameters(),
    )
    return torch.optim.Adam(params, lr=lr)

# Model factory — works for any n_views
def build_models(cfg, device):
    ae_cfg   = cfg['Autoencoder']
    diff_cfg = cfg['diffusion']
    ns_cfg   = cfg['noise_scheduler']
    tr_cfg   = cfg['training']

    archs       = ae_cfg['archs']          # list of [in_dim, h1, h2, …, latent_dim]
    activations = ae_cfg['activations']    # one per view
    n_views     = len(archs)
    latent_dim  = archs[0][-1]            # same for all views
    n_clusters  = tr_cfg['n_clusters']
    use_norm    = ae_cfg.get('use_norm', True)

    # One Autoencoder per view
    autoencoders = [
        Autoencoder(archs[v], activation=activations[v], use_norm=use_norm).to(device)
        for v in range(n_views)
    ]

    # Single attention fusion layer over all views
    attention_layer = AttentionLayer(latent_dim, n_views=n_views).to(device)

    # Diffusion components (one Unet per view + shared noise scheduler)
    unets = [
        Unet(
            emb_size=diff_cfg['emb_size'],      # matches `emb_size` argument
            time_emb=diff_cfg['time_type'],     # matches `time_emb` argument
            out_size=latent_dim                 # matches `out_size` argument
        ).to(device) 
        for _ in range(n_views)
    ]
    scheduler = NoiseScheduler(num_timesteps=ns_cfg['num_timesteps'], beta_schedule=ns_cfg['beta_schedule'],)
    
    imputer = MissingViewImputer(unets, scheduler, device)

    # Clustering head (shared across views)
    view_head = ViewClusterHead(latent_dim, n_clusters).to(device)

    return (autoencoders, attention_layer, unets, scheduler, imputer, view_head,)


def get_mask(n_views, data_len, missing_rate, seed=None):
    """
    Generate an indicator matrix A for incomplete multi-view data.

    Args:
        n_views (int): Number of views.
        data_len (int): Number of samples.
        missing_rate (float): Fraction of missing entries overall (0 <= missing_rate < 1).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Indicator matrix of shape (data_len, n_views) with 1=present, 0=missing.
    """
    if seed is not None:
        np.random.seed(seed)

    # Target fraction of ones per view
    one_rate = 1.0 - missing_rate

    # Initialize mask as zeros
    mask = np.zeros((data_len, n_views), dtype=int)

    # Ensure at least one view per sample
    for i in range(data_len):
        # Choose at least one view randomly
        ones_in_sample = max(1, np.random.binomial(n_views, one_rate))
        ones_in_sample = min(ones_in_sample, n_views)  # cap at n_views
        ones_indices = np.random.choice(n_views, ones_in_sample, replace=False)
        mask[i, ones_indices] = 1

    return mask


def prepare_inputs(x_list, missing_rate, device, seed=None):
    """
    Prepare masked multi-view inputs for training.

    Args:
        x_list (list of np.ndarray): List of views, each of shape (n_samples, n_features).
        missing_rate (float): Fraction of missing entries overall (0 <= missing_rate < 1).
        device (torch.device): Device to place tensors on.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        x_train_list (list of torch.FloatTensor): Masked inputs for each view.
        mask (torch.LongTensor): Indicator matrix (1=present, 0=missing).
    """
    n_views = len(x_list)
    n_samples = x_list[0].shape[0]

    # Generate mask
    mask_np = get_mask(n_views, n_samples, missing_rate, seed=seed)

    # Apply mask to each view and convert to torch tensor
    x_train_list = [
        torch.from_numpy(x_list[v] * mask_np[:, v:v+1]).float().to(device)
        for v in range(n_views)
    ]

    # Convert mask to torch tensor
    mask = torch.from_numpy(mask_np).long().to(device)

    return x_train_list, mask


def parse_missing_rates(raw_value):
    parts = [p.strip() for p in str(raw_value).split(',') if p.strip()]
    if not parts:
        raise ValueError('missing_rate must contain at least one float value.')
    rates = [float(p) for p in parts]
    for r in rates:
        if r < 0.0 or r >= 1.0:
            raise ValueError(f'Invalid missing rate: {r}. Expected 0 <= rate < 1.')
    return rates


def apply_lambda_config(config, lambda_config_path):
    with open(lambda_config_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    params = payload.get('best_params', payload)
    if not isinstance(params, dict):
        raise ValueError('lambda_config must be a JSON object or contain a best_params object.')

    applied = {}
    for k, v in params.items():
        if k.startswith('lambda_') or k in ('mmi_temperature', 'mmi_internal_lambda'):
            config['training'][k] = float(v)
            applied[k] = float(v)

    if not applied:
        raise ValueError('No lambda_*, mmi_temperature, or mmi_internal_lambda keys found in lambda_config.')

    return applied


def safe_rate_str(x):
    return str(x).replace('.', 'p')


def build_checkpoint_path(root_dir, dataset_name, missing_rate, data_seed, tag):
    fname = f"{dataset_name}_mr{safe_rate_str(missing_rate)}_seed{data_seed}_{tag}.pt"
    return os.path.join(root_dir, fname)


def build_metrics_path(root_dir, dataset_name, missing_rate, data_seed, ext):
    fname = f"{dataset_name}_mr{safe_rate_str(missing_rate)}_seed{data_seed}_metrics.{ext}"
    return os.path.join(root_dir, fname)


def save_checkpoint(path, model, optimizer, config, run_seed, data_seed, missing_rate, metrics):
    """
    Save a training checkpoint with model, optimizer, config, seeds, missing rate, and metrics.
    Handles missing metric keys gracefully.
    """
    # Safely extract metrics, defaulting to 0.0 if missing
    metrics_payload = {k: float(metrics.get(k, 0.0)) for k in ['acc', 'nmi', 'ari', 'score']}

    # Build payload for checkpoint
    payload = {
        'config': config,
        'run_seed': int(run_seed),
        'data_seed': int(data_seed),
        'missing_rate': float(missing_rate),
        'metrics': metrics_payload,
        'model_state': model.checkpoint_state(),  # assumes model has this method
        'optimizer_state': optimizer.state_dict(),
    }

    # Save checkpoint
    torch.save(payload, path)
    print('Checkpoint saved:', path)


def save_metrics_history_json(path, history_payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history_payload, f, indent=2)
    print('Metrics JSON saved:', path)


def save_metrics_history_csv(path, history_rows):
    if not history_rows:
        return
    fieldnames = [
        'epoch',
        'loss',
        'rec_loss',
        'df_loss',
        'ce_loss',
        'mmi_loss',
        'cluster_loss',
        'hc_loss',
        'accuracy',
        'NMI',
        'ARI',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print('Metrics CSV saved:', path)


def set_run_seed(run_seed, use_cuda):
    np.random.seed(run_seed)
    random.seed(run_seed)
    os.environ['PYTHONHASHSEED'] = str(run_seed)
    torch.manual_seed(run_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(run_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
