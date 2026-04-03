"""
train.py — End-to-end training script for incomplete multi-view clustering.
Supports any number of views (2, 3, 4, …) via configure.py.

Usage examples
--------------
# Built-in 2-view dataset:
python train.py --dataset HandWritten

# Built-in 3-view dataset:
python train.py --dataset LandUse_21_3View

# Custom 4-view dataset (pass raw .mat / numpy arrays — see "Custom dataset" below):
python train.py --dataset MyData4View \
    --input_dims 64,128,256,512 \
    --latent_dim 128 \
    --n_clusters 10 \
    --missing_rate 0.3

# Override config on the fly:
python train.py --dataset HandWritten --missing_rate 0.5 --epoch 300
"""

import os
import random
import argparse

import torch
import numpy as np
import torch.nn as nn
from sklearn.cluster import KMeans

from datasets import load_data
from trainer import train, get_embeddings
from evaluation import evaluation, get_cluster_sols
from configure import get_default_config, _with_multiview_fields
from utils import (
    build_models,
    build_optimizer,
    prepare_inputs,
    set_run_seed,
    build_metrics_path,
    save_metrics_history_json,
    save_metrics_history_csv,
)


# Config helpers
def build_custom_config(args):
    """
    Build a config dict from CLI args for datasets not in configure.py.
    Use --input_dims to specify per-view feature sizes as comma-separated ints.
    """
    input_dims = [int(d) for d in args.input_dims.split(',')]
    cfg = dict(
        Autoencoder=dict(
            input_dims=input_dims,
            hidden_dims=[1024, 1024, 1024],
            latent_dim=args.latent_dim,
            activation='relu',
            use_norm=True,
        ),
        training=dict(
            seed=args.seed,
            mask_seed=args.mask_seed,
            missing_rate=args.missing_rate,
            batch_size=args.batch_size,
            epoch=args.epoch,
            lr=args.lr,
            lamda_recon=args.lamda_recon,
            lamda_mmi=args.lamda_mmi,
            lamda_diff=args.lamda_diff,
            lamda_cluster=args.lamda_cluster,
            n_clusters=args.n_clusters,
        ),
        diffusion=dict(
            emb_size=128,
            time_type="sinusoidal",
        ),
        noise_scheduler=dict(
            num_timesteps=100,
            beta_schedule="linear",
        ),
    )
    return _with_multiview_fields(cfg)


def maybe_override(cfg, args):
    """Override config fields that were explicitly supplied via CLI."""
    tr = cfg['training']
    if args.lr           is not None: tr['lr']           = args.lr
    if args.seed         is not None: tr['seed']         = args.seed
    if args.epoch        is not None: tr['epoch']        = args.epoch
    if args.mask_seed    is not None: tr['mask_seed']    = args.mask_seed
    if args.batch_size   is not None: tr['batch_size']   = args.batch_size
    if args.n_clusters   is not None: tr['n_clusters']   = args.n_clusters
    if args.missing_rate is not None: tr['missing_rate'] = args.missing_rate

    if args.lamda_mmi    is not None: tr['lamda_mmi']    = args.lamda_mmi
    if args.lamda_diff   is not None: tr['lamda_diff']   = args.lamda_diff
    if args.lamda_recon  is not None: tr['lamda_recon']  = args.lamda_recon
    if args.lamda_cluster is not None: tr['lamda_cluster'] = args.lamda_cluster
    return cfg


# Main training loop
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Config setup: load from configure.py or build from CLI args for custom datasets
    try:
        cfg = get_default_config(args.dataset)
    except Exception:
        # Dataset not in configure.py — build from CLI args
        if not args.input_dims:
            raise ValueError(
                f"Dataset '{args.dataset}' is not in configure.py. "
                "Provide --input_dims (e.g. --input_dims 64,128,256) "
                "and --n_clusters."
            )
        cfg = build_custom_config(args)
    cfg = maybe_override(cfg, args)
    cfg['dataset'] = args.dataset
    if args.source_mode is None:
        args.source_mode = cfg['training'].get('source_mode', 'attention')
    cfg['training']['source_mode'] = args.source_mode
    cfg['training'].setdefault('min_view_presence_ratio', 0.1)
    if args.dataset_root:
        cfg['dataset_root'] = args.dataset_root

    tr_cfg = cfg['training']
    source_mode = tr_cfg['source_mode']
    print(f"\nConfig:\n  dataset={args.dataset}")
    print(f"  views={len(cfg['Autoencoder']['archs'])}, "
          f"latent_dim={cfg['Autoencoder']['archs'][0][-1]}")
    print(f"  n_clusters={tr_cfg['n_clusters']}, "
          f"missing_rate={tr_cfg['missing_rate']}, "
          f"epochs={tr_cfg['epoch']}\n")

    # Reproducibility — set seeds for all randomness sources (data shuffling, mask generation, model initialization)
    set_run_seed(tr_cfg['seed'], torch.cuda.is_available())

    # Data loading and preprocessing
    x_list, y_list = load_data(cfg)
    labels = y_list[0]
    n_views = len(x_list)
    print(f"Loaded {n_views} views, N={x_list[0].shape[0]}")
    for v, x in enumerate(x_list):
        print(f"  View {v}: shape={x.shape}")

    x_views, mask = prepare_inputs(
        x_list,
        missing_rate=tr_cfg['missing_rate'],
        device=device,
        seed=tr_cfg.get('mask_seed'),
        min_view_presence_ratio=tr_cfg.get('min_view_presence_ratio', 0.1),
    )
    print(f"  Mask shape: {mask.shape}, "
          f"overall presence={mask.float().mean():.2%}\n")

    # Build models and optimizer
    (autoencoders, attention_layer, dfs, scheduler, imputer, view_head,) = build_models(cfg, device)

    optimizer = build_optimizer(autoencoders, attention_layer, dfs, view_head, lr=tr_cfg['lr'],)

    # Training Phase — Joint training with clustering and diffusion losses
    # print(f"\n=========================== Training Phase : Joint training ({tr_cfg['epoch']} epochs) ===========================")

    history_rows = train(
        autoencoders=autoencoders, attention_layer=attention_layer, dfs=dfs, scheduler=scheduler, imputer=imputer,
        view_head=view_head, x_views=x_views, mask=mask, optimizer=optimizer, device=device, n_epochs=tr_cfg['epoch'],
        batch_size=tr_cfg['batch_size'], n_clusters=tr_cfg['n_clusters'], lamda_recon=tr_cfg.get('lamda_recon', 1.0),
        lamda_mmi=tr_cfg.get('lamda_mmi', 1.0), lamda_diff=tr_cfg.get('lamda_diff', 1.0), lamda_cluster=tr_cfg.get('lamda_cluster', 1.0),
        mmi_temperature=1.0, conf_threshold=0.6, cluster_temperature=0.05, contrastive_temp=0.1,
        source_mode=source_mode, return_history=True, verbose=True,
    )

        # Evaluation

    print("\nEvaluation")
    z_fused = get_embeddings(
        autoencoders=autoencoders, attention_layer=attention_layer, 
        x_views=x_views, mask=mask, device=device, imputer=imputer, batch_size=256,
        source_mode=source_mode,
    )
        
    z_np = z_fused.detach().cpu().numpy()
    labels_np = labels if isinstance(labels, np.ndarray) else labels.numpy()
    y_pred, _ = get_cluster_sols(z_np, ClusterClass=KMeans, n_clusters=tr_cfg['n_clusters'], init_args={'n_init': 20})
    scores = evaluation(y_pred, labels_np)
    print(f"  ACC={scores['accuracy']:.4f}  NMI={scores['NMI']:.4f}  ARI={scores['ARI']:.4f}")

    # Store final clustering metrics in history for downstream plotting/reporting.
    if history_rows:
        history_rows[-1]["accuracy"] = float(scores["accuracy"])
        history_rows[-1]["NMI"] = float(scores["NMI"])
        history_rows[-1]["ARI"] = float(scores["ARI"])

    # Save checkpoint
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

        metrics_json_path = build_metrics_path(
            args.save_dir, args.dataset, tr_cfg['missing_rate'], tr_cfg['seed'], ext='json'
        )
        metrics_csv_path = build_metrics_path(
            args.save_dir, args.dataset, tr_cfg['missing_rate'], tr_cfg['seed'], ext='csv'
        )
        save_metrics_history_json(
            metrics_json_path,
            {
                "history": history_rows,
                "final_scores": {
                    "accuracy": float(scores["accuracy"]),
                    "NMI": float(scores["NMI"]),
                    "ARI": float(scores["ARI"]),
                },
            },
        )
        save_metrics_history_csv(metrics_csv_path, history_rows)

        ckpt_path = os.path.join(
            args.save_dir,
            f"{args.dataset}_mr{tr_cfg['missing_rate']}_seed{tr_cfg['seed']}.pt",
        )
        torch.save(
            {
                'autoencoders':          [ae.state_dict() for ae in autoencoders],
                'attention_layer':       attention_layer.state_dict(),
                'dfs':                 [unet.state_dict() for unet in dfs],
                'view_head':             view_head.state_dict(),
                'config':                cfg,
                'source_mode':           source_mode,
                'metrics':               scores,
                'history':               history_rows,
            },
            ckpt_path,
        )
        print(f"Checkpoint saved → {ckpt_path}")

    return scores['accuracy'], scores['NMI'], scores['ARI']



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-view incomplete clustering trainer')

    # Dataset
    parser.add_argument('--dataset',      type=str, required=True, help='Dataset name (e.g. HandWritten, LandUse_21, CUB)')
    parser.add_argument('--dataset_root', type=str, default=None,  help='Path to directory containing dataset .mat files')

    # Custom dataset (when not in configure.py)
    parser.add_argument('--input_dims',   type=str, default=None, help='Comma-separated input dims per view, e.g. "64,128,256,512"')
    parser.add_argument('--latent_dim',   type=int, default=128, help='Latent dimension size (default: 128)')

    # Training overrides (all optional — defaults come from configure.py)
    parser.add_argument('--lr',           type=float, default=None, help='Learning rate')
    parser.add_argument('--seed',         type=int,   default=None, help='Random seed')
    parser.add_argument('--epoch',        type=int,   default=None, help='Number of epochs')
    parser.add_argument('--mask_seed',    type=int,   default=None, help='Seed for generating missing data mask')
    parser.add_argument('--n_clusters',   type=int,   default=None, help='Number of clusters')
    parser.add_argument('--batch_size',   type=int,   default=None, help='Batch size')
    parser.add_argument('--missing_rate', type=float, default=None, help='Rate of missing data')

    parser.add_argument('--lamda_mmi',      type=float, default=None, help='Weight for MMI loss')
    parser.add_argument('--lamda_diff',     type=float, default=None, help='Weight for diffusion loss')
    parser.add_argument('--lamda_recon',    type=float, default=None, help='Weight for reconstruction loss')
    parser.add_argument('--lamda_cluster',  type=float, default=None, help='Weight for cluster loss')
    parser.add_argument('--source_mode',    type=str, choices=['mean', 'attention'], default=None, help='Source construction mode for diffusion/recovery (defaults to config value)')

    # Output
    parser.add_argument('--save_dir',     type=str,   default=None, help='Directory to save checkpoint (omit to skip saving)')

    args = parser.parse_args()
    main(args)