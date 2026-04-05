import sys
import random
import itertools
from pathlib import Path

import torch
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from ICDM import ICDM_Model


# Utils
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# Autoencoder Pretraining
def pretrain_autoencoders(model, views, mask, device, epochs=30):
    optimizer = torch.optim.Adam(model.autoencoders.parameters(), lr=1e-3)
    model.autoencoders.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for v in range(len(views)):
            x = views[v]
            m = mask[:, v].bool()

            if m.sum() == 0:
                continue

            x_obs = x[m]
            z = model.autoencoders[v].encoder(x_obs)
            x_rec = model.autoencoders[v].decoder(z)

            loss = torch.nn.functional.mse_loss(x_rec, x_obs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"[Pretrain AE] Epoch {epoch} Loss: {total_loss:.4f}")


# Synthetic Data
def create_synthetic_views(num_samples=24, latent_source_dim=6, nonlinear=True):
    """
    Create a challenging multi-view synthetic dataset.
    
    Args:
        num_samples (int): number of samples
        latent_source_dim (int): latent dimension
        nonlinear (bool): if True, apply nonlinear projection to views

    Returns:
        views (list of np.ndarray): synthetic views
        labels (np.ndarray): ground truth class labels
    """
    # Overlapping classes
    class0 = np.random.normal(-0.5, 0.6, (num_samples // 2, latent_source_dim))
    class1 = np.random.normal(0.5, 0.6, (num_samples - num_samples // 2, latent_source_dim))

    shared_latent = np.concatenate([class0, class1], axis=0).astype(np.float32)
    labels = np.concatenate([
        np.zeros(class0.shape[0], dtype=np.int64),
        np.ones(class1.shape[0], dtype=np.int64),
    ])

    # View projections
    projections = [
        np.random.normal(size=(latent_source_dim, 16)).astype(np.float32),
        np.random.normal(size=(latent_source_dim, 12)).astype(np.float32),
        np.random.normal(size=(latent_source_dim, 10)).astype(np.float32),
    ]
    biases = [
        np.random.normal(scale=0.1, size=(16,)).astype(np.float32),
        np.random.normal(scale=0.1, size=(12,)).astype(np.float32),
        np.random.normal(scale=0.1, size=(10,)).astype(np.float32),
    ]

    views = []
    for proj, bias in zip(projections, biases):
        noise = np.random.normal(scale=0.05, size=(num_samples, proj.shape[1])).astype(np.float32)
        x = shared_latent @ proj + bias + noise

        if nonlinear:
            # Apply elementwise nonlinearities
            # Mix of tanh, sigmoid, and relu per view
            if proj.shape[1] % 3 == 0:
                x[:, :proj.shape[1]//3] = np.tanh(x[:, :proj.shape[1]//3])
                x[:, proj.shape[1]//3:2*proj.shape[1]//3] = 1/(1 + np.exp(-x[:, proj.shape[1]//3:2*proj.shape[1]//3]))
                x[:, 2*proj.shape[1]//3:] = np.maximum(0, x[:, 2*proj.shape[1]//3:])
            else:
                x = np.tanh(x)

        views.append(x.astype(np.float32))

    return views, labels

def create_mask(num_samples, num_views, observed_prob=0.8):
    mask = (np.random.rand(num_samples, num_views) < observed_prob).astype(np.int64)
    empty_rows = np.where(mask.sum(axis=1) == 0)[0]
    if empty_rows.size:
        mask[empty_rows, np.random.randint(0, num_views, size=len(empty_rows))] = 1
    return mask


# Config
def build_config():
    latent_dim = 8
    return {
        "Autoencoder": {
            "archs": [[16, 12, latent_dim], [12, 10, latent_dim], [10, 8, latent_dim]],
            "activations": ["relu", "relu", "relu"],
            "batchnorm": True,
        },
        "training": {
            "seed": 0,
            "batch_size": 8,
            "epoch": 100,
            "lr": 1e-3,
            "mmi_weight": 0.4,
            "cluster_weight": 0.3,
            "rec_weight": 1.0,
            "diff_weight": 0.05,
            "ce_weight": 0.2,
            "hc_weight": 0.2,
            "n_clusters": 2,
            "noise_scale": 0.02,
            "n_eval": 10,
            "save_eval_checkpoint": False,
        },
        "diffusion": {
            "emb_size": latent_dim,
            "time_type": "sinusoidal",
            "out_dims": [latent_dim] * 3,
        },
        "noise_scheduler": {
            "num_timesteps": 60,
            "beta_schedule": "linear",
        },
        "print_num": 10,
    }


# weight scheduler wrapper
def apply_weight_schedule(config, epoch):
    warmup = 30

    scale = min(1.0, epoch / warmup)

    config["training"]["cluster_weight"] = 0.3 * scale
    config["training"]["mmi_weight"] = 0.4 * scale


# Single Run
def run_single(seed):
    set_seed(seed)

    views_np, labels = create_synthetic_views()
    mask_np = create_mask(len(labels), 3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    views = [torch.from_numpy(v).float().to(device) for v in views_np]
    mask = torch.from_numpy(mask_np).long().to(device)

    config = build_config()
    config["training"]["seed"] = seed

    model = ICDM_Model(config, num_views=3)
    model.to_device(device)

    # Pretraining autoencoders before main training loop
    pretrain_autoencoders(model, views, mask, device, epochs=30)

    collapse_stats = {"worst_ratio": 1.0, "epoch": 0}

    def monitor_training(epoch, scores, trained_model):
        apply_weight_schedule(config, epoch)

        per_view = np.asarray(trained_model.debug_rec_loss_per_view_epoch, dtype=np.float64)
        if per_view.size > 0:
            ratio = float(per_view.max() / max(per_view.min(), 1e-8))
            if ratio > collapse_stats["worst_ratio"]:
                collapse_stats["worst_ratio"] = ratio
                collapse_stats["epoch"] = epoch

    optimizer = torch.optim.Adam(
        itertools.chain(
            model.autoencoders.parameters(),
            model.dfs.parameters(),
            model.clusterLayer.parameters(),
            model.AttentionLayer.parameters(),
        ),
        lr=config["training"]["lr"],
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )

    model.train(config, views, [labels], mask, optimizer, device, eval_callback=monitor_training)

    scores = model.evaluation(config, mask, views, [labels], device)
    scores["worst_view_rec_ratio"] = collapse_stats["worst_ratio"]

    return scores


# Multi-seed Evaluation
def run_experiment(seeds=(0, 1, 42, 123)):
    results = []

    print("\nRunning IMVC stability experiment...\n")

    for seed in seeds:
        scores = run_single(seed)

        print(
            f"[Seed {seed}] "
            f"ACC={scores['accuracy']:.4f} | "
            f"NMI={scores['NMI']:.4f} | "
            f"ARI={scores['ARI']:.4f} | "
            f"F1={scores['f_measure']:.4f} | "
            f"ViewRecRatio={scores['worst_view_rec_ratio']:.2f}"
        )

        results.append(scores)

    def agg(metric):
        values = np.array([float(r[metric]) for r in results])
        return values.mean(), values.std()

    print("\n===== FINAL RESULTS (mean +/- std) =====")

    for metric in ["accuracy", "NMI", "ARI", "f_measure", "worst_view_rec_ratio"]:
        mean, std = agg(metric)
        print(f"{metric.upper():<20}: {mean:.4f} +/- {std:.4f}")

    print("========================================\n")


# Main
if __name__ == "__main__":
    run_experiment()




    # print("\n===== FINAL RESULTS =====")
    # for metric in ["accuracy", "NMI", "ARI", "f_measure"]:
    #     vals = np.array([r[metric] for r in results])
    #     print(f"{metric}: {vals.mean():.4f} ± {vals.std():.4f}")