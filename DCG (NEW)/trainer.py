import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# from baseModels import (Autoencoder, AttentionLayer, Unet, NoiseScheduler, MissingViewImputer, CrossViewProjector, ViewClusterHead)
from loss import (MMI, InstanceLoss, ClusterLoss, high_conf_loss_fn, reconstruction_loss_fn)


def _encode_views(autoencoders, x_views, mask):
    """
    Encode each view through its autoencoder.
    Returns:
        latents  : list of (B, latent_dim) tensors — zeroed for missing views
        recons   : list of (B, input_dim_v) tensors — reconstruction of each view
    """
    latents, recons = [], []
    for v, ae in enumerate(autoencoders):
        x_hat, z = ae(x_views[v])           # always forward; mask applied after
        present = mask[:, v].float()         # (B,)
        z     = z     * present.unsqueeze(-1)
        x_hat = x_hat * present.unsqueeze(-1)
        latents.append(z)
        recons.append(x_hat)
    return latents, recons


def _fuse(attention_layer, latents, mask):
    """
    Pass latents through the attention fusion layer.
    Missing-view latents are already zeroed; the attention layer
    will naturally down-weight them via learned scores.
    """
    z_fused = attention_layer(*latents)      # (B, latent_dim)
    return z_fused


def _impute_missing(imputer, z_fused, mask):
    """
    For samples that have at least one missing view, replace z_fused
    with the diffusion-imputed version. Fully-observed samples are unchanged.
    """
    missing_mask = (mask.sum(dim=1) < mask.shape[1])   # (B,) True = has missing view
    if missing_mask.sum() == 0:
        return z_fused

    z_partial   = z_fused[missing_mask]
    z_imputed   = imputer.impute(z_partial, mask=mask[missing_mask])  # diffusion reverse pass
    z_fused     = z_fused.clone()
    z_fused[missing_mask] = z_imputed
    return z_fused


# Phase 1 — Warmup (reconstruction + MMI only, with imputation to include partial samples)
def train_phase1(
    autoencoders,
    attention_layer,
    imputer,
    x_views,            # list of (N, d_v) tensors, already masked
    mask,               # (N, n_views) LongTensor
    optimizer,
    device,
    n_epochs=50,
    batch_size=256,
    mmi_temperature=1.0,
    lamda_recon=1.0,
    lamda_mmi=0.1,
    verbose=True,
):
    """
    Warmup phase: build geometrically stable per-view latents and a
    fused representation before any clustering or diffusion signal.
    Only reconstruction_loss and mmi_loss are active.
    Imputation is applied so partial samples participate.
    """
    N = x_views[0].shape[0]
    dataset = TensorDataset(*x_views, mask)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    for ae in autoencoders:
        ae.train()
    attention_layer.train()

    for epoch in range(n_epochs):
        total_recon = total_mmi = 0.0
        n_batches = 0

        for *batch_views, batch_mask in loader:
            batch_views = [bv.to(device) for bv in batch_views]
            batch_mask  = batch_mask.to(device)

            # Forward
            latents, recons = _encode_views(autoencoders, batch_views, batch_mask)
            z_fused = _fuse(attention_layer, latents, batch_mask)

            # Impute missing samples so they contribute
            z_fused = _impute_missing(imputer, z_fused, batch_mask)

            # Losses
            recon_loss = reconstruction_loss_fn(recons, batch_views, batch_mask)

            mmi_loss = torch.tensor(0.0, device=device)
            n_views  = len(latents)
            for v in range(n_views):
                # Only include samples where view v is present
                present = batch_mask[:, v].bool()
                if present.sum() > 1:
                    mmi_loss = mmi_loss + MMI(z_fused[present], latents[v][present], temperature=mmi_temperature,)
            mmi_loss = mmi_loss / n_views

            loss = lamda_recon * recon_loss + lamda_mmi * mmi_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_mmi   += mmi_loss.item()
            n_batches   += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"[Phase 1] Epoch {epoch+1:>4}/{n_epochs}  "
                f"recon={total_recon/n_batches:.4f}  "
                f"mmi={total_mmi/n_batches:.4f}"
            )


# Phase 2 — Joint training with all losses active
def train_phase2(
    autoencoders,
    attention_layer,
    unets,                   # list of per-view UNets
    scheduler,
    imputer,
    view_head,            # ViewClusterHead (shared across views)
    x_views,
    mask,
    optimizer,
    device,
    n_epochs=100,
    batch_size=256,
    n_clusters=10,
    # loss weights
    lamda_recon=1.0,
    lamda_mmi=0.1,
    lamda_diff=0.5,
    lamda_cluster=1.0,
    # hyperparameters
    mmi_temperature=1.0,
    conf_threshold=0.6,
    cluster_temperature=0.1,
    contrastive_temp=0.1,
    verbose=True,
):
    """
    Joint phase: losses grouped as LR, LD, LI, LC
    """
    N = x_views[0].shape[0]
    n_views = len(x_views)

    dataset = TensorDataset(*x_views, mask)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    cluster_loss_fn = ClusterLoss(class_num=n_clusters, temperature=cluster_temperature, device=device)

    # train mode
    for ae in autoencoders: ae.train()
    attention_layer.train()
    for unet in unets: unet.train()
    view_head.train()

    for epoch in range(n_epochs):
        totals = {"LR":0.0, "LD":0.0, "LI":0.0, "LC":0.0}
        n_batches = 0

        for *batch_views, batch_mask in loader:
            batch_views = [bv.to(device) for bv in batch_views]
            batch_mask  = batch_mask.to(device)
            B           = batch_views[0].shape[0]

            # Encode per-view Z_v, zero missing
            latents, recons = _encode_views(autoencoders, batch_views, batch_mask)
            # Fuse H, impute missing
            z_fused = _fuse(attention_layer, latents, batch_mask)
            z_fused = _impute_missing(imputer, z_fused, batch_mask)

            # LR: reconstruction loss on present views only
            LR = reconstruction_loss_fn(recons, batch_views, batch_mask)

            # LI: MMI loss on present views only, averaged across views
            count = 0
            LI = torch.tensor(0.0, device=device)
            for v in range(n_views):
                present = batch_mask[:, v].bool()
                if present.sum() > 1:
                    LI += MMI(z_fused[present], latents[v][present], temperature=mmi_temperature)
                    count += 1
            LI = LI / max(count, 1)

            # LD: diffusion + contrastive
            fully_observed = (batch_mask.sum(dim=1) == n_views)
            LD = torch.tensor(0.0, device=device)
            if fully_observed.sum() > 1:
                z_clean = z_fused[fully_observed].detach()
                noise = torch.randn_like(z_clean)
                t = torch.randint(0, scheduler.num_timesteps, (z_clean.shape[0],), device=device)
                diff_loss = 0.0
                contrast_loss = 0.0
                for v, unet in enumerate(unets):
                    z_noisy = scheduler.add_noise(z_clean, noise, t, device=device)
                    noise_pred = unet(z_noisy, t.float())
                    diff_loss += F.mse_loss(noise_pred, noise)
                    z_hat0 = scheduler.reconstruct_x0(z_noisy, t, noise_pred)
                    instance_loss_fn = InstanceLoss(batch_size=z_clean.shape[0],
                                                    temperature=contrastive_temp,
                                                    device=device)
                    contrast_loss += instance_loss_fn(z_clean, z_hat0)
                LD = (diff_loss + contrast_loss) / n_views


            # LC: cluster-level consistency + high-confidence pseudo-label supervision
            cluster_loss = torch.tensor(0.0, device=device)
            cluster_pairs = 0
            for i in range(n_views):
                for j in range(i + 1, n_views):
                    both = batch_mask[:, i].bool() & batch_mask[:, j].bool()
                    if both.sum() < n_clusters:
                        continue
                    q_i, _ = view_head(latents[i][both])
                    q_j, _ = view_head(latents[j][both])
                    cluster_loss += cluster_loss_fn(q_i, q_j)
                    cluster_pairs += 1
            if cluster_pairs > 0: cluster_loss = cluster_loss / cluster_pairs

            # --- KL pseudo-label loss with warm-up ---
            if epoch < 10:
                high_conf_loss = torch.tensor(0.0, device=device)
            else:
                high_conf_loss = high_conf_loss_fn(view_head=view_head, latents=latents, mask=batch_mask, threshold=conf_threshold)

            LC = cluster_loss + 0.5 * high_conf_loss

            # total loss
            loss = lamda_recon*LR + lamda_mmi*LI + lamda_diff*LD + lamda_cluster*LC

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            totals["LR"] += LR.item()
            totals["LI"] += LI.item()
            totals["LC"] += LC.item()
            totals["LD"] += LD.item()
            n_batches += 1

        if verbose and (epoch+1) % 10 == 0:
            avg = {k:v/n_batches for k,v in totals.items()}
            print(f"[Phase2] Epoch {epoch+1}/{n_epochs} ==> LOSS={loss.item():.4f} | LR={avg['LR']:.4f}  LD={avg['LD']:.4f}  LI={avg['LI']:.4f}  LC={avg['LC']:.4f}")


# Evaluation helper (after both phases)
@torch.no_grad()
def get_embeddings(autoencoders, attention_layer, imputer, x_views, mask, device, batch_size=512,):
    """
    Returns the complete z_fused for all samples, with imputation applied
    for any missing views. Ready to pass to KMeans or evaluation metrics.
    """
    for ae in autoencoders:
        ae.eval()
    attention_layer.eval()

    N       = x_views[0].shape[0]
    dataset = TensorDataset(*x_views, mask)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_z = []
    for *batch_views, batch_mask in loader:
        batch_views = [bv.to(device) for bv in batch_views]
        batch_mask  = batch_mask.to(device)

        latents, _ = _encode_views(autoencoders, batch_views, batch_mask)
        z_fused    = _fuse(attention_layer, latents, batch_mask)
        z_fused    = _impute_missing(imputer, z_fused, batch_mask)
        all_z.append(z_fused.cpu())

    return torch.cat(all_z, dim=0)   # (N, latent_dim)
