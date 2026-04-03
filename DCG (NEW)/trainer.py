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
    z_fused = attention_layer(*latents, mask=mask)      # (B, latent_dim)
    return z_fused


def _build_condition(attention_layer, latents, mask, source_mode=None):
    """
    Build the diffusion condition from observed view latents.

    source_mode="mean"      -> average over observed views only (optimized for fully-observed samples)
    source_mode="attention" -> use the attention layer with masking
    """
    if source_mode is None:
        raise ValueError("source_mode must be provided explicitly ('mean' or 'attention').")

    if source_mode == "mean":
        stacked = torch.stack(latents, dim=1)  # (B, n_views, latent_dim)
        
        # Optimization: check if all samples are fully-observed
        n_views = mask.shape[1]
        fully_observed = (mask.sum(dim=1) == n_views)
        
        if fully_observed.all():
            # Fast path: simple mean, no masking needed
            return stacked.mean(dim=1)
        
        # Slow path: mask-based averaging for partially-observed samples
        present = mask.unsqueeze(-1).float()
        denom = torch.clamp(present.sum(dim=1), min=1.0)
        return (stacked * present).sum(dim=1) / denom

    if source_mode == "attention":
        return attention_layer(*latents, mask=mask)

    raise ValueError(f"Unknown source_mode: {source_mode}. Expected 'mean' or 'attention'.")


def _impute_missing(imputer, attention_layer, latents, mask, source_mode=None):
    """
    For samples that have at least one missing view, replace the conditioned
    latent with the diffusion-imputed version. Fully-observed samples are unchanged.
    
    Optimization: only call imputer when needed.
    """
    z_condition = _build_condition(attention_layer, latents, mask, source_mode=source_mode)
    
    # Early exit if no missing views
    n_views = mask.shape[1]
    missing_mask = (mask.sum(dim=1) < n_views)   # (B,) True = has missing view
    if not missing_mask.any():
        return z_condition

    # Impute only missing-view samples
    z_partial = z_condition[missing_mask]
    z_imputed = imputer.impute(z_partial, mask=mask[missing_mask])  # diffusion reverse pass
    z_condition = z_condition.clone()
    z_condition[missing_mask] = z_imputed
    return z_condition


# Training Phase — Joint training with all losses active
def train(
    autoencoders,
    attention_layer,
    dfs,                   # list of per-view DFs
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
    source_mode=None,
    return_history=True,
    verbose=True,
):
    """
    Joint phase: losses grouped as LR, LD, LI, LC
    """
    if source_mode is None:
        raise ValueError("source_mode must be provided explicitly ('mean' or 'attention').")

    N = x_views[0].shape[0]
    n_views = len(x_views)

    dataset = TensorDataset(*x_views, mask)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    cluster_loss_fn = ClusterLoss(class_num=n_clusters, temperature=cluster_temperature, device=device)

    # train mode
    for ae in autoencoders: ae.train()
    attention_layer.train()
    for unet in dfs: unet.train()
    view_head.train()

    history_rows = []

    if verbose:
        global_freq = mask.float().mean(dim=0).detach().cpu().tolist()
        freq_msg = " | ".join([f"V{idx}:{freq:.2%}" for idx, freq in enumerate(global_freq)])
        print(f"Per-view presence (dataset): {freq_msg}")

    for epoch in range(n_epochs):
        totals = {
            "LOSS": 0.0,
            "LR": 0.0,
            "LD": 0.0,
            "LD_DIFF": 0.0,
            "LD_CE": 0.0,
            "LI": 0.0,
            "LC": 0.0,
            "LC_CLUSTER": 0.0,
            "LC_HC": 0.0,
        }
        n_batches = 0
        epoch_view_present = torch.zeros(n_views, device=device)
        epoch_total_samples = 0

        for *batch_views, batch_mask in loader:
            batch_views = [bv.to(device) for bv in batch_views]
            batch_mask  = batch_mask.to(device)
            B           = batch_views[0].shape[0]
            epoch_view_present += batch_mask.sum(dim=0)
            epoch_total_samples += B

            # Encode per-view Z_v, zero missing
            latents, recons = _encode_views(autoencoders, batch_views, batch_mask)

            # LR: reconstruction loss on present views only
            LR = reconstruction_loss_fn(recons, batch_views, batch_mask)

            # LI: MMI loss on present views only, averaged across views
            # z_condition includes condition building + imputation for missing views
            z_condition = _impute_missing(imputer, attention_layer, latents, batch_mask, source_mode=source_mode,)

            count = 0
            LI = torch.tensor(0.0, device=device)
            for v in range(n_views):
                present = batch_mask[:, v].bool()
                if present.sum() > 1:
                    LI += MMI(z_condition[present], latents[v][present], temperature=mmi_temperature)
                    count += 1
            LI = LI / max(count, 1)

            # LD: diffusion + contrastive
            fully_observed = (batch_mask.sum(dim=1) == n_views)
            LD = torch.tensor(0.0, device=device)
            if fully_observed.sum() > 1:
                # Fully-observed latents once: (n_views, Bf, latent_dim)
                full_latents = torch.stack([lat[fully_observed] for lat in latents], dim=0)
                b_full = full_latents.shape[1]

                if source_mode not in ("mean", "attention"):
                    raise ValueError(f"Unknown source_mode: {source_mode}. Expected 'mean' or 'attention'.")

                full_view_list = [full_latents[u] for u in range(n_views)]
                if source_mode == "attention" and n_views > 1:
                    _, full_weights = attention_layer(*full_view_list, return_weights=True)  # (Bf, n_views)
                else:
                    full_weights = None

                # Conservative training settings for recovery branch.
                paired_subset_size = min(32, b_full)
                paired_idx = torch.randperm(b_full, device=device)[:paired_subset_size]
                paired_latents = full_latents[:, paired_idx]
                b_pair = paired_latents.shape[1]

                paired_view_list = [paired_latents[u] for u in range(n_views)]
                if source_mode == "attention" and n_views > 1:
                    _, paired_weights = attention_layer(*paired_view_list, return_weights=True)  # (Bp, n_views)
                else:
                    paired_weights = None

                # Precompute leave-one-out helpers to avoid O(N^2) slicing overhead.
                if n_views > 1:
                    full_sum = full_latents.sum(dim=0)
                    paired_sum = paired_latents.sum(dim=0)
                    if source_mode == "attention":
                        full_weighted_sum = (full_latents.permute(1, 0, 2) * full_weights.unsqueeze(-1)).sum(dim=1)
                        paired_weighted_sum = (paired_latents.permute(1, 0, 2) * paired_weights.unsqueeze(-1)).sum(dim=1)

                diff_loss = 0.0
                ce_loss = torch.tensor(0.0, device=device)
                ce_pairs = 0
                instance_loss_fn = InstanceLoss(batch_size=b_pair, temperature=contrastive_temp, device=device)
                t_vectors = [
                    torch.full((b_pair,), tt, dtype=torch.long, device=device)
                    for tt in reversed(range(len(scheduler)))
                ]

                paired_full_mask = torch.ones((b_pair, n_views), device=device, dtype=batch_mask.dtype)
                paired_condition = _build_condition(
                    attention_layer,
                    paired_view_list,
                    paired_full_mask,
                    source_mode=source_mode,
                ).detach()

                for v, _ in enumerate(dfs):
                    # Denoising source from observed latents excluding target view v.
                    if n_views > 1:
                        if source_mode == "mean":
                            z_source = (full_sum - full_latents[v]) / float(n_views - 1)
                        else:
                            w_v = full_weights[:, v].unsqueeze(-1)
                            denom = (1.0 - w_v).clamp_min(1e-8)
                            z_source = (full_weighted_sum - full_latents[v] * w_v) / denom
                    else:
                        z_source = full_latents[v]

                    noise = torch.randn_like(z_source)
                    t = torch.randint(0, scheduler.num_timesteps, (b_full,), device=device)
                    z_noisy = scheduler.add_noise(z_source, noise, t, device=device)
                    noise_pred = imputer.predict_noise(z_noisy, t, v)
                    diff_loss += F.mse_loss(noise_pred, noise)

                    # Recover target view v from other views only (exclude v), trainable on a small paired subset.
                    if n_views <= 1:
                        continue
                    if source_mode == "mean":
                        v_recov = ((paired_sum - paired_latents[v]) / float(n_views - 1)).clone()
                    else:
                        w_v = paired_weights[:, v].unsqueeze(-1)
                        denom = (1.0 - w_v).clamp_min(1e-8)
                        v_recov = ((paired_weighted_sum - paired_latents[v] * w_v) / denom).clone()
                    
                    for t_vec in t_vectors:
                        v_d = imputer.predict_noise(v_recov, t_vec, v)
                        v_recov = scheduler.step(v_d, int(t_vec[0].item()), v_recov)

                    if v_recov.size(0) > 0:
                        # O(N) contrastive alignment: each recovered view aligns to fused paired target.
                        ce_loss += instance_loss_fn(v_recov, paired_condition)
                        ce_pairs += 1

                diff_loss = diff_loss / n_views
                if ce_pairs > 0:
                    ce_loss = ce_loss / ce_pairs
                # Keep cross-view contrastive as a lighter regularizer.
                LD = diff_loss + 1.0 * ce_loss

            # LC: cluster-level consistency + high-confidence pseudo-label supervision
            # O(N) clustering alignment: align each view distribution to fused-condition distribution.
            cluster_loss = torch.tensor(0.0, device=device)
            cluster_terms = 0
            for v in range(n_views):
                present = batch_mask[:, v].bool()
                if present.sum() < n_clusters:
                    continue
                q_v, _ = view_head(latents[v][present])
                q_fused, _ = view_head(z_condition[present].detach())
                cluster_loss += cluster_loss_fn(q_v, q_fused)
                cluster_terms += 1
            if cluster_terms > 0:
                cluster_loss = cluster_loss / cluster_terms

            # --- KL pseudo-label loss with warm-up ---
            # if epoch < 10:
            #     high_conf_loss = torch.tensor(0.0, device=device)
            # else:
            #     high_conf_loss = high_conf_loss_fn(view_head=view_head, latents=latents, mask=batch_mask, threshold=conf_threshold)
            high_conf_loss = high_conf_loss_fn(
                view_head=view_head,
                latents=latents,
                mask=batch_mask,
                threshold=conf_threshold,
                aggregation="entropy_weighted",
            )
            LC = cluster_loss + 0.5 * high_conf_loss

            # total loss
            loss = lamda_recon*LR + lamda_mmi*LI + lamda_diff*LD + lamda_cluster*LC

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # losses
            totals["LR"] += LR.item()
            totals["LI"] += LI.item()
            totals["LC"] += LC.item()
            totals["LD"] += LD.item()
            totals["LOSS"] += loss.item()
            totals["LC_CLUSTER"] += cluster_loss.item()
            totals["LC_HC"] += high_conf_loss.item()
            if fully_observed.sum() > 1:
                totals["LD_DIFF"] += float(diff_loss if not torch.is_tensor(diff_loss) else diff_loss.item())
                totals["LD_CE"] += float(ce_loss.item())
            n_batches += 1

        if verbose and (epoch+1) % 1 == 0:
            avg = {k:v/n_batches for k,v in totals.items()}
            epoch_freq = (epoch_view_present / max(epoch_total_samples, 1)).detach().cpu().tolist()
            freq_msg = " | ".join([f"V{idx}:{freq:.2%}" for idx, freq in enumerate(epoch_freq)])
            print(
                f"Epoch {epoch+1}/{n_epochs} ==> LOSS={loss.item():.4f} | "
                f"LR={avg['LR']:.4f} LD={avg['LD']:.4f} LI={avg['LI']:.4f} LC={avg['LC']:.4f} | "
                f"LD_DIFF={avg['LD_DIFF']:.4f} LD_CE={avg['LD_CE']:.4f} LC_CLUSTER={avg['LC_CLUSTER']:.4f} LC_HC={avg['LC_HC']:.4f} | "
                f"presence: {freq_msg}"
            )

        if return_history:
            avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
            history_rows.append(
                {
                    "epoch": epoch + 1,
                    "loss": avg["LOSS"],
                    "rec_loss": avg["LR"],
                    "df_loss": avg["LD_DIFF"],
                    "ce_loss": avg["LD_CE"],
                    "mmi_loss": avg["LI"],
                    "cluster_loss": avg["LC_CLUSTER"],
                    "hc_loss": avg["LC_HC"],
                    "lc_total": avg["LC"],
                    "ld_total": avg["LD"],
                    "accuracy": None,
                    "NMI": None,
                    "ARI": None,
                }
            )

    return history_rows if return_history else None


# Evaluation helper (after both phases)
@torch.no_grad()
def get_embeddings(autoencoders, attention_layer, imputer, x_views, mask, device, batch_size=512, source_mode=None):
    """
    Returns the complete z_fused for all samples, with imputation applied
    for any missing views. Ready to pass to KMeans or evaluation metrics.
    """
    if source_mode is None:
        raise ValueError("source_mode must be provided explicitly ('mean' or 'attention').")

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
        z_fused    = _impute_missing(imputer, attention_layer, latents, batch_mask, source_mode=source_mode)
        all_z.append(z_fused.cpu())

    return torch.cat(all_z, dim=0)   # (N, latent_dim)
