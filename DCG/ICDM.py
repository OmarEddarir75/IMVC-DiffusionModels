# DCG/ICDM.py
from pathlib import Path
from sklearn.utils import shuffle
from torch.optim.lr_scheduler import ReduceLROnPlateau

from util import *
from loss import *
from baseModels import *
from evaluation import evaluation

class DCG(nn.Module):    
    def __init__(self, config, num_views=None, device="cpu"):
        super().__init__()
        self._config = config
        self._device = device

        self.stop_training = False
        self.debug_assignments = []
        self._degenerate_std_threshold = config['training'].get('degenerate_std_threshold', 1e-6)

        archs, activations = self._parse_autoencoder_config(config['Autoencoder'], num_views)
        out_dims = self._parse_diffusion_config(config['diffusion'], len(archs))
        self._num_views = len(archs)

        if len(activations) != self._num_views or len(out_dims) != self._num_views:
            raise ValueError('Inconsistent number of views in config!')

        latent_dims = [arch[-1] for arch in archs]
        if len(set(latent_dims)) != 1:
            raise ValueError('Inconsistent latent dim!')

        self._latent_dim = latent_dims[0]

        self._cached_view_conditions = [
            torch.sin(
                (i + 1) * torch.arange(1, self._latent_dim + 1).float() / self._latent_dim
            )
            for i in range(self._num_views)
        ]

        # Autoencoders
        self.autoencoders = nn.ModuleList([
            Autoencoder(arch, activation, config['Autoencoder']['batchnorm'])
            for arch, activation in zip(archs, activations)
        ])

        # Diffusion models
        self.dfs = nn.ModuleList([
            Unet(
                config['diffusion']['emb_size'],
                config['diffusion']['time_type'],
                out_dim
            )
            for out_dim in out_dims
        ])

        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(
            config['noise_scheduler']['num_timesteps'],
            device=device
        )

        # Shared cluster layer (linear + softmax) – no extra projection head
        init_mode = config['training'].get('cluster_init', 'kmeans_plus_plus')
        self.clusterLayer = ClusterProject(
            self._latent_dim,
            config['training']['n_clusters'],
            init_mode=init_mode
        )

        self.AttentionLayer = AttentionLayer(self._latent_dim)

        self.to_device(device)

    # ---------- (all helper methods unchanged: _parse_autoencoder_config, _parse_diffusion_config,
    #            _parse_train_args, _parse_eval_args, _empty_latent, _view_condition,
    #            _diffusion_forward, _recover_latent, _is_degenerate_view, _iter_batches,
    #            _diffusion_friendly_imputation, to_device, _compute_latent_codes,
    #            _save_eval_checkpoint, _zero_loss) ----------

    # We only show the changed methods: _get_loss_weights, train, and the two new loss functions.

    def _get_loss_weights(self, config):
        training_cfg = config['training']
        return {
            'rec': training_cfg.get('rec_weight', 1.0),
            'mmi': training_cfg.get('mmi_weight', 1.0),
            'diff': training_cfg.get('diff_weight', 1.0),
            'ccl': training_cfg.get('ccl_weight', 1.0),   # category contrastive
            'kl': training_cfg.get('kl_weight', 1.0),     # KL divergence
            'ce': training_cfg.get('ce_weight', 1.0),     # cross-view contrastive (optional)
        }

    def category_contrastive_loss(self, Q_list, tau_C=1.0):
        """
        Category-level contrastive loss (Eq.16).
        Q_list: list of tensors shape (N, K) – soft assignments for each view.
        """
        n_views = len(Q_list)
        if n_views < 2:
            return torch.tensor(0.0, device=Q_list[0].device)
        total_loss = 0.0
        for m in range(n_views):
            for n in range(n_views):
                if m == n:
                    continue
                Q_m = Q_list[m]   # (N, K)
                Q_n = Q_list[n]
                # For each cluster j
                loss_mn = 0.0
                for j in range(self._n_clusters):
                    # Positive: cosine similarity between the j-th column of Q_m and Q_n
                    pos = torch.exp(F.cosine_similarity(Q_m[:, j], Q_n[:, j], dim=0) / tau_C)
                    # Negative: sum over k != j
                    neg_sum = 0.0
                    for k in range(self._n_clusters):
                        if k != j:
                            neg_sum += torch.exp(F.cosine_similarity(Q_m[:, j], Q_n[:, k], dim=0) / tau_C)
                    loss_mn -= torch.log(pos / (pos + neg_sum))
                total_loss += loss_mn
        return total_loss / (n_views * (n_views - 1))

    def kl_clustering_loss(self, Q_fused, Q_views, temperature=1.0):
        """
        KL divergence loss (Eq.18-20).
        Q_fused: soft assignments from fused representation (N, K)
        Q_views: list of soft assignments from each view (N, K)
        """
        # Stack all assignments (fused + views)
        Q_stack = torch.stack([Q_fused] + Q_views, dim=0)  # (V+1, N, K)
        # High-confidence assignment: elementwise max (Eq.18)
        Q_max = Q_stack.max(dim=0)[0]  # (N, K)
        # Target distribution (Eq.19)
        P = (Q_max ** 2) / (Q_max ** 2).sum(dim=1, keepdim=True)
        P = P / temperature   # optional temperature scaling
        # KL divergence between target and fused assignments (Eq.20)
        kl = (P * torch.log(P / (Q_fused + 1e-8))).sum(dim=1).mean()
        return kl

    def train(self, config, *args, eval_callback=None, epoch_callback=None):
        views, Y_list, mask, optimizer, device = self._parse_train_args(args)
        if self._device != device:
            self.to_device(device)
        self.stop_training = False

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        best_acc, best_nmi, best_ari = 0, 0, 0

        training_cfg = config['training']
        n_eval = max(1, int(training_cfg.get('n_eval', config.get('print_num', 1))))
        save_eval_checkpoint = bool(training_cfg.get('save_eval_checkpoint', False))
        noise_scale = float(training_cfg.get('noise_scale', 0.1))
        self._n_clusters = config['training']['n_clusters']

        for epoch in range(config['training']['epoch'] + 1):
            if self.stop_training:
                print(f"Early stopping – exiting at epoch {epoch}")
                break
            if epoch_callback is not None:
                epoch_callback(epoch, self)

            loss_weights = self._get_loss_weights(config)
            loss_all = loss_rec = loss_mmi = loss_df = loss_ccl = loss_kl = loss_ce = 0.0
            rec_loss_per_view_epoch = [0.0] * self._num_views
            num_batches = 0

            for batch_views, batch_masks in self._iter_batches(views, mask, config['training']['batch_size']):
                if batch_views[0].shape[0] <= 1:
                    continue

                # ---- 1. Impute missing views using diffusion (full batch) ----
                # First, encode observed views to latent
                latent_bank = []
                for v_idx, (ae, bv, bm) in enumerate(zip(self.autoencoders, batch_views, batch_masks)):
                    obs = bm == 1
                    latent = torch.zeros(bv.shape[0], self._latent_dim, device=device)
                    if obs.any():
                        latent[obs] = ae.encoder(bv[obs])
                    # For missing, we will impute later
                    latent_bank.append(latent)

                # Impute missing views using diffusion (like in _compute_latent_codes but per batch)
                for target_idx, diffusion in enumerate(self.dfs):
                    missing = batch_masks[target_idx] == 0
                    if not missing.any():
                        continue
                    # Source views: all except target
                    source_count = torch.stack([batch_masks[v] for v in range(self._num_views) if v != target_idx], dim=0).sum(dim=0)
                    valid_missing = missing & (source_count > 0)
                    if not valid_missing.any():
                        continue
                    # Fuse source latents (mean)
                    fused_source = torch.zeros(valid_missing.sum(), self._latent_dim, device=device)
                    for src_idx in range(self._num_views):
                        if src_idx == target_idx:
                            continue
                        src_avail = valid_missing & (batch_masks[src_idx] == 1)
                        if src_avail.any():
                            fused_source[src_avail[valid_missing]] += latent_bank[src_idx][src_avail]
                    fused_source = fused_source / source_count[valid_missing].unsqueeze(1).float()
                    # Recover missing latents
                    recovered = self._recover_latent(fused_source, diffusion, target_idx, device, fast=False)
                    latent_bank[target_idx][valid_missing] = recovered

                # ---- 2. Forward diffusion loss (using only observed) ----
                diffusion_loss = self._zero_loss(device)
                for v_idx, (ae, diffusion, bv, bm, latent) in enumerate(zip(self.autoencoders, self.dfs, batch_views, batch_masks, latent_bank)):
                    obs = bm == 1
                    if obs.any():
                        lat = latent[obs]  # should be same as ae.encoder(bv[obs])
                        noise = torch.randn_like(lat)
                        timesteps = torch.randint(0, config['noise_scheduler']['num_timesteps'], (lat.shape[0],), device=device).long()
                        noisy = self.noise_scheduler.add_noise(lat, noise, timesteps)
                        noise_pred = self._diffusion_forward(diffusion, noisy, timesteps, v_idx)
                        diffusion_loss += F.mse_loss(noise_pred, noise)
                diffusion_loss = diffusion_loss / self._num_views

                # ---- 3. Attention fusion (using fully imputed latents) ----
                view_mask_tensor = torch.stack(batch_masks, dim=1).to(device=device, dtype=torch.bool)
                fused_latent = self.AttentionLayer(latent_bank, mask=view_mask_tensor)

                # ---- 4. Reconstruction loss (only observed) ----
                per_view_losses = []
                for v_idx, (ae, latent, bv, bm) in enumerate(zip(self.autoencoders, latent_bank, batch_views, batch_masks)):
                    obs = bm == 1
                    if obs.any():
                        rec = ae.decoder(latent[obs])
                        loss_v = F.mse_loss(rec, bv[obs])
                        per_view_losses.append(loss_v)
                        rec_loss_per_view_epoch[v_idx] += loss_v.item()
                reconstruction_loss = torch.stack(per_view_losses).mean() if per_view_losses else self._zero_loss(device)

                # ---- 5. MMI loss (instance-level) ----
                mmi_terms = []
                for latent, bm in zip(latent_bank, batch_masks):
                    obs = bm == 1
                    if obs.sum() <= 1:
                        continue
                    lat = latent[obs]
                    fus = fused_latent[obs]
                    if self._is_degenerate_view(lat) or self._is_degenerate_view(fus):
                        continue
                    mmi_terms.append(MMI(F.softmax(fus, dim=1), F.softmax(lat, dim=1)))
                mmi_loss = sum(mmi_terms) / len(mmi_terms) if mmi_terms else self._zero_loss(device)

                # ---- 6. Category-level losses (using all samples) ----
                # Get soft assignments for fused representation (all samples)
                Q_fused, _ = self.clusterLayer(fused_latent)   # (batch, K)
                # Get soft assignments for each view (all samples, but note: missing views were imputed)
                Q_views = []
                for latent in latent_bank:
                    Q_v, _ = self.clusterLayer(latent)
                    Q_views.append(Q_v)
                # Category contrastive loss
                if len(Q_views) >= 2:
                    ccl_loss = self.category_contrastive_loss(Q_views, tau_C=1.0)
                    # KL divergence loss
                    kl_loss = self.kl_clustering_loss(Q_fused, Q_views)
                else:
                    ccl_loss = self._zero_loss(device)
                    kl_loss = self._zero_loss(device)

                # ---- 7. Cross-view consistency loss (optional, similar to paper's gcl) ----
                # This can be kept as is (your ce_loss) but is not essential for stability.
                ce_terms = []
                ce_criterion_cache = {}
                stacked_latents = torch.stack(latent_bank, dim=0)
                available_float = view_mask_tensor.to(stacked_latents.dtype)
                masked_latents = stacked_latents * available_float.T.unsqueeze(-1)
                latent_sum = masked_latents.sum(dim=0)
                for view_idx, diffusion in enumerate(self.dfs):
                    target_observed = view_mask_tensor[:, view_idx]
                    source_count = available_float.sum(dim=1).long() - target_observed.long()
                    valid = target_observed & (source_count > 0)
                    valid_count = int(valid.sum().item())
                    if valid_count <= 1:
                        continue
                    valid_indices = valid.nonzero(as_tuple=True)[0]
                    source_sum = latent_sum - masked_latents[view_idx]
                    source_latent = source_sum.index_select(0, valid_indices) / source_count.index_select(0, valid_indices).unsqueeze(1).to(source_sum.dtype)
                    recovered_latent = self._recover_latent(source_latent, diffusion, view_idx, device, fast=True)
                    criterion = ce_criterion_cache.get(valid_count)
                    if criterion is None:
                        criterion = InstanceLoss(valid_count, 1.0, device).to(device)
                        ce_criterion_cache[valid_count] = criterion
                    ce_terms.append(criterion(recovered_latent, fused_latent.index_select(0, valid_indices).detach()))
                ce_loss = sum(ce_terms) / len(ce_terms) if ce_terms else self._zero_loss(device)

                # ---- 8. Total loss ----
                loss = (loss_weights['rec'] * reconstruction_loss +
                        loss_weights['diff'] * diffusion_loss +
                        loss_weights['mmi'] * mmi_loss +
                        loss_weights['ccl'] * ccl_loss +
                        loss_weights['kl'] * kl_loss +
                        loss_weights['ce'] * ce_loss)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), config['training'].get('grad_clip_norm', 5.0))
                optimizer.step()

                loss_all += loss.item()
                loss_rec += reconstruction_loss.item()
                loss_df += diffusion_loss.item()
                loss_mmi += mmi_loss.item()
                loss_ccl += ccl_loss.item()
                loss_kl += kl_loss.item()
                loss_ce += ce_loss.item()
                num_batches += 1

            if num_batches > 0:
                scheduler.step(reconstruction_loss.detach())

            if epoch % config['print_num'] == 0:
                print(f"Epoch {epoch}/{config['training']['epoch']} => loss={loss_all:.4f} | rec={loss_rec:.4f} df={loss_df:.4f} mmi={loss_mmi:.4f} ccl={loss_ccl:.4f} kl={loss_kl:.4f} ce={loss_ce:.4f}")

            self.debug_rec_loss_per_view_epoch = rec_loss_per_view_epoch

            if epoch % n_eval == 0:
                scores = self.evaluation(config, mask, views, Y_list, device)
                if eval_callback:
                    eval_callback(epoch, scores, self)
                if save_eval_checkpoint:
                    self._save_eval_checkpoint(config, epoch, scores, optimizer)
                with torch.no_grad():
                    latent_codes = self._compute_latent_codes(mask, views, device)
                    latent_fusion = self.AttentionLayer(latent_codes, mask=mask == 1)
                    y, _ = self.clusterLayer(latent_fusion)
                    self.debug_assignments.append((epoch, y.argmax(dim=1).cpu().numpy().copy()))
                if scores['accuracy'] >= best_acc:
                    best_acc, best_nmi, best_ari = scores['accuracy'], scores['NMI'], scores['ARI']
                if self.stop_training:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return best_acc, best_nmi, best_ari

    def evaluation(self, config, *args):
        mask, views, Y_list, device = self._parse_eval_args(args)

        with torch.no_grad():
            self.autoencoders.eval()
            self.dfs.eval()
            latent_codes = self._compute_latent_codes(mask, views, device)
            latent_fusion = self.AttentionLayer(latent_codes, mask=mask == 1)
            y, _ = self.clusterLayer(latent_fusion)
            y = y.data.cpu().numpy().argmax(1)
            scores = evaluation(y_pred=y, y_true=Y_list[0])
            self.autoencoders.train()
            self.dfs.train()

        return scores