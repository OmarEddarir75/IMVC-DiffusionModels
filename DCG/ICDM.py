from sklearn.utils import shuffle
from pathlib import Path

from loss import *
from evaluation import evaluation
import numpy as np
from baseModels import *
from util import target_l2


class ICDM_Model(nn.Module):    
    def __init__(self, config, num_views=None):
        super().__init__()
        self._config = config

        archs, activations = self._parse_autoencoder_config(config['Autoencoder'], num_views)
        out_dims = self._parse_diffusion_config(config['diffusion'], len(archs))
        self._num_views = len(archs)

        if len(activations) != self._num_views or len(out_dims) != self._num_views:
            raise ValueError('Inconsistent number of views in config!')

        latent_dims = [arch[-1] for arch in archs]
        if len(set(latent_dims)) != 1:
            raise ValueError('Inconsistent latent dim!')

        self._latent_dim = latent_dims[0]
        self.autoencoders = nn.ModuleList([
            Autoencoder(arch, activation, config['Autoencoder']['batchnorm'])
            for arch, activation in zip(archs, activations)
        ])
        self.dfs = nn.ModuleList([
            Unet(config['diffusion']['emb_size'], config['diffusion']['time_type'], out_dim)
            for out_dim in out_dims
        ])
        self.noise_scheduler = NoiseScheduler(config['noise_scheduler']['num_timesteps'])
        self.clusterLayer = ClusterProject(self._latent_dim, config['training']['n_clusters'])
        self.AttentionLayer = AttentionLayer(self._latent_dim)

        for idx, autoencoder in enumerate(self.autoencoders[:2], start=1):
            setattr(self, f'autoencoder{idx}', autoencoder)
        for idx, diffusion in enumerate(self.dfs[:2], start=1):
            setattr(self, f'df{idx}', diffusion)

    def _expand_per_view(self, values, num_views):
        values = list(values)
        if not values:
            return values
        if len(values) >= num_views:
            return values[:num_views]
        return values + [values[-1]] * (num_views - len(values))

    def _parse_autoencoder_config(self, autoencoder_config, num_views=None):
        if 'archs' in autoencoder_config:
            archs = autoencoder_config['archs']
            activations = autoencoder_config.get('activations', 'relu')
        else:
            arch_keys = sorted(
                [key for key in autoencoder_config if key.startswith('arch')],
                key=lambda key: int(key[4:])
            )
            activation_keys = sorted(
                [key for key in autoencoder_config if key.startswith('activations')],
                key=lambda key: int(key[11:])
            )
            archs = [autoencoder_config[key] for key in arch_keys]
            activations = [autoencoder_config[key] for key in activation_keys]

        archs = list(archs)
        if num_views is not None:
            archs = self._expand_per_view(archs, num_views)

        if not isinstance(activations, (list, tuple)):
            activations = [activations] * len(archs)
        else:
            activations = list(activations)
            if num_views is not None:
                activations = self._expand_per_view(activations, len(archs))

        return archs, activations

    def _parse_diffusion_config(self, diffusion_config, num_views=None):
        if 'out_dims' in diffusion_config:
            out_dims = list(diffusion_config['out_dims'])
        else:
            out_dim_keys = sorted(
                [key for key in diffusion_config if key.startswith('out_dim')],
                key=lambda key: int(key[7:])
            )
            out_dims = [diffusion_config[key] for key in out_dim_keys]

        if num_views is not None:
            out_dims = self._expand_per_view(out_dims, num_views)
        return out_dims

    def _parse_train_args(self, args):
        if len(args) == 5 and isinstance(args[0], (list, tuple)):
            views, Y_list, mask, optimizer, device = args
            return list(views), Y_list, mask, optimizer, device
        if len(args) == 6:
            x1_train, x2_train, Y_list, mask, optimizer, device = args
            return [x1_train, x2_train], Y_list, mask, optimizer, device
        raise TypeError('train expects either (views, Y_list, mask, optimizer, device) or (x1, x2, Y_list, mask, optimizer, device).')

    def _parse_eval_args(self, args):
        if len(args) == 4 and isinstance(args[1], (list, tuple)):
            mask, views, Y_list, device = args
            return mask, list(views), Y_list, device
        if len(args) == 5:
            mask, x1_train, x2_train, Y_list, device = args
            return mask, [x1_train, x2_train], Y_list, device
        raise TypeError('evaluation expects either (mask, views, Y_list, device) or (mask, x1, x2, Y_list, device).')

    def _empty_latent(self, device):
        return torch.empty((0, self._latent_dim), device=device)

    def _get_loss_weights(self, config):
        training_cfg = config['training']

        return {
            'rec': training_cfg.get('rec_weight', 1.0),
            # 'rec': 1.0,
            'mmi': training_cfg.get('mmi_weight', 1.0),
            # 'mmi': 1.0,
            'diff': training_cfg.get('diff_weight', 1.0),
            # 'diff': 1.0,
            'cluster': training_cfg.get('cluster_weight', 1.0),
            # 'cluster': 1.0,
            'hc': training_cfg.get('hc_weight', training_cfg.get('cluster_weight', 1.0)),
            # 'hc': 1.0,
            'ce': training_cfg.get('ce_weight', 1.0),
            # 'ce': 1.0,
        }

    def _view_condition(self, batch_size, view_idx, device, dtype):
        positions = torch.arange(1, self._latent_dim + 1, device=device, dtype=dtype)
        phase = (view_idx + 1) * positions / float(self._latent_dim)
        condition = torch.sin(phase).unsqueeze(0)
        return condition.expand(batch_size, -1)

    def _diffusion_forward(self, df, latent, timestep, view_idx):
        conditioned_latent = latent + 0.01 * self._view_condition(
            latent.shape[0], view_idx, latent.device, latent.dtype
        )
        return df(conditioned_latent, timestep)

    def _recover_latent(self, latent, df, view_idx, device):
        if latent.size(0) == 0:
            return latent

        recovered = latent
        for t in range(len(self.noise_scheduler) - 1, -1, -1):
            timestep = torch.full((recovered.shape[0],), t, dtype=torch.long, device=device)
            with torch.no_grad():
                denoised = self._diffusion_forward(df, recovered, timestep, view_idx)
            recovered = self.noise_scheduler.step(denoised, timestep[0], recovered)
        return recovered

    def _is_degenerate_view(self, latent):
        if latent.size(0) <= 1:
            return True
        return torch.all(latent.std(dim=0) < 1e-6).item()

    def _iter_batches(self, views, mask, batch_size):
        shuffled = shuffle(*views, *[mask[:, idx] for idx in range(self._num_views)])
        shuffled_views = list(shuffled[:self._num_views])
        shuffled_masks = list(shuffled[self._num_views:])
        total = views[0].shape[0]

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_views = [view[start:end] for view in shuffled_views]
            batch_masks = [view_mask[start:end] for view_mask in shuffled_masks]
            yield batch_views, batch_masks

    def _diffusion_friendly_imputation(self, batch_views, batch_masks, noise_scale):
        imputed_views = []

        for batch_view, batch_mask in zip(batch_views, batch_masks):
            imputed_view = batch_view.clone()
            observed = batch_mask == 1
            missing = ~observed
            if not missing.any():
                imputed_views.append(imputed_view)
                continue

            if observed.any():
                observed_mean = batch_view[observed].mean(dim=0, keepdim=True)
            else:
                observed_mean = batch_view.mean(dim=0, keepdim=True)

            if noise_scale > 0:
                noise = torch.randn(
                    (int(missing.sum().item()), batch_view.shape[1]),
                    device=batch_view.device,
                    dtype=batch_view.dtype
                ) * noise_scale
                imputed_view[missing] = observed_mean + noise
            else:
                imputed_view[missing] = observed_mean.expand(int(missing.sum().item()), -1)

            imputed_views.append(imputed_view)

        return imputed_views

    def to_device(self, device):
        self.autoencoders.to(device)
        self.dfs.to(device)
        self.clusterLayer.to(device)
        self.AttentionLayer.to(device)

    def _compute_latent_codes(self, mask, views, device):
        num_samples = views[0].shape[0]
        latent_codes = [torch.zeros(num_samples, self._latent_dim, device=device) for _ in range(self._num_views)]

        for view_idx, (autoencoder, batch_view) in enumerate(zip(self.autoencoders, views)):
            observed = mask[:, view_idx] == 1
            if observed.any():
                latent_codes[view_idx][observed] = autoencoder.encoder(batch_view[observed])

        for target_idx, diffusion in enumerate(self.dfs):
            missing = mask[:, target_idx] == 0
            if not missing.any():
                continue

            source_count = (mask[:, torch.arange(self._num_views, device=device) != target_idx] == 1).sum(dim=1)
            valid_missing = missing & (source_count > 0)
            if not valid_missing.any():
                continue

            fused_source = torch.zeros(valid_missing.sum(), self._latent_dim, device=device)
            for source_idx in range(self._num_views):
                if source_idx == target_idx:
                    continue
                source_available = valid_missing & (mask[:, source_idx] == 1)
                if source_available.any():
                    fused_source[source_available[valid_missing]] += latent_codes[source_idx][source_available]
            fused_source = fused_source / source_count[valid_missing].unsqueeze(1).to(fused_source.dtype)
            latent_codes[target_idx][valid_missing] = self._recover_latent(fused_source, diffusion, target_idx, device)

        return latent_codes

    def _save_eval_checkpoint(self, config, epoch, scores, optimizer=None):
        training_cfg = config['training']
        checkpoint_dir = Path(training_cfg.get('checkpoint_dir', Path('DCG') / 'checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"icdm_eval_epoch{epoch}.pt"

        payload = {
            'epoch': epoch,
            'scores': scores,
            'autoencoders_state_dict': self.autoencoders.state_dict(),
            'dfs_state_dict': self.dfs.state_dict(),
            'attention_state_dict': self.AttentionLayer.state_dict(),
            'cluster_state_dict': self.clusterLayer.state_dict(),
        }
        if optimizer is not None:
            payload['optimizer_state_dict'] = optimizer.state_dict()
            torch.save(payload, checkpoint_path)

    def _zero_loss(self, device):
        return torch.zeros(1, device=device).squeeze()

    def train(self, config, *args, eval_callback=None):
        # Parse training inputs: views, labels, mask, optimizer, device
        views, Y_list, mask, optimizer, device = self._parse_train_args(args)

        # Initialize cluster loss criterion
        criterion_cluster = ClusterLoss(config['training']['n_clusters'], 0.5, device).to(device)

        # Get loss weights from config
        loss_weights = self._get_loss_weights(config)

        # Track best scores
        best_acc, best_nmi, best_ari = 0, 0, 0

        # Training parameters
        training_cfg = config['training']
        n_eval = max(1, int(training_cfg.get('n_eval', config.get('print_num', 1))))
        save_eval_checkpoint = bool(training_cfg.get('save_eval_checkpoint', False))
        noise_scale = float(training_cfg.get('noise_scale', 0.1))

        for epoch in range(config['training']['epoch'] + 1):
            # Initialize loss accumulators
            loss_all, loss_rec, loss_mmi, loss_df, loss_cluster, loss_hc, loss_ce = 0, 0, 0, 0, 0, 0, 0
            rec_loss_per_view_epoch = [0.0] * self._num_views

            # Iterate over batches
            for batch_views, batch_masks in self._iter_batches(views, mask, config['training']['batch_size']):
                if batch_views[0].shape[0] <= 1:
                    continue  # Skip tiny batches

                batch_views = self._diffusion_friendly_imputation(batch_views, batch_masks, noise_scale)

                view_mask_tensor = torch.stack(batch_masks, dim=1).to(device=device, dtype=torch.bool)

                # Prepare latent storage
                latent_bank = []
                diffusion_loss = self._zero_loss(device)

                # Diffusion & latent computation per view
                for view_idx, (autoencoder, diffusion, batch_view, batch_mask) in enumerate(
                    zip(self.autoencoders, self.dfs, batch_views, batch_masks)
                ):
                    observed = batch_mask == 1
                    latent_full = torch.zeros(batch_view.shape[0], self._latent_dim, device=device)

                    if observed.any():
                        # Encode observed entries
                        latent = autoencoder.encoder(batch_view[observed])
                        if latent.device != device:
                            latent = latent.to(device=device, non_blocking=True)
                        latent_full[observed] = latent

                        # Add diffusion noise
                        noise = torch.randn_like(latent)
                        timesteps = torch.randint(0, config['noise_scheduler']['num_timesteps'], (latent.shape[0],), device=device).long()
                        noisy = self.noise_scheduler.add_noise(latent, noise, timesteps, device)

                        # Predict noise using diffusion model
                        noise_pred = self._diffusion_forward(diffusion, noisy, timesteps, view_idx)
                        noise_diff = noise_pred - noise
                        diffusion_loss += (noise_diff ** 2).mean(dim=1).mean()

                    latent_bank.append(latent_full)

                # Determine usable samples (at least one view observed)
                available_counts = view_mask_tensor.sum(dim=1)
                usable_mask = available_counts > 0
                usable_count = int(usable_mask.sum().item())
                if usable_count <= 1:
                    continue

                # Fuse latent representations across views
                fused_latent = self.AttentionLayer(latent_bank, mask=view_mask_tensor)
                
                # Reconstruction Loss
                rec_loss_per_view_batch = [0.0] * self._num_views
                rec_loss_per_view_epoch = rec_loss_per_view_epoch  # keep accumulation

                per_view_losses_tensor = []

                for view_idx, (autoencoder, latent_full, batch_view, batch_mask) in enumerate(
                    zip(self.autoencoders, latent_bank, batch_views, batch_masks)
                ):
                    observed = batch_mask == 1
                    if observed.any():
                        reconstructed = autoencoder.decoder(latent_full[observed])

                        # normalized MSE (dimension-invariant)
                        diff = reconstructed - batch_view[observed]
                        view_reconstruction_loss = (diff ** 2).mean(dim=1).mean()

                        # store for balancing
                        per_view_losses_tensor.append(view_reconstruction_loss)

                        # logging
                        loss_val = view_reconstruction_loss.detach().item()
                        rec_loss_per_view_batch[view_idx] = loss_val
                        rec_loss_per_view_epoch[view_idx] += loss_val

                # FINAL reconstruction loss with balancing
                if len(per_view_losses_tensor) > 0:
                    per_view_losses_tensor = torch.stack(per_view_losses_tensor)

                    # mean reconstruction across views
                    reconstruction_loss = per_view_losses_tensor.mean()

                    # Enforce balance between views
                    balance_loss = torch.var(per_view_losses_tensor)

                    # weight can be tuned (0.05–0.2 range)
                    reconstruction_loss = reconstruction_loss + 0.1 * balance_loss
                else:
                    reconstruction_loss = self._zero_loss(device)

                # debug tracking
                self.debug_rec_loss_per_view_batch = rec_loss_per_view_batch

                # Mutual Information Loss (MMI)
                mmi_terms = []
                for latent_full, batch_mask in zip(latent_bank, batch_masks):
                    observed = batch_mask == 1
                    if observed.sum() <= 1:
                        continue
                    latent = latent_full[observed]
                    fused_observed = fused_latent[observed]
                    if self._is_degenerate_view(latent) or self._is_degenerate_view(fused_observed):
                        continue
                    mmi_terms.append(MMI(fused_observed, latent))
                mmi_loss = sum(mmi_terms) / len(mmi_terms) if mmi_terms else self._zero_loss(device)

                # Cluster Loss + HC Loss
                fused_cluster, _ = self.clusterLayer(fused_latent[usable_mask])
                cluster_terms = []
                y_sum = fused_cluster.clone()
                y_count = torch.ones_like(available_counts[usable_mask], dtype=fused_cluster.dtype)

                for latent_full, batch_mask in zip(latent_bank, batch_masks):
                    observed = batch_mask == 1
                    if not observed.any():
                        continue
                    cluster_output, _ = self.clusterLayer(latent_full[observed])

                    shared = usable_mask & observed
                    shared_usable = shared[usable_mask]
                    if shared_usable.any():
                        y_sum[shared_usable] += cluster_output[shared[observed]]
                        y_count[shared_usable] += 1.0

                    shared_count = int(shared.sum().item())
                    if shared_count > config['training']['n_clusters']:
                        cluster_terms.append(criterion_cluster(cluster_output[shared[observed]], fused_cluster[shared_usable].detach()))

                cluster_loss = sum(cluster_terms) / len(cluster_terms) if cluster_terms else self._zero_loss(device)
                y_mean = target_l2(y_sum / y_count.unsqueeze(1))
                fused_cluster = torch.clamp(fused_cluster, min=EPS)
                hc_loss = F.kl_div(fused_cluster.log(), y_mean.detach(), reduction='batchmean')

                # ===== Cross-View Consistency Loss (CE) =====
                ce_terms = []
                stacked_latents = torch.stack(latent_bank, dim=0)
                available_float = view_mask_tensor.to(stacked_latents.dtype)
                masked_latents = stacked_latents * available_float.T.unsqueeze(-1)
                latent_sum = masked_latents.sum(dim=0)

                for view_idx, diffusion in enumerate(self.dfs):
                    target_observed = view_mask_tensor[:, view_idx]
                    source_count = available_counts - target_observed.long()
                    valid = target_observed & (source_count > 0)
                    valid_count = int(valid.sum().item())
                    if valid_count <= 1:
                        continue
                    valid_indices = valid.nonzero(as_tuple=True)[0]
                    source_sum = latent_sum - masked_latents[view_idx]
                    source_latent = source_sum.index_select(0, valid_indices) / source_count.index_select(0, valid_indices).unsqueeze(1).to(source_sum.dtype)
                    recovered_latent = self._recover_latent(source_latent, diffusion, view_idx, device)
                    criterion_instance = InstanceLoss(valid_count, 1.0, device).to(device)
                    ce_terms.append(criterion_instance(recovered_latent, fused_latent.index_select(0, valid_indices).detach()))

                ce_loss = sum(ce_terms) / len(ce_terms) if ce_terms else self._zero_loss(device)

                # CE warmup
                ce_weight = loss_weights['ce']
                if epoch < 20:
                    ce_weight = ce_weight * (epoch / 20.0)

                # ===== Total Loss =====
                loss = (
                    loss_weights['rec'] * reconstruction_loss
                    + loss_weights['diff'] * diffusion_loss
                    + ce_weight * ce_loss
                    + loss_weights['mmi'] * mmi_loss
                    + loss_weights['cluster'] * cluster_loss
                    + loss_weights['hc'] * hc_loss
                )

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()

                # Accumulate metrics for logging
                loss_all += loss.item()
                loss_rec += reconstruction_loss.item()
                loss_df += diffusion_loss.item()
                loss_mmi += mmi_loss.item()
                loss_cluster += cluster_loss.item()
                loss_hc += hc_loss.item()
                loss_ce += ce_loss.item()

            # ===== Logging per epoch =====
            if epoch % config['print_num'] == 0:
                print(
                    f"Epoch: {epoch}/{config['training']['epoch']} ==> loss = {loss_all:.4f} "
                    f"| rec_loss = {loss_rec:.4f} | df_loss = {loss_df:.4f} | mmi_loss = {loss_mmi:.4f} "
                    f"| cluster_loss = {loss_cluster:.4f} | hc_loss = {loss_hc:.4f} | ce_loss = {loss_ce:.4f}"
                )
            self.debug_rec_loss_per_view_epoch = rec_loss_per_view_epoch

            # ===== Evaluation per n_eval epochs =====
            if epoch % n_eval == 0:
                scores = self.evaluation(config, mask, views, Y_list, device)
                if eval_callback is not None:
                    eval_callback(epoch, scores, self)

                if save_eval_checkpoint:
                    self._save_eval_checkpoint(config, epoch, scores, optimizer=optimizer)

                if scores['accuracy'] >= best_acc:
                    best_acc, best_nmi, best_ari = scores['accuracy'], scores['NMI'], scores['ARI']

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
