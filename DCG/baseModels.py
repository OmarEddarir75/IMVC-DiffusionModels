# DCG/baseModels.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        device = x.device
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.tensor([10000.0], device=device)) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size, device=device))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size

class PositionalEmbedding(nn.Module):
    def __init__(self, size: int, type: str, **kwargs):
        super().__init__()

        if type == "sinusoidal":
            self.layer = SinusoidalEmbedding(size, **kwargs)
        else:
            raise ValueError(f"Unknown positional embedding type: {type}")

    def forward(self, x: torch.Tensor):
        return self.layer(x)

class Unet(nn.Module):
    def __init__(self, emb_size: int = 128,
                 time_emb: str = "sinusoidal", out_size: int = 128):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        concat_size = 2*emb_size
        layers = []
        layers.append(nn.Linear(concat_size, 2000))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(2000, 500))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(500, 500))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(500, 2000))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(2000, out_size))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = torch.cat((x, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x

class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, 
                 beta_schedule="linear", device="cpu"):
        """
        Diffusion noise scheduler.
        
        Args:
            num_timesteps: Number of diffusion steps.
            beta_start: Starting beta value.
            beta_end: Ending beta value.
            beta_schedule: Type of beta schedule ("linear").
            device: Device to store tensors on.
        """
        self.device = device
        self.num_timesteps = num_timesteps

        # Beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        else:
            raise NotImplementedError(f"Beta schedule {beta_schedule} not implemented")
        self.betas = betas.to(device)

        # Precompute alphas
        alphas = 1.0 - self.betas
        self.alphas = alphas.to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)

        # Precompute commonly used buffers
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.sqrt_inv_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / 
                                     (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / 
                                     (1.0 - self.alphas_cumprod))

        # Move all buffers to device
        self._move_to_device(device)

    def _move_to_device(self, device):
        """Move all internal tensors to the specified device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_inv_alphas_cumprod = self.sqrt_inv_alphas_cumprod.to(device)
        self.sqrt_inv_alphas_cumprod_minus_one = self.sqrt_inv_alphas_cumprod_minus_one.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)

    def to_device(self, device):
        """Public method to move scheduler to a different device."""
        self._move_to_device(device)
        return self

    def reconstruct_x0(self, x_t, t, noise):
        """
        Reconstruct x0 from x_t and predicted noise.
        
        Args:
            x_t: Noisy sample at time t (batch, dim)
            t: Timestep (scalar int or tensor)
            noise: Predicted noise (batch, dim)
        
        Returns:
            Reconstructed x0 (batch, dim)
        """
        # Ensure t is a scalar for indexing
        t_idx = t.item() if torch.is_tensor(t) else t
        s1 = self.sqrt_inv_alphas_cumprod[t_idx].reshape(-1, 1)
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t_idx].reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        """Compute posterior mean q(x_{t-1} | x_t, x_0)."""
        t_idx = t.item() if torch.is_tensor(t) else t
        s1 = self.posterior_mean_coef1[t_idx].reshape(-1, 1)
        s2 = self.posterior_mean_coef2[t_idx].reshape(-1, 1)
        return s1 * x_0 + s2 * x_t

    def get_variance(self, t):
        """
        Get variance for timestep t. Handles both scalar and tensor t.
        
        Args:
            t: Scalar int, 0D tensor, or 1D tensor of timesteps.
        
        Returns:
            Variance tensor (same shape as t, or scalar if t is scalar).
        """
        if torch.is_tensor(t):
            # Create output tensor on same device as self.betas
            variance = torch.zeros_like(t, dtype=torch.float32, device=self.device)
            non_zero = (t != 0)
            if non_zero.any():
                t_nonzero = t[non_zero]
                # Ensure indices are on CPU for indexing? No, if t is on GPU, need to move? 
                # Actually, indexing with GPU tensor works if the indexed tensor is also on GPU.
                # But for safety, we can move to CPU for indexing then back. Better: ensure self.betas etc are on same device.
                # Since all buffers are on self.device, and t may be on a different device, we need to handle.
                # We'll assume t is on self.device (caller should ensure). If not, move temporarily.
                orig_device = t.device
                if orig_device != self.device:
                    t_nonzero = t_nonzero.to(self.device)
                var = (self.betas[t_nonzero] * 
                       (1.0 - self.alphas_cumprod_prev[t_nonzero]) / 
                       (1.0 - self.alphas_cumprod[t_nonzero])).clamp(min=1e-20)
                if orig_device != self.device:
                    var = var.to(orig_device)
                variance[non_zero] = var
            return variance
        else:
            # Scalar t
            if t == 0:
                return torch.tensor(0.0, device=self.device)
            variance = self.betas[t] * (1.0 - self.alphas_cumprod_prev[t]) / (1.0 - self.alphas_cumprod[t])
            return variance.clamp(min=1e-20)

    def _step_single(self, model_output, t, sample):
        """
        Single step denoising for a single timestep t (scalar).
        """
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = torch.sqrt(self.get_variance(t))
            pred_prev_sample = pred_prev_sample + variance * noise
        return pred_prev_sample

    def step(self, model_output, timestep, sample):
        """
        Apply one denoising step. Handles both scalar and batched timesteps.
        
        Args:
            model_output: Predicted noise (batch, dim)
            timestep: Scalar int, 0D tensor, or 1D tensor of timesteps (batch,)
            sample: Current noisy sample (batch, dim)
        
        Returns:
            Denoised sample for previous timestep (batch, dim)
        """
        if torch.is_tensor(timestep) and timestep.numel() > 1:
            # Batch of timesteps: check if all same
            unique_ts = torch.unique(timestep)
            if len(unique_ts) == 1:
                # All same, process in batch
                t_val = unique_ts[0].item()
                return self._step_single(model_output, t_val, sample)
            else:
                # Different timesteps, process each group separately
                result = torch.zeros_like(sample)
                for t_val in unique_ts:
                    mask = (timestep == t_val)
                    if mask.any():
                        t_int = t_val.item()
                        result[mask] = self._step_single(model_output[mask], t_int, sample[mask])
                return result
        else:
            # Scalar timestep
            t = timestep.item() if torch.is_tensor(timestep) else timestep
            return self._step_single(model_output, t, sample)

    def add_noise(self, x_start, x_noise, timesteps):
        """
        Add noise to x_start according to the diffusion process.
        
        Args:
            x_start: Clean data (batch, dim)
            x_noise: Random noise (batch, dim)
            timesteps: Timestep(s) for each sample. Can be scalar int, 0D tensor, or 1D tensor.
        
        Returns:
            Noisy version of x_start (batch, dim)
        """
        # Ensure timesteps is on the same device as scheduler tensors
        if torch.is_tensor(timesteps):
            if timesteps.device != self.device:
                timesteps = timesteps.to(self.device)
            # Ensure timesteps is 1D or scalar
            if timesteps.dim() == 0:
                timesteps = timesteps.unsqueeze(0)
            s1 = self.sqrt_alphas_cumprod[timesteps].view(-1, 1)
            s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1)
        else:
            # Scalar int
            s1 = self.sqrt_alphas_cumprod[timesteps].view(-1, 1)
            s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1)
        
        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps
    
class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self, encoder_dim, activation='relu', batchnorm=True):
        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        def get_activation():
            if self._activation == 'sigmoid':
                return nn.Sigmoid()
            elif self._activation == 'leakyrelu':
                return nn.LeakyReLU(0.2, inplace=True)
            elif self._activation == 'tanh':
                return nn.Tanh()
            elif self._activation == 'relu':
                return nn.ReLU()
            elif self._activation == 'gelu':
                return nn.GELU()
            elif self._activation in ['silu', 'swish']:
                return nn.SiLU()
            else:
                raise ValueError(f'Unknown activation type {self._activation}')
        
        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.LayerNorm(encoder_dim[i + 1]))
                encoder_layers.append(get_activation())

        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if i < self._dim - 1: # Usually we don't apply BN/Act on the final output layer of decoder unless specified
                if self._batchnorm:
                    decoder_layers.append(nn.LayerNorm(decoder_dim[i + 1]))
                decoder_layers.append(get_activation())

        # decoder_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent

class AttentionLayer(nn.Module):
    def __init__(self, latent_dim):
        super(AttentionLayer, self).__init__()
        self._latent_dim = latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.LayerNorm(self._latent_dim),
            nn.ReLU(),
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.LayerNorm(self._latent_dim),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(self._latent_dim, 1, bias=True)

    def _stack_views(self, *views):
        if len(views) == 1:
            stacked = views[0]
            if isinstance(stacked, (list, tuple)):
                stacked = torch.stack(stacked, dim=1)
        else:
            stacked = torch.stack(views, dim=1)

        if stacked.dim() == 2:
            stacked = stacked.unsqueeze(1)

        if stacked.dim() != 3:
            raise ValueError("Expected views with shape [batch, num_views, latent_dim].")
        return stacked

    def forward(self, *views, mask=None, tau=10.0):
        h = self._stack_views(*views)
        batch_size, num_views, latent_dim = h.shape

        flat_h = h.reshape(batch_size * num_views, latent_dim)
        logits = self.output_layer(self.mlp(flat_h)).reshape(batch_size, num_views) / tau

        if mask is not None:
            mask = mask.to(device=h.device, dtype=torch.bool)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(batch_size, -1)
            if mask.shape != (batch_size, num_views):
                raise ValueError("Mask must have shape [batch, num_views].")
            logits = logits.masked_fill(~mask, -1e9)

        logits = logits - logits.max(dim=1, keepdim=True).values
        weights = F.softmax(logits, dim=1)

        if mask is not None:
            weights = weights * mask.to(weights.dtype)

        return (weights.unsqueeze(-1) * h).sum(dim=1)

class ClusterProject(nn.Module):
    """Projection head for clustering."""
    def __init__(self, latent_dim, n_clusters):
        super(ClusterProject, self).__init__()
        self._latent_dim = latent_dim
        self._n_clusters = n_clusters
        self.cluster_projector = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.LayerNorm(self._latent_dim),
            nn.ReLU(),
        )
        self.cluster = nn.Sequential(
            nn.Linear(self._latent_dim, self._n_clusters),
            # nn.Sigmoid(),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        z = self.cluster_projector(x)
        y = self.cluster(z)
        return y, z

