import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    """
    AutoEncoder module for multi-view representation learning.
    """

    def __init__(self, encoder_dims, activation='silu', use_norm=True, dropout=0.0):
        super(Autoencoder, self).__init__()

        self.encoder_dims = encoder_dims
        self.latent_dim = encoder_dims[-1]
        self.use_norm = use_norm
        self.dropout = dropout
        self.activation_name = activation.lower()

        def get_activation():
            if self.activation_name == 'relu':
                return nn.ReLU(inplace=True)
            elif self.activation_name == 'leakyrelu':
                return nn.LeakyReLU(0.2, inplace=True)
            elif self.activation_name == 'tanh':
                return nn.Tanh()
            elif self.activation_name == 'sigmoid':
                return nn.Sigmoid()
            elif self.activation_name == 'silu':
                return nn.SiLU()
            else:
                raise ValueError(f'Unknown activation: {activation}')

        # encoder
        encoder_layers = []
        num_enc_layers = len(encoder_dims) - 1

        for i in range(num_enc_layers):
            in_dim = encoder_dims[i]
            out_dim = encoder_dims[i + 1]

            encoder_layers.append(nn.Linear(in_dim, out_dim))

            if i < num_enc_layers - 1:
                if self.use_norm:
                    encoder_layers.append(nn.LayerNorm(out_dim))

                encoder_layers.append(get_activation())

                if self.dropout > 0:
                    encoder_layers.append(nn.Dropout(self.dropout))

        self.encoder_net = nn.Sequential(*encoder_layers)

        # decoder
        decoder_dims = list(reversed(encoder_dims))
        decoder_layers = []
        num_dec_layers = len(decoder_dims) - 1

        for i in range(num_dec_layers):
            in_dim = decoder_dims[i]
            out_dim = decoder_dims[i + 1]

            decoder_layers.append(nn.Linear(in_dim, out_dim))

            if i < num_dec_layers - 1:
                if self.use_norm:
                    decoder_layers.append(nn.LayerNorm(out_dim))

                decoder_layers.append(get_activation())

                if self.dropout > 0:
                    decoder_layers.append(nn.Dropout(self.dropout))

        self.decoder_net = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """
        Encode input to latent space.
        """
        z = self.encoder_net(x)
        z = torch.tanh(z)
        return z

    def decode(self, z):
        """
        Decode latent to input space.
        """
        x_hat = self.decoder_net(z)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class AttentionLayer(nn.Module):
    def __init__(self, latent_dim, n_views=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_views = n_views
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * n_views, latent_dim * n_views),
            nn.LayerNorm(latent_dim * n_views),
            nn.SiLU(),
            nn.Linear(latent_dim * n_views, latent_dim * n_views),
            nn.LayerNorm(latent_dim * n_views),
            nn.SiLU(),
            nn.Linear(latent_dim * n_views, n_views),
        )
        self.output_layer = nn.Sigmoid()

    def forward(self, *views, tau=0.5, return_weights=False):
        
        assert len(views) == self.n_views, f"Expected {self.n_views} views, got {len(views)}"

        for v in views:
            assert v.shape[-1] == self.latent_dim, "Latent dim mismatch across views"

        h = torch.cat(views, dim=-1)
        act = self.output_layer(self.mlp(h))
        w = F.softmax(act / tau, dim=-1)  # attention weights over views
        stacked = torch.stack(views, dim=1)
        shared = (stacked * w.unsqueeze(-1)).sum(dim=1)

        if return_weights:
            return shared, w

        return shared


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        device = x.device
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size, device=device))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class TimeEmbed(nn.Module):
    def __init__(self, dim: int, type: str, **kwargs):
        super().__init__()
        if type == "sinusoidal":
            self.embed = SinusoidalEmbedding(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
        else:
            raise ValueError(f"Unknown positional embedding type: {type}")            

    def forward(self, t: torch.Tensor):
        return self.mlp(self.embed(t))


class Unet(nn.Module):
    def __init__(self, emb_size: int = 128, time_emb: str = "sinusoidal", out_size: int = 128):
        super().__init__()

        self.time_mlp = TimeEmbed(emb_size, time_emb)
        concat_size = 2*emb_size
        layers = []
        layers.append(nn.Linear(concat_size, 2000))
        layers.append(nn.LayerNorm(2000))
        layers.append(nn.SiLU())

        layers.append(nn.Linear(2000, 500))
        layers.append(nn.LayerNorm(500))
        layers.append(nn.SiLU())

        layers.append(nn.Linear(500, 500))
        layers.append(nn.LayerNorm(500))
        layers.append(nn.SiLU())

        layers.append(nn.Linear(500, 2000))
        layers.append(nn.LayerNorm(2000))
        layers.append(nn.SiLU())

        layers.append(nn.Linear(2000, out_size))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = torch.cat((x, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x


class NoiseScheduler():
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear"):

        self.num_timesteps = num_timesteps
        
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "cosine":
            timesteps = torch.arange(num_timesteps + 1, dtype=torch.float32) / num_timesteps
            alphas_bar = torch.cos((timesteps + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            self.betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])

        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, axis=0)
        self.alphas_bar_prev = F.pad(self.alphas_bar[:-1], (1, 0), value=1.)

        self.sqrt_alphas_bar = self.alphas_bar ** 0.5
        self.sqrt_one_minus_alphas_bar = (1 - self.alphas_bar) ** 0.5

        self.sqrt_inv_alphas_bar = torch.sqrt(1 / self.alphas_bar)
        self.sqrt_inv_alphas_bar_minus_one = torch.sqrt(1 / self.alphas_bar - 1)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_bar_prev) / (1. - self.alphas_bar)
        self.posterior_mean_coef2 = (1. - self.alphas_bar_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_bar)

    def _as_index_tensor(self, t, device=None):
        if torch.is_tensor(t):
            idx = t.long()
        else:
            idx = torch.tensor(t, dtype=torch.long)

        if device is not None:
            idx = idx.to(device)
        return idx

    def reconstruct_x0(self, x_t, t, noise):
        # Ensure t is a tensor index for scheduler buffer lookup.
        t_idx = self._as_index_tensor(t, device=self.sqrt_inv_alphas_bar.device).cpu()
        s1 = self.sqrt_inv_alphas_bar[t_idx].to(x_t.device)
        s2 = self.sqrt_inv_alphas_bar_minus_one[t_idx].to(x_t.device)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        # Ensure t is a tensor index for scheduler buffer lookup.
        t_idx = self._as_index_tensor(t, device=self.posterior_mean_coef1.device).cpu()
        s1 = self.posterior_mean_coef1[t_idx].to(x_0.device)
        s2 = self.posterior_mean_coef2[t_idx].to(x_t.device)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t, device):
        t_idx = self._as_index_tensor(t).item()
        if t_idx == 0:
            return 0

        variance = (self.betas[t_idx] * (1. - self.alphas_bar_prev[t_idx]) / (1. - self.alphas_bar[t_idx])).to(device)
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = self._as_index_tensor(timestep).item()
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)
        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t, sample.device) ** 0.5) * noise
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps, device):
        # Scheduler buffers live on CPU by default; index them on CPU first.
        idx = timesteps.long().cpu()
        s1 = self.sqrt_alphas_bar[idx].to(device)
        s2 = self.sqrt_one_minus_alphas_bar[idx].to(device)
        s1 = s1.view(-1, 1)
        s2 = s2.view(-1, 1)
        out1 = s1 * x_start
        out2 = s2 * x_noise
        out = out1 + out2
        return out

    def __len__(self):
        return self.num_timesteps
    

class MissingViewImputer:
    """
    Uses trained UNets + NoiseScheduler to impute clean latent vectors
    for samples where one or more views are missing.
    Supports multiple UNets (one per view).
    """
    def __init__(self, unets, scheduler, device):
        """
        unets: list of UNets, one per view
        scheduler: NoiseScheduler
        device: torch device
        """
        self.unets = unets
        self.scheduler = scheduler
        self.device = device
        self.n_views = len(unets)

    @torch.no_grad()
    def impute(self, z_observed, mask=None, n_steps=50, noise_level=0.3):
        """
        z_observed: (B, latent_dim) fused latent from observed views only
        mask: (B, n_views) tensor, 1 if view present, 0 if missing
        n_steps: number of reverse diffusion steps
        noise_level: fraction of total timesteps to corrupt (0.0–1.0)
        """
        t_start = int(noise_level * self.scheduler.num_timesteps)
        z_imputed = z_observed.clone()
        imputed_sum = torch.zeros_like(z_observed)
        imputed_count = torch.zeros(z_observed.size(0), 1, device=z_observed.device, dtype=z_observed.dtype)

        for v, unet in enumerate(self.unets):
            # Determine which samples are missing for this view
            if mask is not None:
                missing_idx = (mask[:, v] == 0).nonzero(as_tuple=True)[0]
                if len(missing_idx) == 0:
                    continue
            else:
                missing_idx = torch.arange(z_observed.size(0), device=z_observed.device)

            x = z_observed[missing_idx].clone()

            # Initial corruption
            noise = torch.randn_like(x)
            t_tensor = torch.full((x.size(0),), t_start, device=self.device, dtype=torch.long)
            x = self.scheduler.add_noise(x, noise, t_tensor, device=self.device)

            # Reverse diffusion
            for t in reversed(range(t_start)):
                t_tensor = torch.full((x.size(0),), t, device=self.device, dtype=torch.long)
                noise_pred = unet(x, t_tensor.float())
                x = self.scheduler.step(noise_pred, t, x)

            # Accumulate per-view imputations and blend once at the end.
            imputed_sum[missing_idx] += x
            imputed_count[missing_idx] += 1.0

        valid = imputed_count.squeeze(-1) > 0
        if valid.any():
            z_imputed[valid] = imputed_sum[valid] / imputed_count[valid]

        return z_imputed


class CrossViewProjector(nn.Module):
    """
    Maps the latent of view i into the latent space of view j.
    One projector per ordered pair (i→j). Enables cross_view_loss
    without requiring views to share input dimensionality.
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z):
        return self.net(z)


class ViewClusterHead(nn.Module):
    """
    Lightweight per-view cluster projection head.
    Each view develops its own assignment confidence independently.
    """
    def __init__(self, latent_dim, n_clusters, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, n_clusters),
        )

    def forward(self, z):
        logits = self.net(z)
        q = F.softmax(logits / self.temperature, dim=-1)
        return q, logits
    
