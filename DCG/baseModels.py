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

class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t].to(x_t.device)
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t].to(x_t.device)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t].to(x_0.device)
        s2 = self.posterior_mean_coef2[t].to(x_t.device)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0
        
        # Ensure t is a tensor for indexing if needed, but here it's likely an int or scalar tensor
        variance = (self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t]))
        if torch.is_tensor(t):
            variance = variance.to(t.device)
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)
        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps, device):
        s1 = self.sqrt_alphas_cumprod[timesteps].to(device)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].to(device)
        s1 = s1.view(-1, 1)
        s2 = s2.view(-1, 1)
        out1 = s1 * x_start
        out2 = s2 * x_noise
        out = out1 + out2
        return out

    def __len__(self):
        return self.num_timesteps

class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):

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
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
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
                    decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
                decoder_layers.append(get_activation())

        decoder_layers.append(nn.Softmax(dim=1))
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
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(),
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.BatchNorm1d(self._latent_dim),
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
            nn.BatchNorm1d(self._latent_dim),
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

