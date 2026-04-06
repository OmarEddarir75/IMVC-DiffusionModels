# DCG/loss.py
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = sys.float_info.epsilon


def compute_joint(view1, view2, EPS=EPS):
    """
    Compute a numerically stable joint probability matrix P.

    Args:
        view1: Tensor of shape [batch_size, k], row-stochastic (sum=1 per row).
        view2: Tensor of shape [batch_size, k], row-stochastic.
        EPS: Small constant for numerical stability.

    Returns:
        p_ij: Symmetric joint probability matrix of shape [k, k] summing to 1.
    """
    bn, k = view1.size()
    assert view2.size() == (bn, k)

    # Normalize each row to sum to 1 (ensures stochasticity)
    view1 = view1 / view1.sum(dim=1, keepdim=True).clamp(min=EPS)
    view2 = view2 / view2.sum(dim=1, keepdim=True).clamp(min=EPS)

    # Joint: outer product per sample, then average over batch
    p_i_j = torch.bmm(view1.unsqueeze(2), view2.unsqueeze(1))  # [batch, k, k]
    p_i_j = p_i_j.mean(dim=0)                                  # average over batch

    # Symmetrize
    p_i_j = (p_i_j + p_i_j.t()) / 2.0

    # Normalize to sum to 1
    p_i_j = p_i_j / p_i_j.sum().clamp(min=EPS)
    return p_i_j


def MMI(view1, view2, lamb=1.0, EPS=EPS):
    """
    Mutual information (MI) based loss between two latent views.
    With lamb=1.0, this equals -I(view1; view2) (negative mutual information).
    Minimizing this loss maximizes the dependence between views.

    Args:
        view1: Tensor of shape [batch, k], latent codes (e.g., fused view).
        view2: Tensor of shape [batch, k], latent codes (e.g., single view).
        lamb: Weight for marginal entropy terms. lamb=1 gives standard -MI,
              lamb=0 removes marginal terms (only cross-entropy part).
        EPS: Small constant for numerical stability.

    Returns:
        Scalar loss value (negative mutual information or a variant).
    """
    bn, k = view1.size()
    assert view2.size() == (bn, k), "view1 and view2 must have the same shape"

    # Compute joint probability matrix
    p_i_j = compute_joint(view1, view2, EPS=EPS)
    p_i_j = 0.5 * p_i_j + 0.5 * p_i_j.detach()
    assert p_i_j.size() == (k, k)

    # Compute marginals
    p_i = p_i_j.sum(dim=1, keepdim=True)  # [k, 1]
    p_j = p_i_j.sum(dim=0, keepdim=True)  # [1, k]

    # Clamp for numerical stability
    p_i_j = p_i_j.clamp(min=EPS)
    p_i = p_i.clamp(min=EPS)
    p_j = p_j.clamp(min=EPS)

    # MMI loss formula: -p_ij * (log p_ij - lamb*log p_i - lamb*log p_j)
    loss_matrix = -p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_i) - lamb * torch.log(p_j))
    loss = loss_matrix.sum()
    return loss


class InstanceLoss(nn.Module):
    """Instance-level contrastive loss (NT-Xent) for cross-view consistency."""
    def __init__(self, batch_size, temperature=0.5, device='cpu'):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self._mask_correlated_samples(batch_size).to(device)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def _mask_correlated_samples(self, batch_size):
        """Create mask to exclude positive pairs from negatives."""
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: Latent codes from first view [batch_size, dim]
            z_j: Latent codes from second view [batch_size, dim]
        Returns:
            Contrastive loss (scalar).
        """
        N = 2 * self.batch_size
        z = torch.cat([z_i, z_j], dim=0)  # [2*B, D]

        # Cosine similarity scaled by temperature
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)   # positives between z_i and z_j
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).view(N, 1)
        negative_samples = sim[self.mask].view(N, -1)

        logits = torch.cat([positive_samples, negative_samples], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=z.device)
        loss = self.criterion(logits, labels)
        return loss


class ClusterLoss(nn.Module):
    """Cluster-level contrastive loss with entropy regularization."""
    def __init__(self, class_num, temperature=0.1, device='cpu'):
        super().__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self._mask_correlated_clusters(class_num).to(device)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def _mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N), dtype=torch.bool)
        mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        return mask

    def forward(self, c_i, c_j, alpha=1.0):
        """
        Args:
            c_i: Cluster assignment probabilities for first view [batch, class_num]
            c_j: Cluster assignment probabilities for second view [batch, class_num]
            alpha: Weight for entropy regularization term.
        Returns:
            Cluster contrastive loss + entropy regularization.
        """
        # Entropy regularization (encourages balanced clusters)
        p_i = c_i.sum(0)
        p_i /= p_i.sum()
        p_j = c_j.sum(0)
        p_j /= p_j.sum()

        log_k = math.log(p_i.size(0))
        ne_loss = log_k + (p_i * torch.log(p_i + EPS)).sum()
        ne_loss += log_k + (p_j * torch.log(p_j + EPS)).sum()

        # Contrastive similarity between cluster prototypes
        c_i, c_j = c_i.T, c_j.T  # [class_num, dim]
        c = torch.cat([c_i, c_j], dim=0)  # [2*class_num, dim]
        c = F.normalize(c, dim=1)
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat([sim_i_j, sim_j_i], dim=0).view(2 * self.class_num, 1)
        negative_clusters = sim[self.mask].view(2 * self.class_num, -1)

        logits = torch.cat([positive_clusters, negative_clusters], dim=1)
        labels = torch.zeros(2 * self.class_num, dtype=torch.long, device=c.device)
        loss = self.criterion(logits, labels) / logits.size(0)

        return loss + alpha * ne_loss / self.class_num



# def MMI(view1, view2):
#     """Maximise cosine similarity between fused and single-view latents."""
#     v1 = F.normalize(view1, dim=1)
#     v2 = F.normalize(view2, dim=1)
#     cos_sim = (v1 * v2).sum(dim=1)
#     return -cos_sim.mean()