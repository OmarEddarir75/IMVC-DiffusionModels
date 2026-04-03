import math
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

EPS = sys.float_info.epsilon

def _as_probabilities(view, temperature=1.0):
    """Map arbitrary latent vectors to valid per-sample categorical probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return F.softmax(view / temperature, dim=1)


def compute_joint(z_fused, latents, EPS=EPS):
    """Compute a valid, symmetric joint probability matrix P."""

    bn, k = z_fused.size()
    assert (latents.size(0) == bn and latents.size(1) == k)

    p_i_j = torch.matmul(z_fused.t(), latents) / float(bn)
    p_i_j = (p_i_j + p_i_j.t()) / 2.0
    p_i_j = torch.clamp(p_i_j, min=EPS)
    p_i_j = p_i_j / torch.clamp(p_i_j.sum(), min=EPS)

    return p_i_j


def MMI(z_fused, latents, lamb=1.0, temperature=1.0, EPS=EPS):
    """Mutual Information between latent representations, to maximize."""
    _, k = z_fused.size()
    z_fused_prob = _as_probabilities(z_fused, temperature)
    z_prob       = _as_probabilities(latents, temperature)
    
    p_i_j = compute_joint(z_fused_prob, z_prob, EPS)  # joint distribution
    assert p_i_j.size() == (k, k)

    # marginal distributions
    p_i = p_i_j.sum(dim=1, keepdim=True)
    p_j = p_i_j.sum(dim=0, keepdim=True)

    # clamp to avoid log(0)
    p_i_j = torch.clamp(p_i_j, min=EPS)
    p_i = torch.clamp(p_i, min=EPS)
    p_j = torch.clamp(p_j, min=EPS)

    # standard MI formula
    mi = p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_j) - lamb * torch.log(p_i))
    mi = mi.sum()
    mi = mi / max(math.log(k), EPS)

    return -mi

class InstanceLoss(nn.Module):
    """Instance-level contrast loss"""
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), device=self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z_i = F.normalize(z_i, dim=1, eps=EPS)
        z_j = F.normalize(z_j, dim=1, eps=EPS)
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    """Cluster-level contrast loss"""
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N), device=self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j, alpha=1.0):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        p_i = torch.clamp(p_i, min=EPS)
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        p_j = torch.clamp(p_j, min=EPS)
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + alpha * ne_loss
    

def cross_view_loss_fn(latents, projectors):
    loss, pairs = 0.0, 0
    for i in range(len(latents)):
        for j in range(len(latents)):
            if i == j: continue
            z_i_in_j = projectors[(i, j)](latents[i])
            loss += F.mse_loss(z_i_in_j, latents[j].detach())
            pairs += 1
    return loss / pairs


def high_conf_loss_fn(view_head, latents, mask, threshold=0.8, aggregation="entropy_weighted"):
    """
    KL-based high-confidence multi-view clustering loss.

    Args:
        view_head : shared ViewClusterHead
        latents   : list of (B, latent_dim)
        mask      : (B, n_views) 1=present, 0=missing
        threshold : confidence threshold
    """
    n_views = len(latents)
    device  = latents[0].device

    view_probs = []
    view_logits = []

    # Forward pass (detach for target construction) 
    for v in range(n_views):
        present = mask[:, v].bool()

        q = torch.zeros(latents[v].shape[0], view_head.net[-1].out_features, device=device)
        logits = torch.zeros_like(q)

        if present.sum() > 0:
            q_v, logits_v = view_head(latents[v][present].detach())
            q[present] = q_v
            logits[present] = logits_v

        view_probs.append(q)
        view_logits.append(logits)

    # (n_views, B, K)
    stacked = torch.stack(view_probs, dim=0)

    # Mask missing views
    presence = mask.t().unsqueeze(-1).float()  # (n_views, B, 1)
    stacked = stacked * presence

    # Aggregate across views with a stable reducer.
    if aggregation not in ("mean", "entropy_weighted"):
        raise ValueError(f"Unknown aggregation: {aggregation}. Expected 'mean' or 'entropy_weighted'.")

    if aggregation == "mean":
        weights = presence
    else:
        # Lower entropy => higher confidence => larger weight.
        k = stacked.shape[-1]
        entropy = -(stacked.clamp_min(EPS) * torch.log(stacked.clamp_min(EPS))).sum(dim=-1, keepdim=True)
        norm = max(math.log(k), EPS)
        confidence = 1.0 - (entropy / norm)
        confidence = torch.clamp(confidence, min=0.0)
        weights = confidence * presence

    weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(EPS)
    q_agg = (stacked * weights).sum(dim=0)

    # Confidence filtering 
    conf_score = q_agg.max(dim=1).values
    sample_mask = conf_score >= threshold

    if sample_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    # Build soft target P
    p = q_agg ** 2
    p = p / (p.sum(dim=1, keepdim=True) + 1e-8)

    # KL loss across views 
    loss = torch.tensor(0.0, device=device)
    count = 0

    for v in range(n_views):
        valid = sample_mask & mask[:, v].bool()
        if valid.sum() == 0:
            continue

        _, logits_v = view_head(latents[v][valid])

        # more stable than log(q)
        log_q_v = F.log_softmax(logits_v / view_head.temperature, dim=-1)

        p_valid = p[valid]

        loss += F.kl_div(log_q_v, p_valid, reduction='batchmean')
        count += 1

    return loss / max(count, 1)

def reconstruction_loss_fn(recon_views, x_views, mask):
    loss = 0.0
    count = 0
    for v in range(len(recon_views)):
        if mask[:, v].sum() > 0:
            loss += F.mse_loss(recon_views[v][mask[:, v] == 1], x_views[v][mask[:, v] == 1])
            count += 1
    return loss / max(count, 1)
