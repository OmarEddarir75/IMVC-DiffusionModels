import random

import numpy as np
import torch


def set_random_seed(seed=0):
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def expand_per_view(values, num_views):
    """Repeat the last item until a per-view setting reaches ``num_views``."""
    values = list(values)
    if not values:
        return values
    if len(values) >= num_views:
        return values[:num_views]
    return values + [values[-1]] * (num_views - len(values))


def get_indexed_config_values(config, prefix):
    """Collect values from keys like ``prefix1``, ``prefix2``, ... in numeric order."""
    keys = sorted(
        (key for key in config if key.startswith(prefix)),
        key=lambda key: int(key[len(prefix):])
    )
    return [config[key] for key in keys]


def target_l2(q):
    """Sharpen soft assignments while avoiding division by zero."""
    weight = q * q
    return (weight.t() / weight.sum(dim=1).clamp_min(1e-12)).t()
