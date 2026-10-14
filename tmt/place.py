import numpy as np
import torch

from .train import mean_latency

GRID = np.round(np.arange(-0.10, 0.1001, 0.002), 4)


def shifted(theta, delta, upper=None):
    t = theta + delta
    return t.clamp(min=-1.0) if upper is None else torch.minimum(t.clamp(min=-1.0), upper)


def place(vit, images, table, budget, theta, upper=None, grid=GRID):
    cache = {}
    g = lambda i: cache.setdefault(i, mean_latency(vit, images, table, shifted(theta, float(grid[i]), upper)))
    lo, hi = 0, len(grid) - 1
    if not g(lo) <= budget <= g(hi):
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if g(mid) <= budget:
            lo = mid
        else:
            hi = mid
    p = (g(hi) - budget) / (g(hi) - g(lo)) if g(hi) != g(lo) else 1.0
    return dict(low=shifted(theta, float(grid[lo]), upper), high=shifted(theta, float(grid[hi]), upper), p=float(p))


def choose(ids, p, salt=12345):
    h = (np.asarray(ids, dtype=np.uint64) * np.uint64(6364136223846793005) + np.uint64(salt)) & np.uint64(0xFFFFFFFFFFFF)
    return h.astype(np.float64) / float(0x1000000000000) < p


def thresholds_for(ids, placed):
    low = torch.as_tensor(choose(ids, placed["p"]))[:, None]
    return torch.where(low, placed["low"][None], placed["high"][None])
