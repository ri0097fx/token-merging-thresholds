import math

import numpy as np
import torch
import torch.nn.functional as F

from .merge import tome_counts


def kl_distance(out, ref):
    lp = out.float().log_softmax(-1)
    lq = ref.float().log_softmax(-1)
    return (lq.exp() * (lq - lp)).sum(-1)


def cosine_distance(out, ref):
    return 1 - (F.normalize(out.float(), dim=-1) * F.normalize(ref.float(), dim=-1)).sum(-1)


DISTANCES = {"kl": kl_distance, "cos": cosine_distance}


def autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


def batches(images, size):
    if torch.is_tensor(images):
        for b in range(0, len(images), size):
            yield images[b:b + size]
    else:
        for b in images:
            yield b[0] if isinstance(b, (list, tuple)) else b


@torch.no_grad()
def original(vit, img):
    return vit(img, counts=[0] * vit.depth)[0]


@torch.no_grad()
def caps(vit, images, r, q=0.99, batch=250):
    device = vit.mean.device
    best = [[] for _ in range(vit.depth)]
    hook = lambda l, b: best[l].append(torch.where(torch.isfinite(b), b, torch.full_like(b, -2.0)).max(1).values.float().cpu())
    for x in batches(images, batch):
        with autocast(device):
            vit(x.to(device), counts=tome_counts(vit.tokens, vit.depth, r), hook=hook)
    return torch.tensor([float(np.quantile(torch.cat(b).numpy(), q)) for b in best])


@torch.no_grad()
def initial_price(vit, images, table, r, distance, batch=250):
    device = vit.mean.device
    d = round(2 * (vit.tokens - 1) / 196 * 12 / vit.depth)
    full = table.static([0] * vit.depth, vit.tokens)
    obj = {}
    for rr in (r - d, r + d):
        ks = tome_counts(vit.tokens, vit.depth, rr)
        vals = []
        for x in batches(images, batch):
            x = x.to(device)
            with autocast(device):
                vals.append(distance(vit(x, counts=ks)[0], original(vit, x)))
        obj[rr] = float(torch.cat(vals).mean())
    return (obj[r + d] - obj[r - d]) / ((table.budget(r - d, vit.tokens) - table.budget(r + d, vit.tokens)) / full)


@torch.no_grad()
def mean_latency(vit, images, table, thresholds, batch=500):
    device = vit.mean.device
    lat = []
    for x in batches(images, batch):
        with autocast(device):
            _, n_in, k = vit(x.to(device), thresholds)
        lat.append(table.latency(n_in, k))
    return float(torch.cat(lat).mean())


def initial_level(vit, images, table, budget, upper, steps=16):
    lo, hi = -1.0, 1.0
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if mean_latency(vit, images, table, torch.minimum(torch.full_like(upper, mid), upper)) > budget:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def train(vit, loader, table, budget, mu0, upper, level, distance, steps=3000, perturbations=8, sigma=(0.05, 0.02),
          lr=0.02 * 0.05 / 0.3, eta=0.05, clip=5.0, seed=0, log=None):
    device = vit.mean.device
    torch.manual_seed(seed)
    upper = upper.to(device)
    theta = torch.full((vit.depth,), level, device=device).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    gen = torch.Generator(device=device).manual_seed(seed)
    full = table.static([0] * vit.depth, vit.tokens)
    mu, ema, step = mu0, None, 0
    while step < steps:
        for img in loader:
            if step >= steps:
                break
            img = (img[0] if isinstance(img, (list, tuple)) else img).to(device)
            ref = original(vit, img).repeat_interleave(perturbations, 0)
            x = img.repeat_interleave(perturbations, 0)
            sig = sigma[0] * (sigma[1] / sigma[0]) ** (step / max(1, steps - 1))
            eps = sig * torch.randn(x.shape[0], vit.depth, device=device, generator=gen)
            rows = torch.minimum(torch.maximum(theta.detach()[None] + eps, torch.full_like(eps, -1.0)), upper)
            with autocast(device):
                out, n_in, k = vit(x, rows)
            lat = table.latency(n_in, k).float()
            reward = -distance(out, ref) - mu * lat / full
            adv = reward.view(-1, perturbations)
            adv = (adv - adv.mean(1, keepdim=True)).view(-1)
            m = adv.abs().mean().item()
            ema = m if ema is None else 0.99 * ema + 0.01 * m
            adv = adv.clamp(-clip * ema, clip * ema)
            opt.zero_grad()
            theta.grad = -(adv[:, None] * eps).mean(0) / sig ** 2
            opt.step()
            with torch.no_grad():
                theta.copy_(torch.minimum(torch.maximum(theta, torch.full_like(theta, -1.0)), upper))
            mu = mu * math.exp(eta * (lat.mean().item() / budget - 1))
            step += 1
            if log is not None and step % 25 == 0:
                log(step, theta.detach().cpu().tolist(), mu, lat.mean().item())
    return theta.detach().cpu(), mu
