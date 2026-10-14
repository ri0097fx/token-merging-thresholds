import json
import statistics
import time

import numpy as np
import torch

from .merge import attention, merge_single, tome_counts


def timed(fn, reps, warm=3):
    for i in range(warm):
        fn(i)
    ts = []
    for i in range(reps):
        t0 = time.perf_counter()
        fn(i)
        ts.append(time.perf_counter() - t0)
    return 1e3 * statistics.median(ts)


@torch.no_grad()
def measure_parts(vit, reps=36, seed=0):
    blocks = list(vit.model.blocks)
    C, N0 = vit.model.embed_dim, vit.tokens
    g = torch.Generator().manual_seed(seed)
    attn, merge, mlp = [0.0] * (N0 + 1), [0.0] * (N0 + 1), [0.0] * (N0 + 1)
    for N in range(1, N0 + 1):
        x = torch.randn(1, N, C, generator=g)
        s = torch.rand(1, N, generator=g) + 0.5
        attn[N] = timed(lambda i: attention(blocks[i % len(blocks)], x, s), reps)
        metrics = [attention(b, x, s)[1] for b in blocks]
        merge[N] = timed(lambda i: merge_single(x, s, metrics[i % len(metrics)], lambda best: N // 4), reps)
        mlp[N] = timed(lambda i: x + blocks[i % len(blocks)].ls2(blocks[i % len(blocks)].mlp(blocks[i % len(blocks)].norm2(x))), reps)
    H, W = vit.model.patch_embed.img_size
    img = torch.randn(1, 3, H, W, generator=g)
    xf, sf = torch.randn(1, 50, C, generator=g), torch.ones(1, 50)
    return dict(attn=attn, merge=merge, mlp=mlp, stem=timed(lambda i: vit.embed(img), reps),
                head=timed(lambda i: vit.head(xf, sf), reps), overhead=0.0)


def isotonic(y):
    blocks = [[v, 1] for v in y]
    out = []
    for v, w in blocks:
        out.append([v, w])
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2 = out.pop()
            v1, w1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    return [v for v, w in out for _ in range(w)]


class LatencyTable:
    def __init__(self, parts, depth):
        self.parts, self.depth = parts, depth
        self.attn = torch.tensor(parts["attn"], dtype=torch.float64) + parts["overhead"]
        self.merge = torch.tensor(parts["merge"], dtype=torch.float64)
        self.mlp = torch.tensor(parts["mlp"], dtype=torch.float64)
        self.const = parts["stem"] + parts["head"]

    def latency(self, n_in, k):
        d = n_in.device
        return self.const + (self.attn.to(d)[n_in] + self.merge.to(d)[n_in] + self.mlp.to(d)[n_in - k]).sum(1)

    def static(self, ks, n0):
        n = torch.tensor([[n0]])
        n_in = [n0]
        for k in ks[:-1]:
            n_in.append(n_in[-1] - k)
        return float(self.latency(torch.tensor([n_in]), torch.tensor([ks]))[0])

    def budget(self, r, n0):
        return self.static(tome_counts(n0, self.depth, r), n0)

    def save(self, path):
        json.dump(self.parts, open(path, "w"))

    @classmethod
    def load(cls, path, depth):
        return cls(json.load(open(path)), depth)


@torch.no_grad()
def calibrate(vit, parts, images, r, repeats=3):
    measured, counts = [], []
    for ks_rule in (tome_counts(vit.tokens, vit.depth, 0), tome_counts(vit.tokens, vit.depth, r)):
        for img in images:
            ts = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                ks = run_counts(vit, img, ks_rule)
                ts.append(1e3 * (time.perf_counter() - t0))
            measured.append(statistics.median(ts))
            counts.append(ks)
    n0 = vit.tokens

    def fit(p):
        table = LatencyTable(dict(p, overhead=0.0), vit.depth)
        pred = [table.static(ks, n0) for ks in counts]
        return float(np.mean([(m - q) / vit.depth for m, q in zip(measured, pred)]))
    parts = dict(parts, overhead=fit(parts))
    mono = dict(parts, **{k: [0.0] + isotonic(parts[k][1:]) for k in ("attn", "merge", "mlp")})
    mono["overhead"] = fit(mono)
    return mono


@torch.no_grad()
def run_counts(vit, img, ks):
    x, s = vit.embed(img)
    for blk, k in zip(vit.model.blocks, ks):
        x, metric = attention(blk, x, s)
        x, s, _ = merge_single(x, s, metric, lambda best, k=k: k)
        x = x + blk.ls2(blk.mlp(blk.norm2(x)))
    vit.head(x, s)
    return ks
