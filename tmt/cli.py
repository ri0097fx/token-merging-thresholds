import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
from torchvision.datasets import ImageFolder

from .latency import LatencyTable, calibrate, measure_parts
from .merge import ThresholdViT
from .place import place
from .train import DISTANCES, caps, initial_level, initial_price, train


def subset(folder, transform, count, seed, batch=250):
    ds = ImageFolder(folder, transform=transform)
    idx = np.sort(np.random.default_rng(seed).choice(len(ds), min(count, len(ds)), replace=False))
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=batch, num_workers=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deit_small_patch16_224.fb_in1k")
    ap.add_argument("--train", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--r", type=int, default=12)
    ap.add_argument("--distance", choices=sorted(DISTANCES), default="kl")
    ap.add_argument("--table")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--perturbations", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="thresholds.json")
    args = ap.parse_args()

    cpu = ThresholdViT(args.model)
    transform = timm.data.create_transform(**cpu.data_config)
    heldout = subset(args.heldout, transform, 2000, 0)
    if args.table and Path(args.table).exists():
        table = LatencyTable.load(args.table, cpu.depth)
    else:
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        first = next(iter(subset(args.heldout, transform, 150, 0, 150)))[0]
        parts = calibrate(cpu, measure_parts(cpu), [x[None] for x in first], args.r)
        table = LatencyTable(parts, cpu.depth)
        table.save(args.table or "latency_table.json")
        torch.set_num_threads(threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit = ThresholdViT(args.model).to(device)
    distance = DISTANCES[args.distance]
    budget = table.budget(args.r, vit.tokens)
    train_ds = ImageFolder(args.train, transform=transform)
    calib = subset(args.train, transform, 20000, 5000, 500)
    upper = caps(vit, subset(args.train, transform, 5000, 0), args.r)
    mu0 = initial_price(vit, heldout, table, args.r, distance)
    level = initial_level(vit, calib, table, budget, upper)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=4)
    theta, mu = train(vit, loader, table, budget, mu0, upper, level, distance, steps=args.steps,
                      perturbations=args.perturbations, seed=args.seed,
                      log=lambda s, t, m, l: print(f"step {s} mu={m:.4g} latency={l:.2f}/{budget:.2f} ms", flush=True))
    placed = place(vit, calib, table, budget, theta, upper.cpu())
    out = dict(model=args.model, r=args.r, budget_ms=budget, theta=theta.tolist(), caps=upper.tolist(), mu=mu)
    if placed is not None:
        out.update(low=placed["low"].tolist(), high=placed["high"].tolist(), p=placed["p"])
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
