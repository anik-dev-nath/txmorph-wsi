#!/usr/bin/env python3
"""Benchmark the SimCLR input pipeline: CPU augmentation vs GPU augmentation.

Measures end-to-end training throughput (tiles/s, including the ResNet-34
forward+backward) for both paths so the choice is evidence-based rather than
assumed. Run on the GPU server:

    PYTHONPATH=src python scripts/bench_simclr_pipeline.py --steps 12
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from txmorph.data.gpu_augment import two_views  # noqa: E402
from txmorph.data.tile_dataset import SimCLRTransform, list_tiles  # noqa: E402
from txmorph.encoders.simclr import SimCLREncoder, nt_xent  # noqa: E402
from txmorph.training.simclr import _RawTileLoader, _ThreadedSimCLRLoader  # noqa: E402


def _train_steps(encoder, opt, batches, steps, device, augment=None,
                 channels_last=False):
    """Run `steps` optimiser steps, return (tiles_processed, seconds)."""
    use_bf16 = torch.cuda.is_bf16_supported()
    n = 0
    torch.cuda.synchronize()
    t0 = time.time()
    for i, item in enumerate(batches):
        if i >= steps:
            break
        if augment is None:
            v1, v2 = item
            v1, v2 = v1.to(device), v2.to(device)
        else:
            tiles = item.to(device)
            with torch.no_grad():
                v1, v2 = augment(tiles)
        if channels_last:
            v1 = v1.contiguous(memory_format=torch.channels_last)
            v2 = v2.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            _, z1 = encoder(v1)
            _, z2 = encoder(v2)
            loss = nt_xent(z1, z2, 0.07)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        n += v1.shape[0]
    torch.cuda.synchronize()
    return n, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="/home/anik-server/data/tiles")
    ap.add_argument("--batch-size", type=int, default=384)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    paths = list_tiles(args.tiles)
    print(f"[bench] {len(paths)} tiles, batch={args.batch_size}, "
          f"steps={args.steps}, threads={args.threads}", flush=True)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    results = {}

    # --- current path: augmentation in numpy/PIL on the CPU worker threads ---
    torch.backends.cudnn.benchmark = False
    enc = SimCLREncoder(proj_dim=128).to(dev)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3)
    cpu_loader = _ThreadedSimCLRLoader(paths, args.batch_size, SimCLRTransform(),
                                       num_threads=args.threads, seed=0)
    it = cpu_loader.epoch_iterator()
    _train_steps(enc, opt, it, 2, dev)                       # warm-up
    n, sec = _train_steps(enc, opt, it, args.steps, dev)
    results["cpu_aug"] = n / sec
    print(f"[bench] CPU-aug : {n} tiles in {sec:.1f}s = {n/sec:.0f} tiles/s", flush=True)
    del enc, opt, cpu_loader, it
    torch.cuda.empty_cache()

    # --- new path: threads only np.load, augmentation as batched GPU kernels ---
    torch.backends.cudnn.benchmark = True
    enc = SimCLREncoder(proj_dim=128).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(0)
    gpu_loader = _RawTileLoader(paths, args.batch_size,
                                num_threads=args.threads, seed=0)
    it = gpu_loader.epoch_iterator(0)
    aug = lambda t: two_views(t, gen)                        # noqa: E731
    _train_steps(enc, opt, it, 3, dev, augment=aug, channels_last=True)   # warm-up
    n, sec = _train_steps(enc, opt, it, args.steps, dev, augment=aug,
                          channels_last=True)
    results["gpu_aug"] = n / sec
    print(f"[bench] GPU-aug : {n} tiles in {sec:.1f}s = {n/sec:.0f} tiles/s", flush=True)

    speedup = results["gpu_aug"] / max(results["cpu_aug"], 1e-9)
    epoch_min_cpu = len(paths) / results["cpu_aug"] / 60
    epoch_min_gpu = len(paths) / results["gpu_aug"] / 60
    print(f"\n[bench] speedup = {speedup:.1f}x")
    print(f"[bench] projected epoch time: CPU-aug {epoch_min_cpu:.0f} min  ->  "
          f"GPU-aug {epoch_min_gpu:.0f} min")


if __name__ == "__main__":
    main()
