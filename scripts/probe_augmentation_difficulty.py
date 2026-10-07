#!/usr/bin/env python3
"""Check that an augmentation setting gives SimCLR a non-trivial task.

NT-Xent at chance is ln(2N-1) (~6.6 at batch 384). The v4 run settled at ~0.017,
meaning the trained encoder matched positives almost perfectly -- the sign of a
shortcut, not of a good representation. This probe scores the *existing*
checkpoint under different augmentation settings: a setting that still yields a
near-zero loss has not fixed the shortcut, whatever else it changes.

    PYTHONPATH=src python scripts/probe_augmentation_difficulty.py --batches 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from txmorph.data.gpu_augment import two_views  # noqa: E402
from txmorph.data.tile_dataset import list_tiles  # noqa: E402
from txmorph.encoders.simclr import SimCLREncoder, nt_xent  # noqa: E402
from txmorph.training.simclr import _RawTileLoader  # noqa: E402
from txmorph.utils import ckpt  # noqa: E402

SETTINGS = [
    ("v4 baseline  (fast jitter, crop 0.6-1.0)", dict(stain_jitter="fast",  crop_scale=(0.6, 1.0))),
    ("wider crop   (fast jitter, crop 0.2-1.0)", dict(stain_jitter="fast",  crop_scale=(0.2, 1.0))),
    ("macenko only (sigma 0.05, crop 0.6-1.0)",  dict(stain_jitter="macenko", stain_sigma=0.05, crop_scale=(0.6, 1.0))),
    ("proposed     (macenko 0.05, crop 0.2-1.0)", dict(stain_jitter="macenko", stain_sigma=0.05, crop_scale=(0.2, 1.0))),
    ("proposed+    (macenko 0.15, crop 0.2-1.0)", dict(stain_jitter="macenko", stain_sigma=0.15, crop_scale=(0.2, 1.0))),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="/home/anik-server/data/tiles")
    ap.add_argument("--ckpt", default="/home/anik-server/checkpoints/simclr.pt")
    ap.add_argument("--batch-size", type=int, default=384)
    ap.add_argument("--batches", type=int, default=6)
    args = ap.parse_args()

    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    paths = list_tiles(args.tiles)
    enc = SimCLREncoder(proj_dim=128).to(dev).to(memory_format=torch.channels_last)
    if Path(args.ckpt).exists():
        ckpt.maybe_resume(args.ckpt, enc, None, map_location=dev)
        print(f"[probe] loaded {args.ckpt}")
    enc.eval()

    print(f"[probe] chance loss = ln(2N-1) = "
          f"{torch.log(torch.tensor(2.0 * args.batch_size - 1)):.2f}\n")
    for name, kw in SETTINGS:
        loader = _RawTileLoader(paths, args.batch_size, num_threads=4, seed=123)
        gen = torch.Generator(device=dev).manual_seed(0)
        losses, t0, n = [], time.time(), 0
        with torch.no_grad():
            for i, tiles in enumerate(loader.epoch_iterator(0)):
                if i >= args.batches:
                    break
                tiles = tiles.to(dev)
                v1, v2 = two_views(tiles, gen, **kw)
                v1 = v1.contiguous(memory_format=torch.channels_last)
                v2 = v2.contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, z1 = enc(v1)
                    _, z2 = enc(v2)
                    losses.append(float(nt_xent(z1, z2, 0.07)))
                n += tiles.shape[0]
        torch.cuda.synchronize()
        rate = n / (time.time() - t0)
        mean = sum(losses) / len(losses)
        print(f"[probe] {name:<44} loss {mean:6.3f}   {rate:4.0f} tiles/s")


if __name__ == "__main__":
    main()
