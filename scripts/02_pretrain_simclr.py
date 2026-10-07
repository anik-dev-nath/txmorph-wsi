"""02_pretrain_simclr.py - contrastively pretrain the tile encoder (heavy GPU job)."""
import argparse
from pathlib import Path

import _bootstrap


def main():
    ap = argparse.ArgumentParser(description="Pretrain SimCLR tile encoder.")
    _bootstrap.common_args(ap)
    args = ap.parse_args()
    cfg = _bootstrap.load(args)

    from txmorph.training.simclr import train_simclr
    ec = cfg.get("simclr", cfg.get("encoder", {}))
    ckpt_path = str(Path(cfg.paths.ckpt) / "simclr.pt")
    train_simclr(
        tiles_dir=cfg.paths.tiles, ckpt_path=ckpt_path,
        epochs=ec.get("epochs", 100), batch_size=ec.get("batch_size", 512),
        lr=ec.get("lr", 1e-3), temperature=ec.get("temperature", 0.07),
        proj_dim=ec.get("proj_dim", 128), device=cfg.get("device", "cuda"),
        num_workers=ec.get("num_workers", cfg.get("num_workers", 4)),
        seed=cfg.get("seed", 0),
    )
    print(f"[simclr] wrote {ckpt_path}")


if __name__ == "__main__":
    main()
