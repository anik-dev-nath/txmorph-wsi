"""00_setup.py - create the artifact directories and print the resolved config."""
import argparse
from pathlib import Path

import _bootstrap


def main():
    ap = argparse.ArgumentParser(description="Set up dirs/paths and print config.")
    _bootstrap.common_args(ap)
    cfg = _bootstrap.load(ap.parse_args())

    for key in ("tiles", "tile_emb", "runs", "ckpt"):
        p = cfg.paths.get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)
            print(f"[setup] {key}: {p}")
    paired = cfg.paths.get("paired")
    if paired:
        Path(paired).parent.mkdir(parents=True, exist_ok=True)
    print("[setup] done")


if __name__ == "__main__":
    main()
