"""04_train_aggregator.py - train attention-MIL with the subtype aux task; freeze.

Expects the slide manifest CSV (``paths.slides``, falling back to ``paths.clinical``)
with columns ``slide_id, subtype`` for the auxiliary classifier.
"""
import argparse
from pathlib import Path

import _bootstrap


def main():
    ap = argparse.ArgumentParser(description="Train attention-MIL aggregator.")
    _bootstrap.common_args(ap)
    args = ap.parse_args()
    cfg = _bootstrap.load(args)

    import pandas as pd
    from txmorph.training.mil import train_aggregator

    manifest = cfg.paths.get("slides", cfg.paths.clinical)
    clinical = pd.read_csv(manifest)
    subtypes = sorted(clinical["subtype"].dropna().unique())
    code = {s: i for i, s in enumerate(subtypes)}
    slide_ids = clinical["slide_id"].tolist()
    labels = [code[s] for s in clinical["subtype"]]

    mc = cfg.get("mil", cfg.get("aggregator", {}))
    ckpt_path = str(Path(cfg.paths.ckpt) / "mil.pt")
    train_aggregator(slide_ids, labels, cfg.paths.tile_emb, ckpt_path,
                     n_subtypes=len(subtypes), epochs=mc.get("epochs", 50),
                     lr=mc.get("lr", 1e-4), device=cfg.get("device", "cuda"),
                     seed=cfg.get("seed", 0))
    print(f"[mil] wrote {ckpt_path}")


if __name__ == "__main__":
    main()
