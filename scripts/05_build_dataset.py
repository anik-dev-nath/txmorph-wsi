"""05_build_dataset.py - build paired.parquet with patient-level folds.

Expects a clinical manifest CSV (``paths.clinical``) with at least:
``patient_id, pre_slide_id, cohort, drug_id, drug_smiles, pcr,
rfs_time, rfs_event, receptor_subtype, stage``.
``post_slide_id`` is optional (single-timepoint mode for IMPRESS/free cohorts).
"""
import argparse
from pathlib import Path

import _bootstrap


def main():
    ap = argparse.ArgumentParser(description="Build paired.parquet.")
    _bootstrap.common_args(ap)
    args = ap.parse_args()
    cfg = _bootstrap.load(args)

    if Path(cfg.paths.paired).exists() and not args.force:
        print(f"[dataset] {cfg.paths.paired} exists; use --force to rebuild")
        return

    import pandas as pd
    from txmorph.data.paired_dataset import build_paired_dataset
    from txmorph.training.mil import load_aggregator

    clinical = pd.read_csv(cfg.paths.clinical)
    mc = cfg.get("mil", cfg.get("aggregator", {}))
    pooling = mc.get("pooling", "mean")

    mil = None
    if pooling == "mil":
        # Sized from the checkpoint, not from a manifest column: script 04 trains the
        # subtype head off slides.csv['subtype'], so re-deriving it here from
        # clinical.csv['receptor_subtype'] mismatched whenever the two columns had
        # different cardinality (they usually do).
        mil = load_aggregator(str(Path(cfg.paths.ckpt) / "mil.pt"),
                              device=cfg.get("device", "cpu"))

    dc = cfg.get("diffusion", {})
    df = build_paired_dataset(clinical, cfg.paths.tile_emb, mil, cfg.paths.paired,
                              n_folds=dc.get("n_folds", 5), seed=cfg.get("seed", 0),
                              device=cfg.get("device", "cpu"), pooling=pooling)
    print(f"[dataset] slide pooling: {pooling}")
    print(f"[dataset] wrote {cfg.paths.paired} ({len(df)} patients)")


if __name__ == "__main__":
    main()
