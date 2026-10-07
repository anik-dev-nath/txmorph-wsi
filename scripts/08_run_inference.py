"""08_run_inference.py - per-patient inference -> biomarkers.

Paired mode: sample N=100 z_post per patient, derive biomarkers via zone counting.
Single-timepoint mode: compute P(pCR) via likelihood ratio under pCR=1 vs pCR=0.
Auto-detects mode from paired.parquet (all z_post NaN → single-timepoint).
"""
import argparse
import json
from pathlib import Path

import _bootstrap


def main():
    ap = argparse.ArgumentParser(description="Run per-patient inference.")
    _bootstrap.common_args(ap)
    args = ap.parse_args()
    cfg = _bootstrap.load(args)

    import numpy as np
    from txmorph.data.paired_dataset import load_paired
    from txmorph.drug.features import build_drug_features
    from txmorph.inference.run import run_inference

    df, z_pre, z_post = load_paired(cfg.paths.paired)

    # Auto-detect single-timepoint
    single_timepoint = bool(np.all(np.isnan(z_post)))
    if single_timepoint:
        print("[inference] single-timepoint mode (likelihood-based P(pCR))")

    dc = cfg.get("diffusion", {})
    drcfg = cfg.get("drug", {})
    variant = drcfg.get("variant", "onehot")
    drug_feats, meta = build_drug_features(
        df, variant=variant, radius=drcfg.get("morgan_radius", 2),
        n_bits=drcfg.get("morgan_bits", 2048),
        combination=drcfg.get("combination", "bitwise_or"))

    out_dir = Path(cfg.paths.runs) / "samples"
    records = run_inference(
        z_pre, z_post, drug_feats, df["pcr"].to_numpy(), df["fold"].to_numpy(),
        cfg.paths.ckpt, str(out_dir), n_folds=dc.get("n_folds", 5),
        n_samples=dc.get("n_samples", 100), dim=dc.get("dim", 512),
        cond_dim=dc.get("cond_dim", 512), T=dc.get("T", 1000),
        schedule=dc.get("schedule", "cosine"), cosine_s=dc.get("cosine_s", 0.008),
        conditioning=dc.get("conditioning", "film"), n_blocks=dc.get("n_blocks", 3),
        width=dc.get("width", 512), drug_variant=variant,
        num_drugs=meta.get("num_drugs", 16), n_bits=meta.get("n_bits", 2048),
        clip_x0=dc.get("clip_x0"), device=cfg.get("device", "cuda"), force=args.force,
        single_timepoint=single_timepoint)

    summary = [{"index": r["index"], "fold": r["fold"], "pcr": r["pcr"],
                "p_pcr": r["p_pcr"], "sigma2_tx": r["sigma2_tx"]} for r in records]
    (out_dir / "biomarkers.json").write_text(json.dumps(summary, indent=2))
    print(f"[inference] {len(records)} patients -> {out_dir}")


if __name__ == "__main__":
    main()
