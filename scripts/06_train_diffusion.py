"""06_train_diffusion.py - train the conditional latent diffusion, 5-fold CV.

Auto-detects single-timepoint mode: if all z_post values are NaN (IMPRESS /
free cohorts), trains class-conditional p(z_pre | drug, pCR) instead of
p(z_post | z_pre, drug).
"""
import argparse

import numpy as np

import _bootstrap


def main():
    ap = argparse.ArgumentParser(description="Train conditional diffusion (5-fold).")
    _bootstrap.common_args(ap)
    args = ap.parse_args()
    cfg = _bootstrap.load(args)

    from txmorph.data.paired_dataset import load_paired
    from txmorph.drug.features import build_drug_features
    from txmorph.training.diffusion import train_diffusion_cv

    df, z_pre, z_post = load_paired(cfg.paths.paired)

    # Auto-detect single-timepoint: all z_post are NaN
    single_timepoint = bool(np.all(np.isnan(z_post)))
    if single_timepoint:
        print("[diffusion] single-timepoint mode detected (all z_post NaN)")
        print("[diffusion] training class-conditional p(z_pre | drug, pCR)")

    dc = cfg.get("diffusion", {})
    drcfg = cfg.get("drug", {})
    variant = drcfg.get("variant", "onehot")
    drug_feats, meta = build_drug_features(
        df, variant=variant, radius=drcfg.get("morgan_radius", 2),
        n_bits=drcfg.get("morgan_bits", 2048),
        combination=drcfg.get("combination", "bitwise_or"))

    kw = dict(
        n_folds=dc.get("n_folds", 5), dim=dc.get("dim", 512),
        cond_dim=dc.get("cond_dim", 512), T=dc.get("T", 1000),
        schedule=dc.get("schedule", "cosine"), cosine_s=dc.get("cosine_s", 0.008),
        lambda_vlb=dc.get("lambda_vlb", 1e-3), conditioning=dc.get("conditioning", "film"),
        n_blocks=dc.get("n_blocks", 3), width=dc.get("width", 512),
        drug_variant=variant, num_drugs=meta.get("num_drugs", 16),
        n_bits=meta.get("n_bits", 2048), lr=dc.get("lr", 2e-4),
        batch_size=dc.get("batch_size", 64), device=cfg.get("device", "cuda"),
        seed=cfg.get("seed", 0),
        single_timepoint=single_timepoint,
    )
    if single_timepoint:
        kw["pcr"] = df["pcr"].to_numpy()

    train_diffusion_cv(
        z_pre, z_post, drug_feats, df["fold"].to_numpy(), cfg.paths.ckpt, **kw)
    print(f"[diffusion] wrote {dc.get('n_folds', 5)} fold checkpoints to {cfg.paths.ckpt}")


if __name__ == "__main__":
    main()
