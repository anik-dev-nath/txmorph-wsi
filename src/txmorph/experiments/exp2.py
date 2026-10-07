"""Experiment 2 - Uncertainty -> RFS (CLAUDE.md S11-Exp2).

Cox PH: RFS ~ sigma2_tx (continuous) + pcr + subtype + stage. Report HR/CI/p for
sigma2_tx, Harrell C, Schoenfeld PH check. KM + log-rank for high/low sigma2_tx
quartiles WITHIN pCR achievers (exploratory, low event count).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..eval.survival import cox_ph, schoenfeld_test, km_logrank, quartile_groups


def run_exp2(records, paired_df, covariates=("sigma2_tx", "pcr", "subtype", "stage")):
    """records from inference.run; paired_df has rfs_time/rfs_event/subtype/stage."""
    idx = np.array([r["index"] for r in records])
    sigma2 = np.array([r["sigma2_tx"] for r in records])

    df = paired_df.iloc[idx].reset_index(drop=True).copy()
    df["sigma2_tx"] = sigma2
    # encode categoricals to numeric codes for the Cox design
    if "subtype" not in df and "receptor_subtype" in df:
        df["subtype"] = df["receptor_subtype"]
    for cat in ("subtype", "stage"):
        if cat in df:
            df[cat] = pd.Categorical(df[cat]).codes
    df["pcr"] = df["pcr"].astype(int)

    cox = cox_ph(df, covariates=[c for c in covariates if c in df])
    try:
        cox["schoenfeld"] = schoenfeld_test(cox["model"], df)
    except Exception as e:                                  # PH check is best-effort
        cox["schoenfeld"] = f"unavailable: {e}"

    # within-pCR KM on sigma2_tx quartiles (exploratory)
    pcr_df = df[df["pcr"] == 1].copy()
    km = None
    if len(pcr_df) >= 8:
        pcr_df["sigma2_group"] = quartile_groups(pcr_df["sigma2_tx"].to_numpy())
        grp = pcr_df.dropna(subset=["sigma2_group"])
        if grp["sigma2_group"].nunique() == 2:
            km = km_logrank(grp, "sigma2_group")
    return {"cox": cox, "km_within_pcr": km}


def run(cfg):
    from ..data.paired_dataset import load_paired
    from ..inference.run import run_inference
    df, z_pre, z_post = load_paired(cfg.paths.paired)
    records = run_inference(z_pre, z_post, df["drug_id"].to_numpy(),
                            df["pcr"].to_numpy(), df["fold"].to_numpy(),
                            cfg.paths.ckpt, out_dir=f"{cfg.paths.runs}/samples")
    return run_exp2(records, df)
