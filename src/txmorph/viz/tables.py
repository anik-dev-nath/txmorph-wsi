"""Manuscript tables (CLAUDE.md S15): emit LaTeX .tex + CSV.

T1 cohort characteristics, T2 Exp-1 metrics, T3 Cox results, T4 ablations. Each
writer returns the written paths. CONSORT flow is generated programmatically from
paired.parquet provenance.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np


def _write(df, out_base: str):
    """Write a pandas DataFrame as both .csv and .tex. Returns (csv, tex)."""
    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    csv, tex = base.with_suffix(".csv"), base.with_suffix(".tex")
    df.to_csv(csv, index=False)
    tex.write_text(df.to_latex(index=False, float_format="%.3f"))
    return str(csv), str(tex)


def table_cohort(paired_df, out_base: str):
    """T1: N, pCR rate, subtype distribution, follow-up per cohort."""
    import pandas as pd
    rows = []
    for cohort, g in paired_df.groupby("cohort"):
        rows.append({
            "cohort": cohort, "n": len(g),
            "pcr_rate": float(g["pcr"].astype(int).mean()),
            "median_rfs": float(g["rfs_time"].median()) if "rfs_time" in g else np.nan,
            "n_subtypes": int(g["receptor_subtype"].nunique())
            if "receptor_subtype" in g else 0,
        })
    return _write(pd.DataFrame(rows), out_base)


def table_exp1(exp1_result, out_base: str):
    """T2: AUROC/AUPRC/ECE/Brier, TxMorph vs baseline, mean +/- 95% CI."""
    import pandas as pd
    rows = []
    for model in ("txmorph", "baseline"):
        r = exp1_result[model]
        row = {"model": model}
        for m in ("auroc", "auprc", "ece", "brier"):
            pt, lo, hi = r[m]
            row[m] = f"{pt:.3f} ({lo:.3f}-{hi:.3f})"
        rows.append(row)
    return _write(pd.DataFrame(rows), out_base)


def table_cox(exp2_result, out_base: str):
    """T3: HR, 95% CI, p per covariate; C-index."""
    import pandas as pd
    cov = exp2_result["cox"]["covariates"]
    rows = [{"covariate": k, "hr": v["hr"], "ci_low": v["ci_low"],
             "ci_high": v["ci_high"], "p": v["p"]} for k, v in cov.items()]
    df = pd.DataFrame(rows)
    df.attrs["c_index"] = exp2_result["cox"]["c_index"]
    return _write(df, out_base)


def table_ablations(exp5_result, out_base: str):
    """T4: each variant's metric delta."""
    import pandas as pd
    rows = []
    for name, r in exp5_result.items():
        pt, lo, hi = r["variant"]
        rows.append({"variant": name, "metric": r["metric"], "delta": r["delta"],
                     "variant_value": f"{pt:.3f} ({lo:.3f}-{hi:.3f})"})
    return _write(pd.DataFrame(rows), out_base)


def consort_flow(paired_df, out_base: str, exclusions: dict | None = None):
    """CONSORT-style cohort accounting figure from paired.parquet provenance."""
    import matplotlib.pyplot as plt
    from .style import set_style, save_figure, PALETTE

    set_style()
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.axis("off")
    steps = [f"Assessed: {len(paired_df) + sum((exclusions or {}).values())} patients"]
    for reason, n in (exclusions or {}).items():
        steps.append(f"Excluded ({reason}): {n}")
    steps.append(f"Included: {len(paired_df)} patients")
    for cohort, g in paired_df.groupby("cohort"):
        steps.append(f"  {cohort}: {len(g)}")

    y = 0.95
    for s in steps:
        ax.text(0.05, y, s, fontsize=8, va="top",
                bbox=dict(boxstyle="round", fc="white", ec=PALETTE["muted"]))
        y -= 0.9 / max(len(steps), 1)
    ax.set_title("CONSORT cohort flow")
    return save_figure(fig, out_base)
