"""Survival analysis (Experiment 2): Cox PH, KM, log-rank, Harrell C, Schoenfeld
(CLAUDE.md S11-Exp2).

Uses lifelines. ``sigma2_tx`` enters the Cox model as a CONTINUOUS covariate to
preserve power; quartile grouping is reserved for the KM plot only. The PH
assumption is checked via Schoenfeld residuals. lifelines is imported lazily so
the module imports on a bare box.
"""
from __future__ import annotations

import numpy as np


def cox_ph(df, duration_col: str = "rfs_time", event_col: str = "rfs_event",
           covariates=("sigma2_tx", "pcr", "subtype", "stage")):
    """Fit Cox PH. Returns a dict with per-covariate HR/CI/p, C-index, and the
    fitted ``CoxPHFitter`` (for Schoenfeld residuals / plotting).
    """
    from lifelines import CoxPHFitter

    cols = [duration_col, event_col, *covariates]
    data = df[cols].dropna().copy()
    cph = CoxPHFitter()
    cph.fit(data, duration_col=duration_col, event_col=event_col)
    summ = cph.summary
    results = {}
    for cov in covariates:
        if cov in summ.index:
            results[cov] = {
                "hr": float(np.exp(summ.loc[cov, "coef"])),
                "ci_low": float(np.exp(summ.loc[cov, "coef lower 95%"])),
                "ci_high": float(np.exp(summ.loc[cov, "coef upper 95%"])),
                "p": float(summ.loc[cov, "p"]),
            }
    return {"covariates": results, "c_index": float(cph.concordance_index_),
            "model": cph}


def schoenfeld_test(cph, df, duration_col: str = "rfs_time",
                    event_col: str = "rfs_event"):
    """Proportional-hazards check via Schoenfeld residuals. Returns the test table."""
    data = df.dropna().copy()
    return cph.check_assumptions(data, show_plots=False)


def km_logrank(df, group_col: str, duration_col: str = "rfs_time",
               event_col: str = "rfs_event"):
    """Kaplan-Meier curves per group + a log-rank test across two groups.

    Returns {"curves": {group: KaplanMeierFitter}, "logrank_p": float}.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    curves = {}
    groups = sorted(df[group_col].dropna().unique())
    for g in groups:
        sub = df[df[group_col] == g]
        kmf = KaplanMeierFitter()
        kmf.fit(sub[duration_col], sub[event_col], label=str(g))
        curves[g] = kmf

    logrank_p = float("nan")
    if len(groups) == 2:
        a = df[df[group_col] == groups[0]]
        b = df[df[group_col] == groups[1]]
        res = logrank_test(a[duration_col], b[duration_col],
                           a[event_col], b[event_col])
        logrank_p = float(res.p_value)
    return {"curves": curves, "logrank_p": logrank_p, "groups": groups}


def quartile_groups(values, low_q: float = 0.25, high_q: float = 0.75):
    """Label a continuous vector into 'low'/'high' by its quartiles (NaN between).

    For the KM plot of high vs low ``sigma2_tx`` within pCR achievers.
    """
    v = np.asarray(values, dtype=float)
    lo, hi = np.quantile(v, low_q), np.quantile(v, high_q)
    out = np.full(v.shape, None, dtype=object)
    out[v <= lo] = "low"
    out[v >= hi] = "high"
    return out
