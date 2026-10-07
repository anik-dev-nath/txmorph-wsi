"""Multiple-testing correction + association helpers (CLAUDE.md S11, S14).

Benjamini-Hochberg FDR across the many gene/pathway association tests (Exp 3),
plus a thin correlation helper. Pure numpy; no statsmodels hard dep for BH.
"""
from __future__ import annotations

import numpy as np


def benjamini_hochberg(pvalues, alpha: float = 0.05):
    """BH-FDR. Returns (rejected_bool, qvalues) aligned to input order."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity of q from the largest rank down
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    qvalues = np.empty(n, dtype=float)
    qvalues[order] = q
    rejected = qvalues <= alpha
    return rejected, qvalues


def pointbiserial(binary_membership, continuous_expr):
    """Correlation of a binary cluster membership with a continuous expression
    vector. Returns (r, p) via a two-sided t-approximation.
    """
    from scipy import stats
    x = np.asarray(binary_membership).astype(float)
    y = np.asarray(continuous_expr, dtype=float)
    r, p = stats.pointbiserialr(x, y)
    return float(r), float(p)
