"""Classification metrics with bootstrap CIs (CLAUDE.md S11, S13).

AUROC/AUPRC plus a generic bootstrap-CI wrapper (1000 resamples) so every headline
number is reported as mean +/- 95% CI, not a single point. Pure numpy (+ the torch
``auroc`` in calibration.py for the rank-based variant); no sklearn hard dep here.
"""
from __future__ import annotations

import numpy as np


def auroc(scores, labels) -> float:
    """Rank-based AUROC (Mann-Whitney U). NaN if a class is absent."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {}
    start = 0
    for j, c in enumerate(counts):
        avg[j] = (start + 1 + start + c) / 2.0
        start += c
    ranks = np.array([avg[v] for v in inv])
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = ranks[labels == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auprc(scores, labels) -> float:
    """Average precision (area under precision-recall), the minority-class metric."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    # step-wise AP: sum precision * delta-recall
    prev_recall = 0.0
    ap = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_recall)
        prev_recall = r
    return float(ap)


def bootstrap_ci(metric_fn, *arrays, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0):
    """Bootstrap CI for ``metric_fn(*resampled_arrays)``.

    Returns (point_estimate, ci_low, ci_high). All arrays are resampled with the
    same indices (paired), preserving score-label correspondence.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    point = metric_fn(*arrays)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = metric_fn(*[a[idx] for a in arrays])
        if not np.isnan(val):
            stats.append(val)
    if not stats:
        return point, float("nan"), float("nan")
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return float(point), lo, hi
