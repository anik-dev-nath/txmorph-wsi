"""Minibatch optimal-transport coupling between unpaired pre- and post-treatment
slide latents (CLAUDE.md S1, unpaired reframing 2026-09-08).

WHY THIS MODULE EXISTS
    The scientific claim is p(z_post | z_pre, drug). No open cohort provides
    per-patient (z_pre, z_post) pairs: IMPRESS is pre-treatment biopsies only and
    Post-NAT-BRCA is post-treatment resections from *different* patients. So the
    pairs needed to train a conditional generative model do not exist in the data.

    Minibatch OT supplies them. Within a batch we solve the assignment that
    minimises total squared distance between the pre and post samples, and treat
    the matched indices as pseudo-pairs. Training a flow on OT pseudo-pairs
    converges to a valid transport between the two marginals (Tong et al.,
    "Improving and generalizing flow-based generative models with minibatch
    optimal transport"), even though no individual pseudo-pair is a real patient.

    What this buys and what it does not: the learned map is correct *in
    distribution*. It is NOT a claim that patient i's post-treatment slide looks
    like post sample j. Per-patient correspondence is unverifiable on unpaired
    data and must never be claimed -- see ``docs/LIMITATIONS.md``.

STRATIFICATION
    Post-NAT-BRCA contains receptor subtypes absent from IMPRESS (e.g. HR+/HER2-),
    and coupling a HER2+ biopsy to an HR+/HER2- resection would fabricate a
    transition that no patient underwent. ``groups`` restricts the assignment so
    mass only moves within a stratum. Always pass it.
"""
from __future__ import annotations

import numpy as np


def _sqdist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise squared euclidean cost matrix, shape (len(x), len(y))."""
    x2 = (x ** 2).sum(1)[:, None]
    y2 = (y ** 2).sum(1)[None, :]
    return np.maximum(x2 + y2 - 2.0 * x @ y.T, 0.0)


def ot_coupling(x: np.ndarray, y: np.ndarray, groups_x=None, groups_y=None,
                mode: str = "exact", seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Couple rows of ``x`` (pre) to rows of ``y`` (post).

    Returns ``(ix, iy)`` index arrays of equal length: ``x[ix[k]]`` is paired with
    ``y[iy[k]]``. Not every row need appear when the batches differ in size.

    ``mode``:
      ``"exact"``       Hungarian assignment on squared euclidean cost. This is
                        the OT plan for equal-mass discrete measures.
      ``"independent"`` random pairing, ignoring cost. This is the ablation that
                        isolates what OT contributes -- flow matching still
                        trains, but against an arbitrary coupling.

    ``groups_x`` / ``groups_y`` restrict pairing to matching strata (e.g. receptor
    subtype). When given, the assignment is solved separately inside each stratum
    and concatenated.
    """
    if mode not in ("exact", "independent"):
        raise ValueError(f"unknown coupling mode: {mode}")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(f"x{x.shape} and y{y.shape} must be 2-D with equal width")

    if groups_x is None and groups_y is None:
        return _couple_block(x, y, np.arange(len(x)), np.arange(len(y)), mode, seed)
    if groups_x is None or groups_y is None:
        raise ValueError("pass both groups_x and groups_y, or neither")

    gx = np.asarray(groups_x)
    gy = np.asarray(groups_y)
    if len(gx) != len(x) or len(gy) != len(y):
        raise ValueError("groups must align with x and y")

    ix_all, iy_all = [], []
    for g in np.unique(gx):
        rx = np.where(gx == g)[0]
        ry = np.where(gy == g)[0]
        if len(rx) == 0 or len(ry) == 0:
            # a stratum present on one side only contributes no pairs; silently
            # dropping it is correct -- we cannot transport what we cannot observe
            continue
        a, b = _couple_block(x[rx], y[ry], rx, ry, mode, seed)
        ix_all.append(a)
        iy_all.append(b)
    if not ix_all:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    return np.concatenate(ix_all), np.concatenate(iy_all)


def _couple_block(xb, yb, rx, ry, mode, seed):
    n, m = len(xb), len(yb)
    k = min(n, m)
    if mode == "independent":
        rng = np.random.default_rng(seed)
        return rx[rng.permutation(n)[:k]], ry[rng.permutation(m)[:k]]
    from scipy.optimize import linear_sum_assignment
    cost = _sqdist(xb, yb)
    r, c = linear_sum_assignment(cost)      # rectangular-safe; matches min(n, m)
    return rx[r], ry[c]


def coupling_cost(x: np.ndarray, y: np.ndarray, ix: np.ndarray,
                  iy: np.ndarray) -> float:
    """Mean squared distance of a coupling. Diagnostic: the exact coupling must
    score <= the independent one, which is the sanity check that OT is doing
    anything at all."""
    if len(ix) == 0:
        return float("nan")
    d = np.asarray(x, dtype=np.float64)[ix] - np.asarray(y, dtype=np.float64)[iy]
    return float((d ** 2).sum(1).mean())
