"""Morphology -> expression bridge: predict pathway signature scores from a slide latent.

Exp 3 needs expression aligned to the *same* patients it clusters, which no treatment
cohort provides (see ``build_tcga_bridge_manifest.py``). This module learns
``slide latent -> signature score`` on TCGA-BRCA, where both modalities exist per
patient, so the predictor can then score IMPRESS/Post-NAT slides that have no RNA-seq.

Three things this has to get right, because the regime is n ~ 10^2 cases against a
512-d latent (p >> n):

* **Ridge, with alpha chosen inside the training fold.** Unregularised regression at
  p >> n interpolates the training cases exactly and generalises at chance.
* **Case-level folds and train-fold-only standardisation**, the same leakage rule the
  rest of the project uses for patients.
* **A permutation null.** At this n and p a held-out correlation of a few tenths can
  arise from noise alone. ``permutation_pvalue`` refits the whole CV pipeline on
  shuffled targets, so the reported p reflects the fitting procedure rather than an
  analytic assumption. Exp 3 should not lean on a signature whose bridge fails here.

Predicted scores are a weaker instrument than measured RNA-seq and must be described
that way wherever they are used.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

_ALPHAS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5)


@dataclass
class BridgeResult:
    """Held-out performance of one signature's predictor."""
    signature: str
    pearson: float
    spearman: float
    spearman_ci: tuple = (float("nan"), float("nan"))
    p_permutation: float = float("nan")
    n: int = 0
    oof: np.ndarray = field(default=None, repr=False)

    @property
    def passes(self) -> bool:
        """Usable downstream: positive rank correlation that beats the permutation null."""
        return bool(self.spearman > 0 and self.p_permutation < 0.05)


def _folds_by_case(n: int, n_folds: int, seed: int) -> np.ndarray:
    """Seeded case-level fold assignment (one row == one case here)."""
    rng = np.random.default_rng(seed)
    folds = np.arange(n) % n_folds
    rng.shuffle(folds)
    return folds


def _fit_predict_fold(X_tr, y_tr, X_te, alphas):
    """Standardise on the training fold only, pick alpha by inner CV, predict test."""
    from sklearn.linear_model import RidgeCV

    mu, sd = X_tr.mean(0, keepdims=True), X_tr.std(0, keepdims=True) + 1e-8
    ym = y_tr.mean()
    model = RidgeCV(alphas=alphas)
    model.fit((X_tr - mu) / sd, y_tr - ym)
    return model.predict((X_te - mu) / sd) + ym


def cross_val_predict(Z, y, n_folds: int = 5, seed: int = 0, alphas=_ALPHAS):
    """Out-of-fold predictions of one signature score from slide latents."""
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if Z.shape[0] != y.shape[0]:
        raise ValueError(f"Z has {Z.shape[0]} rows but y has {y.shape[0]}")
    n_folds = int(min(n_folds, Z.shape[0]))
    folds = _folds_by_case(len(y), n_folds, seed)
    oof = np.zeros_like(y)
    for k in range(n_folds):
        tr, te = folds != k, folds == k
        if not te.any() or tr.sum() < 2:
            continue
        oof[te] = _fit_predict_fold(Z[tr], y[tr], Z[te], alphas)
    return oof


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _pearson(a, b) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def permutation_pvalue(Z, y, observed: float, n_perm: int = 200, n_folds: int = 5,
                       seed: int = 0) -> float:
    """Fraction of label shuffles whose CV Spearman reaches ``observed``.

    The *whole* CV fit is repeated per shuffle, so alpha selection and fold-wise
    standardisation are inside the null. A cheaper null that permuted only the final
    predictions would understate how much a p >> n ridge can overfit.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float64).ravel()
    hits = 0
    for i in range(n_perm):
        perm = rng.permutation(len(y))
        r = _spearman(cross_val_predict(Z, y[perm], n_folds=n_folds, seed=seed + i + 1),
                      y[perm])
        if np.isfinite(r) and r >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)          # add-one: never reports p = 0


def _bootstrap_ci(pred, true, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI for Spearman, resampling held-out cases."""
    rng = np.random.default_rng(seed)
    n = len(true)
    if n < 3:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = _spearman(pred[idx], true[idx])
        if np.isfinite(r):
            stats.append(r)
    if not stats:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def train_bridge_cv(Z, scores, signature_names, n_folds: int = 5, seed: int = 0,
                    n_perm: int = 200, n_boot: int = 1000, alphas=_ALPHAS):
    """Fit + evaluate one predictor per signature. Returns [BridgeResult].

    ``Z``      (n_cases, d) slide latents, one row per case.
    ``scores`` (n_cases, n_signatures) measured signature scores from RNA-seq.

    Evaluation is entirely out-of-fold; nothing here is fit on data it is scored on.
    """
    Z = np.asarray(Z, dtype=np.float64)
    S = np.asarray(scores, dtype=np.float64)
    if S.ndim == 1:
        S = S[:, None]
    if Z.shape[0] != S.shape[0]:
        raise ValueError(f"{Z.shape[0]} latents vs {S.shape[0]} score rows")
    if S.shape[1] != len(signature_names):
        raise ValueError(f"{S.shape[1]} score columns vs "
                         f"{len(signature_names)} signature names")

    results = []
    for k, name in enumerate(signature_names):
        y = S[:, k]
        oof = cross_val_predict(Z, y, n_folds=n_folds, seed=seed, alphas=alphas)
        rho = _spearman(oof, y)
        results.append(BridgeResult(
            signature=name,
            pearson=_pearson(oof, y),
            spearman=rho,
            spearman_ci=_bootstrap_ci(oof, y, n_boot=n_boot, seed=seed),
            p_permutation=(permutation_pvalue(Z, y, rho, n_perm=n_perm,
                                              n_folds=n_folds, seed=seed)
                           if np.isfinite(rho) and n_perm > 0 else float("nan")),
            n=len(y), oof=oof))
    return results


def fit_full_bridge(Z, scores, alphas=_ALPHAS):
    """Refit on ALL cases for downstream application. Returns ``predict(Z_new)``.

    Only for scoring cohorts that were never part of this fit (IMPRESS/Post-NAT).
    Reported performance must come from :func:`train_bridge_cv`, never from this.
    """
    from sklearn.linear_model import RidgeCV

    Z = np.asarray(Z, dtype=np.float64)
    S = np.asarray(scores, dtype=np.float64)
    if S.ndim == 1:
        S = S[:, None]
    mu, sd = Z.mean(0, keepdims=True), Z.std(0, keepdims=True) + 1e-8
    ym = S.mean(0, keepdims=True)
    models = []
    for k in range(S.shape[1]):
        m = RidgeCV(alphas=alphas)
        m.fit((Z - mu) / sd, S[:, k] - ym[0, k])
        models.append(m)

    def predict(Z_new):
        Zn = (np.asarray(Z_new, dtype=np.float64) - mu) / sd
        return np.stack([m.predict(Zn) + ym[0, k] for k, m in enumerate(models)], axis=1)

    return predict
