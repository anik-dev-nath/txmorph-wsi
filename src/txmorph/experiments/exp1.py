"""Experiment 1 - Calibrated pCR (CLAUDE.md S11-Exp1).

P(pCR) = fraction of the 100 samples in the (train-fold-fit) pCR zone. Metrics:
AUROC, AUPRC (prominent - minority class), ECE (15 bins), Brier, each with
bootstrap 95% CIs. Baseline: an attention-MIL-style logistic on z_pre. Claim:
match AUROC, beat calibration.
"""
from __future__ import annotations

import numpy as np

from ..eval.classification import auroc, auprc, bootstrap_ci
from ..eval.calibration import expected_calibration_error, brier_score, reliability_curve


def evaluate_probabilities(p_pcr, labels, n_boot: int = 1000, seed: int = 0):
    """Compute AUROC/AUPRC/ECE/Brier with bootstrap CIs for a probability vector."""
    import torch
    p = np.asarray(p_pcr, dtype=float)
    y = np.asarray(labels).astype(int)

    def _ece(pp, yy):
        return expected_calibration_error(torch.tensor(pp), torch.tensor(yy))

    def _brier(pp, yy):
        return brier_score(torch.tensor(pp), torch.tensor(yy))

    metrics = {
        "auroc": bootstrap_ci(auroc, p, y, n_boot=n_boot, seed=seed),
        "auprc": bootstrap_ci(auprc, p, y, n_boot=n_boot, seed=seed),
        "ece": bootstrap_ci(_ece, p, y, n_boot=n_boot, seed=seed),
        "brier": bootstrap_ci(_brier, p, y, n_boot=n_boot, seed=seed),
    }
    conf, acc, cnt = reliability_curve(torch.tensor(p), torch.tensor(y))
    metrics["reliability"] = {"conf": conf, "acc": acc, "count": cnt}
    return metrics


def baseline_zpre_logistic(z_pre, pcr, folds, seed: int = 0, calibration: str = "none",
                           tune_C: bool = True):
    """Cross-validated discriminative baseline: logistic pCR from ``z_pre``.

    Fit per fold on train patients, predict held-out; returns out-of-fold probs.

    **Configuration matters here, and an earlier version got it wrong.** The
    original baseline fit ``LogisticRegression(class_weight="balanced")`` on raw
    512-d embeddings with scikit-learn's default ``C=1.0`` and no standardisation.
    With ~100 training patients and 512 features that is badly under-regularised:
    it overfits, emits over-confident probabilities, and so scores a poor ECE
    (0.161) *by misconfiguration*. Comparing a generative model's calibration
    against that is a strawman comparison, and it is what the original "2x better
    calibration" claim rested on. A properly configured baseline reaches ECE
    0.091 uncalibrated and 0.059 with Platt scaling -- better than the generative
    model's 0.084 (`scripts/analysis/calib_test.py`).

    Everything below is fit **inside the training fold only**:
      * standardisation (mean/sd from train patients);
      * the L2 strength ``C``, by inner stratified CV over a log grid;
      * the calibration map, when ``calibration`` is ``"platt"`` or ``"isotonic"``.

    Parameters
    ----------
    calibration : ``"none"`` | ``"platt"`` | ``"isotonic"``.
    tune_C : select ``C`` by inner CV (default). ``False`` pins ``C=0.01``, the
        value the diagnostic scripts used, for exact comparability with them.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    z_pre = np.asarray(z_pre, dtype=float)
    pcr = np.asarray(pcr).astype(int)
    folds = np.asarray(folds)
    oof = np.zeros(len(pcr), dtype=float)

    for k in sorted(np.unique(folds)):
        tr, te = folds != k, folds == k
        base = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced",
                                       random_state=seed)),
        ])
        if tune_C:
            n_min = int(np.bincount(pcr[tr]).min())
            inner = StratifiedKFold(n_splits=min(5, max(2, n_min)), shuffle=True,
                                    random_state=seed)
            est = GridSearchCV(base, {"clf__C": np.logspace(-4, 1, 12)},
                               scoring="roc_auc", cv=inner, n_jobs=1)
        else:
            base.set_params(clf__C=0.01)
            est = base

        if calibration in ("platt", "isotonic"):
            method = "sigmoid" if calibration == "platt" else "isotonic"
            n_min = int(np.bincount(pcr[tr]).min())
            est = CalibratedClassifierCV(est, method=method,
                                         cv=min(5, max(2, n_min)))
        est.fit(z_pre[tr], pcr[tr])
        oof[te] = est.predict_proba(z_pre[te])[:, 1]
    return oof


def run_exp1(records, z_pre, folds, n_boot: int = 1000, seed: int = 0):
    """records: from inference.run (need p_pcr, pcr, index). Returns TxMorph vs baseline."""
    idx = np.array([r["index"] for r in records])
    p_pcr = np.array([r["p_pcr"] for r in records])
    y = np.array([r["pcr"] for r in records])

    tx = evaluate_probabilities(p_pcr, y, n_boot=n_boot, seed=seed)
    base_probs = baseline_zpre_logistic(np.asarray(z_pre)[idx], y,
                                        np.asarray(folds)[idx], seed=seed)
    base = evaluate_probabilities(base_probs, y, n_boot=n_boot, seed=seed)
    return {"txmorph": tx, "baseline": base}


def run(cfg):
    from ..data.paired_dataset import load_paired
    from ..inference.run import run_inference
    df, z_pre, z_post = load_paired(cfg.paths.paired)
    drug_feats = df["drug_id"].to_numpy()
    records = run_inference(z_pre, z_post, drug_feats, df["pcr"].to_numpy(),
                            df["fold"].to_numpy(), cfg.paths.ckpt,
                            out_dir=f"{cfg.paths.runs}/samples")
    return run_exp1(records, z_pre, df["fold"].to_numpy())
