"""pCR-zone logistic g(z) = sigmoid(w.z + b), fit on the TRAINING FOLD ONLY
(CLAUDE.md S5, S6.6, S11-Exp1).

The zone is a linear classifier of pCR from the observed post-treatment embedding
``z_post``. At inference, P(pCR) = fraction of the 100 sampled z_post that fall in
the zone. Fitting it inside the train fold (never on test patients) is a hard
no-leakage requirement checked by ``tests/test_leakage.py``.
"""
from __future__ import annotations

import numpy as np

from .derive import PCRZone


def fit_pcr_zone(z_post_train: np.ndarray, pcr_train: np.ndarray,
                 threshold: float = 0.5, C: float = 1.0, max_iter: int = 1000,
                 device: str = "cpu") -> PCRZone:
    """Fit the pCR-zone logistic on training-fold (z_post, pcr). Returns a PCRZone.

    z_post_train: (n, d) float. pcr_train: (n,) bool/int.
    """
    import torch
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(z_post_train, dtype=np.float64)
    y = np.asarray(pcr_train).astype(int)
    clf = LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced")
    clf.fit(X, y)
    w = torch.tensor(clf.coef_.ravel(), dtype=torch.float32, device=device)
    b = float(clf.intercept_[0])
    return PCRZone(w=w, b=b, threshold=threshold)
