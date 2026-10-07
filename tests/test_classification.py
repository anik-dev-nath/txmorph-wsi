"""AUROC/AUPRC + bootstrap CI (numpy, no torch)."""
import numpy as np
from txmorph.eval.classification import auroc, auprc, bootstrap_ci


def test_auroc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0


def test_auroc_ties_half():
    y = np.array([0, 1])
    assert auroc(np.array([0.5, 0.5]), y) == 0.5


def test_auprc_all_positive_first():
    y = np.array([1, 1, 0, 0])
    assert abs(auprc(np.array([0.9, 0.8, 0.2, 0.1]), y) - 1.0) < 1e-9


def test_auprc_matches_sklearn():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 100)
    s = y * 0.5 + rng.random(100)
    assert abs(auprc(s, y) - average_precision_score(y, s)) < 1e-6


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = y * 0.6 + rng.random(200)
    pt, lo, hi = bootstrap_ci(auroc, s, y, n_boot=300, seed=0)
    assert lo <= pt <= hi
    assert 0.0 <= lo <= hi <= 1.0
