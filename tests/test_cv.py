import numpy as np
import pytest
from txmorph.utils.cv import assign_folds, assert_no_leakage, train_val_masks


def _rows_per_patient(n_patients=200, max_rows=3, seed=0):
    """Build patient_ids with 1-3 rows each (mimics multiple slides/timepoints)."""
    rng = np.random.default_rng(seed)
    pids, labels = [], []
    for p in range(n_patients):
        y = int(rng.random() < 0.3)              # ~30% pCR
        for _ in range(rng.integers(1, max_rows + 1)):
            pids.append(f"P{p:04d}"); labels.append(y)
    return pids, labels


def test_no_patient_spans_folds():
    pids, labels = _rows_per_patient()
    folds = assign_folds(pids, labels, n_folds=5, seed=0)
    assert_no_leakage(pids, folds)               # must not raise
    # explicit: every patient's rows share one fold
    seen = {}
    for p, f in zip(pids, folds):
        if p in seen:
            assert seen[p] == f
        seen[p] = f


def test_determinism_and_seed_sensitivity():
    pids, labels = _rows_per_patient()
    a = assign_folds(pids, labels, seed=0)
    b = assign_folds(pids, labels, seed=0)
    c = assign_folds(pids, labels, seed=1)
    assert np.array_equal(a, b)                  # same seed -> identical
    assert not np.array_equal(a, c)              # different seed -> different


def test_stratification_balances_pcr_rate():
    pids, labels = _rows_per_patient(n_patients=400)
    labels = np.array(labels)
    folds = assign_folds(pids, labels, n_folds=5, seed=0)
    overall = labels.mean()
    for k in range(5):
        rate = labels[folds == k].mean()
        assert abs(rate - overall) < 0.10        # class balance preserved per fold


def test_all_folds_nonempty_and_in_range():
    pids, labels = _rows_per_patient()
    folds = assign_folds(pids, labels, n_folds=5, seed=2)
    assert set(np.unique(folds)) == set(range(5))


def test_guard_catches_injected_leakage():
    pids = ["P1", "P1", "P2", "P2"]
    leaky = [0, 1, 0, 0]                          # P1 spans folds 0 and 1
    with pytest.raises(AssertionError):
        assert_no_leakage(pids, leaky)


def test_train_val_masks_partition():
    folds = np.array([0, 1, 2, 0, 1])
    tr, va = train_val_masks(folds, val_fold=1)
    assert np.array_equal(va, folds == 1)
    assert np.array_equal(tr, ~va)
    assert (tr & va).sum() == 0 and (tr | va).all()
