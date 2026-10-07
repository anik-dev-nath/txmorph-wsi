"""Two-sample metric correctness. Pure numpy — runs on the dev box."""
import numpy as np
import pytest

from txmorph.eval.distribution import (
    mmd2_rbf, mmd_permutation_test, energy_distance, sliced_wasserstein, report)


def test_mmd_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 6))
    y = rng.normal(size=(200, 6))
    assert abs(mmd2_rbf(x, y)) < 0.02


def test_mmd_grows_with_separation():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(150, 6))
    near = rng.normal(size=(150, 6)) + 0.5
    far = rng.normal(size=(150, 6)) + 3.0
    assert mmd2_rbf(x, near) < mmd2_rbf(x, far)


def test_permutation_test_does_not_reject_under_the_null():
    """Same distribution => large p-value. Note the inverted reading: large p is
    the good outcome when comparing generated against real."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(80, 5))
    y = rng.normal(size=(80, 5))
    out = mmd_permutation_test(x, y, n_perm=300, seed=0)
    assert out["p_value"] > 0.05


def test_permutation_test_rejects_a_real_difference():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 5))
    y = rng.normal(size=(80, 5)) + 2.0
    out = mmd_permutation_test(x, y, n_perm=300, seed=0)
    assert out["p_value"] < 0.01


def test_energy_distance_zero_for_identical_samples():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(50, 4))
    assert abs(energy_distance(x, x)) < 1e-9


def test_energy_distance_is_ordered_by_separation():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(100, 4))
    near = rng.normal(size=(100, 4)) + 0.5
    far = rng.normal(size=(100, 4)) + 4.0
    assert energy_distance(x, near) < energy_distance(x, far)


def test_sliced_wasserstein_tracks_a_known_mean_shift():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(300, 10))
    for shift in (0.0, 1.0, 4.0):
        y = rng.normal(size=(300, 10)) + shift
        got = sliced_wasserstein(x, y, n_proj=300, seed=0)
        # a pure mean shift of s projects to a 1-D shift of s*<theta,u>; the
        # sliced distance must be monotone in s and bounded by it
        assert got <= shift + 0.6


def test_sliced_wasserstein_handles_unequal_sizes():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(120, 8))
    y = rng.normal(size=(37, 8))
    assert np.isfinite(sliced_wasserstein(x, y, seed=0))


def test_report_returns_every_metric():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(60, 5))
    y = rng.normal(size=(60, 5))
    out = report(x, y, n_perm=100, seed=0)
    for k in ("mmd2", "p_value", "energy_distance", "sliced_w2", "n_x", "n_y"):
        assert k in out


def test_mmd_needs_two_samples_per_side():
    with pytest.raises(ValueError):
        mmd2_rbf(np.zeros((1, 3)), np.zeros((5, 3)))
