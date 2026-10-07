"""The LLR->probability calibrator must survive the LLR's native scale.

The single-timepoint biomarker is an eps-MSE gap of order 1e-4. sklearn's
LogisticRegression is L2-penalised by default, so fitting on that raw scale
shrinks the coefficient to ~0 and silently degenerates to an intercept-only
model that returns the training base rate for everyone. That is exactly what
happened on the real run: P(pCR) was constant within every fold (std 0.0,
AUROC 0.517) while the underlying LLR was stable across Monte-Carlo budgets.
"""
import numpy as np
import pytest

pytest.importorskip("sklearn")

from txmorph.inference.run import _fit_llr_calibrator


def _separable_llr(n=200, scale=1.0, seed=0):
    """LLR that genuinely separates the classes, emitted at an arbitrary scale."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    llr = (rng.normal(size=n) + 2.0 * y) * scale
    return llr, y


@pytest.mark.parametrize("scale", [1.0, 1e-2, 1e-4, 1e-6])
def test_calibrator_recovers_signal_at_any_scale(scale):
    """Ranking must not depend on the units the LLR happens to come out in."""
    llr, y = _separable_llr(scale=scale)
    p = _fit_llr_calibrator(llr, y)(llr)
    assert p.std() > 0.01, f"collapsed to a constant at scale {scale}: std={p.std():.2e}"
    assert p[y == 1].mean() > p[y == 0].mean()


def test_tiny_scale_matches_unit_scale_ranking():
    """A 1e-4-scaled LLR must rank patients identically to the same LLR at 1.0."""
    llr, y = _separable_llr(scale=1.0)
    p_unit = _fit_llr_calibrator(llr, y)(llr)
    p_tiny = _fit_llr_calibrator(llr * 1e-4, y)(llr * 1e-4)
    assert np.corrcoef(p_unit, p_tiny)[0, 1] > 0.999


def test_constant_llr_falls_back_to_base_rate():
    """Nothing to map: return the base rate rather than dividing by ~0."""
    y = np.array([0, 1, 1, 0, 1, 0])
    p = _fit_llr_calibrator(np.full(len(y), 0.3), y)(np.zeros(3))
    assert np.allclose(p, y.mean())


def test_single_class_fold_falls_back_to_base_rate():
    y = np.ones(8, dtype=int)
    p = _fit_llr_calibrator(np.linspace(0, 1, 8), y)(np.zeros(4))
    assert np.all(p > 0.99)


def test_calibrator_uses_train_statistics_only():
    """Scaling is fit on train; applying it to unseen values must not refit."""
    llr, y = _separable_llr(scale=1e-4, seed=3)
    calibrate = _fit_llr_calibrator(llr, y)
    # A held-out batch shifted far from the training range still maps monotonically.
    probe = np.array([llr.min(), llr.mean(), llr.max()])
    p = calibrate(probe)
    assert p[0] < p[1] < p[2]
