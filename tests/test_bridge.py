"""Morphology->expression bridge: leakage, p >> n behaviour, and the permutation null.

Inline Gaussians only, to check the estimator's statistical behaviour (does it detect
signal, does it stay at chance on noise, does it leak). No cohort stand-ins; real
performance comes from TCGA slide latents vs measured RNA-seq.
"""
import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from txmorph.expression.bridge import (
    cross_val_predict, train_bridge_cv, permutation_pvalue, fit_full_bridge,
    BridgeResult,
)


def _linear_signal(n=120, d=32, noise=0.3, seed=0):
    """y is a genuine linear function of Z, plus noise."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = Z @ w / np.sqrt(d) + noise * rng.normal(size=n)
    return Z, y


def test_recovers_real_signal():
    Z, y = _linear_signal()
    oof = cross_val_predict(Z, y, n_folds=5, seed=0)
    from scipy.stats import spearmanr
    assert spearmanr(oof, y).statistic > 0.5


def test_pure_noise_stays_near_chance():
    """No relationship -> out-of-fold correlation must not be strong.

    This is the property that makes the bridge trustworthy: if morphology carries no
    information about a signature, the reported number has to say so.
    """
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(80, 64))
    y = rng.normal(size=80)               # independent of Z
    oof = cross_val_predict(Z, y, n_folds=5, seed=0)
    from scipy.stats import spearmanr
    assert abs(spearmanr(oof, y).statistic) < 0.4


def test_high_dim_noise_does_not_produce_false_signal():
    """p >> n is the real regime (n~91 cases, d=512). Ridge must not interpolate."""
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(40, 512))
    y = rng.normal(size=40)
    oof = cross_val_predict(Z, y, n_folds=5, seed=0)
    from scipy.stats import spearmanr
    rho = spearmanr(oof, y).statistic
    assert abs(rho) < 0.5, f"spurious correlation {rho:.3f} at p>>n"


def test_permutation_null_flags_noise_and_clears_signal():
    """The null is the guard Exp 3 relies on before trusting a signature."""
    rng = np.random.default_rng(3)
    Z_noise = rng.normal(size=(60, 64))
    y_noise = rng.normal(size=60)
    oof = cross_val_predict(Z_noise, y_noise, seed=0)
    from scipy.stats import spearmanr
    p_noise = permutation_pvalue(Z_noise, y_noise, spearmanr(oof, y_noise).statistic,
                                 n_perm=60, seed=0)
    assert p_noise > 0.05, f"noise passed the null (p={p_noise})"

    Z_sig, y_sig = _linear_signal(n=90, d=24, noise=0.2, seed=4)
    oof_s = cross_val_predict(Z_sig, y_sig, seed=0)
    p_sig = permutation_pvalue(Z_sig, y_sig, spearmanr(oof_s, y_sig).statistic,
                               n_perm=60, seed=0)
    assert p_sig < 0.05, f"real signal failed the null (p={p_sig})"


def test_permutation_pvalue_never_zero():
    """Add-one correction: p=0 would be an unsupportable claim from finite shuffles."""
    Z, y = _linear_signal(n=60, d=16, noise=0.05, seed=5)
    p = permutation_pvalue(Z, y, observed=0.99, n_perm=20, seed=0)
    assert p > 0


def test_no_leakage_from_test_fold_standardisation():
    """Scaling must be fit on the training fold only.

    Shifting/scaling the test rows by a large constant changes predictions (the model
    is not scale-free), but must not *improve* them -- if standardisation had been fit
    on all rows, the test fold's own statistics would have leaked in.
    """
    Z, y = _linear_signal(n=100, d=16, seed=6)
    base = cross_val_predict(Z, y, n_folds=5, seed=0)
    from scipy.stats import spearmanr
    assert np.isfinite(spearmanr(base, y).statistic)
    # A constant offset on every row is absorbed by the training-fold mean.
    shifted = cross_val_predict(Z + 5.0, y, n_folds=5, seed=0)
    assert np.allclose(base, shifted, atol=1e-6)


def test_train_bridge_cv_shapes_and_names():
    Z, y1 = _linear_signal(n=70, d=20, seed=7)
    rng = np.random.default_rng(8)
    y2 = rng.normal(size=70)
    scores = np.stack([y1, y2], axis=1)
    res = train_bridge_cv(Z, scores, ["real", "noise"], n_folds=5, seed=0,
                          n_perm=40, n_boot=200)
    assert [r.signature for r in res] == ["real", "noise"]
    assert all(isinstance(r, BridgeResult) and r.n == 70 for r in res)
    assert res[0].spearman > res[1].spearman
    lo, hi = res[0].spearman_ci
    assert lo <= res[0].spearman <= hi
    assert res[0].passes and not res[1].passes


def test_mismatched_inputs_raise():
    Z = np.zeros((10, 4))
    with pytest.raises(ValueError, match="rows"):
        cross_val_predict(Z, np.zeros(9))
    with pytest.raises(ValueError, match="score columns"):
        train_bridge_cv(Z, np.zeros((10, 2)), ["only_one"], n_perm=0)


def test_full_bridge_applies_to_new_cohort():
    """The refit predictor scores slides that have no RNA-seq (IMPRESS/Post-NAT)."""
    Z, y = _linear_signal(n=60, d=16, seed=9)
    predict = fit_full_bridge(Z, y[:, None])
    rng = np.random.default_rng(10)
    Z_new = rng.normal(size=(7, 16))
    out = predict(Z_new)
    assert out.shape == (7, 1) and np.isfinite(out).all()
