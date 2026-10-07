"""Balanced two-sample comparison. Pure numpy — runs on the dev box.

The point of report_balanced is that an unbalanced permutation test rejects on a
trivially small difference purely because one side has many more samples. These
tests pin that behaviour down so the fix cannot silently regress.
"""
import numpy as np

from txmorph.eval.distribution import (mmd_permutation_test, report_balanced)


def test_unbalanced_test_rejects_where_balanced_does_not():
    """The motivating failure: same distributions, tiny shift, but 100:1 sizes.

    This is exactly the situation the transport evaluation was in (2,500 generated
    vs ~20 real) and is why every fold reported p=0.002.
    """
    rng = np.random.default_rng(0)
    real = rng.normal(size=(20, 8))
    gen = rng.normal(size=(2000, 8)) + 0.35        # small systematic offset

    unbalanced = mmd_permutation_test(gen, real, n_perm=300, seed=0)
    balanced = report_balanced(gen, real, n_rep=15, n_perm=300, seed=0)

    assert unbalanced["p_value"] < 0.05            # over-powered: rejects
    assert balanced["p_value_median"] > unbalanced["p_value"]


def test_balanced_does_not_reject_when_distributions_match():
    rng = np.random.default_rng(1)
    real = rng.normal(size=(25, 6))
    gen = rng.normal(size=(2500, 6))
    out = report_balanced(gen, real, n_rep=20, n_perm=300, seed=0)
    assert out["p_value_median"] > 0.05
    assert out["reject_rate_05"] < 0.35


def test_balanced_still_rejects_a_large_real_difference():
    """Sanity: the fix must not make the test blind."""
    rng = np.random.default_rng(2)
    real = rng.normal(size=(25, 6))
    gen = rng.normal(size=(2500, 6)) + 3.0
    out = report_balanced(gen, real, n_rep=20, n_perm=300, seed=0)
    assert out["p_value_median"] < 0.05
    assert out["reject_rate_05"] > 0.8


def test_sides_are_equal_size():
    rng = np.random.default_rng(3)
    real = rng.normal(size=(17, 4))
    gen = rng.normal(size=(900, 4))
    out = report_balanced(gen, real, n_rep=5, n_perm=100, seed=0)
    assert out["n_per_side"] == 17
    assert out["n_real"] == 17
    assert out["n_gen_pool"] == 900


def test_reproducible_under_seed():
    rng = np.random.default_rng(4)
    real = rng.normal(size=(20, 5))
    gen = rng.normal(size=(500, 5))
    a = report_balanced(gen, real, n_rep=8, n_perm=100, seed=7)
    b = report_balanced(gen, real, n_rep=8, n_perm=100, seed=7)
    assert a["p_value_median"] == b["p_value_median"]
    assert a["mmd2_mean"] == b["mmd2_mean"]
