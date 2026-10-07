"""Benjamini-Hochberg FDR correctness (numpy)."""
import numpy as np
from txmorph.eval.stats import benjamini_hochberg


def test_bh_matches_reference():
    p = np.array([0.001, 0.01, 0.02, 0.2, 0.5, 0.9])
    rej, q = benjamini_hochberg(p, alpha=0.05)
    # q_i = p_i * n / rank, monotone-adjusted
    assert np.allclose(q, [0.006, 0.03, 0.04, 0.3, 0.6, 0.9], atol=1e-6)
    assert rej.tolist() == [True, True, True, False, False, False]


def test_bh_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.random(50)
    _, q = benjamini_hochberg(p)
    order = np.argsort(p)
    qs = q[order]
    assert np.all(qs[:-1] <= qs[1:] + 1e-12)        # non-decreasing in p-order
    assert np.all((q >= 0) & (q <= 1))


def test_bh_all_null_none_rejected():
    p = np.full(20, 0.9)
    rej, _ = benjamini_hochberg(p)
    assert not rej.any()
