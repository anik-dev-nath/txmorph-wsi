"""Exp4 stat core: predicted-vs-observed per-arm Spearman + bootstrap CI.

Pure numpy/scipy (no torch); the fold-aware counterfactual re-inference in
``run_exp4`` is exercised on the server (needs trained checkpoints)."""
import numpy as np

from txmorph.experiments.exp4 import compare_to_observed


def test_compare_to_observed_monotone_rho_is_one():
    pred = {"a": 0.10, "b": 0.30, "c": 0.50, "d": 0.70}
    obs = {"a": 0.15, "b": 0.25, "c": 0.55, "d": 0.80}
    r = compare_to_observed(pred, obs, n_boot=200, seed=0)
    assert r["spearman_rho"] == 1.0
    lo, hi = r["ci"]
    assert -1.0 <= lo <= hi <= 1.0
    assert r["arms"] == ["a", "b", "c", "d"]
    assert np.allclose(r["pred"], [0.10, 0.30, 0.50, 0.70])


def test_compare_to_observed_uses_only_shared_arms():
    pred = {"a": 0.2, "b": 0.4, "x": 0.9}     # 'x' has no observed rate
    obs = {"a": 0.1, "b": 0.5}
    r = compare_to_observed(pred, obs, n_boot=50)
    assert set(r["arms"]) == {"a", "b"}
