"""Exp5 ablation-delta tabulation (needs torch for the calibration metrics)."""
import numpy as np
import pytest

pytest.importorskip("torch")

from txmorph.experiments.exp5 import ablation_delta, run_exp5


def _records(rng, n=200, sep=0.0):
    """Synthetic held-out records: p_pcr correlated with pcr by ``sep``."""
    y = (rng.random(n) < 0.3).astype(int)
    p = np.clip(0.5 + sep * (y - 0.5) + 0.1 * rng.standard_normal(n), 0, 1)
    return [{"p_pcr": float(pi), "pcr": int(yi)} for pi, yi in zip(p, y)]


def test_ablation_delta_negative_when_variant_is_worse():
    rng = np.random.default_rng(0)
    full = _records(rng, sep=0.6)          # informative full model
    weak = _records(rng, sep=0.0)          # ablated model (no signal)
    d = ablation_delta(full, weak, metric="auprc", n_boot=200, seed=0)
    assert d["metric"] == "auprc"
    assert d["delta"] == d["variant"][0] - d["full"][0]
    assert d["delta"] < 0.0                # removing signal hurts AUPRC


def test_run_exp5_tabulates_each_variant():
    rng = np.random.default_rng(1)
    full = _records(rng, sep=0.6)
    variants = {"no_drug": _records(rng, sep=0.1),
                "n_samples_10": _records(rng, sep=0.55)}
    out = run_exp5(full, variants, metric="auprc", n_boot=100, seed=0)
    assert set(out) == {"no_drug", "n_samples_10"}
    for r in out.values():
        assert "delta" in r and "full" in r and "variant" in r
