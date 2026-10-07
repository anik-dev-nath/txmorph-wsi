"""Exp 3's single-timepoint variant: clustering residual-disease latents.

Inline Gaussians only, to check the estimator behaves (finds planted structure,
caps K sensibly at small n, applies FDR, parses Post-NAT's multi-focus sizes).
Real phenotypes come from Post-NAT-BRCA slides.
"""
import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from txmorph.experiments.exp3 import (
    run_exp3_residual_disease, associate_clinical, _first_float,
)


def _planted(n_per=25, dim=12, gap=6.0, seed=0):
    """Two well-separated groups, with a clinical variable that tracks them."""
    rng = np.random.default_rng(seed)
    Z = np.vstack([rng.normal(size=(n_per, dim)),
                   rng.normal(size=(n_per, dim)) + gap])
    true = np.r_[np.zeros(n_per, int), np.ones(n_per, int)]
    meta = {
        "cellularity": list(10 + 40 * true + rng.normal(0, 2, len(true))),
        "residual_size_mm": list(5 + 20 * true + rng.normal(0, 1, len(true))),
        "response_bc": ["NDR" if t else "PDR" for t in true],
        "receptor_subtype": ["HR+/HER2-"] * len(true),
    }
    return Z, meta, true


def test_recovers_planted_structure_and_associations():
    Z, meta, true = _planted()
    res = run_exp3_residual_disease(Z, meta, seed=0)
    assert "error" not in res
    assert res["k"] >= 2
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(true, res["labels"]) > 0.8
    assert res["stability_ari"] > 0.5
    sig = [a["variable"] for a in res["clinical_assoc"] if a.get("significant_fdr")]
    assert "cellularity" in sig


def test_refuses_when_too_few_cases():
    """Clustering 8 patients would be noise dressed as phenotypes."""
    res = run_exp3_residual_disease(np.random.default_rng(0).normal(size=(8, 10)),
                                    {"cellularity": list(range(8))})
    assert "error" in res


def test_k_is_capped_by_sample_size():
    """An 8-component GMM on ~50 patients yields components smaller than the
    dimension; BIC will pick them and they mean nothing."""
    Z, meta, _ = _planted(n_per=12)
    res = run_exp3_residual_disease(Z, meta, seed=0, max_k=8)
    assert res["k_range_searched"][1] <= 8
    assert res["k"] <= max(2, len(Z) // 12)


def test_unassociated_clusters_are_reported_as_such():
    """Structure with no clinical correlate must not come back 'significant'."""
    rng = np.random.default_rng(3)
    Z, meta, _ = _planted(gap=6.0, seed=3)
    meta = dict(meta)
    meta["cellularity"] = list(rng.normal(size=len(Z)))       # unrelated
    meta["residual_size_mm"] = list(rng.normal(size=len(Z)))
    meta["response_bc"] = list(rng.choice(["PDR", "NDR"], size=len(Z)))
    res = run_exp3_residual_disease(Z, meta, seed=0)
    assoc = {a["variable"]: a for a in res["clinical_assoc"]}
    assert not assoc["cellularity"].get("significant_fdr", False)


def test_multifocal_size_parsing():
    """Post-NAT writes multifocal residual disease as '4.5,1'; take the largest."""
    assert _first_float("4.5,1") == 4.5
    assert _first_float("9") == 9.0
    assert np.isnan(_first_float(""))
    assert np.isnan(_first_float(None))
    assert np.isnan(_first_float("not a number"))


def test_associate_clinical_applies_fdr():
    labels = np.r_[np.zeros(20, int), np.ones(20, int)]
    rng = np.random.default_rng(0)
    meta = {"cellularity": list(rng.normal(size=40)),
            "residual_size_mm": list(rng.normal(size=40)),
            "ln_involved": list(rng.normal(size=40)),
            "response_bc": list(rng.choice(["PDR", "NDR"], size=40))}
    rows = associate_clinical(labels, meta)
    assert rows and all("p_fdr" in r for r in rows)
    for r in rows:                       # FDR-adjusted p is never below raw p
        assert r["p_fdr"] >= r["p"] - 1e-12
