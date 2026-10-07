"""Kill-switch gate: latent selection + separation metrics (CLAUDE.md S10).

The gate is mandatory and runs on real cohorts, so the latent it separates has
to be the one the cohort actually provides. These use small inline Gaussians to
check the selection logic and the metric, never as cohort stand-ins.
"""
import numpy as np
import pytest

from txmorph.viz.umap_check import select_gate_latent, separation_metrics


def _two_clusters(n=60, dim=8, gap=3.0, seed=0):
    """Two separable Gaussian blobs + their labels."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    z = rng.normal(size=(n, dim)) + gap * y[:, None]
    return z, y


def test_selects_z_post_when_paired():
    """Paired mode: z_post is finite, so the gate separates it."""
    z_pre, _ = _two_clusters()
    z_post = z_pre + 1.0
    z, name = select_gate_latent(z_pre, z_post)
    assert name == "z_post"
    assert np.array_equal(z, z_post)


def test_falls_back_to_z_pre_when_z_post_is_nan():
    """Single-timepoint: paired_dataset fills z_post with NaN -> use z_pre.

    Without this the gate hands all-NaN to silhouette/logistic/UMAP and dies.
    """
    z_pre, _ = _two_clusters()
    z_post = np.full_like(z_pre, np.nan)
    z, name = select_gate_latent(z_pre, z_post)
    assert name == "z_pre"
    assert np.array_equal(z, z_pre)


def test_partial_z_post_falls_back_rather_than_passing_nan():
    """A mixed table still must not send NaN rows downstream."""
    z_pre, _ = _two_clusters()
    z_post = z_pre.copy()
    z_post[0] = np.nan
    z, name = select_gate_latent(z_pre, z_post)
    assert name == "z_pre"
    assert np.isfinite(z).all()


def test_raises_when_no_finite_latent():
    """Both latents unusable is a build error, not something to paper over."""
    nan = np.full((4, 8), np.nan)
    with pytest.raises(ValueError, match="finite latent"):
        select_gate_latent(nan, nan)


def test_separation_metrics_detect_signal_and_its_absence():
    """AUROC clears the GO threshold on separable data and sits near chance on noise."""
    z, y = _two_clusters(gap=4.0)
    m = separation_metrics(z, y, seed=0)
    assert m["separation_auroc"] > 0.65          # GO threshold (CLAUDE.md S10)
    assert m["silhouette"] > 0.0

    rng = np.random.default_rng(1)
    noise = rng.normal(size=z.shape)
    m0 = separation_metrics(noise, y, seed=0)
    assert m0["separation_auroc"] < 0.65         # NO-GO: representation lacks signal
