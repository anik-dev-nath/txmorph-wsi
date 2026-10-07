"""Tests for single-timepoint diffusion adaptation.

Tests ContextBuilder with pCR conditioning, estimate_nll, and class-conditional
training convergence on a toy 2-D Gaussian with pCR-dependent cluster structure.
"""
import numpy as np
import pytest
import torch

from txmorph.diffusion.context import ContextBuilder
from txmorph.diffusion.schedule import NoiseSchedule
from txmorph.diffusion.gaussian_diffusion import GaussianDiffusion
from txmorph.diffusion.denoiser import LatentDenoiser
from txmorph.utils.seed import seed_everything


# ---------- ContextBuilder with pCR ----------

def test_context_builder_no_pcr():
    """ContextBuilder without pCR (paired mode) produces correct shape."""
    cb = ContextBuilder(z_dim=8, drug_dim=4, pcr_dim=0, out_dim=8)
    z = torch.randn(3, 8)
    d = torch.randn(3, 4)
    c = cb(z, d)
    assert c.shape == (3, 8)


def test_context_builder_with_pcr():
    """ContextBuilder with pCR=1 dim produces correct shape."""
    cb = ContextBuilder(z_dim=8, drug_dim=4, pcr_dim=1, out_dim=8)
    z = torch.randn(3, 8)
    d = torch.randn(3, 4)
    pcr = torch.ones(3, 1)
    c = cb(z, d, pcr)
    assert c.shape == (3, 8)


def test_context_builder_pcr_none_fallback():
    """ContextBuilder with pcr_dim=0 ignores pcr_embed even if passed."""
    cb = ContextBuilder(z_dim=8, drug_dim=4, pcr_dim=0, out_dim=8)
    z = torch.randn(2, 8)
    d = torch.randn(2, 4)
    c1 = cb(z, d)
    c2 = cb(z, d, None)
    assert torch.allclose(c1, c2)


def test_single_timepoint_context_excludes_z_pre():
    """With z_dim=0 the context must not depend on z_pre at all.

    In single-timepoint mode z_pre is the *target*. If it also reaches the
    context, the denoiser can reconstruct the noise from the conditioning alone
    and stops using the pCR bit, which is what the likelihood ratio is built
    from (toy data: AUROC 0.436 leaked vs 0.864 clean).
    """
    cb = ContextBuilder(z_dim=0, drug_dim=4, pcr_dim=1, out_dim=8)
    d = torch.randn(3, 4)
    p = torch.ones(3, 1)
    c_a = cb(torch.randn(3, 8), d, p)
    c_b = cb(torch.randn(3, 8) * 100, d, p)      # wildly different z_pre
    assert torch.allclose(c_a, c_b), "context changed with z_pre -- target leaked in"


def test_context_builder_raises_when_pcr_missing():
    """A pcr_dim>0 builder must refuse to silently drop the pCR condition."""
    cb = ContextBuilder(z_dim=0, drug_dim=4, pcr_dim=1, out_dim=8)
    with pytest.raises(ValueError):
        cb(torch.randn(2, 8), torch.randn(2, 4))


def test_context_builder_pcr_changes_output():
    """Different pCR values produce different contexts."""
    cb = ContextBuilder(z_dim=8, drug_dim=4, pcr_dim=1, out_dim=8)
    z = torch.randn(1, 8)
    d = torch.randn(1, 4)
    c1 = cb(z, d, torch.ones(1, 1))
    c0 = cb(z, d, torch.zeros(1, 1))
    assert not torch.allclose(c1, c0)


# ---------- estimate_nll ----------

def test_estimate_nll_shape():
    """estimate_nll returns (B,) tensor."""
    seed_everything(0)
    dim = 4
    sch = NoiseSchedule(T=50, schedule="cosine")
    diff = GaussianDiffusion(sch)
    model = LatentDenoiser(dim=dim, width=32, cond_dim=8, n_blocks=1)
    model.eval()

    z0 = torch.randn(3, dim)
    c = torch.randn(3, 8)
    with torch.no_grad():
        nll = diff.estimate_nll(model, z0, c, n_mc=5)
    assert nll.shape == (3,)
    assert torch.isfinite(nll).all()


def test_nll_contrast_shares_randomness_across_hypotheses():
    """Two identical contexts must score *identically* under nll_contrast.

    This is what common random numbers buys: with independent draws per
    hypothesis the two columns differ by pure MC noise, and any likelihood ratio
    built on them is noise. Scoring the same context twice isolates that -- the
    gap has to be exactly zero.
    """
    seed_everything(0)
    dim = 8
    sch = NoiseSchedule(T=100, schedule="cosine")
    diff = GaussianDiffusion(sch)
    model = LatentDenoiser(dim=dim, width=32, cond_dim=8, n_blocks=1)
    model.eval()

    z0 = torch.randn(6, dim)
    c = torch.randn(6, 8)
    with torch.no_grad():
        nll = diff.nll_contrast(model, z0, [c, c.clone()], n_mc=8)

    assert nll.shape == (2, 6)
    assert torch.allclose(nll[0], nll[1], atol=1e-6), \
        "identical contexts scored differently -- randomness is not shared"


def test_nll_contrast_cuts_gap_noise_vs_independent_draws():
    """Sharing the draws must shrink the noise on the hypothesis gap by ~10x+.

    This is the whole point of ``nll_contrast``. The likelihood-ratio P(pCR)
    depends on the *difference* between two NLLs; if each is sampled
    independently, that difference carries the MC noise of both and the biomarker
    degenerates to chance (measured end-to-end on weakly-separated toy data:
    AUROC 0.504 independent vs 0.598 paired). Comparing the replicate spread of
    the gap under each scheme guards the fix without needing a trained model.
    """
    seed_everything(3)
    dim = 8
    sch = NoiseSchedule(T=100, schedule="cosine")
    diff = GaussianDiffusion(sch)
    model = LatentDenoiser(dim=dim, width=32, cond_dim=8, n_blocks=1)
    model.eval()

    z0 = torch.randn(4, dim)
    c_a = torch.randn(4, 8)
    c_b = c_a + 0.5 * torch.randn(4, 8)

    shared_gaps, independent_gaps = [], []
    with torch.no_grad():
        for rep in range(6):
            g = torch.Generator()
            g.manual_seed(100 + rep)
            nll = diff.nll_contrast(model, z0, [c_a, c_b], n_mc=16, generator=g)
            shared_gaps.append(nll[1] - nll[0])

            # independent draws per hypothesis -- what the estimator must not do
            g_a = torch.Generator(); g_a.manual_seed(200 + rep)
            g_b = torch.Generator(); g_b.manual_seed(900 + rep)
            n_a = diff.nll_contrast(model, z0, [c_a], n_mc=16, generator=g_a)[0]
            n_b = diff.nll_contrast(model, z0, [c_b], n_mc=16, generator=g_b)[0]
            independent_gaps.append(n_b - n_a)

    shared_noise = torch.stack(shared_gaps).std(dim=0).mean()
    independent_noise = torch.stack(independent_gaps).std(dim=0).mean()

    assert shared_noise * 10 < independent_noise, (
        f"common random numbers barely helped: shared spread {shared_noise:.2e} "
        f"vs independent {independent_noise:.2e}")


def test_estimate_nll_deterministic_seed():
    """estimate_nll is consistent across calls with same seed."""
    seed_everything(42)
    dim = 4
    sch = NoiseSchedule(T=50, schedule="cosine")
    diff = GaussianDiffusion(sch)
    model = LatentDenoiser(dim=dim, width=32, cond_dim=8, n_blocks=1)
    model.eval()

    z0 = torch.randn(2, dim)
    c = torch.randn(2, 8)
    with torch.no_grad():
        seed_everything(0)
        nll1 = diff.estimate_nll(model, z0, c, n_mc=10)
        seed_everything(0)
        nll2 = diff.estimate_nll(model, z0, c, n_mc=10)
    assert torch.allclose(nll1, nll2)


# ---------- Class-conditional training convergence ----------

def test_class_conditional_training_converges():
    """Train class-conditional diffusion on 2-D toy data with pCR labels.

    pCR=1 patients cluster near (2, 2), pCR=0 near (-2, -2). After training,
    the model should assign lower NLL to in-distribution points under the
    correct class condition.
    """
    seed_everything(42)
    dim = 2
    n_per_class = 50
    T = 100

    # Generate toy data: pCR=1 around (2,2), pCR=0 around (-2,-2)
    z_pcr1 = torch.randn(n_per_class, dim) * 0.5 + torch.tensor([2.0, 2.0])
    z_pcr0 = torch.randn(n_per_class, dim) * 0.5 + torch.tensor([-2.0, -2.0])
    z_all = torch.cat([z_pcr1, z_pcr0])
    pcr_all = torch.cat([torch.ones(n_per_class), torch.zeros(n_per_class)])

    # Drug features: constant (single drug)
    drug_dim = 4
    d_drug = torch.zeros(z_all.shape[0], drug_dim)

    # Build model with pCR conditioning. z_dim=0: the context is [drug || pCR],
    # never z_pre, because z_pre is what we are denoising.
    cond_dim = 8
    cb = ContextBuilder(z_dim=0, drug_dim=drug_dim, pcr_dim=1, out_dim=cond_dim)
    denoiser = LatentDenoiser(dim=dim, width=64, cond_dim=cond_dim, n_blocks=2)
    sch = NoiseSchedule(T=T, schedule="cosine")
    diff = GaussianDiffusion(sch, lambda_vlb=1e-3)

    params = list(cb.parameters()) + list(denoiser.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    # Train
    n = z_all.shape[0]
    for epoch in range(150):
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            z0 = z_all[idx]
            pcr_embed = pcr_all[idx].unsqueeze(-1)
            c = cb(z0, d_drug[idx], pcr_embed)  # context includes z_pre (=z0) + drug + pcr
            out = diff.loss(denoiser, z0, c)
            opt.zero_grad()
            out["total"].backward()
            opt.step()

    # Evaluate: NLL should be lower for correct condition
    denoiser.eval()
    cb.eval()
    test_z1 = torch.tensor([[2.0, 2.0]])  # a pCR=1 point
    test_z0 = torch.tensor([[-2.0, -2.0]])  # a pCR=0 point
    d_test = torch.zeros(1, drug_dim)

    # Both hypotheses must be scored with nll_contrast, i.e. on shared draws.
    # Differencing two independent estimate_nll calls compares numbers whose MC
    # noise is far larger than the gap between the conditions.
    c_one = cb(test_z1, d_test, torch.ones(1, 1))
    c_zero = cb(test_z1, d_test, torch.zeros(1, 1))
    with torch.no_grad():
        g = torch.Generator()
        g.manual_seed(0)
        nll_1 = diff.nll_contrast(denoiser, test_z1, [c_one, c_zero],
                                  n_mc=200, generator=g)
        g.manual_seed(0)
        nll_0 = diff.nll_contrast(denoiser, test_z0, [c_one, c_zero],
                                  n_mc=200, generator=g)

    assert nll_1[0].item() < nll_1[1].item(), (
        f"pCR=1 point: NLL(pcr=1)={nll_1[0].item():.4f} should be < "
        f"NLL(pcr=0)={nll_1[1].item():.4f}")
    assert nll_0[1].item() < nll_0[0].item(), (
        f"pCR=0 point: NLL(pcr=0)={nll_0[1].item():.4f} should be < "
        f"NLL(pcr=1)={nll_0[0].item():.4f}")


def test_diffusion_module_single_timepoint():
    """DiffusionModule creates correct architecture in single-timepoint mode."""
    from txmorph.training.diffusion import DiffusionModule

    mod = DiffusionModule(
        dim=8, cond_dim=8, drug_variant="onehot", num_drugs=4,
        n_bits=64, n_blocks=1, width=32, conditioning="film",
        device="cpu", single_timepoint=True)

    assert mod.single_timepoint
    assert mod.context.pcr_dim == 1
    assert mod.context.z_dim == 0, "z_pre must be excluded from the context"

    # Test forward pass with pCR
    z = torch.randn(2, 8)
    d_drug = mod.drug(torch.tensor([0, 1]))
    pcr = torch.ones(2, 1)
    c = mod.context(z, d_drug, pcr)
    assert c.shape == (2, 8)


def test_diffusion_module_paired_mode():
    """DiffusionModule in paired mode has pcr_dim=0."""
    from txmorph.training.diffusion import DiffusionModule

    mod = DiffusionModule(
        dim=8, cond_dim=8, drug_variant="onehot", num_drugs=4,
        n_bits=64, n_blocks=1, width=32, conditioning="film",
        device="cpu", single_timepoint=False)

    assert not mod.single_timepoint
    assert mod.context.pcr_dim == 0
