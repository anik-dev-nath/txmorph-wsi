import torch
from txmorph.diffusion.schedule import NoiseSchedule, extract
from txmorph.diffusion.gaussian_diffusion import GaussianDiffusion
from txmorph.diffusion.denoiser import LatentDenoiser
from txmorph.utils.seed import seed_everything


def make(T=200):
    return GaussianDiffusion(NoiseSchedule(T, "cosine"))


def test_forward_marginal_moments():
    """z_t ~ N(sqrt(abar) z0, (1-abar) I) -- check empirically."""
    seed_everything(0)
    diff = make()
    z0 = torch.tensor([[3.0, -2.0]]).expand(40000, 2).contiguous()
    t = torch.full((40000,), 120, dtype=torch.long)
    zt = diff.q_sample(z0, t)
    abar = diff.sch.alphas_cumprod[120].item()
    exp_mean = (abar ** 0.5) * torch.tensor([3.0, -2.0])
    assert torch.allclose(zt.mean(0), exp_mean, atol=0.05)
    assert abs(zt.var(0).mean().item() - (1 - abar)) < 0.05


def test_posterior_variance_independent_rederivation():
    """Re-derive q(z_{t-1}|z_t,z0) variance from primitives (precision combine)."""
    sch = NoiseSchedule(T=300, schedule="cosine")
    for t in [1, 50, 150, 299]:
        beta_t = sch.betas[t].item()
        alpha_t = sch.alphas[t].item()
        prior_var = (1 - sch.alphas_cumprod_prev[t]).item()   # Var q(z_{t-1}|z0)
        like_var = beta_t                                      # Var q(z_t|z_{t-1})
        # posterior precision = 1/prior_var + alpha_t/like_var
        post_var = 1.0 / (1.0 / prior_var + alpha_t / like_var)
        assert abs(post_var - sch.posterior_variance[t].item()) < 1e-4


def test_eps_inversion_and_mean_consistency():
    """x0_from_eps recovers z0; p-mean (true eps) == q_posterior mean."""
    seed_everything(1)
    diff = make()
    z0 = torch.randn(8, 2)
    t = torch.randint(1, diff.T, (8,))
    eps = torch.randn_like(z0)
    zt = diff.q_sample(z0, t, eps)
    x0_rec = diff.predict_x0_from_eps(zt, t, eps)
    assert torch.allclose(x0_rec, z0, atol=1e-4)
    mean_q, _, _ = diff.q_posterior(z0, zt, t)
    # mean computed from the recovered x0 must match the posterior mean
    mean_from_eps, _, _ = diff.q_posterior(x0_rec, zt, t)
    assert torch.allclose(mean_q, mean_from_eps, atol=1e-4)


def test_loss_runs_and_backprops():
    seed_everything(2)
    diff = make()
    model = LatentDenoiser(dim=2, width=64, cond_dim=8, n_blocks=2)
    z0 = torch.randn(16, 2)
    out = diff.loss(model, z0, c=None)
    out["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0 and torch.isfinite(out["total"])


def test_end_to_end_recovers_2d_gaussian():
    """Train DDPM on a 2-D Gaussian; sampled moments should match (fixed-var)."""
    seed_everything(0)
    diff = make(T=200)
    true_mean = torch.tensor([2.0, -1.0])
    true_std = torch.tensor([0.5, 1.0])

    def data(n):
        return true_mean + true_std * torch.randn(n, 2)

    model = LatentDenoiser(dim=2, width=96, cond_dim=8, n_blocks=2)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    for _ in range(6000):
        z0 = data(256)
        loss = diff.loss(model, z0, c=None)["total"]
        opt.zero_grad(); loss.backward(); opt.step()

    samples = diff.sample(model, n=4000, dim=2, c=None, learned_var=False, clip_x0=8.0)
    assert torch.allclose(samples.mean(0), true_mean, atol=0.25), samples.mean(0)
    ratio = samples.std(0) / true_std
    assert torch.all((ratio > 0.6) & (ratio < 1.45)), samples.std(0)
