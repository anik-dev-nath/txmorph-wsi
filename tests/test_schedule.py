import torch
from txmorph.diffusion.schedule import NoiseSchedule


def test_cumprod_monotone_and_endpoints():
    sch = NoiseSchedule(T=1000, schedule="cosine")
    acp = sch.alphas_cumprod
    assert torch.all(acp[1:] <= acp[:-1] + 1e-6)      # non-increasing
    assert acp[0] < 1.0 and acp[-1] < acp[0]
    assert acp[-1] < 0.05                              # ~ pure noise at T


def test_alpha_beta_relationship():
    sch = NoiseSchedule(T=500, schedule="cosine")
    assert torch.allclose(sch.alphas, 1.0 - sch.betas, atol=1e-6)
    # abar_t == alpha_t * abar_{t-1}
    assert torch.allclose(sch.alphas_cumprod, sch.alphas * sch.alphas_cumprod_prev, atol=1e-5)


def test_posterior_variance_bounded_by_beta():
    sch = NoiseSchedule(T=500, schedule="cosine")
    # beta_tilde_t <= beta_t for all t>=1
    assert torch.all(sch.posterior_variance[1:] <= sch.betas[1:] + 1e-6)
