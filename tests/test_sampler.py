import torch
from txmorph.diffusion.schedule import NoiseSchedule
from txmorph.diffusion.gaussian_diffusion import GaussianDiffusion
from txmorph.diffusion.denoiser import LatentDenoiser
from txmorph.diffusion.sampler import sample_patient
from txmorph.utils.seed import seed_everything


def test_sample_shape_and_determinism():
    diff = GaussianDiffusion(NoiseSchedule(50, "cosine"))
    model = LatentDenoiser(dim=16, width=32, cond_dim=16, n_blocks=2)
    ctx = torch.randn(1, 16)
    seed_everything(7)
    s1 = sample_patient(diff, model, ctx, n=100, dim=16)
    seed_everything(7)
    s2 = sample_patient(diff, model, ctx, n=100, dim=16)
    assert s1.shape == (100, 16)
    assert torch.allclose(s1, s2)            # deterministic under fixed seed
