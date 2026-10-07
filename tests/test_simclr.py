import torch
import torch.nn as nn
from txmorph.encoders.simclr import nt_xent, SimCLREncoder
from txmorph.utils.seed import seed_everything


def test_ntxent_finite_and_backprops():
    seed_everything(0)
    z1 = torch.randn(32, 64, requires_grad=True)
    z2 = torch.randn(32, 64, requires_grad=True)
    loss = nt_xent(z1, z2, temperature=0.07)
    loss.backward()
    assert torch.isfinite(loss) and z1.grad is not None


def test_ntxent_aligned_lower_than_shuffled():
    """Loss should be lower when view pairs match than when they are shuffled."""
    seed_everything(1)
    base = torch.randn(64, 32)
    aligned = nt_xent(base, base + 0.01 * torch.randn_like(base))      # positives ~ identical
    perm = torch.randperm(64)
    shuffled = nt_xent(base, (base + 0.01 * torch.randn_like(base))[perm])
    assert aligned < shuffled


def test_encoder_shapes_with_dummy_backbone():
    feat_dim = 48
    backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, feat_dim))
    enc = SimCLREncoder(backbone=backbone, feat_dim=feat_dim, proj_dim=16)
    x = torch.randn(10, 3, 8, 8)
    h, z = enc(x)
    assert h.shape == (10, feat_dim)
    assert z.shape == (10, 16)
    assert torch.allclose(z.norm(dim=1), torch.ones(10), atol=1e-5)    # L2-normalized
