"""Cross-attention conditioning variant of the denoiser (Exp-5 ablation).

Verifies the crossattn path builds, returns the two heads with correct shapes,
handles the null-context branch, and is genuinely different from FiLM."""
import pytest

torch = pytest.importorskip("torch")

from txmorph.diffusion.denoiser import LatentDenoiser


def _inputs(b=4, dim=512, cond_dim=512):
    zt = torch.randn(b, dim)
    t = torch.randint(0, 1000, (b,))
    c = torch.randn(b, cond_dim)
    return zt, t, c


def test_crossattn_forward_shapes():
    net = LatentDenoiser(dim=512, width=256, cond_dim=512, n_blocks=3,
                         conditioning="crossattn")
    zt, t, c = _inputs()
    eps, v = net(zt, t, c)
    assert eps.shape == (4, 512)
    assert v.shape == (4, 512)
    assert torch.isfinite(eps).all() and torch.isfinite(v).all()


def test_crossattn_null_context_branch():
    net = LatentDenoiser(width=256, conditioning="crossattn")
    zt, t, _ = _inputs()
    eps, v = net(zt, t, None)                 # c=None -> learned null token
    assert eps.shape == (4, 512) and v.shape == (4, 512)


def test_crossattn_differs_from_film():
    torch.manual_seed(0)
    zt, t, c = _inputs()
    film = LatentDenoiser(width=256, conditioning="film")
    cross = LatentDenoiser(width=256, conditioning="crossattn")
    ef, _ = film(zt, t, c)
    ec, _ = cross(zt, t, c)
    assert not torch.allclose(ef, ec)


def test_unknown_conditioning_raises():
    with pytest.raises(ValueError):
        LatentDenoiser(conditioning="bogus")
