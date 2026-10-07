"""Tests for GPU-side SimCLR augmentation (src/txmorph/data/gpu_augment.py).

Runs on CPU tensors -- the ops are device-agnostic, so correctness checks here
also cover the CUDA path. torch-guarded so the dev box can skip cleanly.
"""
import pytest

torch = pytest.importorskip("torch")

from txmorph.data.gpu_augment import (          # noqa: E402
    _BLUR_K, _crop_flip, _gaussian_blur, _stain_matrices, augment_batch,
    macenko_jitter_batch, two_views,
)


def _tiles(b=4, px=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (b, px, px, 3), generator=g, dtype=torch.uint8)


def _stained_tiles(b=3, px=64, seed=0):
    """Synthetic H&E-like tiles: real stain structure, unlike uniform noise."""
    import numpy as np

    from txmorph.data.stain import _REF_STAINS, od_to_rgb

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(b):
        conc = rng.gamma(2.0, 0.4, size=(2, px * px)).astype(np.float32)
        od = (_REF_STAINS @ conc).T
        out.append(od_to_rgb(od).reshape(px, px, 3))
    return torch.from_numpy(np.stack(out))


def test_augment_batch_shape_dtype_and_range():
    out = augment_batch(_tiles(), torch.Generator().manual_seed(1))
    assert out.shape == (4, 3, 256, 256)
    assert out.dtype == torch.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_out_px_resizes():
    out = augment_batch(_tiles(px=256), torch.Generator().manual_seed(1), out_px=128)
    assert out.shape == (4, 3, 128, 128)


def test_deterministic_under_fixed_generator_seed():
    t = _tiles()
    a = augment_batch(t, torch.Generator().manual_seed(7))
    b = augment_batch(t, torch.Generator().manual_seed(7))
    torch.testing.assert_close(a, b)


def test_different_seeds_give_different_views():
    t = _tiles()
    a = augment_batch(t, torch.Generator().manual_seed(1))
    b = augment_batch(t, torch.Generator().manual_seed(2))
    assert not torch.allclose(a, b)


def test_two_views_are_independent_augmentations():
    v1, v2 = two_views(_tiles(), torch.Generator().manual_seed(3))
    assert v1.shape == v2.shape == (4, 3, 256, 256)
    assert not torch.allclose(v1, v2)


def test_blur_with_zero_probability_is_identity():
    """sharp samples get a delta kernel, so p=0 must leave the input untouched."""
    x = torch.rand(3, 3, 32, 32)
    out = _gaussian_blur(x, torch.Generator().manual_seed(0), prob=0.0)
    torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)


def test_blur_kernel_covers_three_sigma():
    """_BLUR_K must span +/-3 sigma at the largest sigma the sampler draws (2.0)."""
    assert _BLUR_K // 2 >= 3 * 2.0 - 1e-9 or _BLUR_K >= 13


def test_blur_preserves_mean_and_reduces_variance():
    x = torch.rand(8, 3, 64, 64)
    out = _gaussian_blur(x, torch.Generator().manual_seed(5), prob=1.0)
    # normalised kernel + reflect padding => mean preserved, high frequencies lost
    torch.testing.assert_close(out.mean(), x.mean(), atol=2e-3, rtol=0)
    assert float(out.var()) < float(x.var())


def test_crop_flip_stays_within_tile():
    """A constant-bordered tile must not sample the reflection padding."""
    x = torch.full((6, 3, 64, 64), 0.5)
    out = _crop_flip(x, 64, torch.Generator().manual_seed(11))
    torch.testing.assert_close(out, torch.full_like(out, 0.5), atol=1e-5, rtol=0)


def test_stain_matrix_matches_numpy_reference():
    """Batched eigh-based estimate must agree with data.stain's SVD version."""
    import numpy as np

    from txmorph.data.stain import _stain_matrix, rgb_to_od

    tiles = _stained_tiles(b=3, px=64, seed=1)
    od = -torch.log10((tiles.float().reshape(3, -1, 3) + 1.0) / 240.0)
    gpu = _stain_matrices(od, beta=0.15, alpha=1.0, n_sample=4096,
                          gen=torch.Generator().manual_seed(0))
    for i in range(3):
        ref = _stain_matrix(rgb_to_od(tiles[i].numpy()).reshape(-1, 3))
        for col in range(2):                     # compare direction, ignore sign
            cos = abs(float(np.dot(gpu[i, :, col].numpy(), ref[:, col])))
            assert cos > 0.97, f"tile {i} stain {col}: cosine {cos:.3f}"


def test_macenko_jitter_is_near_identity_at_zero_sigma():
    """sigma=0 means unperturbed concentrations, so the tile must round-trip."""
    tiles = _stained_tiles(b=2, px=64, seed=2)
    out = macenko_jitter_batch(tiles, torch.Generator().manual_seed(0), sigma=0.0)
    assert out.shape == (2, 64, 64, 3)
    assert float((out - tiles.float()).abs().mean()) < 3.0      # /255 levels


def test_macenko_jitter_perturbs_and_stays_in_range():
    tiles = _stained_tiles(b=3, px=64, seed=3)
    out = macenko_jitter_batch(tiles, torch.Generator().manual_seed(0), sigma=0.25)
    assert 0.0 <= float(out.min()) and float(out.max()) <= 255.0
    assert float((out - tiles.float()).abs().mean()) > 1.0
    assert torch.isfinite(out).all()


def test_macenko_handles_blank_tile_without_nan():
    """An all-white tile has no valid OD pixels; must fall back, not produce NaN."""
    blank = torch.full((2, 32, 32, 3), 255, dtype=torch.uint8)
    out = macenko_jitter_batch(blank, torch.Generator().manual_seed(0), sigma=0.1)
    assert torch.isfinite(out).all()


def test_augment_batch_macenko_path():
    out = augment_batch(_stained_tiles(b=2, px=64), torch.Generator().manual_seed(0),
                        out_px=64, stain_jitter="macenko")
    assert out.shape == (2, 3, 64, 64)
    assert torch.isfinite(out).all()
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_wider_crop_scale_makes_views_less_similar():
    """The whole point of widening crop scale: positive pairs must diverge more."""
    tiles = _stained_tiles(b=6, px=64, seed=4)
    narrow = two_views(tiles, torch.Generator().manual_seed(0), out_px=64,
                       crop_scale=(0.6, 1.0))
    wide = two_views(tiles, torch.Generator().manual_seed(0), out_px=64,
                     crop_scale=(0.2, 1.0))
    d_narrow = float((narrow[0] - narrow[1]).abs().mean())
    d_wide = float((wide[0] - wide[1]).abs().mean())
    assert d_wide > d_narrow


def test_matches_cpu_transform_statistics():
    """GPU path should land in the same intensity regime as SimCLRTransform."""
    from txmorph.data.tile_dataset import SimCLRTransform
    import numpy as np

    tile = (np.random.default_rng(0).random((256, 256, 3)) * 255).astype(np.uint8)
    cpu_v1, _ = SimCLRTransform(rng=np.random.default_rng(0))(tile)
    gpu = augment_batch(torch.from_numpy(tile)[None], torch.Generator().manual_seed(0),
                        stain_jitter="fast", crop_scale=(0.6, 1.0))
    assert 0.0 <= float(gpu.min()) and float(gpu.max()) <= 1.0
    # both pipelines jitter around the tile's own level; compare loose bounds only
    assert abs(float(gpu.mean()) - float(cpu_v1.mean())) < 0.45
