"""Macenko OD transforms + normalize/jitter shape & dtype (numpy)."""
import numpy as np
from txmorph.data.stain import (
    rgb_to_od, od_to_rgb, macenko_normalize, macenko_jitter, _stain_matrix,
    estimate_stain_matrix,
)


def _synthetic_he_tile(seed=0, px=64):
    """A small H&E-like tile: two stain-colored blobs on a light background."""
    rng = np.random.default_rng(seed)
    tile = np.full((px, px, 3), 220, dtype=np.uint8)          # light background
    tile[:px // 2] = [150, 80, 180]                            # hematoxylin-ish
    tile[px // 2:] = [220, 120, 140]                           # eosin-ish
    noise = rng.integers(-8, 8, tile.shape)
    return np.clip(tile + noise, 0, 255).astype(np.uint8)


def test_od_roundtrip():
    tile = _synthetic_he_tile()
    od = rgb_to_od(tile)
    back = od_to_rgb(od)
    assert back.shape == tile.shape and back.dtype == np.uint8
    assert np.abs(back.astype(int) - tile.astype(int)).mean() < 2.0


def test_stain_matrix_shape_and_norm():
    tile = _synthetic_he_tile()
    od = rgb_to_od(tile).reshape(-1, 3)
    stains = _stain_matrix(od)
    assert stains.shape == (3, 2)
    assert np.allclose(np.linalg.norm(stains, axis=0), 1.0, atol=1e-3)


def test_normalize_shape_dtype():
    tile = _synthetic_he_tile()
    out = macenko_normalize(tile)
    assert out.shape == tile.shape and out.dtype == np.uint8


def test_jitter_is_seeded_and_differs():
    tile = _synthetic_he_tile()
    a = macenko_jitter(tile, rng=np.random.default_rng(0))
    b = macenko_jitter(tile, rng=np.random.default_rng(0))
    c = macenko_jitter(tile, rng=np.random.default_rng(1))
    assert np.array_equal(a, b)                                # deterministic per seed
    assert not np.array_equal(a, c)                            # varies across seeds


def test_estimate_stain_matrix_accepts_image_or_pixels():
    """estimate_stain_matrix works on (H,W,3) and (N,3) and matches _stain_matrix."""
    tile = _synthetic_he_tile()
    S_img = estimate_stain_matrix(tile)
    S_px = estimate_stain_matrix(tile.reshape(-1, 3))
    assert S_img.shape == (3, 2) and S_px.shape == (3, 2)
    assert np.allclose(S_img, S_px, atol=1e-5)                 # same pixels -> same matrix
    # equivalent to calling the internal estimator directly
    S_ref = _stain_matrix(rgb_to_od(tile).reshape(-1, 3))
    assert np.allclose(S_img, S_ref, atol=1e-5)


def test_cached_stains_equivalent_to_per_tile():
    """Passing the tile's own precomputed matrix == per-tile estimation (no SVD)."""
    tile = _synthetic_he_tile()
    S = estimate_stain_matrix(tile)
    legacy = macenko_normalize(tile)                           # estimates internally
    cached = macenko_normalize(tile, stains=S)                 # SVD skipped
    assert cached.shape == tile.shape and cached.dtype == np.uint8
    assert np.array_equal(legacy, cached)


def test_cached_stains_valid_across_tiles():
    """A per-slide matrix applied to a different tile still yields a valid image."""
    slide_matrix = estimate_stain_matrix(_synthetic_he_tile(seed=0))
    other = _synthetic_he_tile(seed=3)
    out = macenko_normalize(other, stains=slide_matrix)
    assert out.shape == other.shape and out.dtype == np.uint8
