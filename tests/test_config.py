"""Config loader: merge order, group exposure (leaf + parent), CLI overrides."""
import numpy as np  # noqa: F401  (kept for parity with other tests)
from txmorph.utils.config import load_config, _coerce, _deep_merge


def test_deep_merge_nested():
    a = {"x": {"a": 1, "b": 2}, "y": 3}
    b = {"x": {"b": 20, "c": 30}}
    out = _deep_merge(a, b)
    assert out == {"x": {"a": 1, "b": 20, "c": 30}, "y": 3}


def test_coerce_types():
    assert _coerce("3") == 3
    assert _coerce("1e-4") == 1e-4
    assert _coerce("true") is True
    assert _coerce("null") is None
    assert _coerce("resnet34") == "resnet34"


def test_group_exposed_under_leaf_and_parent():
    cfg = load_config("diffusion/ddpm")
    assert cfg.diffusion.T == cfg.ddpm.T           # both names available
    assert cfg.T == cfg.diffusion.T                # and at top level


def test_paths_merged_under_key():
    cfg = load_config()
    assert "paths" in cfg and cfg.paths.tiles.endswith("tiles")


def test_override_applies():
    cfg = load_config("diffusion/ddpm", overrides=["diffusion.T=200"])
    assert cfg.diffusion.T == 200
