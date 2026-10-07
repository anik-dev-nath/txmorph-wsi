"""Lightweight config loading (CLAUDE.md S4: config-driven, seeded, logged).

We keep runtime deps light: configs are plain YAML under ``configs/``. This
loader merges, in order, ``default.yaml`` -> ``paths.yaml`` (under key ``paths``)
-> any named group files (e.g. ``diffusion/ddpm``) -> ``key=value`` CLI overrides,
and returns an attribute-accessible dict. This avoids a hard Hydra dependency
while preserving the "everything from configs/" rule.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
import ast
import yaml


class DotDict(dict):
    """dict with attribute access; nested dicts are wrapped on read."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return DotDict(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v


def _find_config_dir(start: Path | None = None) -> Path:
    """Walk up from cwd/start to find the ``configs/`` directory."""
    here = (start or Path.cwd()).resolve()
    for base in (here, *here.parents):
        cand = base / "configs"
        if cand.is_dir():
            return cand
    raise FileNotFoundError("could not locate a 'configs/' directory")


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(val: str) -> Any:
    """Parse a CLI override value ('3', '1e-4', 'true', 'null', 'text')."""
    low = val.lower()
    if low in ("null", "none", "~"):
        return None
    if low in ("true", "false"):
        return low == "true"
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def _apply_override(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(
    *groups: str,
    overrides: Iterable[str] | None = None,
    config_dir: str | Path | None = None,
) -> DotDict:
    """Merge default + paths + named group configs + CLI overrides.

    groups:    e.g. "diffusion/ddpm", "encoder/simclr" (paths relative to configs/,
               ".yaml" optional).
    overrides: iterable of "a.b=c" strings (e.g. from argparse REMAINDER).
    """
    cdir = Path(config_dir) if config_dir else _find_config_dir()
    cfg: dict = {}

    default = cdir / "default.yaml"
    if default.exists():
        base = load_yaml(default)
        base.pop("defaults", None)                 # Hydra 'defaults:' is a no-op here
        cfg = _deep_merge(cfg, base)

    paths = cdir / "paths.yaml"
    if paths.exists():
        cfg = _deep_merge(cfg, {"paths": load_yaml(paths)})

    for g in groups:
        p = cdir / (g if g.endswith((".yaml", ".yml")) else f"{g}.yaml")
        if not p.exists():
            raise FileNotFoundError(f"config group not found: {p}")
        # expose the group under its leaf name, its parent group name (e.g.
        # "diffusion/ddpm" -> also under "diffusion"), AND at the top level.
        parts = Path(g.removesuffix(".yaml").removesuffix(".yml")).parts
        data = load_yaml(p)
        cfg = _deep_merge(cfg, {parts[-1]: data})           # leaf ("ddpm")
        if len(parts) > 1:
            cfg = _deep_merge(cfg, {parts[0]: data})        # group ("diffusion")
        cfg = _deep_merge(cfg, data)                        # top level

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"bad override (expected key=value): {ov!r}")
        k, v = ov.split("=", 1)
        _apply_override(cfg, k.strip(), _coerce(v.strip()))

    return DotDict(cfg)
