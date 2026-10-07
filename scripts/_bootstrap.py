"""Shared script bootstrap: put ``src/`` on the path and expose config loading.

Scripts are thin CLI wrappers; the real logic lives in ``src/txmorph``. Importing
this module makes ``import txmorph`` work whether or not the package is pip-installed.
"""
from __future__ import annotations
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from txmorph.utils.config import load_config          # noqa: E402
from txmorph.utils.seed import seed_everything        # noqa: E402


def common_args(ap):
    """Add the shared --group/--override/--force flags to an ArgumentParser."""
    ap.add_argument("--group", action="append", default=[],
                    help="named config group (e.g. diffusion/ddpm); repeatable")
    ap.add_argument("--override", action="append", default=[],
                    help="config override key=value; repeatable")
    ap.add_argument("--force", action="store_true", help="recompute existing artifacts")
    return ap


def load(args):
    """Load the merged config for the given CLI args and seed everything."""
    cfg = load_config(*args.group, overrides=args.override)
    seed_everything(cfg.get("seed", 0))
    return cfg
