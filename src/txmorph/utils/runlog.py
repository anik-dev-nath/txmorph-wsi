"""Per-run logging (CLAUDE.md S5, S13): config.yaml + env.txt + metrics.json.

Every entrypoint lands in ``runs/{date}_{user}_{name}/`` and records the exact
config, environment (pip freeze + GPU + git SHA), and final metrics so any run is
reproducible from its own directory.
"""
from __future__ import annotations
from datetime import date
from pathlib import Path
import getpass
import json
import subprocess
import sys


def new_run_dir(runs_root: str, name: str) -> Path:
    """Create and return ``runs/{date}_{user}_{name}/``."""
    user = getpass.getuser()
    d = Path(runs_root) / f"{date.today().isoformat()}_{user}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _env_text() -> str:
    lines = [f"python: {sys.version.splitlines()[0]}"]
    try:
        import torch
        lines.append(f"torch: {torch.__version__}")
        if torch.cuda.is_available():
            lines.append(f"gpu: {torch.cuda.get_device_name(0)}")
    except Exception:
        lines.append("torch: unavailable")
    for cmd, label in [(["git", "rev-parse", "HEAD"], "git_sha"),
                       ([sys.executable, "-m", "pip", "freeze"], "pip_freeze")]:
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            lines.append(f"--- {label} ---\n{out.strip()}")
        except Exception:
            lines.append(f"--- {label} --- unavailable")
    return "\n".join(lines)


def log_run(run_dir: Path, config: dict, metrics: dict | None = None) -> None:
    """Write config.yaml, env.txt, and (optionally) metrics.json into ``run_dir``."""
    import yaml
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dict(config)))
    (run_dir / "env.txt").write_text(_env_text())
    if metrics is not None:
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
