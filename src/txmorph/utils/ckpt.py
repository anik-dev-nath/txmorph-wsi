"""Checkpoint save/resume: step + optimizer + model + RNG state (CLAUDE.md S2.2).

Long GPU jobs on a shared server can be interrupted; every training loop should
checkpoint each epoch and auto-resume if a checkpoint exists. This module is a
thin, dependency-light wrapper around ``torch.save``/``torch.load`` that also
persists python/numpy/torch RNG state so a resumed run is bit-for-bit continuous.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import random

import numpy as np
import torch


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state: dict) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        # set_rng_state_all requires CPU ByteTensors; a checkpoint loaded with
        # map_location="cuda" yields cuda tensors, so move them back to CPU.
        cuda_states = [s.cpu() if torch.is_tensor(s) else s for s in state["torch_cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def save(path: str | Path, model, optimizer=None, step: int = 0, epoch: int = 0,
         extra: dict | None = None) -> None:
    """Atomically write a checkpoint (model+optimizer+step+epoch+RNG+extra)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "model": model.state_dict() if hasattr(model, "state_dict") else model,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "epoch": epoch,
        "rng": _rng_state(),
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)                                  # atomic on same filesystem


def load(path: str | Path, model=None, optimizer=None, map_location="cpu",
         restore_rng: bool = True) -> dict:
    """Load a checkpoint; optionally restore model/optimizer/RNG in place.

    Returns the raw state dict (so callers can read step/epoch/extra).
    """
    state = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None and state.get("model") is not None:
        model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if restore_rng and state.get("rng"):
        _set_rng_state(state["rng"])
    return state


def maybe_resume(path: str | Path, model, optimizer=None, map_location="cpu") -> dict | None:
    """Resume from ``path`` if it exists, else return None (fresh start)."""
    path = Path(path)
    if path.exists():
        return load(path, model, optimizer, map_location=map_location)
    return None
