"""Inference engine (Block 6): draw N=100 reverse trajectories per patient.

Batched: all N trajectories advance together, one model call per timestep.
"""
from __future__ import annotations
import torch

from .gaussian_diffusion import GaussianDiffusion


@torch.no_grad()
def sample_patient(diffusion: GaussianDiffusion, model, context, n=100,
                   dim=512, device="cpu", learned_var=True, clip_x0=None):
    """context: (1, cond_dim) fused [z_pre||drug] context c, or None.
    Returns (n, dim) sampled z_post vectors.
    clip_x0: clamp predicted x0 each step for stability on unbounded latents
    (set from the empirical embedding scale, e.g. ~6x the per-dim std)."""
    model.eval()
    c = None if context is None else context.to(device)
    return diffusion.sample(model, n=n, dim=dim, c=c, device=device,
                            learned_var=learned_var, clip_x0=clip_x0)
