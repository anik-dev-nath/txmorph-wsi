"""Deterministic conditional autoencoder - the Exp-5 ablation of the diffusion path.

Replaces the stochastic conditional diffusion ``p(z_post | z_pre, drug)`` with a
deterministic point predictor ``f(c) -> z_post`` (MSE-trained), where the context
``c = LayerNorm(Linear([z_pre || d_drug]))`` is built exactly as for the diffusion
model. At inference every "sample" is identical, so ``sigma2_tx == 0`` and P(pCR)
collapses to ``g(f(c)) in {0, 1}``. This isolates the value of modelling the full
response *distribution* (and thus the uncertainty biomarker) over a point estimate.

Matched in capacity to ``LatentDenoiser`` (same ``width``/``n_blocks``, residual
MLP blocks) so the ablation swaps *only* the generative mechanism, not the model
size.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class LatentRegressor(nn.Module):
    """c (cond_dim) -> residual MLP -> z_post (dim). No timestep, no noise."""

    def __init__(self, dim: int = 512, width: int = 512, cond_dim: int = 512,
                 n_blocks: int = 3):
        super().__init__()
        self.inp = nn.Linear(cond_dim, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(width, width),
                          nn.SiLU(), nn.Linear(width, width))
            for _ in range(n_blocks)])
        self.out = nn.Linear(width, dim)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        h = self.inp(c)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)
