"""Noise schedules for the latent diffusion model.

Implements the cosine schedule (Nichol & Dhariwal, 2021) and a linear fallback,
precomputing every per-timestep quantity the forward/reverse processes need.
All quantities are derived in float64 for numerical stability and exposed as
float32 tensors. See TxMorph-WSI_Equations_and_Notation.md (Sections 2-4).
"""
from __future__ import annotations
import math
import torch


def cosine_betas(T: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Cosine schedule: abar(t) = cos(((t/T + s)/(1+s)) * pi/2)^2."""
    def abar(t: int) -> float:
        return math.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    betas = [min(1.0 - abar(i) / abar(i - 1), max_beta) for i in range(1, T + 1)]
    return torch.tensor(betas, dtype=torch.float64)


def linear_betas(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)


class NoiseSchedule:
    """Holds all per-timestep diffusion coefficients as float32 buffers.

    Index convention: t in [0, T-1] addresses step (t+1) in the 1..T math, i.e.
    betas[0] = beta_1. posterior quantities at index 0 use alphas_cumprod_prev=1.
    """

    def __init__(self, T: int, schedule: str = "cosine", s: float = 0.008):
        self.T = T
        self.schedule = schedule
        if schedule == "cosine":
            betas = cosine_betas(T, s)
        elif schedule == "linear":
            betas = linear_betas(T)
        else:
            raise ValueError(f"unknown schedule {schedule}")

        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)                      # alphas_cumprod  (abar_t)
        acp_prev = torch.cat([torch.ones(1, dtype=torch.float64), acp[:-1]])

        # posterior variance  beta_tilde_t = (1 - abar_{t-1})/(1 - abar_t) * beta_t
        post_var = betas * (1.0 - acp_prev) / (1.0 - acp)
        # posterior mean coefficients (for q(z_{t-1}|z_t, z_0))
        post_mean_c0 = betas * torch.sqrt(acp_prev) / (1.0 - acp)
        post_mean_ct = (1.0 - acp_prev) * torch.sqrt(alphas) / (1.0 - acp)

        f32 = torch.float32
        self.betas = betas.to(f32)
        self.alphas = alphas.to(f32)
        self.alphas_cumprod = acp.to(f32)
        self.alphas_cumprod_prev = acp_prev.to(f32)
        self.sqrt_alphas_cumprod = torch.sqrt(acp).to(f32)
        self.sqrt_one_minus_acp = torch.sqrt(1.0 - acp).to(f32)
        self.posterior_variance = post_var.to(f32)
        # clamp log at index 0 (post_var[0] can be ~0) to avoid -inf
        self.posterior_log_variance = torch.log(
            torch.clamp(post_var, min=1e-20)
        ).to(f32)
        self.posterior_mean_coef0 = post_mean_c0.to(f32)
        self.posterior_mean_coeft = post_mean_ct.to(f32)
        self.log_betas = torch.log(betas).to(f32)

    def to(self, device):
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self


def extract(arr: torch.Tensor, t: torch.Tensor, broadcast_shape) -> torch.Tensor:
    """Gather arr[t] and reshape to broadcast against a tensor of broadcast_shape."""
    out = arr.to(t.device)[t].float()
    return out.reshape(t.shape[0], *([1] * (len(broadcast_shape) - 1)))
