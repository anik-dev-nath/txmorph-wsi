"""Denoiser for the 512-d latent (Block 5).

The latent is a vector, not a spatial grid, so this is an MLP-style residual
"U-Net": down-then-up width with skip connections in feature space. Timestep is
injected via a sinusoidal embedding; the context c (= [z_pre || drug] fused) is
injected by FiLM (default) or cross-attention (stub).

Implementation note (see TxMorph-WSI_DDPM_Deep_Dive.md S3.3): with a *single*
context token, cross-attention degenerates to a learned additive injection, so
FiLM (which adds a multiplicative scale) is at least as expressive and is the
default. The ``conditioning='crossattn'`` variant (Exp-5 ablation) therefore
attends over *two* context tokens - the timestep embedding and the fused context
``c`` - so the attention is non-degenerate and genuinely differs from FiLM.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class FiLMResBlock(nn.Module):
    """Residual MLP block with FiLM (scale+shift) conditioning."""

    def __init__(self, width: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.lin1 = nn.Linear(width, width)
        self.lin2 = nn.Linear(width, width)
        self.act = nn.SiLU()
        self.film = nn.Linear(cond_dim, 2 * width)

    def forward(self, h, cond):
        scale, shift = self.film(cond).chunk(2, dim=-1)
        x = self.norm(h)
        x = x * (1 + scale) + shift
        x = self.lin2(self.act(self.lin1(self.act(x))))
        return h + x


class CrossAttnResBlock(nn.Module):
    """Residual block conditioning via multi-head cross-attention.

    The hidden state ``h`` is the (length-1) query sequence; the keys/values are
    the context tokens ``[temb, cemb]`` (timestep + fused context). With two KV
    tokens the softmax is non-degenerate, so this is a genuine cross-attention
    variant of the FiLM block (Exp-5 ablation), not the single-token degenerate
    case. An attention residual is followed by a standard MLP residual.
    """

    def __init__(self, width: int, cond_dim: int, n_heads: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, n_heads, batch_first=True)
        self.norm_m = nn.LayerNorm(width)
        self.lin1 = nn.Linear(width, width)
        self.lin2 = nn.Linear(width, width)
        self.act = nn.SiLU()

    def forward(self, h, ctx_tokens):
        # h: (B, width); ctx_tokens: (B, K, width)
        q = self.norm_q(h).unsqueeze(1)                 # (B, 1, width)
        a, _ = self.attn(q, ctx_tokens, ctx_tokens)     # (B, 1, width)
        h = h + a.squeeze(1)
        x = self.lin2(self.act(self.lin1(self.act(self.norm_m(h)))))
        return h + x


class LatentDenoiser(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        width: int = 512,
        cond_dim: int = 512,
        n_blocks: int = 3,
        conditioning: str = "film",
    ):
        super().__init__()
        self.conditioning = conditioning
        self.dim = dim
        self.cond_dim = cond_dim
        te = width
        self.t_mlp = nn.Sequential(
            nn.Linear(te, te), nn.SiLU(), nn.Linear(te, te)
        )
        self.c_mlp = nn.Sequential(
            nn.Linear(cond_dim, te), nn.SiLU(), nn.Linear(te, te)
        )
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))  # used when c is None
        self.in_proj = nn.Linear(dim, width)
        if conditioning == "film":
            self.blocks = nn.ModuleList(
                [FiLMResBlock(width, te) for _ in range(n_blocks)]
            )
        elif conditioning == "crossattn":
            self.blocks = nn.ModuleList(
                [CrossAttnResBlock(width, te) for _ in range(n_blocks)]
            )
        else:
            raise ValueError(conditioning)
        self.out_norm = nn.LayerNorm(width)
        self.head_eps = nn.Linear(width, dim)
        self.head_v = nn.Linear(width, dim)
        self.te = te

    def forward(self, zt, t, c=None):
        temb = self.t_mlp(timestep_embedding(t, self.te))
        if c is None:
            c = self.null_cond.unsqueeze(0).expand(zt.shape[0], -1)
        cemb = self.c_mlp(c)
        h = self.in_proj(zt)
        if self.conditioning == "crossattn":
            ctx_tokens = torch.stack([temb, cemb], dim=1)   # (B, 2, width)
            for blk in self.blocks:
                h = blk(h, ctx_tokens)
        else:
            cond = temb + cemb
            for blk in self.blocks:
                h = blk(h, cond)
        h = self.out_norm(h)
        return self.head_eps(h), self.head_v(h)
