"""Conditioning context builder c = LayerNorm(Linear([z_pre || d_drug])) (CLAUDE.md S6.5).

Fuses the pre-treatment slide embedding (512-d) and the drug embedding (128-d)
into the 512-d context vector consumed by the denoiser. Supports classifier-free
guidance by randomly dropping the context to the denoiser's null token during
training (``p_uncond``): callers pass ``c=None`` for the dropped rows.

**Paired mode** (default): the target is ``z_post`` and the context carries
``z_pre``, so the model learns ``p(z_post | z_pre, drug)``.

**Single-timepoint mode** (``z_dim=0, pcr_dim=1``): there is no ``z_post``, so the
target becomes ``z_pre`` itself and the context is built from ``[drug || pCR]``
*only*. Excluding ``z_pre`` is not a detail -- it is what makes the model a
class-conditional density ``p(z_pre | drug, pCR)`` whose likelihood ratio between
pCR=1 and pCR=0 is the biomarker. Feeding ``z_pre`` into the context while
denoising ``z_pre`` lets the denoiser recover the noise from the context alone
(``eps = (z_t - sqrt(abar) z_pre) / sqrt(1-abar)``), so it never needs the pCR bit
and the ratio collapses: measured on weakly-separated toy data, AUROC 0.436 with
``z_pre`` in the context versus 0.864 without.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ContextBuilder(nn.Module):
    """[z_pre || d_drug (|| pcr_embed)] -> LayerNorm(Linear) -> c (out_dim).

    ``z_dim=0`` drops ``z_pre`` from the context (single-timepoint mode); the
    ``z_pre`` argument to :meth:`forward` is then ignored, so call sites are
    identical in both modes. ``pcr_dim>0`` appends the pCR label (1 dim for a
    scalar).
    """

    def __init__(self, z_dim: int = 512, drug_dim: int = 128,
                 pcr_dim: int = 0, out_dim: int = 512):
        super().__init__()
        self.z_dim = z_dim
        self.pcr_dim = pcr_dim
        self.proj = nn.Linear(z_dim + drug_dim + pcr_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, z_pre: torch.Tensor, d_drug: torch.Tensor,
                pcr_embed: torch.Tensor | None = None) -> torch.Tensor:
        parts = [z_pre] if self.z_dim > 0 else []
        parts.append(d_drug)
        if self.pcr_dim > 0:
            if pcr_embed is None:
                raise ValueError("ContextBuilder was built with pcr_dim>0 but no "
                                 "pcr_embed was passed; the pCR condition would be "
                                 "silently dropped")
            parts.append(pcr_embed)
        return self.norm(self.proj(torch.cat(parts, dim=-1)))
