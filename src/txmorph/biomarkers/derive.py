"""Derived biomarkers from the N sampled z_post vectors (Block 6).

Four readouts:
  z_post_bar : mean              -> expected morphology
  sigma2_tx  : mean per-dim var  -> response uncertainty (= tr(Cov)/d)
  P(pCR)     : fraction in zone   -> calibrated probability (g = pCR-zone logistic)
  residual r : obs - mean         -> resistance signal (non-pCR patients)
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class Biomarkers:
    z_post_bar: torch.Tensor
    sigma2_tx: float
    p_pcr: float
    residual: torch.Tensor | None


class PCRZone:
    """Logistic g(z) = sigmoid(w.z + b). FIT ON TRAINING FOLD ONLY (no leakage)."""

    def __init__(self, w: torch.Tensor, b: float, threshold: float = 0.5):
        self.w, self.b, self.thr = w, b, threshold

    def proba(self, z):
        return torch.sigmoid(z @ self.w + self.b)

    def in_zone(self, z):
        return (self.proba(z) > self.thr)


def derive_biomarkers(samples: torch.Tensor, pcr_zone: PCRZone,
                      z_post_obs: torch.Tensor | None = None) -> Biomarkers:
    """samples: (N, d).  z_post_obs: (d,) observed post-treatment embedding."""
    mean = samples.mean(dim=0)
    sigma2 = samples.var(dim=0, unbiased=False).mean().item()
    p_pcr = pcr_zone.in_zone(samples).float().mean().item()
    residual = None if z_post_obs is None else (z_post_obs - mean)
    return Biomarkers(z_post_bar=mean, sigma2_tx=sigma2, p_pcr=p_pcr, residual=residual)
