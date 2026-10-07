"""Calibration & classification metrics (Experiment 1).

ECE, Brier, reliability curve, AUROC. Pure torch so they are unit-testable.
Bootstrap CIs are added in experiments/exp1_calibration.py.
"""
from __future__ import annotations
import torch


def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    return torch.mean((probs - labels.float()) ** 2).item()


def expected_calibration_error(probs, labels, n_bins: int = 15) -> float:
    probs = probs.float()
    labels = labels.float()
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = 0.0
    n = probs.numel()
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += (mask.float().sum() / n) * torch.abs(acc - conf)
    return float(ece)


def reliability_curve(probs, labels, n_bins: int = 15):
    """Return (bin_confidence, bin_accuracy, bin_count) for a reliability diagram."""
    bins = torch.linspace(0, 1, n_bins + 1)
    conf, acc, cnt = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            conf.append(float("nan")); acc.append(float("nan")); cnt.append(0)
        else:
            conf.append(probs[mask].mean().item())
            acc.append(labels[mask].float().mean().item())
            cnt.append(int(mask.sum()))
    return conf, acc, cnt


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUROC (Mann-Whitney)."""
    scores = scores.float()
    labels = labels.long()
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    n_pos = (labels == 1).sum().item()
    n_neg = (labels == 0).sum().item()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[labels == 1].sum().item()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
