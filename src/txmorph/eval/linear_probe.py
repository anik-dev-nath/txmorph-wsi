"""Linear-probe evaluation gate for the frozen SimCLR encoder (CLAUDE.md S6.2).

Protocol (Chen et al., 2020 style): freeze the encoder, extract features on a
labeled dataset (NCT-CRC-100K, 9 tissue classes), train ONLY a linear classifier
on the frozen features, report top-1 accuracy. Gate: must clear ~0.90 before
extracting embeddings for the rest of the project.

Dataset-agnostic: works on any (features, labels). Pure torch (no sklearn) so it
is unit-testable. On the real run, feed encoder.embed() outputs over NCT-CRC-100K.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class ProbeResult:
    accuracy: float
    passed: bool
    target: float
    per_class_acc: list


@torch.no_grad()
def extract_features(encoder, loader, device="cpu"):
    """Run a frozen encoder over a dataloader -> (features (N,d), labels (N,))."""
    encoder.eval()
    feats, labels = [], []
    for x, y in loader:
        h = encoder.embed(x.to(device)) if hasattr(encoder, "embed") else encoder(x.to(device))
        if isinstance(h, tuple):
            h = h[0]
        feats.append(h.cpu()); labels.append(y.cpu())
    return torch.cat(feats), torch.cat(labels)


def _standardize(train, *others, eps=1e-6):
    mu = train.mean(0, keepdim=True)
    sd = train.std(0, keepdim=True) + eps
    return [(t - mu) / sd for t in (train, *others)]


def linear_probe(train_feats, train_labels, val_feats, val_labels,
                 n_classes: int, target: float = 0.90,
                 epochs: int = 300, lr: float = 1e-2, seed: int = 0) -> ProbeResult:
    """Train a single linear layer on frozen features; report val top-1 accuracy."""
    torch.manual_seed(seed)
    train_feats, val_feats = _standardize(train_feats.float(), val_feats.float())
    clf = nn.Linear(train_feats.shape[1], n_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(clf(train_feats), train_labels.long())
        loss.backward(); opt.step()

    with torch.no_grad():
        pred = clf(val_feats).argmax(1)
        acc = (pred == val_labels).float().mean().item()
        per_class = []
        for c in range(n_classes):
            m = val_labels == c
            per_class.append((pred[m] == c).float().mean().item() if m.any() else float("nan"))
    return ProbeResult(accuracy=acc, passed=acc >= target, target=target,
                       per_class_acc=per_class)


def evaluate_encoder_probe(encoder, train_loader, val_loader, n_classes,
                           target: float = 0.90, device="cpu", **kw) -> ProbeResult:
    """Convenience: extract frozen features then run the linear probe (the gate)."""
    tf, tl = extract_features(encoder, train_loader, device)
    vf, vl = extract_features(encoder, val_loader, device)
    return linear_probe(tf, tl, vf, vl, n_classes=n_classes, target=target, **kw)
