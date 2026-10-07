import torch
from txmorph.eval.linear_probe import linear_probe, extract_features
from txmorph.utils.seed import seed_everything


def _make_classes(n_per, d, n_classes, sep, noise, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = sep * torch.randn(n_classes, d, generator=g)
    feats, labels = [], []
    for c in range(n_classes):
        feats.append(centers[c] + noise * torch.randn(n_per, d, generator=g))
        labels.append(torch.full((n_per,), c))
    return torch.cat(feats), torch.cat(labels)


def test_probe_passes_on_separable_features():
    """A good representation (separable classes) must clear the 0.90 gate."""
    seed_everything(0)
    f, y = _make_classes(n_per=300, d=64, n_classes=9, sep=4.0, noise=1.0)
    n = f.shape[0]; perm = torch.randperm(n)
    tr, va = perm[: n * 4 // 5], perm[n * 4 // 5:]
    res = linear_probe(f[tr], y[tr], f[va], y[va], n_classes=9, target=0.90)
    assert res.passed and res.accuracy > 0.90, res.accuracy


def test_probe_fails_on_noise_features():
    """A bad representation (features independent of labels) must NOT pass."""
    seed_everything(1)
    f = torch.randn(2700, 64)
    y = torch.randint(0, 9, (2700,))
    n = f.shape[0]; perm = torch.randperm(n)
    tr, va = perm[: n * 4 // 5], perm[n * 4 // 5:]
    res = linear_probe(f[tr], y[tr], f[va], y[va], n_classes=9, target=0.90)
    assert not res.passed and res.accuracy < 0.30, res.accuracy   # ~ chance (1/9)


def test_extract_features_from_encoder():
    """extract_features pulls frozen embeddings + labels from a loader."""
    import torch.nn as nn
    enc = nn.Linear(10, 8)
    enc.embed = lambda x: enc(x)           # mimic SimCLREncoder.embed
    loader = [(torch.randn(16, 10), torch.randint(0, 3, (16,))) for _ in range(4)]
    feats, labels = extract_features(enc, loader)
    assert feats.shape == (64, 8) and labels.shape == (64,)
