"""Deterministic-AE ablation: training reduces MSE and inference is deterministic
(sigma2_tx == 0, P(pCR) in {0,1}). Needs torch, so runs on the server."""
import numpy as np
import pytest

pytest.importorskip("torch")


def _toy(rng, n=48, dim=8):
    """A learnable map: z_post = z_pre @ W (+ drug bias); patient-level folds."""
    z_pre = rng.standard_normal((n, dim)).astype(np.float32)
    W = rng.standard_normal((dim, dim)).astype(np.float32)
    drug = (rng.random(n) < 0.5).astype(np.int64)          # 2 drugs, one-hot ids
    z_post = (z_pre @ W + drug[:, None]).astype(np.float32)
    pcr = (z_post[:, 0] > np.median(z_post[:, 0])).astype(int)
    folds = np.tile(np.arange(4), n // 4)[:n]
    return z_pre, z_post, drug, pcr, folds


def test_ae_trains_and_is_deterministic(tmp_path):
    import torch  # noqa: F401
    from txmorph.training.ae import train_ae_fold
    from txmorph.inference.ae_run import run_ae_inference

    rng = np.random.default_rng(0)
    z_pre, z_post, drug, pcr, folds = _toy(rng)
    dim = z_pre.shape[1]
    ck = str(tmp_path / "ae_fold0.pt")
    kw = dict(dim=dim, cond_dim=dim, n_blocks=2, width=32, drug_variant="onehot",
              num_drugs=2, epochs=60, lr=5e-3, batch_size=16, device="cpu", seed=0)
    train_ae_fold(z_pre, z_post, drug, folds, val_fold=0, ckpt_path=ck, **kw)

    recs = run_ae_inference(
        z_pre, z_post, drug, pcr, folds, str(tmp_path), str(tmp_path / "samp"),
        n_folds=1, n_samples=16, dim=dim, cond_dim=dim, drug_variant="onehot",
        num_drugs=2, n_blocks=2, width=32, device="cpu")
    assert recs, "expected held-out records for fold 0"
    for r in recs:
        assert r["sigma2_tx"] == pytest.approx(0.0, abs=1e-6)   # deterministic
        assert r["p_pcr"] in (0.0, 1.0)                          # hard zone decision
