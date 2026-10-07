"""Deterministic conditional-AE training (Exp-5 ablation of the diffusion path).

Mirrors ``training/diffusion.py`` but trains an MSE point predictor ``f(c) -> z_post``
instead of the denoiser. Same context builder + drug encoder, 5-fold patient-level
CV, checkpoint+resume each epoch, one checkpoint per fold (``ae_fold{k}.pt``).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from ..diffusion.autoencoder import LatentRegressor
from ..diffusion.context import ContextBuilder
from ..drug.encoder import OneHotDrugEncoder, SmilesDrugEncoder
from ..utils import ckpt
from ..utils.cv import train_val_masks


def _build_drug_encoder(variant: str, num_drugs: int, n_bits: int, device: str):
    if variant == "onehot":
        return OneHotDrugEncoder(num_drugs=num_drugs).to(device)
    if variant == "smiles":
        return SmilesDrugEncoder(n_bits=n_bits).to(device)
    raise ValueError(f"unknown drug variant: {variant}")


class AEModule:
    """Bundle of the trainable deterministic-AE parameters (parallels DiffusionModule)."""

    def __init__(self, dim, cond_dim, drug_variant, num_drugs, n_bits,
                 n_blocks, width, device):
        import torch.nn as nn
        self.regressor = LatentRegressor(dim=dim, width=width, cond_dim=cond_dim,
                                         n_blocks=n_blocks)
        self.context = ContextBuilder(z_dim=dim, out_dim=cond_dim)
        self.drug = _build_drug_encoder(drug_variant, num_drugs, n_bits, device)
        self.module = nn.ModuleList([self.regressor, self.context, self.drug]).to(device)

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()


def train_ae_fold(z_pre, z_post, drug_feats, folds, val_fold: int, ckpt_path: str,
                  dim: int = 512, cond_dim: int = 512, conditioning: str = "film",
                  n_blocks: int = 3, width: int = 512, drug_variant: str = "onehot",
                  num_drugs: int = 16, n_bits: int = 2048, epochs: int = 500,
                  lr: float = 1e-4, batch_size: int = 64, device: str = "cuda",
                  seed: int = 0, log_every: int = 50, **_ignore):
    """Train one CV fold's AE (train = all folds except ``val_fold``).

    ``conditioning`` (and any diffusion-only kwargs in ``_ignore``) are accepted so
    the call signature parallels ``train_diffusion_fold``; the AE simply consumes the
    context ``c`` and has no FiLM/cross-attn distinction.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    train_mask, _ = train_val_masks(folds, val_fold)
    zc = torch.from_numpy(np.asarray(z_pre, dtype=np.float32))[train_mask].to(device)
    z0 = torch.from_numpy(np.asarray(z_post, dtype=np.float32))[train_mask].to(device)
    dfeat = torch.from_numpy(np.asarray(drug_feats))[train_mask].to(device)
    dfeat = dfeat.long() if drug_variant == "onehot" else dfeat.float()

    mod = AEModule(dim, cond_dim, drug_variant, num_drugs, n_bits,
                   n_blocks, width, device)
    opt = torch.optim.AdamW(mod.parameters(), lr=lr)

    start_epoch = 0
    state = ckpt.maybe_resume(ckpt_path, mod, opt, map_location=device)
    if state is not None:
        start_epoch = state.get("epoch", 0) + 1

    n = z0.shape[0]
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    mod.train()
    for epoch in range(start_epoch, epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            zb, cb, db = z0[idx], zc[idx], dfeat[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                d_drug = mod.drug(db)
                c = mod.context(cb, d_drug)
                pred = mod.regressor(c)
                loss = F.mse_loss(pred, zb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch % log_every == 0:
            print(f"[ae fold{val_fold}] epoch {epoch} mse {loss.item():.4f}")
        ckpt.save(ckpt_path, mod, opt, step=0, epoch=epoch,
                  extra={"val_fold": val_fold})
    return mod


def train_ae_cv(z_pre, z_post, drug_feats, folds, ckpt_dir: str, n_folds: int = 5, **kw):
    """Train all folds; write ``ae_fold{k}.pt`` each. Returns paths."""
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(n_folds):
        p = str(Path(ckpt_dir) / f"ae_fold{k}.pt")
        train_ae_fold(z_pre, z_post, drug_feats, folds, k, p, **kw)
        paths.append(p)
    return paths
