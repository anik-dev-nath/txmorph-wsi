"""Deterministic-AE held-out inference (Exp-5 ablation).

Point prediction ``f(c) -> z_post``; every one of the N "samples" is identical, so
``sigma2_tx == 0`` and ``P(pCR) = g(pred) in {0, 1}``. Mirrors ``inference/run.py``
(same fold loop, same train-only pCR-zone fit, same record schema) so the ablation
plugs straight into the Exp-5 delta tabulation.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from ..training.ae import AEModule
from ..biomarkers.derive import derive_biomarkers
from ..biomarkers.pcr_zone import fit_pcr_zone
from ..utils import ckpt
from ..utils.cv import train_val_masks


def load_ae_fold(ckpt_path: str, dim, cond_dim, drug_variant, num_drugs, n_bits,
                 n_blocks, width, device):
    """Load one fold's trained AEModule from ``ckpt_path`` (eval mode)."""
    mod = AEModule(dim, cond_dim, drug_variant, num_drugs, n_bits,
                   n_blocks, width, device)
    ckpt.load(ckpt_path, mod, map_location=device, restore_rng=False)
    mod.eval()
    return mod


def run_ae_inference(z_pre, z_post, drug_feats, pcr, folds, ckpt_dir: str,
                     out_dir: str, n_folds: int = 5, n_samples: int = 100,
                     dim: int = 512, cond_dim: int = 512, drug_variant: str = "onehot",
                     num_drugs: int = 16, n_bits: int = 2048, n_blocks: int = 3,
                     width: int = 512, device: str = "cpu", force: bool = False,
                     **_ignore):
    """Held-out AE inference over all folds. Same record schema as ``run_inference``."""
    import torch

    z_pre = np.asarray(z_pre, dtype=np.float32)
    z_post = np.asarray(z_post, dtype=np.float32)
    drug_feats = np.asarray(drug_feats)
    pcr = np.asarray(pcr).astype(int)
    folds = np.asarray(folds)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    records = []
    for k in range(n_folds):
        train_mask, val_mask = train_val_masks(folds, k)
        # pCR-zone: fit on this fold's TRAIN patients only (no leakage)
        zone = fit_pcr_zone(z_post[train_mask], pcr[train_mask], device=device)

        ckpt_path = str(Path(ckpt_dir) / f"ae_fold{k}.pt")
        mod = load_ae_fold(ckpt_path, dim, cond_dim, drug_variant, num_drugs,
                           n_bits, n_blocks, width, device)

        for i in np.where(val_mask)[0]:
            out_path = Path(out_dir) / f"{i}.npy"
            if out_path.exists() and not force:
                samples = torch.from_numpy(np.load(out_path)).to(device)
            else:
                zc = torch.from_numpy(z_pre[i:i + 1]).to(device)
                db = torch.from_numpy(drug_feats[i:i + 1]).to(device)
                db = db.long() if drug_variant == "onehot" else db.float()
                with torch.no_grad():
                    d_drug = mod.drug(db)
                    c = mod.context(zc, d_drug)
                    pred = mod.regressor(c)                 # (1, dim) point estimate
                samples = pred.expand(n_samples, dim).contiguous()
                np.save(out_path, samples.cpu().numpy())

            z_obs = torch.from_numpy(z_post[i]).to(device)
            bm = derive_biomarkers(samples, zone, z_obs)
            records.append({
                "index": int(i), "fold": int(k), "pcr": int(pcr[i]),
                "p_pcr": bm.p_pcr, "sigma2_tx": bm.sigma2_tx,
                "z_post_bar": bm.z_post_bar.cpu().numpy(),
                "residual": None if bm.residual is None else bm.residual.cpu().numpy(),
            })
    return records
