"""Conditional latent-diffusion training loop, 5-fold patient-level CV
(CLAUDE.md S6.5, S7, S8 step 7).

**Paired mode** (default): trains p(z_post | z_pre, drug). Context is
c = LayerNorm(Linear([z_pre || d_drug])).

**Single-timepoint mode** (``single_timepoint=True``): trains class-conditional
p(z_pre | drug, pCR). Context is c = LayerNorm(Linear([z_pre || d_drug || pcr])).
The target z_0 = z_pre and pCR is appended to the context. At inference, NLL is
evaluated under pCR=1 and pCR=0 to compute P(pCR) via Bayes' rule.

Classifier-free guidance: drop context to the null token with probability
``p_uncond``. bf16 autocast; checkpoint + resume every epoch; one checkpoint per
fold (train on the other folds).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from ..diffusion.schedule import NoiseSchedule
from ..diffusion.gaussian_diffusion import GaussianDiffusion
from ..diffusion.denoiser import LatentDenoiser
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


class DiffusionModule:
    """Bundle of the trainable conditional-diffusion parameters."""

    def __init__(self, dim, cond_dim, drug_variant, num_drugs, n_bits,
                 n_blocks, width, conditioning, device,
                 single_timepoint: bool = False):
        import torch.nn as nn
        self.single_timepoint = single_timepoint
        self.denoiser = LatentDenoiser(dim=dim, width=width, cond_dim=cond_dim,
                                       n_blocks=n_blocks, conditioning=conditioning)
        # Single-timepoint: target is z_pre, so z_pre must stay OUT of the context
        # or the denoiser reads the answer from its own conditioning and ignores
        # the pCR bit entirely (see diffusion/context.py).
        self.context = ContextBuilder(
            z_dim=0 if single_timepoint else dim, out_dim=cond_dim,
            pcr_dim=1 if single_timepoint else 0)
        self.drug = _build_drug_encoder(drug_variant, num_drugs, n_bits, device)
        self.module = nn.ModuleList([self.denoiser, self.context, self.drug]).to(device)

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


def train_diffusion_fold(z_pre, z_post, drug_feats, folds, val_fold: int,
                         ckpt_path: str, dim: int = 512, cond_dim: int = 512,
                         T: int = 1000, schedule: str = "cosine", cosine_s: float = 0.008,
                         lambda_vlb: float = 1e-3, conditioning: str = "film",
                         n_blocks: int = 3, width: int = 512,
                         drug_variant: str = "onehot", num_drugs: int = 16,
                         n_bits: int = 2048, epochs: int = 500, lr: float = 1e-4,
                         batch_size: int = 64, p_uncond: float = 0.1,
                         device: str = "cuda", seed: int = 0, log_every: int = 50,
                         single_timepoint: bool = False, pcr=None,
                         unlabeled=None, pretrain_epochs: int = 0,
                         unlabeled_drug_id: int | None = None):
    """Train one CV fold's diffusion model (train = all folds except ``val_fold``).

    When ``single_timepoint=True``: target z_0 = z_pre (not z_post), and the pCR
    label is included in the context so the model learns class-conditional densities
    p(z_pre | drug, pCR). ``pcr`` must be an (N,) int array of labels.

    **Semi-supervised density** (``unlabeled`` + ``pretrain_epochs > 0``): the
    denoiser is first fit unconditionally -- pCR set to the null token -- on the
    training fold *plus* ``unlabeled`` slide embeddings, then fine-tuned
    conditionally on the labeled training fold only. This is the one structural
    advantage the generative framing has over a discriminative baseline: a
    logistic regression cannot consume slides that carry no pCR label. Measured
    at +0.025 AUROC with only 53 unlabeled slides
    (``scripts/19_architecture_ablation.py``).

    ``unlabeled`` must contain **no** patient from any CV fold and none from the
    external validation cohort; it is caller-supplied and not checked here, so
    build it with ``scripts/20_build_unlabeled_corpus.sh``, which enforces that.
    """
    import torch

    torch.manual_seed(seed)
    train_mask, _ = train_val_masks(folds, val_fold)
    zc = torch.from_numpy(np.asarray(z_pre, dtype=np.float32))[train_mask].to(device)
    dfeat = torch.from_numpy(np.asarray(drug_feats))[train_mask].to(device)
    if drug_variant == "onehot":
        dfeat = dfeat.long()
    else:
        dfeat = dfeat.float()

    if single_timepoint:
        z0 = zc.clone()  # target = z_pre
        assert pcr is not None, "pcr labels required for single-timepoint mode"
        pcr_labels = torch.from_numpy(
            np.asarray(pcr, dtype=np.float32))[train_mask].to(device)
    else:
        z0 = torch.from_numpy(np.asarray(z_post, dtype=np.float32))[train_mask].to(device)
        pcr_labels = None

    sch = NoiseSchedule(T=T, schedule=schedule, s=cosine_s)
    diffusion = GaussianDiffusion(sch, lambda_vlb=lambda_vlb).to(device)
    mod = DiffusionModule(dim, cond_dim, drug_variant, num_drugs, n_bits,
                          n_blocks, width, conditioning, device,
                          single_timepoint=single_timepoint)
    opt = torch.optim.AdamW(mod.parameters(), lr=lr)

    start_epoch = 0
    state = ckpt.maybe_resume(ckpt_path, mod, opt, map_location=device)
    if state is not None:
        start_epoch = state.get("epoch", 0) + 1

    n = z0.shape[0]
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    mod.train()

    # ---- semi-supervised phase -------------------------------------------
    # Fit p(z) unconditionally on labeled-train + unlabeled slides, then fall
    # through to the conditional fine-tune below. Skipped when resuming, since
    # the restored checkpoint already carries it.
    if (single_timepoint and unlabeled is not None and len(unlabeled)
            and pretrain_epochs > 0 and state is None):
        un = torch.from_numpy(np.asarray(unlabeled, dtype=np.float32)).to(device)
        if un.shape[1] != z0.shape[1]:
            raise ValueError(f"unlabeled dim {un.shape[1]} != latent dim {z0.shape[1]}")

        # Unlabeled slides have no regimen. Giving them an existing drug id would
        # teach the density that that drug looks like generic breast morphology,
        # so they get their own reserved slot instead.
        if drug_variant == "onehot":
            if unlabeled_drug_id is None:
                unlabeled_drug_id = num_drugs - 1
            observed = {int(v) for v in np.asarray(drug_feats).reshape(-1)}
            if unlabeled_drug_id in observed:
                raise ValueError(
                    f"unlabeled_drug_id={unlabeled_drug_id} collides with an "
                    f"observed drug id; raise num_drugs or pass a free slot")
            d_un = torch.full((len(un),), int(unlabeled_drug_id),
                              dtype=torch.long, device=device)
        else:
            d_un = torch.zeros((len(un), dfeat.shape[1]),
                               dtype=torch.float32, device=device)

        z_pre_all = torch.cat([z0, un])
        d_all = torch.cat([dfeat, d_un])
        m = z_pre_all.shape[0]
        print(f"[diff fold{val_fold}] semi-supervised pretrain: "
              f"{n} labeled + {len(un)} unlabeled = {m} slides, "
              f"{pretrain_epochs} epochs", flush=True)
        for epoch in range(pretrain_epochs):
            perm = torch.randperm(m, device=device)
            for i in range(0, m, batch_size):
                idx = perm[i:i + batch_size]
                zb, db = z_pre_all[idx], d_all[idx]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    # pCR is the null token throughout: this phase learns the
                    # marginal p(z | drug), not either class-conditional.
                    null_pcr = torch.full((len(idx), 1), -1.0, device=device)
                    c = mod.context(zb, mod.drug(db), null_pcr)
                    loss = diffusion.loss(mod.denoiser, zb, c)["total"]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            if epoch % log_every == 0:
                print(f"[diff fold{val_fold}] pretrain epoch {epoch} "
                      f"total {loss.item():.4f}", flush=True)

    for epoch in range(start_epoch, epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            zb, cb, db = z0[idx], zc[idx], dfeat[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                d_drug = mod.drug(db)
                if single_timepoint:
                    pcr_embed = pcr_labels[idx].unsqueeze(-1)  # (B, 1)
                    # Classifier-free guidance: drop pCR with p_uncond
                    if p_uncond > 0:
                        drop = torch.rand(pcr_embed.shape[0], device=device) < p_uncond
                        pcr_embed = pcr_embed.clone()
                        pcr_embed[drop] = -1.0  # null token
                    c = mod.context(cb, d_drug, pcr_embed)
                else:
                    c = mod.context(cb, d_drug)
                if p_uncond > 0 and not single_timepoint:
                    drop = torch.rand(c.shape[0], device=device) < p_uncond
                    c[drop] = 0.0
                out = diffusion.loss(mod.denoiser, zb, c)
                loss = out["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch % log_every == 0:
            print(f"[diff fold{val_fold}] epoch {epoch} "
                  f"total {loss.item():.4f} simple {out['simple'].item():.4f} "
                  f"vlb {out['vlb'].item():.4f}")
        ckpt.save(ckpt_path, mod, opt, step=0, epoch=epoch,
                  extra={"val_fold": val_fold, "single_timepoint": single_timepoint})
    return mod, diffusion


def train_diffusion_cv(z_pre, z_post, drug_feats, folds, ckpt_dir: str,
                       n_folds: int = 5, **kw):
    """Train all folds; write ``diffusion_fold{k}.pt`` each. Returns paths."""
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(n_folds):
        p = str(Path(ckpt_dir) / f"diffusion_fold{k}.pt")
        train_diffusion_fold(z_pre, z_post, drug_feats, folds, k, p, **kw)
        paths.append(p)
    return paths
