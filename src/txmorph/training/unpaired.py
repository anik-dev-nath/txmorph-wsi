"""Unpaired training of the ORIGINAL conditional design p(z_post | z_pre, drug).

WHY THIS EXISTS (CLAUDE.md S1; unpaired reframing 2026-09-08)
    The project's scientific claim needs (z_pre, z_post) pairs. No open cohort has
    them -- IMPRESS is pre-treatment biopsies, Post-NAT-BRCA is post-treatment
    resections from different patients. The July pivot dropped z_post entirely and
    turned the project into a pCR classifier, which is not the contribution.

    This module restores the original design on unpaired data. Each minibatch, an
    optimal-transport assignment between the pre batch and a post batch supplies
    pseudo-pairs, and the EXISTING, TESTED conditional DDPM trains on them
    unchanged. Nothing in diffusion/ is modified: same schedule, same denoiser,
    same context builder c = LayerNorm(Linear([z_pre || d_drug])), same sampler,
    same four biomarkers. Only where the training target comes from changes.

    Consequence worth stating: sigma^2_tx becomes a real quantity again. In the
    single-timepoint path it was hard-coded to p(1-p) (docs/LIMITATIONS.md S2);
    here it is the spread of sampled post-treatment morphology, as designed.

WHAT IS AND IS NOT LEARNED
    The map is correct in distribution, not per patient. No pseudo-pair is a real
    patient's before/after, and no claim of per-patient correspondence is
    supportable on unpaired data. Evaluate with the two-sample metrics in
    eval/distribution.py against HELD-OUT post samples, never by reconstruction.

LEAKAGE
    Two independent splits are enforced: pre-side patients are split by the frozen
    patient-level folds, and the post pool has its own seeded split. A post sample
    in the validation pool is never coupled during training, or the distributional
    evaluation would be scoring the model on its own training data.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from ..diffusion.schedule import NoiseSchedule
from ..diffusion.gaussian_diffusion import GaussianDiffusion
from ..transport.coupling import ot_coupling, coupling_cost
from ..utils import ckpt
from ..utils.cv import train_val_masks
from .diffusion import DiffusionModule


def split_post_pool(n_post: int, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Seeded fold assignment for the post pool, mirroring the pre-side scheme."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(n_post) % n_folds)


def train_unpaired_fold(z_pre, drug_feats, z_post_pool, folds, post_folds,
                        val_fold: int, ckpt_path: str,
                        groups_pre=None, groups_post=None,
                        dim: int = 512, cond_dim: int = 512, T: int = 1000,
                        schedule: str = "cosine", cosine_s: float = 0.008,
                        lambda_vlb: float = 1e-3, conditioning: str = "film",
                        n_blocks: int = 3, width: int = 512,
                        drug_variant: str = "onehot", num_drugs: int = 16,
                        n_bits: int = 2048, epochs: int = 500, lr: float = 1e-4,
                        batch_size: int = 64, p_uncond: float = 0.1,
                        coupling: str = "exact", post_batch_mult: float = 1.0,
                        subspace_dim: int | None = None,
                        device: str = "cuda", seed: int = 0, log_every: int = 50):
    """Train one CV fold of p(z_post | z_pre, drug) from unpaired data.

    ``coupling='independent'`` is the ablation that removes OT and pairs at
    random; it isolates what the transport actually contributes.
    ``post_batch_mult`` draws a larger post batch than the pre batch so the
    assignment has slack to choose from rather than being forced into a
    near-bijection.
    """
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    z_pre = np.asarray(z_pre, dtype=np.float32)
    z_post_pool = np.asarray(z_post_pool, dtype=np.float32)
    drug_feats = np.asarray(drug_feats)

    tr_mask, _ = train_val_masks(folds, val_fold)
    pre_idx = np.where(tr_mask)[0]
    post_idx = np.where(post_folds != val_fold)[0]     # post training pool only
    if len(pre_idx) == 0 or len(post_idx) == 0:
        raise ValueError(f"fold {val_fold}: empty pre ({len(pre_idx)}) or "
                         f"post ({len(post_idx)}) training set")

    gp = None if groups_pre is None else np.asarray(groups_pre)
    gq = None if groups_post is None else np.asarray(groups_post)

    # ---- in-fold standardisation -------------------------------------------
    # The cosine schedule assumes z_0 is roughly unit-scale. Raw mean-pooled
    # SimCLR embeddings are not: sampling then converges toward N(0,I) at a
    # magnitude nothing like the real latents, and every two-sample test rejects
    # for a trivial scale reason rather than a distributional one. Statistics come
    # from the TRAINING split of each side only, so held-out post slides do not
    # inform the transform used to score them.
    mu_pre = z_pre[pre_idx].mean(0)
    sd_pre = z_pre[pre_idx].std(0) + 1e-6
    mu_post = z_post_pool[post_idx].mean(0)
    sd_post = z_post_pool[post_idx].std(0) + 1e-6
    z_pre = ((z_pre - mu_pre) / sd_pre).astype(np.float32)
    z_post_pool = ((z_post_pool - mu_post) / sd_post).astype(np.float32)

    # ---- in-fold dimensionality reduction ----------------------------------
    # Measured (scripts/analysis/ddpm_capacity_probe.py): with ~93 post slides the
    # denoiser cannot fit a 512-d density at all -- loss stalls at 1.0 (= emitting
    # zero) or diverges to 4.5. At 16-64 dims it reaches ~0.51-0.53, matching the
    # 0.525 an N(0,I) control attains, i.e. it is finally learning. Estimating a
    # 512-d density from ~100 samples is the binding statistical problem, so both
    # sides are projected into a low-dimensional subspace.
    #
    # Two separate PCAs, each fit on its own side's TRAINING split only: the pre
    # and post cohorts are different patients from different institutions and do
    # not share a basis. Held-out rows are only ever transformed, never fitted.
    pca_pre = pca_post = None
    if subspace_dim is not None and subspace_dim < z_pre.shape[1]:
        from sklearn.decomposition import PCA
        pca_pre = PCA(n_components=subspace_dim, random_state=seed).fit(z_pre[pre_idx])
        pca_post = PCA(n_components=subspace_dim,
                       random_state=seed).fit(z_post_pool[post_idx])
        z_pre = pca_pre.transform(z_pre).astype(np.float32)
        z_post_pool = pca_post.transform(z_post_pool).astype(np.float32)
        # re-standardise: PCA components carry very unequal variance, and the
        # cosine schedule assumes unit scale per dimension
        s_pre = z_pre[pre_idx].std(0) + 1e-6
        s_post = z_post_pool[post_idx].std(0) + 1e-6
        z_pre = (z_pre / s_pre).astype(np.float32)
        z_post_pool = (z_post_pool / s_post).astype(np.float32)
        if dim != subspace_dim:
            raise ValueError(f"dim={dim} must equal subspace_dim={subspace_dim}")
        print(f"[unpaired fold{val_fold}] subspace {subspace_dim}-d "
              f"(pre EVR {pca_pre.explained_variance_ratio_.sum():.3f}, "
              f"post EVR {pca_post.explained_variance_ratio_.sum():.3f})", flush=True)

    zc_all = torch.from_numpy(z_pre).to(device)
    dfeat_all = torch.from_numpy(drug_feats)
    dfeat_all = (dfeat_all.long() if drug_variant == "onehot"
                 else dfeat_all.float()).to(device)
    zpost_all = torch.from_numpy(z_post_pool).to(device)

    sch = NoiseSchedule(T=T, schedule=schedule, s=cosine_s)
    diffusion = GaussianDiffusion(sch, lambda_vlb=lambda_vlb).to(device)
    mod = DiffusionModule(dim, cond_dim, drug_variant, num_drugs, n_bits,
                          n_blocks, width, conditioning, device,
                          single_timepoint=False)   # paired design: z_pre in context
    opt = torch.optim.AdamW(mod.parameters(), lr=lr)

    start_epoch = 0
    state = ckpt.maybe_resume(ckpt_path, mod, opt, map_location=device)
    if state is not None:
        start_epoch = state.get("epoch", 0) + 1

    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    n = len(pre_idx)
    post_bs = max(2, int(round(batch_size * post_batch_mult)))
    mod.train()
    last_cost = float("nan")

    for epoch in range(start_epoch, epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            b_pre = pre_idx[perm[i:i + batch_size]]
            take = min(post_bs, len(post_idx))
            b_post = rng.choice(post_idx, size=take, replace=False)

            # OT pseudo-pairs, recomputed every batch so the coupling tracks the
            # current sample rather than being frozen once at start-up
            ix, iy = ot_coupling(
                z_pre[b_pre], z_post_pool[b_post],
                groups_x=None if gp is None else gp[b_pre],
                groups_y=None if gq is None else gq[b_post],
                mode=coupling, seed=int(rng.integers(1 << 30)))
            if len(ix) == 0:
                continue     # no shared stratum in this batch
            last_cost = coupling_cost(z_pre[b_pre], z_post_pool[b_post], ix, iy)

            src = torch.as_tensor(b_pre[ix], device=device)
            dst = torch.as_tensor(b_post[iy], device=device)
            zb = zpost_all[dst]                 # target z_0 = matched z_post
            cb = zc_all[src]                    # condition on the real z_pre
            db = dfeat_all[src]

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                c = mod.context(cb, mod.drug(db))
                if p_uncond > 0:
                    drop = torch.rand(c.shape[0], device=device) < p_uncond
                    c = c.clone()
                    c[drop] = 0.0
                out = diffusion.loss(mod.denoiser, zb, c)
                loss = out["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if epoch % log_every == 0:
            print(f"[unpaired fold{val_fold}] epoch {epoch} "
                  f"total {loss.item():.4f} simple {out['simple'].item():.4f} "
                  f"ot_cost {last_cost:.3f}", flush=True)
        ckpt.save(ckpt_path, mod, opt, step=0, epoch=epoch,
                  extra={"val_fold": val_fold, "unpaired": True,
                         "coupling": coupling,
                         # inference must reproduce the exact in-fold transform
                         "mu_pre": mu_pre, "sd_pre": sd_pre,
                         "mu_post": mu_post, "sd_post": sd_post,
                         "pca_pre": pca_pre, "pca_post": pca_post,
                         "s_pre": None if pca_pre is None else s_pre,
                         "s_post": None if pca_post is None else s_post})
    return mod, diffusion


def train_unpaired_cv(z_pre, drug_feats, z_post_pool, folds, post_folds,
                      ckpt_dir: str, n_folds: int = 5, **kw):
    """Train all folds; write ``unpaired_fold{k}.pt``. Returns the paths."""
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(n_folds):
        p = str(Path(ckpt_dir) / f"unpaired_fold{k}.pt")
        train_unpaired_fold(z_pre, drug_feats, z_post_pool, folds, post_folds,
                            k, p, **kw)
        paths.append(p)
    return paths
