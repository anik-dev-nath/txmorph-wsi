"""Per-patient inference: sample N=100 z_post, derive the four biomarkers
(CLAUDE.md S6.6, S8 step 9).

**Paired mode**: for each held-out (test-fold) patient, reload that fold's
diffusion checkpoint, build context from z_pre + drug, draw N samples, derive
P(pCR), sigma2_tx, z_post_bar, residual r. pCR-zone logistic fit on the fold's
TRAINING patients only.

**Single-timepoint mode** (``single_timepoint=True``): no z_post is available.
P(pCR) is computed via likelihood ratio — the denoising NLL under the pCR=1
context vs pCR=0 context, combined via Bayes' rule. sigma2_tx is the uncertainty
in the NLL estimate. Residual is not computed.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from ..diffusion.schedule import NoiseSchedule
from ..diffusion.gaussian_diffusion import GaussianDiffusion
from ..diffusion.sampler import sample_patient
from ..training.diffusion import DiffusionModule
from ..biomarkers.derive import derive_biomarkers
from ..biomarkers.pcr_zone import fit_pcr_zone
from ..utils import ckpt
from ..utils.cv import train_val_masks


def load_fold_module(ckpt_path: str, dim, cond_dim, drug_variant, num_drugs,
                     n_bits, n_blocks, width, conditioning, device,
                     single_timepoint: bool = False):
    """Load one fold's trained DiffusionModule from ``ckpt_path`` (eval mode)."""
    mod = DiffusionModule(dim, cond_dim, drug_variant, num_drugs, n_bits,
                          n_blocks, width, conditioning, device,
                          single_timepoint=single_timepoint)
    ckpt.load(ckpt_path, mod, map_location=device, restore_rng=False)
    mod.eval()
    return mod


_load_fold_module = load_fold_module  # backwards-compatible alias


def load_fold_modules(ckpt_dir: str, folds, z_post, pcr, n_folds: int = 5, *,
                      dim: int = 512, cond_dim: int = 512, drug_variant: str = "onehot",
                      num_drugs: int = 16, n_bits: int = 2048, n_blocks: int = 3,
                      width: int = 512, conditioning: str = "film", device: str = "cpu",
                      single_timepoint: bool = False, z_pre=None):
    """Load every fold's module and fit each fold's train-only pCR zone.

    Returns ``(modules, zones)`` lists indexed by fold ``k``. The zone for fold
    ``k`` is fit on that fold's TRAINING patients only (no leakage).

    In single-timepoint mode, the zone is fit on z_pre (not z_post).
    """
    z_fit = np.asarray(z_pre if single_timepoint else z_post, dtype=np.float32)
    pcr = np.asarray(pcr).astype(int)
    folds = np.asarray(folds)
    modules, zones = [], []
    for k in range(n_folds):
        train_mask, _ = train_val_masks(folds, k)
        zones.append(fit_pcr_zone(z_fit[train_mask], pcr[train_mask], device=device))
        ckpt_path = str(Path(ckpt_dir) / f"diffusion_fold{k}.pt")
        modules.append(load_fold_module(
            ckpt_path, dim, cond_dim, drug_variant, num_drugs, n_bits,
            n_blocks, width, conditioning, device,
            single_timepoint=single_timepoint))
    return modules, zones


def sample_counterfactual(diffusion, mod, z_pre_row, drug_feat, drug_variant: str,
                          n_samples: int = 100, dim: int = 512, device: str = "cpu",
                          clip_x0: float | None = None):
    """Draw ``n_samples`` z_post for one patient's ``z_pre`` under an arbitrary drug.

    ``drug_feat`` is a scalar drug id (onehot) or an ``(n_bits,)`` fingerprint
    (smiles). Used for counterfactual re-inference in Exp-4 (vary the drug, hold
    ``z_pre``). Returns a ``(n_samples, dim)`` tensor.
    """
    import torch
    zc = torch.from_numpy(np.asarray(z_pre_row, dtype=np.float32)[None]).to(device)
    db = torch.as_tensor(np.asarray(drug_feat)[None]).to(device)
    db = db.long() if drug_variant == "onehot" else db.float()
    with torch.no_grad():
        d_drug = mod.drug(db)
        c = mod.context(zc, d_drug)
    return sample_patient(diffusion, mod.denoiser, c, n=n_samples, dim=dim,
                          device=device, clip_x0=clip_x0)


def single_timepoint_scores(diffusion, mod, z_rows, drug_rows, drug_variant: str,
                            device, n_mc: int = 50, n_reps: int = 5, seed: int = 0):
    """Batched log-likelihood-ratio score for pCR, one row per patient.

    Scores ``z_pre`` under the pCR=1 and pCR=0 contexts and returns their NLL gap
    ``llr = nll(pCR=0) - nll(pCR=1)`` (positive favours pCR). Both hypotheses share
    timesteps and noise via :meth:`GaussianDiffusion.nll_contrast` — differencing
    two independently-sampled NLLs buries the signal under MC noise and yields a
    chance-level biomarker.

    ``llr`` is an unnormalised score, not a probability: the eps-MSE gap has no
    natural scale, so a softmax over it would pin every patient near 0.5 and wreck
    the calibration result that is this project's headline. Callers map ``llr`` to
    a probability with a logistic fit on the TRAINING fold only (see
    ``run_inference``). ``mc_se`` is the across-replicate standard error of ``llr``,
    a diagnostic: it must be small relative to the spread of ``llr`` across
    patients, otherwise the estimate is noise.

    Returns ``(llr, mc_se)`` as float arrays of length ``len(z_rows)``.
    """
    import torch

    zc = torch.from_numpy(np.asarray(z_rows, dtype=np.float32)).to(device)
    db = torch.from_numpy(np.asarray(drug_rows)).to(device)
    db = db.long() if drug_variant == "onehot" else db.float()
    n = zc.shape[0]

    with torch.no_grad():
        d_drug = mod.drug(db)
        c1 = mod.context(zc, d_drug, torch.ones(n, 1, device=device))
        c0 = mod.context(zc, d_drug, torch.zeros(n, 1, device=device))
        gen = torch.Generator(device=device)
        reps = []
        for r in range(n_reps):
            gen.manual_seed(seed * 7919 + r)
            nll = diffusion.nll_contrast(mod.denoiser, zc, [c1, c0],
                                         n_mc=n_mc, generator=gen)
            reps.append(nll[1] - nll[0])
        stacked = torch.stack(reps)

    llr = stacked.mean(dim=0).float().cpu().numpy()
    mc_se = (stacked.std(dim=0) / np.sqrt(n_reps)).float().cpu().numpy()
    return llr, mc_se


def _fit_llr_calibrator(llr_train, pcr_train):
    """1-D logistic mapping the LLR score to a probability (train fold only).

    The LLR is **standardised on the training fold before fitting**, and that is
    load-bearing rather than cosmetic. The raw score is an eps-MSE gap of order
    1e-4; sklearn's logistic is L2-penalised by default, so on that scale the
    penalty crushes the coefficient to ~0 and the fit degenerates to an
    intercept, returning the training base rate for every patient. That failure
    is silent and looks like a calibrated-but-uninformative model: P(pCR) came
    out constant within each fold (std 0.0, AUROC 0.517) even though the
    underlying LLR was a stable per-patient quantity
    (corr(n_mc=50, n_mc=400) = 0.96).

    Scaler statistics come from the training fold only, so no test information
    reaches the mapping.
    """
    from sklearn.linear_model import LogisticRegression

    x = np.asarray(llr_train, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(pcr_train).astype(int)
    if len(np.unique(y)) < 2:            # degenerate fold: fall back to base rate
        rate = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        return lambda v: np.full(len(np.asarray(v)), rate, dtype=float)

    mu = float(x.mean())
    sd = float(x.std())
    if not np.isfinite(sd) or sd < 1e-12:   # genuinely constant LLR: nothing to map
        rate = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        return lambda v: np.full(len(np.asarray(v)), rate, dtype=float)

    lr = LogisticRegression(max_iter=1000)
    lr.fit((x - mu) / sd, y)
    return lambda v: lr.predict_proba(
        (np.asarray(v, dtype=np.float64).reshape(-1, 1) - mu) / sd)[:, 1]


def run_inference(z_pre, z_post, drug_feats, pcr, folds, ckpt_dir: str,
                  out_dir: str, n_folds: int = 5, n_samples: int = 100,
                  dim: int = 512, cond_dim: int = 512, T: int = 1000,
                  schedule: str = "cosine", cosine_s: float = 0.008,
                  lambda_vlb: float = 1e-3, conditioning: str = "film",
                  n_blocks: int = 3, width: int = 512, drug_variant: str = "onehot",
                  num_drugs: int = 16, n_bits: int = 2048, clip_x0: float | None = None,
                  device: str = "cpu", force: bool = False,
                  single_timepoint: bool = False, n_mc: int = 50, n_reps: int = 5):
    """Run held-out inference over all folds. Returns a dict of per-patient records.

    Each record: {index, fold, p_pcr, sigma2_tx, z_post_bar, residual, pcr}.
    Samples written to ``out_dir/{i}.npy`` (paired mode only).

    When ``single_timepoint=True``, P(pCR) is computed via likelihood ratio
    instead of sampling + zone counting.
    """
    import torch

    z_pre = np.asarray(z_pre, dtype=np.float32)
    z_post = np.asarray(z_post, dtype=np.float32)
    drug_feats = np.asarray(drug_feats)
    pcr = np.asarray(pcr).astype(int)
    folds = np.asarray(folds)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sch = NoiseSchedule(T=T, schedule=schedule, s=cosine_s)
    diffusion = GaussianDiffusion(sch, lambda_vlb=lambda_vlb).to(device)

    records = []
    for k in range(n_folds):
        train_mask, val_mask = train_val_masks(folds, k)

        ckpt_path = str(Path(ckpt_dir) / f"diffusion_fold{k}.pt")
        mod = load_fold_module(ckpt_path, dim, cond_dim, drug_variant, num_drugs,
                               n_bits, n_blocks, width, conditioning, device,
                               single_timepoint=single_timepoint)

        if single_timepoint:
            # Score every patient in the fold, then map LLR -> probability with a
            # logistic fit on the TRAINING patients only (no leakage). Training
            # rows are scored purely to fit that calibrator; only val rows are
            # emitted as records.
            tr_idx = np.where(train_mask)[0]
            va_idx = np.where(val_mask)[0]
            llr_tr, _ = single_timepoint_scores(
                diffusion, mod, z_pre[tr_idx], drug_feats[tr_idx],
                drug_variant, device, n_mc=n_mc, n_reps=n_reps, seed=k)
            llr_va, mc_se_va = single_timepoint_scores(
                diffusion, mod, z_pre[va_idx], drug_feats[va_idx],
                drug_variant, device, n_mc=n_mc, n_reps=n_reps, seed=k)
            calibrate = _fit_llr_calibrator(llr_tr, pcr[tr_idx])
            p_va = calibrate(llr_va)

            for j, i in enumerate(va_idx):
                p = float(p_va[j])
                records.append({
                    "index": int(i), "fold": int(k), "pcr": int(pcr[i]),
                    "p_pcr": p,
                    # Predictive (Bernoulli) uncertainty of the calibrated call.
                    # NOT the paired design's sampled-morphology variance -- there
                    # is no z_post to spread over here; see CLAUDE.md S11 Exp 2.
                    "sigma2_tx": p * (1.0 - p),
                    "llr": float(llr_va[j]), "mc_se": float(mc_se_va[j]),
                    "z_post_bar": None, "residual": None,
                })
            continue

        # Paired mode: pCR zone fit on z_post from training fold
        zone = fit_pcr_zone(z_post[train_mask], pcr[train_mask], device=device)

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
                samples = sample_patient(diffusion, mod.denoiser, c,
                                         n=n_samples, dim=dim, device=device,
                                         clip_x0=clip_x0)
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
