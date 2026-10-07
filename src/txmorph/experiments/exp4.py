"""Experiment 4 - In-silico drug comparison (CLAUDE.md S11-Exp4).

Per patient, re-infer under each ISPY2 arm (vary d_drug, hold z_pre); population
mean P(pCR) per arm -> rank -> Spearman rho vs the observed trial ranking. Bootstrap
CI over patients. Report as SUGGESTIVE (n~6 arms, low power); state the
counterfactual caveat (only the administered arm is observed).
"""
from __future__ import annotations

import numpy as np


def per_arm_pcr_rates(sample_fn, z_pre, arm_drug_feats, zone, dim: int = 512):
    """Mean P(pCR) per arm across patients.

    sample_fn(z_pre_row, drug_feat) -> (N, dim) samples for one patient/arm.
    arm_drug_feats: {arm_name: drug_feat}. zone: fitted PCRZone.
    Returns {arm: {"rate": float, "per_patient": np.ndarray}}.
    """
    import torch
    z_pre = np.asarray(z_pre, dtype=np.float32)
    out = {}
    for arm, dfeat in arm_drug_feats.items():
        per_pt = []
        for i in range(len(z_pre)):
            samples = sample_fn(z_pre[i], dfeat)
            if not torch.is_tensor(samples):
                samples = torch.as_tensor(samples)
            per_pt.append(zone.in_zone(samples).float().mean().item())
        per_pt = np.asarray(per_pt)
        out[arm] = {"rate": float(per_pt.mean()), "per_patient": per_pt}
    return out


def compare_to_observed(pred_rates, observed_rates, n_boot: int = 1000, seed: int = 0):
    """Spearman rho between predicted and observed per-arm pCR rates + bootstrap CI.

    pred_rates/observed_rates: dict {arm: rate}. Bootstrap resamples arms.
    """
    from scipy import stats
    arms = [a for a in pred_rates if a in observed_rates]
    pred = np.array([pred_rates[a] for a in arms])
    obs = np.array([observed_rates[a] for a in arms])
    rho, p = stats.spearmanr(pred, obs)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arms), size=len(arms))
        if len(np.unique(idx)) < 2:
            continue
        r, _ = stats.spearmanr(pred[idx], obs[idx])
        if not np.isnan(r):
            boots.append(r)
    ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))) if boots \
        else (float("nan"), float("nan"))
    return {"spearman_rho": float(rho), "p": float(p), "ci": ci, "arms": arms,
            "pred": pred.tolist(), "observed": obs.tolist()}


def run_exp4(z_pre, z_post, pcr, folds, arm_drug_feats, observed_rates, ckpt_dir,
             *, n_folds: int = 5, n_samples: int = 100, dim: int = 512,
             cond_dim: int = 512, T: int = 1000, schedule: str = "cosine",
             cosine_s: float = 0.008, lambda_vlb: float = 1e-3,
             conditioning: str = "film", n_blocks: int = 3, width: int = 512,
             drug_variant: str = "onehot", num_drugs: int = 16, n_bits: int = 2048,
             clip_x0: float | None = None, device: str = "cpu", seed: int = 0):
    """In-silico arm comparison via fold-correct counterfactual re-inference.

    For every patient we hold ``z_pre`` fixed and re-infer under each arm's drug
    feature, using *that patient's held-out fold* model and pCR zone (no leakage).
    The population-mean P(pCR) per arm is ranked against ``observed_rates`` by
    Spearman rho (bootstrap CI over arms).

    ``arm_drug_feats``: ``{arm: drug_feat}`` (scalar id for onehot, (n_bits,) for
    smiles). ``observed_rates``: ``{arm: observed_pcr_rate}``. Returns the
    ``compare_to_observed`` dict augmented with per-arm predicted rates.
    """
    from ..diffusion.schedule import NoiseSchedule
    from ..diffusion.gaussian_diffusion import GaussianDiffusion
    from ..inference.run import load_fold_modules, sample_counterfactual

    z_pre = np.asarray(z_pre, dtype=np.float32)
    folds = np.asarray(folds)

    sch = NoiseSchedule(T=T, schedule=schedule, s=cosine_s)
    diffusion = GaussianDiffusion(sch, lambda_vlb=lambda_vlb).to(device)
    modules, zones = load_fold_modules(
        ckpt_dir, folds, z_post, pcr, n_folds, dim=dim, cond_dim=cond_dim,
        drug_variant=drug_variant, num_drugs=num_drugs, n_bits=n_bits,
        n_blocks=n_blocks, width=width, conditioning=conditioning, device=device)

    pred_rates, per_arm_patient = {}, {}
    for arm, dfeat in arm_drug_feats.items():
        per_pt = np.empty(len(z_pre), dtype=np.float32)
        for i in range(len(z_pre)):
            k = int(folds[i])
            samples = sample_counterfactual(
                diffusion, modules[k], z_pre[i], dfeat, drug_variant,
                n_samples=n_samples, dim=dim, device=device, clip_x0=clip_x0)
            per_pt[i] = zones[k].in_zone(samples).float().mean().item()
        pred_rates[arm] = float(per_pt.mean())
        per_arm_patient[arm] = per_pt

    result = compare_to_observed(pred_rates, observed_rates, seed=seed)
    result["pred_rates"] = pred_rates
    result["per_patient"] = {a: v.tolist() for a, v in per_arm_patient.items()}
    return result


def run(cfg):
    raise NotImplementedError(
        "Exp4 is orchestrated by run_exp4(); call it from 09_run_experiments.py "
        "(arms + observed rates come from configs/experiment/exp4.yaml)."
    )
