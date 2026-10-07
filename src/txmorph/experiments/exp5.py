"""Experiment 5 - Ablations (CLAUDE.md S11-Exp5).

Each variant re-trains/re-infers with one component swapped, then re-runs the Exp-1
metrics and reports the delta (with CIs) vs the full model:
  - no drug conditioning
  - diffusion -> deterministic autoencoder
  - SMILES -> one-hot drug encoder
  - N=10 vs 100 samples
  - SimCLR -> random encoder
  - cross-attention -> FiLM/adaLN conditioning

This module provides the delta-tabulation core; each variant's records are produced
by re-running the training/inference pipeline with the corresponding config
override (wired in 09_run_experiments.py).
"""
from __future__ import annotations

import numpy as np

from .exp1 import evaluate_probabilities

# config overrides that define each ablation variant (applied on top of the full model)
ABLATION_VARIANTS = {
    "no_drug": {"drop_drug_conditioning": True},
    "deterministic_ae": {"model": "autoencoder"},
    "onehot_drug": {"diffusion.drug_variant": "onehot"},
    "n_samples_10": {"diffusion.n_samples": 10},
    "random_encoder": {"encoder.random_init": True},
    "crossattn": {"diffusion.conditioning": "crossattn"},
}


def ablation_delta(full_records, variant_records, metric: str = "auprc",
                   n_boot: int = 1000, seed: int = 0):
    """Delta of ``metric`` (variant - full) with each model's bootstrap CI."""
    def _probs(recs):
        return (np.array([r["p_pcr"] for r in recs]),
                np.array([r["pcr"] for r in recs]))

    pf, yf = _probs(full_records)
    pv, yv = _probs(variant_records)
    full = evaluate_probabilities(pf, yf, n_boot=n_boot, seed=seed)[metric]
    var = evaluate_probabilities(pv, yv, n_boot=n_boot, seed=seed)[metric]
    return {"metric": metric, "full": full, "variant": var,
            "delta": var[0] - full[0]}


def run_exp5(full_records, variant_records: dict, metric: str = "auprc",
             n_boot: int = 1000, seed: int = 0):
    """variant_records: {variant_name: records}. Returns per-variant metric deltas."""
    return {name: ablation_delta(full_records, recs, metric, n_boot, seed)
            for name, recs in variant_records.items()}


# diffusion-based variants runnable from paired.parquet alone (retrain and/or
# re-infer here); `retrain` marks whether the model must be re-trained per fold.
_RUNNABLE = {
    "no_drug": {"retrain": True},        # drug signal removed (constant drug token)
    "onehot_drug": {"retrain": True},    # swap SMILES -> one-hot drug encoder
    "crossattn": {"retrain": True},      # FiLM -> cross-attention conditioning
    "n_samples_10": {"retrain": False},  # re-infer full model with N=10 (paired only)
    # Single-timepoint replacements for the ablations the paired design assumed.
    # P(pCR) is a likelihood ratio here, so the Monte-Carlo budget -- not the number
    # of drawn samples -- is what controls estimator variance; `n_mc_10` tests the
    # same thing `n_samples_10` was meant to.
    "n_mc_10": {"retrain": False},
    # Slide pooling: the MIL's attention is fit on a subtype auxiliary task. At the
    # Week-12 gate it separated subtype at 0.973 but pCR at only 0.602, where mean
    # pooling of the same bags reached 0.691. This variant quantifies that cost end
    # to end rather than at the gate.
    "mil_pooling": {"retrain": True},
}
# `deterministic_ae` is handled specially (AE path, not diffusion) and runs from
# paired.parquet alone. `random_encoder` runs only when a random-init-encoder paired
# table is supplied via paths.paired_random.
_NEEDS_ARTIFACT = {
    "random_encoder": "needs embeddings from a random-init encoder "
                      "(a separate paired table); set paths.paired_random to enable.",
}


def _variant_setup(name, df, dc, drc, base_drug_variant):
    """Return (drug_feats, drug_variant, num_drugs, n_bits, conditioning, n_samples,
    retrain) for a runnable variant, or raise KeyError if unknown."""
    from ..drug.features import build_drug_features
    n_bits = drc.get("morgan_bits", 2048)
    conditioning = dc.get("conditioning", "film")
    n_samples = dc.get("n_samples", 100)

    if name == "no_drug":
        feats, meta = build_drug_features(
            df, variant=base_drug_variant, radius=drc.get("morgan_radius", 2),
            n_bits=n_bits, combination=drc.get("combination", "bitwise_or"))
        feats = np.zeros_like(np.asarray(feats))          # constant null drug
        num_drugs = max(1, meta.get("num_drugs", 1))
        return feats, base_drug_variant, num_drugs, n_bits, conditioning, n_samples, True
    if name == "onehot_drug":
        feats, meta = build_drug_features(df, variant="onehot")
        return feats, "onehot", meta["num_drugs"], n_bits, conditioning, n_samples, True
    if name == "crossattn":
        feats, meta = build_drug_features(
            df, variant=base_drug_variant, radius=drc.get("morgan_radius", 2),
            n_bits=n_bits, combination=drc.get("combination", "bitwise_or"))
        num_drugs = meta.get("num_drugs", 16)
        return feats, base_drug_variant, num_drugs, n_bits, "crossattn", n_samples, True
    if name in ("n_samples_10", "n_mc_10", "mil_pooling"):
        feats, meta = build_drug_features(
            df, variant=base_drug_variant, radius=drc.get("morgan_radius", 2),
            n_bits=n_bits, combination=drc.get("combination", "bitwise_or"))
        num_drugs = meta.get("num_drugs", 16)
        n = 10 if name == "n_samples_10" else n_samples
        retrain = name == "mil_pooling"      # different latents -> must refit
        return feats, base_drug_variant, num_drugs, n_bits, conditioning, n, retrain
    raise KeyError(name)


def _run_diffusion_variant(name, z_pre, z_post, feats, pcr, folds, *, base_ckpt_dir,
                           work_dir, dc, cond, n_samples, retrain, dvar, num_drugs,
                           n_bits, tc, device, force, seed, single_timepoint=False,
                           n_mc: int = 50):
    """Retrain (if needed) + re-infer one diffusion-based variant; return records.

    ``single_timepoint`` must match the mode the full model was trained in, or the
    ablation deltas compare models with different targets and conditioning.
    """
    from pathlib import Path
    from ..training.diffusion import train_diffusion_cv
    from ..inference.run import run_inference

    ck_dir = base_ckpt_dir if not retrain else str(Path(work_dir) / "ckpt" / name)
    out_dir = str(Path(work_dir) / "samples" / name)
    if retrain:
        print(f"[exp5] retraining variant {name} (cond={cond}, drug={dvar})")
        Path(ck_dir).mkdir(parents=True, exist_ok=True)
        train_kw = {}
        if single_timepoint:                    # target is z_pre; pCR is the condition
            train_kw["pcr"] = pcr
        train_diffusion_cv(
            z_pre, z_post, feats, folds, ck_dir, n_folds=dc.get("n_folds", 5),
            dim=dc.get("dim", 512), cond_dim=dc.get("cond_dim", 512),
            T=dc.get("T", 1000), schedule=dc.get("schedule", "cosine"),
            cosine_s=dc.get("cosine_s", 0.008), lambda_vlb=dc.get("lambda_vlb", 1e-3),
            conditioning=cond, n_blocks=dc.get("n_blocks", 3),
            width=dc.get("width", 512), drug_variant=dvar, num_drugs=num_drugs,
            n_bits=n_bits, epochs=tc.get("epochs", 500), lr=tc.get("lr", 2e-4),
            batch_size=tc.get("batch_size", 64), p_uncond=tc.get("p_uncond", 0.1),
            device=device, seed=seed, single_timepoint=single_timepoint, **train_kw)
    return run_inference(
        z_pre, z_post, feats, pcr, folds, ck_dir, out_dir,
        n_folds=dc.get("n_folds", 5), n_samples=n_samples, dim=dc.get("dim", 512),
        cond_dim=dc.get("cond_dim", 512), T=dc.get("T", 1000),
        schedule=dc.get("schedule", "cosine"), cosine_s=dc.get("cosine_s", 0.008),
        lambda_vlb=dc.get("lambda_vlb", 1e-3), conditioning=cond,
        n_blocks=dc.get("n_blocks", 3), width=dc.get("width", 512),
        drug_variant=dvar, num_drugs=num_drugs, n_bits=n_bits,
        clip_x0=dc.get("clip_x0"), device=device, force=force,
        single_timepoint=single_timepoint, n_mc=n_mc)


def _run_deterministic_ae(df, z_pre, z_post, *, work_dir, dc, drc, tc,
                          base_drug_variant, device, force, seed):
    """Train a deterministic conditional AE + infer deterministically; return records."""
    from pathlib import Path
    from ..drug.features import build_drug_features
    from ..training.ae import train_ae_cv
    from ..inference.ae_run import run_ae_inference

    pcr = df["pcr"].to_numpy()
    folds = df["fold"].to_numpy()
    n_bits = drc.get("morgan_bits", 2048)
    feats, meta = build_drug_features(
        df, variant=base_drug_variant, radius=drc.get("morgan_radius", 2),
        n_bits=n_bits, combination=drc.get("combination", "bitwise_or"))
    num_drugs = meta.get("num_drugs", 16)
    ck_dir = str(Path(work_dir) / "ckpt" / "deterministic_ae")
    out_dir = str(Path(work_dir) / "samples" / "deterministic_ae")
    Path(ck_dir).mkdir(parents=True, exist_ok=True)
    print("[exp5] training deterministic-AE variant (point predictor f(c)->z_post)")
    train_ae_cv(
        z_pre, z_post, feats, folds, ck_dir, n_folds=dc.get("n_folds", 5),
        dim=dc.get("dim", 512), cond_dim=dc.get("cond_dim", 512),
        conditioning=dc.get("conditioning", "film"), n_blocks=dc.get("n_blocks", 3),
        width=dc.get("width", 512), drug_variant=base_drug_variant, num_drugs=num_drugs,
        n_bits=n_bits, epochs=tc.get("epochs", 500), lr=tc.get("lr", 2e-4),
        batch_size=tc.get("batch_size", 64), device=device, seed=seed)
    return run_ae_inference(
        z_pre, z_post, feats, pcr, folds, ck_dir, out_dir,
        n_folds=dc.get("n_folds", 5), n_samples=dc.get("n_samples", 100),
        dim=dc.get("dim", 512), cond_dim=dc.get("cond_dim", 512),
        drug_variant=base_drug_variant, num_drugs=num_drugs, n_bits=n_bits,
        n_blocks=dc.get("n_blocks", 3), width=dc.get("width", 512),
        device=device, force=force)


def _run_alt_table_variant(name, paired_path, *, work_dir, dc, drc, tc,
                           base_drug_variant, device, force, seed):
    """Retrain + re-infer on an alternative paired table; return records.

    Used by variants that change the *latents* rather than the model:
    ``random_encoder`` (untrained encoder) and ``mil_pooling`` (attention pooling
    instead of mean). Each loads its own (z_pre, z_post, folds, pcr, drug) from its
    own table, so nothing from the main pipeline's latents leaks in. Folds are
    seeded from patient_id, so they match the full model's and the comparison stays
    patient-aligned.
    """
    from ..data.paired_dataset import load_paired
    from ..drug.features import build_drug_features

    import numpy as np

    rdf, rz_pre, rz_post = load_paired(paired_path)
    folds = rdf["fold"].to_numpy()
    pcr = rdf["pcr"].to_numpy()
    # Detected from this table, not inherited: an alternative-latent run is a wholly
    # separate pipeline and could have been built in either mode.
    single_timepoint = bool(np.all(np.isnan(rz_post)))
    n_bits = drc.get("morgan_bits", 2048)
    feats, meta = build_drug_features(
        rdf, variant=base_drug_variant, radius=drc.get("morgan_radius", 2),
        n_bits=n_bits, combination=drc.get("combination", "bitwise_or"))
    num_drugs = meta.get("num_drugs", 16)
    cond = dc.get("conditioning", "film")
    print(f"[exp5] retraining diffusion on {name} latents ({paired_path})")
    return _run_diffusion_variant(
        name, rz_pre, rz_post, feats, pcr, folds,
        base_ckpt_dir="", work_dir=work_dir, dc=dc, cond=cond,
        n_samples=dc.get("n_samples", 100), retrain=True, dvar=base_drug_variant,
        num_drugs=num_drugs, n_bits=n_bits, tc=tc, device=device, force=force, seed=seed,
        single_timepoint=single_timepoint)


def run_exp5_orchestrated(df, z_pre, z_post, full_records, *, base_ckpt_dir, work_dir,
                          dc, drc, tc, paired_random: str | None = None,
                          paired_mil: str | None = None,
                          variants=None, metric: str = "auprc", device: str = "cpu",
                          force: bool = False, seed: int = 0):
    """Retrain/re-infer each ablation variant, then tabulate metric deltas vs full.

    ``dc`` diffusion config, ``drc`` drug config, ``tc`` training config (dicts).
    Diffusion variants (``_RUNNABLE``) and ``deterministic_ae`` run from
    paired.parquet alone; ``random_encoder`` needs ``paired_random``. Variants whose
    prerequisites are absent are skipped and reported under ``"skipped"``.
    Returns ``{"deltas": {...}, "skipped": {...}}``.
    """
    import numpy as np

    if variants is None:
        variants = list(_RUNNABLE) + ["deterministic_ae", "random_encoder"]
    base_drug_variant = drc.get("variant", "onehot")
    pcr = df["pcr"].to_numpy()
    folds = df["fold"].to_numpy()
    # Ablations must be trained and scored in the same mode as the full model they
    # are differenced against.
    single_timepoint = bool(np.all(np.isnan(z_post)))

    variant_records, skipped = {}, {}
    for name in variants:
        try:
            if name == "deterministic_ae":
                if single_timepoint:
                    skipped[name] = ("the deterministic AE is a point predictor of "
                                     "z_post, which single-timepoint cohorts do not "
                                     "have; the diffusion-vs-deterministic contrast "
                                     "needs the paired design.")
                    print(f"[exp5] SKIP {name}: {skipped[name]}")
                    continue
                recs = _run_deterministic_ae(
                    df, z_pre, z_post, work_dir=work_dir, dc=dc, drc=drc, tc=tc,
                    base_drug_variant=base_drug_variant, device=device,
                    force=force, seed=seed)
            elif name == "random_encoder":
                if not paired_random:
                    skipped[name] = _NEEDS_ARTIFACT[name]
                    print(f"[exp5] SKIP {name}: {_NEEDS_ARTIFACT[name]}")
                    continue
                recs = _run_alt_table_variant(
                    "random_encoder", paired_random, work_dir=work_dir, dc=dc,
                    drc=drc, tc=tc, base_drug_variant=base_drug_variant,
                    device=device, force=force, seed=seed)
            elif name in _RUNNABLE:
                if name == "onehot_drug" and base_drug_variant == "onehot":
                    # The ablation swaps SMILES -> one-hot. If the full model is
                    # already one-hot (IMPRESS ships no SMILES), this retrains the
                    # same model and reports a delta of exactly 0 -- an artifact
                    # that reads as a real null.
                    skipped[name] = ("the full model already uses the one-hot drug "
                                     "encoder (no SMILES in this cohort), so this "
                                     "variant is identical to it.")
                    print(f"[exp5] SKIP {name}: {skipped[name]}")
                    continue
                if name == "n_samples_10" and single_timepoint:
                    # Single-timepoint P(pCR) is a likelihood ratio, not a count over
                    # sampled z_post, so n_samples never enters it. `n_mc_10` is the
                    # replacement that tests the same estimator-variance question.
                    skipped[name] = ("n_samples does not affect single-timepoint "
                                     "P(pCR) (likelihood ratio, not sampling); "
                                     "superseded by n_mc_10 in this mode.")
                    print(f"[exp5] SKIP {name}: {skipped[name]}")
                    continue
                if name == "n_mc_10" and not single_timepoint:
                    skipped[name] = ("n_mc only applies to the single-timepoint "
                                     "likelihood ratio; the paired design uses "
                                     "n_samples_10 instead.")
                    print(f"[exp5] SKIP {name}: {skipped[name]}")
                    continue
                if name == "mil_pooling":
                    if not paired_mil:
                        skipped[name] = ("needs a MIL-pooled paired table; set "
                                         "paths.paired_mil (build with script 05 and "
                                         "aggregator.pooling=mil).")
                        print(f"[exp5] SKIP {name}: {skipped[name]}")
                        continue
                    recs = _run_alt_table_variant(
                        "mil_pooling", paired_mil, work_dir=work_dir, dc=dc, drc=drc,
                        tc=tc, base_drug_variant=base_drug_variant, device=device,
                        force=force, seed=seed)
                    variant_records[name] = recs
                    continue
                feats, dvar, num_drugs, n_bits, cond, n_samples, retrain = _variant_setup(
                    name, df, dc, drc, base_drug_variant)
                recs = _run_diffusion_variant(
                    name, z_pre, z_post, feats, pcr, folds,
                    base_ckpt_dir=base_ckpt_dir, work_dir=work_dir, dc=dc, cond=cond,
                    n_samples=n_samples, retrain=retrain, dvar=dvar, num_drugs=num_drugs,
                    n_bits=n_bits, tc=tc, device=device, force=force, seed=seed,
                    single_timepoint=single_timepoint,
                    n_mc=10 if name == "n_mc_10" else 50)
            else:
                skipped[name] = f"unknown variant {name!r}"
                continue
        except FileNotFoundError as e:
            skipped[name] = f"missing artifact: {e}"
            print(f"[exp5] SKIP {name}: {e}")
            continue
        variant_records[name] = recs

    return {"deltas": run_exp5(full_records, variant_records, metric, seed=seed),
            "skipped": skipped}


def run(cfg):
    raise NotImplementedError(
        "Exp5 is orchestrated by run_exp5_orchestrated(); call it from "
        "09_run_experiments.py with the full-model records + diffusion/drug/train "
        "configs."
    )
