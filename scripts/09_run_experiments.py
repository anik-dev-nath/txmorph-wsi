"""09_run_experiments.py - run experiments + emit figures/tables.

Runs the full-model held-out inference once (cached), then the selected
experiments:
  exp1 calibration | exp2 survival | exp3 resistance clusters |
  exp4 in-silico drug comparison (counterfactual re-inference per arm) |
  exp5 ablations (retrain/re-infer each variant)
Results + figures + tables are written under a new run dir. Select with
``--experiment {all,exp1,exp2,exp3,exp4,exp5}`` (default: all).
"""
import argparse
import json
from pathlib import Path

import _bootstrap


def _arm_features(arms, variant, meta, radius, n_bits, combination):
    """Build {arm: drug_feat} + {arm: observed_pcr} from the exp4 arm config.

    Arms whose drug can't be mapped into the trained model's vocabulary are
    skipped (reported by the caller). ``drug_feat`` is a vocab index (onehot) or
    a Morgan fingerprint vector (smiles).
    """
    from txmorph.drug.features import build_smiles_features
    feats, observed, missing = {}, {}, []
    vocab = meta.get("vocab", {})
    for name, spec in arms.items():
        observed[name] = float(spec["observed_pcr"])
        if variant == "onehot":
            raw = spec.get("drug_id")
            key = raw if raw in vocab else (
                type(next(iter(vocab)))(raw) if vocab else raw)
            if key not in vocab:
                missing.append(name)
                observed.pop(name)
                continue
            feats[name] = int(vocab[key])
        else:
            feats[name] = build_smiles_features(
                [spec["smiles"]], radius, n_bits, combination)[0]
    return feats, observed, missing


def main():
    ap = argparse.ArgumentParser(description="Run experiments + figures/tables.")
    _bootstrap.common_args(ap)
    ap.add_argument("--name", default="experiments")
    ap.add_argument("--experiment", default="all",
                    choices=["all", "exp1", "exp2", "exp3", "exp4", "exp5"])
    args = ap.parse_args()
    cfg = _bootstrap.load(args)

    from txmorph.data.paired_dataset import load_paired
    from txmorph.drug.features import build_drug_features
    from txmorph.inference.run import run_inference
    from txmorph.experiments import exp1, exp2, exp3, exp4, exp5
    from txmorph.viz import figures, tables
    from txmorph.utils.config import load_yaml, _find_config_dir
    from txmorph.utils.runlog import new_run_dir, log_run

    which = args.experiment
    want = {"exp1", "exp2", "exp3", "exp4", "exp5"} if which == "all" else {which}

    import numpy as np

    df, z_pre, z_post = load_paired(cfg.paths.paired)
    folds = df["fold"].to_numpy()
    pcr = df["pcr"].to_numpy()
    # Same auto-detection as scripts 06/08: all-NaN z_post means the cohort has no
    # post-treatment slide, so P(pCR) comes from the likelihood ratio rather than
    # zone-counting sampled z_post. Getting this wrong silently scores the headline
    # experiment off an all-NaN pCR zone.
    single_timepoint = bool(np.all(np.isnan(z_post)))
    if single_timepoint:
        print("[experiments] single-timepoint mode (likelihood-based P(pCR))")
    dc = dict(cfg.get("diffusion", {}))
    drc = dict(cfg.get("drug", {}))
    variant = drc.get("variant", "onehot")
    drug_feats, meta = build_drug_features(
        df, variant=variant, radius=drc.get("morgan_radius", 2),
        n_bits=drc.get("morgan_bits", 2048),
        combination=drc.get("combination", "bitwise_or"))
    device = cfg.get("device", "cuda")

    run_dir = new_run_dir(cfg.paths.runs, args.name)

    # Full-model held-out inference (needed by exp1/2/3/5). Every schedule and
    # architecture argument must match what 06 trained with -- they are rebuilt
    # from config here, not stored in the checkpoint, so a default that disagrees
    # with the training config (e.g. T) loads the weights under the wrong schedule.
    records = run_inference(
        z_pre, z_post, drug_feats, pcr, folds, cfg.paths.ckpt,
        str(run_dir / "samples"), n_folds=dc.get("n_folds", 5),
        n_samples=dc.get("n_samples", 100), dim=dc.get("dim", 512),
        cond_dim=dc.get("cond_dim", 512), T=dc.get("T", 1000),
        schedule=dc.get("schedule", "cosine"), cosine_s=dc.get("cosine_s", 0.008),
        lambda_vlb=dc.get("lambda_vlb", 1e-3),
        conditioning=dc.get("conditioning", "film"),
        n_blocks=dc.get("n_blocks", 3), width=dc.get("width", 512),
        drug_variant=variant,
        num_drugs=meta.get("num_drugs", 16), n_bits=meta.get("n_bits", 2048),
        clip_x0=dc.get("clip_x0"), device=device, force=args.force,
        single_timepoint=single_timepoint)

    results = {}
    if "exp1" in want:
        results["exp1"] = exp1.run_exp1(records, z_pre, folds)
    if "exp2" in want:
        # Cox needs at least some observed follow-up. IMPRESS ships no survival
        # endpoint (rfs_time/rfs_event are empty for every patient), and lifelines
        # fails deep inside its Newton-Raphson with a ZeroDivisionError rather than
        # a usable message when handed zero rows.
        n_surv = int(df[["rfs_time", "rfs_event"]].notna().all(axis=1).sum()) \
            if {"rfs_time", "rfs_event"}.issubset(df.columns) else 0
        n_events = int(df["rfs_event"].fillna(0).astype(float).sum()) \
            if "rfs_event" in df.columns else 0
        if n_surv < 10 or n_events < 5:
            print(f"[exp2] skipped: {n_surv} patients with follow-up, {n_events} events "
                  "— this cohort carries no survival endpoint (CLAUDE.md marks Exp 2 "
                  "exploratory under the single-timepoint pivot)")
        else:
            results["exp2"] = exp2.run_exp2(records, df)
    if "exp3" in want:
        results["exp3"] = exp3.run_exp3(records)

    if "exp4" in want and single_timepoint:
        # Exp 4 ranks arms by the mean P(pCR) of counterfactually sampled z_post.
        # Single-timepoint cohorts have no z_post to sample, and IMPRESS is not a
        # multi-arm trial -- CLAUDE.md defers Exp 4 to future work under this pivot.
        print("[exp4] skipped: counterfactual arm ranking needs the paired design "
              "and a multi-arm trial (future work under the single-timepoint pivot)")
    elif "exp4" in want:
        cdir = _find_config_dir()
        arms = (load_yaml(cdir / "experiment" / "exp4.yaml") or {}).get("arms") or {}
        if not arms:
            print("[exp4] no arms configured (configs/experiment/exp4.yaml); skipping")
        else:
            arm_feats, observed, missing = _arm_features(
                arms, variant, meta, drc.get("morgan_radius", 2),
                drc.get("morgan_bits", 2048), drc.get("combination", "bitwise_or"))
            if missing:
                print(f"[exp4] arms not in trained drug vocab, skipped: {missing}")
            if len(arm_feats) < 2:
                print("[exp4] fewer than 2 usable arms; skipping")
            else:
                results["exp4"] = exp4.run_exp4(
                    z_pre, z_post, pcr, folds, arm_feats, observed, cfg.paths.ckpt,
                    n_folds=dc.get("n_folds", 5), n_samples=dc.get("n_samples", 100),
                    dim=dc.get("dim", 512), cond_dim=dc.get("cond_dim", 512),
                    T=dc.get("T", 1000), schedule=dc.get("schedule", "cosine"),
                    cosine_s=dc.get("cosine_s", 0.008),
                    lambda_vlb=dc.get("lambda_vlb", 1e-3),
                    conditioning=dc.get("conditioning", "film"),
                    n_blocks=dc.get("n_blocks", 3), width=dc.get("width", 512),
                    drug_variant=variant, num_drugs=meta.get("num_drugs", 16),
                    n_bits=meta.get("n_bits", 2048), clip_x0=dc.get("clip_x0"),
                    device=device, seed=cfg.get("seed", 0))

    if "exp5" in want:
        cdir = _find_config_dir()
        e5 = load_yaml(cdir / "experiment" / "exp5.yaml") or {}
        tc = {"epochs": dc.get("epochs", 500), "lr": dc.get("lr", 2e-4),
              "batch_size": dc.get("batch_size", 64),
              "p_uncond": dc.get("p_uncond", 0.1)}
        exp5_out = exp5.run_exp5_orchestrated(
            df, z_pre, z_post, records, base_ckpt_dir=cfg.paths.ckpt,
            work_dir=str(run_dir / "ablations"), dc=dc, drc=drc, tc=tc,
            paired_random=cfg.paths.get("paired_random"),
            paired_mil=cfg.paths.get("paired_mil"),
            variants=e5.get("variants"), metric=e5.get("metric", "auprc"),
            device=device, force=args.force, seed=cfg.get("seed", 0))
        results["exp5"] = exp5_out["deltas"]
        results["exp5_skipped"] = exp5_out["skipped"]

    # figures + tables for whatever ran
    fig_dir = run_dir / "figures"
    figures.make_all_figures(results, str(fig_dir))
    tbl = run_dir / "tables"
    if "exp1" in results:
        tables.table_exp1(results["exp1"], str(tbl / "T2_exp1"))
    if "exp2" in results:
        tables.table_cox(results["exp2"], str(tbl / "T3_cox"))
    if "exp5" in results:
        tables.table_ablations(results["exp5"], str(tbl / "T4_ablations"))
    tables.table_cohort(df, str(tbl / "T1_cohort"))

    metrics = {
        "experiment": which,
        "exp1": results.get("exp1"),
        "exp2_cox": results.get("exp2", {}).get("cox", {}).get("covariates")
        if "exp2" in results else None,
        "exp3_k": results.get("exp3", {}).get("k") if "exp3" in results else None,
        "exp4": {k: v for k, v in results.get("exp4", {}).items()
                 if k != "per_patient"} if "exp4" in results else None,
        "exp5": results.get("exp5"),
        "exp5_skipped": results.get("exp5_skipped"),
    }
    log_run(run_dir, dict(cfg), metrics)
    (run_dir / "results.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"[experiments] ran {sorted(want)} -> {run_dir}")


if __name__ == "__main__":
    main()
