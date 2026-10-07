"""12_external_validation.py - apply the frozen model to the HER2 Yale arm.

Independent external validation (CLAUDE.md S11 cross-cutting, S14). The Yale
trastuzumab-response cohort is 85 pre-treatment biopsies the model has never seen:
it is excluded from SimCLR pretraining via the pinned corpus manifest, absent from
the MIL's training manifest, and absent from every diffusion fold.

Protocol, and why each part is what it is:

* **Nothing is fit on the external cohort.** For each fold k we take that fold's
  diffusion model and fit its LLR->probability calibrator on fold k's *IMPRESS
  training* patients, then apply both to all 85 external cases. Fold probabilities
  are averaged. Every fold is "training" with respect to this cohort, so the
  ensemble is legitimate rather than a form of test-time selection.
* **The discriminative baseline gets the same deal** -- trained on all 126 IMPRESS
  patients, applied once to the external set -- so the comparison is like for like.
* **The endpoint is not identical to pCR.** Yale labels trastuzumab *response*
  (36 responder / 49 non-responder), while the model was trained on pathologic
  complete response in IMPRESS. This is a related but distinct endpoint and the
  transfer must be described that way; it is a genuine external test of the
  morphology->response signal, not a like-for-like replication of Exp 1.

    python scripts/12_external_validation.py
"""
import argparse
import json
from pathlib import Path

import _bootstrap


def main():
    ap = argparse.ArgumentParser(description="External validation on the HER2 Yale arm.")
    _bootstrap.common_args(ap)
    ap.add_argument("--clinical", default="/home/anik-server/data/clinical_her2_yale.csv")
    ap.add_argument("--emb-dir", default=None, help="default: <tile_emb>_external")
    ap.add_argument("--name", default="external_her2_yale")
    ap.add_argument("--n-mc", type=int, default=50)
    args = ap.parse_args()
    cfg = _bootstrap.load(args)

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.linear_model import LogisticRegression

    from txmorph.encoders.simclr import SimCLREncoder
    from txmorph.encoders.extract import extract_slide_embeddings
    from txmorph.data.embeddings import read_slide_bag
    from txmorph.data.paired_dataset import load_paired
    from txmorph.drug.features import build_drug_features
    from txmorph.diffusion.schedule import NoiseSchedule
    from txmorph.diffusion.gaussian_diffusion import GaussianDiffusion
    from txmorph.inference.run import (load_fold_module, single_timepoint_scores,
                                       _fit_llr_calibrator)
    from txmorph.utils.cv import train_val_masks
    from txmorph.utils import ckpt
    from txmorph.eval.classification import auroc, auprc, bootstrap_ci
    from txmorph.eval.calibration import expected_calibration_error, brier_score
    from txmorph.utils.runlog import new_run_dir

    device = cfg.get("device", "cuda")
    emb_dir = args.emb_dir or (str(cfg.paths.tile_emb).rstrip("/") + "_external")
    Path(emb_dir).mkdir(parents=True, exist_ok=True)

    ext = pd.read_csv(args.clinical)
    tiles_root = Path(cfg.paths.tiles)
    ec = cfg.get("simclr", cfg.get("encoder", {}))

    # --- embed the external slides with the SAME frozen encoder ---------------
    encoder = SimCLREncoder(proj_dim=ec.get("proj_dim", 128))
    ckpt.load(str(Path(cfg.paths.ckpt) / "simclr.pt"), encoder, map_location=device,
              restore_rng=False)
    encoder = encoder.to(device).eval()

    keep, Z = [], []
    for i, r in enumerate(ext.itertuples(), 1):
        sid = str(r.pre_slide_id)
        d = tiles_root / sid
        if not (d / "done.json").exists():
            continue
        extract_slide_embeddings(encoder, str(d), emb_dir, sid,
                                 batch_size=ec.get("extract_batch_size", 256),
                                 device=device,
                                 num_workers=ec.get("num_workers", cfg.get("num_workers", 3)),
                                 force=args.force)
        Z.append(read_slide_bag(sid, emb_dir).astype(np.float32).mean(axis=0))
        keep.append(r.Index)
        if i % 25 == 0:
            print(f"[external] embedded {i}/{len(ext)}", flush=True)
    ext = ext.loc[keep].reset_index(drop=True)
    Z_ext = np.stack(Z)
    y_ext = ext["pcr"].astype(int).to_numpy()
    print(f"[external] {len(ext)} cases | positives {y_ext.sum()}/{len(y_ext)} "
          f"| latents {Z_ext.shape}")

    # --- internal cohort: models, calibrators, and the baseline ---------------
    df, z_pre, z_post = load_paired(cfg.paths.paired)
    folds = df["fold"].to_numpy()
    pcr = df["pcr"].astype(int).to_numpy()
    dc = cfg.get("diffusion", {})
    drc = cfg.get("drug", {})
    variant = drc.get("variant", "onehot")
    feats_int, meta = build_drug_features(df, variant=variant,
                                          radius=drc.get("morgan_radius", 2),
                                          n_bits=drc.get("morgan_bits", 2048),
                                          combination=drc.get("combination", "bitwise_or"))
    # Map the external regimen into the *trained* vocabulary; an unseen drug id
    # would otherwise index a randomly-initialised embedding row.
    vocab = meta.get("vocab", {})
    ext_drug = []
    for v in ext["drug_id"].astype(str):
        if v not in vocab:
            raise SystemExit(f"[external] drug_id {v!r} is not in the trained vocab "
                             f"{sorted(vocab)}; cannot score it honestly")
        ext_drug.append(int(vocab[v]))
    ext_drug = np.asarray(ext_drug)

    sch = NoiseSchedule(T=dc.get("T", 1000), schedule=dc.get("schedule", "cosine"),
                        s=dc.get("cosine_s", 0.008))
    diffusion = GaussianDiffusion(sch, lambda_vlb=dc.get("lambda_vlb", 1e-3)).to(device)

    n_folds = dc.get("n_folds", 5)
    p_folds = []
    for k in range(n_folds):
        mod = load_fold_module(
            str(Path(cfg.paths.ckpt) / f"diffusion_fold{k}.pt"),
            dc.get("dim", 512), dc.get("cond_dim", 512), variant,
            meta.get("num_drugs", 16), meta.get("n_bits", 2048),
            dc.get("n_blocks", 3), dc.get("width", 512),
            dc.get("conditioning", "film"), device, single_timepoint=True)
        tr, _ = train_val_masks(folds, k)
        ti = np.where(tr)[0]
        llr_tr, _ = single_timepoint_scores(diffusion, mod, z_pre[ti], feats_int[ti],
                                            variant, device, n_mc=args.n_mc, seed=k)
        llr_ex, _ = single_timepoint_scores(diffusion, mod, Z_ext, ext_drug,
                                            variant, device, n_mc=args.n_mc, seed=k)
        p_folds.append(_fit_llr_calibrator(llr_tr, pcr[ti])(llr_ex))
        print(f"[external] fold {k}: scored {len(llr_ex)} external cases", flush=True)
    p_tx = np.mean(p_folds, axis=0)

    # Discriminative baseline, trained on ALL internal patients, applied once.
    mu, sd = z_pre.mean(0), z_pre.std(0) + 1e-8
    base = LogisticRegression(max_iter=3000, C=0.01, class_weight="balanced")
    base.fit((z_pre - mu) / sd, pcr)
    p_base = base.predict_proba((Z_ext - mu) / sd)[:, 1]

    def _metrics(p, y):
        t = torch.tensor

        def ci(fn):                       # bootstrap_ci -> (point, lo, hi)
            point, lo, hi = bootstrap_ci(fn, p, y, n_boot=1000, seed=0)
            return {"mean": float(point), "lo": float(lo), "hi": float(hi)}

        return {
            "auroc": ci(auroc),
            "auprc": ci(auprc),
            "ece": ci(lambda a, b: expected_calibration_error(t(a), t(b))),
            "brier": ci(lambda a, b: brier_score(t(a), t(b))),
        }

    out = {"n": int(len(y_ext)), "positives": int(y_ext.sum()),
           "endpoint": "trastuzumab response (NOT identical to pCR; see module docstring)",
           "txmorph": _metrics(p_tx, y_ext), "baseline": _metrics(p_base, y_ext)}

    run_dir = new_run_dir(cfg.paths.runs, args.name)
    (run_dir / "external_metrics.json").write_text(json.dumps(out, indent=2, default=str))
    np.savez(run_dir / "external_predictions.npz", p_txmorph=p_tx, p_baseline=p_base,
             y=y_ext, patient_id=ext["patient_id"].to_numpy())

    def _fmt(v):
        return f"{v['mean']:.3f} [{v['lo']:.3f},{v['hi']:.3f}]"

    print(f"\n[external] HER2 Yale, n={len(y_ext)} ({y_ext.sum()} responders)")
    print(f"{'metric':8s} {'TxMorph (generative)':>26s} {'baseline (logistic)':>26s}")
    for k in ("auroc", "auprc", "ece", "brier"):
        print(f"{k:8s} {_fmt(out['txmorph'][k]):>26s} {_fmt(out['baseline'][k]):>26s}")
    print(f"\n[external] wrote {run_dir}")


if __name__ == "__main__":
    main()
