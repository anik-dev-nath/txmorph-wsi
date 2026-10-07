"""21_unpaired_transport.py - train and evaluate p(z_post | z_pre, drug) on
UNPAIRED cohorts (CLAUDE.md S1; unpaired reframing 2026-09-08).

This restores the project's original conditional-generative design. No open cohort
provides per-patient (z_pre, z_post) pairs -- IMPRESS is pre-treatment biopsies,
Post-NAT-BRCA is post-treatment resections from different patients -- so minibatch
optimal transport supplies pseudo-pairs and the existing tested DDPM trains on them
unchanged (see src/txmorph/training/unpaired.py).

    PRE  side: IMPRESS, 126 patients, paired.parquet          (z_pre, drug, subtype)
    POST side: Post-NAT-BRCA, 96 slides / 53 patients         (z_post)

EVALUATION
    No per-patient ground truth exists, so reconstruction error is not measurable
    and is never reported. The model is scored by two-sample tests between its
    generated post-treatment latents and HELD-OUT real ones (eval/distribution.py).
    A LARGE permutation p-value is the good outcome: generated samples are not
    distinguishable from real held-out ones.

ABLATION
    ``--coupling independent`` replaces optimal transport with random pairing. It
    is the control that isolates what the transport contributes; everything else is
    held fixed.

Usage (server):
    PYTHONPATH=src python scripts/21_unpaired_transport.py --strata subtype
    PYTHONPATH=src python scripts/21_unpaired_transport.py --strata subtype --coupling independent
    PYTHONPATH=src python scripts/21_unpaired_transport.py --strata tnbc_only
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (adds src/ to sys.path)


def load_post_pool(emb_dir: str, slides_csv: str, clinical_csv: str):
    """Mean-pool each post slide's tile bag; join subtype and patient id.

    Mean pooling, deliberately: the delivered pre-side latents in paired.parquet
    are byte-identical to a plain mean over the tile bag (docs/LIMITATIONS.md S11),
    so the post side must be pooled the same way or the two sides would live in
    different spaces and the transport would be fitting an artifact.
    """
    import pandas as pd
    from txmorph.data.embeddings import read_slide_bag, list_slides

    slides = pd.read_csv(slides_csv, dtype={"slide_id": str})
    clin = pd.read_csv(clinical_csv, dtype={"patient_id": str})
    meta = slides.merge(clin[["patient_id", "receptor_subtype", "drug_id"]],
                        on="patient_id", how="left")
    meta = meta.set_index("slide_id")

    z, pid, sub = [], [], []
    for sid in sorted(list_slides(emb_dir)):
        if sid not in meta.index:
            continue
        bag = read_slide_bag(sid, emb_dir)
        z.append(np.asarray(bag, dtype=np.float32).mean(axis=0))
        pid.append(str(meta.loc[sid, "patient_id"]))
        sub.append(str(meta.loc[sid, "receptor_subtype"]))
    if not z:
        raise SystemExit(f"no post embeddings matched {slides_csv} in {emb_dir}")
    return np.stack(z), np.array(pid), np.array(sub)


def patient_level_folds(patient_ids: np.ndarray, n_folds: int, seed: int):
    """Assign folds at PATIENT level: 96 slides come from 53 patients, and a
    patient's slides must never straddle the train/held-out boundary."""
    uniq = np.unique(patient_ids)
    rng = np.random.default_rng(seed)
    assign = {p: i % n_folds for i, p in enumerate(rng.permutation(uniq))}
    return np.array([assign[p] for p in patient_ids])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    _bootstrap.common_args(ap)
    ap.add_argument("--post-emb", default="/home/anik-server/data/tile_emb_unlabeled")
    ap.add_argument("--post-slides", default="/home/anik-server/data/slides_postnat.csv")
    ap.add_argument("--post-clinical", default="/home/anik-server/data/clinical_postnat.csv")
    ap.add_argument("--strata", default="subtype",
                    choices=["subtype", "none", "tnbc_only"],
                    help="'subtype' couples only within receptor subtype (default, "
                         "biologically correct); 'none' ignores subtype and is the "
                         "scale-up sensitivity; 'tnbc_only' restricts both sides to "
                         "TNBC, the one stratum with real mass on both sides")
    ap.add_argument("--coupling", default="exact", choices=["exact", "independent"])
    ap.add_argument("--clip-x0", type=float, default=6.0,
                    help="clamp predicted x0 to +-this many per-dim std "
                         "(CLAUDE.md S7). Latents are standardised in-fold, so "
                         "6.0 is the prescribed 6x. Set 0 to disable.")
    ap.add_argument("--subspace-dim", type=int, default=32,
                    help="PCA subspace the diffusion runs in, fit per training "
                         "fold. Measured: at 512-d the denoiser never learns "
                         "(loss 1.0-4.5); at 16-64 it reaches ~0.52, the same "
                         "floor an N(0,I) control attains. 0 disables.")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="1e-3 measured best; the 2e-4 config default does not "
                         "converge on this data size (ddpm_capacity_probe.py)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = _bootstrap.load(args)

    import torch
    from txmorph.data.paired_dataset import load_paired
    from txmorph.drug.features import build_drug_features
    from txmorph.training.unpaired import train_unpaired_cv, split_post_pool
    from txmorph.diffusion.schedule import NoiseSchedule
    from txmorph.diffusion.gaussian_diffusion import GaussianDiffusion
    from txmorph.diffusion.sampler import sample_patient
    from txmorph.training.diffusion import DiffusionModule
    from txmorph.eval.distribution import (report as dist_report,
                                           report_balanced)
    from txmorph.utils import ckpt as ckpt_mod

    dc = cfg.get("diffusion", {})
    device = cfg.get("device", "cuda")
    seed = cfg.get("seed", 0)
    n_folds = dc.get("n_folds", 5)

    # ---------------- pre side ----------------
    df, z_pre, _ = load_paired(cfg.paths.paired)
    sub_pre = df["receptor_subtype"].astype(str).to_numpy()
    folds = df["fold"].to_numpy()
    drcfg = cfg.get("drug", {})
    variant = drcfg.get("variant", "onehot")
    drug_feats, dmeta = build_drug_features(
        df, variant=variant, radius=drcfg.get("morgan_radius", 2),
        n_bits=drcfg.get("morgan_bits", 2048),
        combination=drcfg.get("combination", "bitwise_or"))

    # ---------------- post side ----------------
    z_post, post_pid, sub_post = load_post_pool(
        args.post_emb, args.post_slides, args.post_clinical)
    post_folds = patient_level_folds(post_pid, n_folds, seed)

    if args.strata == "tnbc_only":
        mpre, mpost = sub_pre == "TNBC", sub_post == "TNBC"
        z_pre, drug_feats, folds, sub_pre = (z_pre[mpre], drug_feats[mpre],
                                             folds[mpre], sub_pre[mpre])
        df = df[mpre]
        z_post, sub_post, post_folds = z_post[mpost], sub_post[mpost], post_folds[mpost]
    gp = gq = None
    if args.strata == "subtype":
        gp, gq = sub_pre, sub_post

    shared = sorted(set(sub_pre) & set(sub_post))
    print(f"[transport] pre n={len(z_pre)}  post n={len(z_post)} "
          f"({len(np.unique(post_pid))} patients)")
    print(f"[transport] strata={args.strata} coupling={args.coupling} "
          f"shared subtypes={shared}")
    if args.strata == "subtype":
        usable = int(np.isin(sub_post, list(shared)).sum())
        print(f"[transport] post slides in a shared subtype: {usable}/{len(sub_post)} "
              f"-- the rest cannot be coupled and are dropped")

    # Effective model dimension: the PCA subspace when enabled, else the raw
    # latent width. Measured at 512-d the denoiser never learns on this sample
    # size (scripts/analysis/ddpm_capacity_probe.py).
    sub = args.subspace_dim if args.subspace_dim and args.subspace_dim > 0 else None
    model_dim = sub or dc.get("dim", 512)
    print(f"[transport] model_dim={model_dim} lr={args.lr} "
          f"(subspace={'off' if sub is None else sub})")

    tag = f"{args.strata}_{args.coupling}_d{model_dim}"
    out_dir = Path(args.out or (Path(cfg.paths.runs) / f"unpaired_{tag}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_dir = out_dir / "ckpt"

    # ---------------- train ----------------
    train_unpaired_cv(
        z_pre, drug_feats, z_post, folds, post_folds, str(ck_dir), n_folds=n_folds,
        groups_pre=gp, groups_post=gq, coupling=args.coupling,
        dim=model_dim, cond_dim=dc.get("cond_dim", 512),
        subspace_dim=sub,
        T=dc.get("T", 1000), schedule=dc.get("schedule", "cosine"),
        lambda_vlb=dc.get("lambda_vlb", 1e-3),
        conditioning=dc.get("conditioning", "film"),
        n_blocks=dc.get("n_blocks", 3), width=dc.get("width", 512),
        drug_variant=variant, num_drugs=dmeta.get("num_drugs", 16),
        n_bits=dmeta.get("n_bits", 2048), epochs=args.epochs,
        lr=args.lr, batch_size=dc.get("batch_size", 64),
        device=device, seed=seed)

    # ---------------- sample + evaluate ----------------
    sch = NoiseSchedule(T=dc.get("T", 1000), schedule=dc.get("schedule", "cosine"),
                        s=dc.get("cosine_s", 0.008))
    diffusion = GaussianDiffusion(sch, lambda_vlb=dc.get("lambda_vlb", 1e-3)).to(device)

    results, gen_mean_all, gen_all, real_all = {}, [], [], []
    for k in range(n_folds):
        mod = DiffusionModule(model_dim, dc.get("cond_dim", 512), variant,
                              dmeta.get("num_drugs", 16), dmeta.get("n_bits", 2048),
                              dc.get("n_blocks", 3), dc.get("width", 512),
                              dc.get("conditioning", "film"), device,
                              single_timepoint=False)
        st = ckpt_mod.maybe_resume(str(ck_dir / f"unpaired_fold{k}.pt"), mod, None,
                                   map_location=device)
        if st is None:
            print(f"[transport] fold {k}: no checkpoint, skipping")
            continue
        mod.eval()

        # Reproduce the fold's in-fold standardisation. The model was trained in
        # that space, so both the conditioning and the real post latents it is
        # scored against must be mapped into it -- otherwise every two-sample
        # test rejects on a scale mismatch rather than on distribution shape.
        ex = st.get("extra", {})
        mu_pre, sd_pre = ex["mu_pre"], ex["sd_pre"]
        mu_post, sd_post = ex["mu_post"], ex["sd_post"]
        pca_pre, pca_post = ex.get("pca_pre"), ex.get("pca_post")

        def to_model_space(x, mu, sd, pca, s):
            """Apply the fold's frozen transform. Held-out rows are transformed
            with the training-fit PCA, never re-fitted."""
            x = (np.asarray(x) - mu) / sd
            if pca is not None:
                x = pca.transform(x) / s
            return x.astype(np.float32)

        va = np.where(folds == k)[0]
        real = to_model_space(z_post[post_folds == k], mu_post, sd_post,
                              pca_post, ex.get("s_post"))
        if len(real) < 2:
            print(f"[transport] fold {k}: only {len(real)} held-out post slides, "
                  f"skipping the two-sample test")
            continue

        # Sample ALL held-out patients in ONE batched reverse pass.
        # diffusion.sample accepts a per-row context, so looping per patient costs
        # len(va) x T sequential model calls instead of T. The denoiser is tiny,
        # so that loop is dominated by kernel-launch overhead: it pinned ~2 CPU
        # cores while the GPU idled at 4%. Batching is ~len(va)x fewer launches
        # and actually saturates the device (CLAUDE.md S6.5: draw the samples as
        # "a single batched pass per timestep").
        #
        # clip_x0: latents are standardised in-fold (per-dim std = 1), so
        # CLAUDE.md S7's "clamp predicted x0 to ~6x the per-dim embedding std" is
        # literally 6.0 here. Without it reverse sampling diverges and every
        # two-sample test rejects on magnitude rather than distribution shape
        # (observed: sliced-W2 ~25,000 in a unit-variance space).
        z_pre_s = to_model_space(z_pre, mu_pre, sd_pre, pca_pre, ex.get("s_pre"))
        zc = torch.from_numpy(z_pre_s[va]).to(device)
        db = torch.from_numpy(drug_feats[va])
        db = (db.long() if variant == "onehot" else db.float()).to(device)
        with torch.no_grad():
            c_all = mod.context(zc, mod.drug(db))               # (n_val, cond_dim)
        c_rep = c_all.repeat_interleave(args.n_samples, dim=0)   # (n_val*S, cond)

        flat = sample_patient(diffusion, mod.denoiser, c_rep, n=c_rep.shape[0],
                              dim=model_dim, device=device,
                              clip_x0=(args.clip_x0 or None))
        gen = flat.cpu().numpy().reshape(len(va), args.n_samples, -1)
        gen_mean = gen.mean(axis=1)               # per-patient expected morphology

        # Balanced: subsample the generated pool down to len(real). Comparing
        # every draw (n_val x S) against ~20 real slides is a ~100:1 imbalance
        # that makes the permutation test reject on any microscopic bias.
        r_all = report_balanced(gen.reshape(-1, gen.shape[-1]), real,
                                n_rep=25, n_perm=500, seed=seed)
        r_mean = report_balanced(gen_mean, real, n_rep=25, n_perm=500, seed=seed)
        results[f"fold{k}"] = {"all_samples": r_all, "patient_means": r_mean,
                               "n_val_pre": int(len(va)), "n_real_post": int(len(real))}
        print(f"[transport] fold {k}: MMD2 {r_all['mmd2_mean']:.4f} "
              f"p={r_all['p_value_median']:.3f} "
              f"(reject {r_all['reject_rate_05']:.2f}) | "
              f"means p={r_mean['p_value_median']:.3f} | "
              f"SW2 {r_all['sliced_w2_mean']:.3f} | n_real={len(real)}")

        gen_all.append(gen.reshape(-1, gen.shape[-1]))
        gen_mean_all.append(gen_mean)
        real_all.append(real)

    if gen_all:
        pooled = {
            "all_samples": report_balanced(np.vstack(gen_all), np.vstack(real_all),
                                           n_rep=50, n_perm=1000, seed=seed),
            "patient_means": report_balanced(np.vstack(gen_mean_all),
                                             np.vstack(real_all),
                                             n_rep=50, n_perm=1000, seed=seed),
        }
        results["pooled"] = pooled
        print(f"\n[transport] POOLED  MMD2 {pooled['all_samples']['mmd2']:.4f} "
              f"p={pooled['all_samples']['p_value']:.3f} | "
              f"energy {pooled['all_samples']['energy_distance']:.4f} | "
              f"SW2 {pooled['all_samples']['sliced_w2']:.3f}")
        print("[transport] reminder: a LARGE p-value is the favourable outcome -- it "
              "means generated and real post latents are not distinguishable.")

    results["config"] = {"strata": args.strata, "coupling": args.coupling,
                         "epochs": args.epochs, "n_samples": args.n_samples,
                         "n_pre": int(len(z_pre)), "n_post": int(len(z_post)),
                         "shared_subtypes": shared, "seed": int(seed)}
    (out_dir / "transport_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"[transport] wrote {out_dir / 'transport_metrics.json'}")


if __name__ == "__main__":
    main()
