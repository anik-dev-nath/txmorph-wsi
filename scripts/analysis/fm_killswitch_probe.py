"""Kill-switch on a foundation-model latent: does pCR separation beat 0.691?

The project's single scientific bet is that the encoder's latent carries
treatment-response structure (CLAUDE.md S1). On the from-scratch SimCLR latent it
separates pCR at 0.691 and *institution* at 0.94 -- the diagnosis in
docs/STATUS_FOR_REVIEW.md. This scores the same quantity on a foundation-model
latent, holding everything else fixed.

Held fixed against the SimCLR run so the comparison is exact:
  * the same 126 patients and the same seeded patient-level folds, read from the
    existing paired.parquet -- not re-assigned;
  * the same probe (L2 logistic, train-fold standardisation, out-of-fold scores);
  * mean pooling of the tile bag. Not MIL: the MIL attention was trained on the
    subtype auxiliary task against the *SimCLR* features, so reusing it here would
    confound encoder with aggregator. pool_diag.py also found mean pooling beat
    MIL on this metric (0.691 vs 0.602), so this is the stronger comparator.

Reports the gate (CLAUDE.md S10: AUROC >= 0.65 to proceed, and this plan's target
of 0.75 to declare the encoder swap a success) plus the cohort-confounding control
-- a better encoder should raise pCR separation *without* raising institution
separation.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load_pooled(emb_dir: str, slide_ids, pool: str = "mean") -> np.ndarray:
    """Mean-pool each slide's tile bag -> (n_slides, feat_dim)."""
    import h5py
    out = []
    for sid in slide_ids:
        path = os.path.join(emb_dir, f"{sid}.h5")
        with h5py.File(path, "r") as h:
            e = np.asarray(h["emb"], dtype=np.float32)
        out.append(e.mean(0) if pool == "mean" else e.max(0))
    return np.stack(out)


def oof_auroc(X, y, folds, seed=0):
    """Out-of-fold AUROC from an L2 logistic probe, train-fold scaling only."""
    s = np.zeros(len(y))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        m = LogisticRegression(max_iter=5000, C=0.01, class_weight="balanced",
                               random_state=seed)
        m.fit((X[tr] - mu) / sd, y[tr])
        s[va] = m.decision_function((X[va] - mu) / sd)
    return roc_auc_score(y, s), s


def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            b.append(roc_auc_score(y[i], s[i]))
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--paired", default="/home/anik-server/data/paired.parquet",
                    help="source of the frozen patient-level fold assignment")
    ap.add_argument("--slides-csv", default="/home/anik-server/data/slides.csv")
    ap.add_argument("--external-emb", default=None,
                    help="optional second cohort dir, for the confounding control")
    ap.add_argument("--label", default="encoder")
    args = ap.parse_args()

    df = pd.read_parquet(args.paired, columns=["patient_id", "pcr", "fold"])
    slides = pd.read_csv(args.slides_csv)
    scol = "slide_id"
    pcol = "patient_id"
    df = df.merge(slides[[pcol, scol]].drop_duplicates(pcol), on=pcol, how="left")

    have = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(args.emb_dir, "*.h5"))}
    df = df[df[scol].isin(have)].reset_index(drop=True)
    print(f"[{args.label}] n = {len(df)} patients with embeddings "
          f"| pCR {int(df.pcr.sum())} ({df.pcr.mean():.3f})")

    X = load_pooled(args.emb_dir, df[scol].tolist())
    y = df.pcr.to_numpy().astype(int)
    folds = df.fold.to_numpy()
    print(f"[{args.label}] pooled latent {X.shape}")

    auc, s = oof_auroc(X, y, folds)
    lo, hi = boot_ci(y, s)
    print(f"\n=== KILL-SWITCH: pCR separation ===")
    print(f"  {args.label:22s} AUROC {auc:.3f}  [{lo:.3f}, {hi:.3f}]")
    print(f"  {'SimCLR v5 (reference)':22s} AUROC 0.691  [0.601, 0.778]")
    print(f"  delta {auc - 0.691:+.3f}")
    print(f"  CLAUDE.md S10 gate (>=0.65): {'PASS' if auc >= 0.65 else 'FAIL'}")
    print(f"  encoder-swap target (>=0.75): {'MET' if auc >= 0.75 else 'NOT MET'}")

    if args.external_emb:
        ext_ids = [os.path.basename(p)[:-3]
                   for p in sorted(glob.glob(os.path.join(args.external_emb, "*.h5")))]
        if ext_ids:
            Xe = load_pooled(args.external_emb, ext_ids)
            Xa = np.vstack([X, Xe])
            ya = np.r_[np.zeros(len(X)), np.ones(len(Xe))]
            fa = np.r_[folds, np.random.default_rng(0).integers(0, 5, len(Xe))]
            ca, _ = oof_auroc(Xa, ya.astype(int), fa)
            print(f"\n=== CONTROL: institution separability ===")
            print(f"  cohort-discrimination AUROC {ca:.3f}  (SimCLR v5: 0.94)")
            print("  lower is better -- a good encoder separates biology, not sites")


if __name__ == "__main__":
    main()
