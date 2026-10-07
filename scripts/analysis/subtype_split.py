"""subtype_split.py - does pooling HER2+ and TNBC into one model cost us AUROC?

WHY: the published IMPRESS benchmark (Huang et al., npj Precis Oncol 2023) reports
pCR AUC **per subtype** -- 0.8975 HER2+ and 0.7674 TNBC -- while this project pools
all 126 patients into a single model and reports 0.588 (generative) / 0.691
(discriminative). Pooling is a plausible cause: HER2+ and TNBC have different
response biology, and drug_id is perfectly collinear with subtype here
(docs/LIMITATIONS.md S6), so a pooled model cannot separate regimen from subtype.

This measures, on the existing frozen folds and latents, with no retraining:
  1. pooled AUROC (the current number, as a control)
  2. per-subtype AUROC of the pooled model's own out-of-fold scores
  3. per-subtype AUROC of probes fit *within* each subtype
  4. the same for a subtype-stratified discriminative baseline

(2) vs (3) is the question: if fitting within subtype beats slicing the pooled
model's predictions, the pooled model is genuinely losing signal and the fix is to
stratify. Everything is fit inside the training fold only.

Usage (server):
    PYTHONPATH=src python scripts/analysis/subtype_split.py
"""
from __future__ import annotations
import argparse

import numpy as np


def probe_auroc(z, y, folds, seed=0):
    """Out-of-fold scores from a standardised, regularised logistic probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scores = np.full(len(y), np.nan)
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        if tr.sum() == 0 or va.sum() == 0:
            continue
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(z[tr])
        lr = LogisticRegression(C=0.01, max_iter=5000, random_state=seed)
        lr.fit(sc.transform(z[tr]), y[tr])
        scores[va] = lr.predict_proba(sc.transform(z[va]))[:, 1]
    return scores


def auroc_ci(scores, y, n_boot=2000, seed=0):
    from txmorph.eval.classification import auroc, bootstrap_ci
    m = ~np.isnan(scores)
    if len(np.unique(y[m])) < 2:
        return {"auroc": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": int(m.sum())}
    out = bootstrap_ci(auroc, scores[m], y[m], n_boot=n_boot, seed=seed)
    if isinstance(out, dict):
        val = out.get("value", out.get("mean"))
        lo, hi = out.get("lo", out.get("ci_lo")), out.get("hi", out.get("ci_hi"))
    else:
        val, lo, hi = out
    return {"auroc": float(val), "lo": float(lo), "hi": float(hi), "n": int(m.sum())}


def fmt(tag, r):
    if np.isnan(r["auroc"]):
        print(f"  {tag:<34} n={r['n']:<4} (single class - undefined)")
    else:
        print(f"  {tag:<34} n={r['n']:<4} AUROC {r['auroc']:.3f} "
              f"[{r['lo']:.3f},{r['hi']:.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", default="/home/anik-server/data/paired.parquet")
    args = ap.parse_args()

    from txmorph.data.paired_dataset import load_paired

    df, z_pre, _ = load_paired(args.paired)
    y = df["pcr"].to_numpy().astype(int)
    folds = df["fold"].to_numpy()
    sub = df["receptor_subtype"].astype(str).to_numpy()

    print(f"n={len(y)}  pCR={y.mean():.3f}")
    print("subtype counts:")
    for s in sorted(set(sub)):
        m = sub == s
        print(f"  {s:<12} n={m.sum():<4} pCR={y[m].mean():.3f}")

    print("\n=== 1. pooled model (current design) ===")
    pooled = probe_auroc(z_pre, y, folds)
    fmt("pooled, all patients", auroc_ci(pooled, y))

    print("\n=== 2. pooled model's scores, sliced by subtype ===")
    print("    (does the ONE pooled model rank correctly inside each subtype?)")
    for s in sorted(set(sub)):
        m = sub == s
        fmt(f"pooled scores | {s}", auroc_ci(pooled[m], y[m]))

    print("\n=== 3. probe fit WITHIN each subtype (stratified) ===")
    print("    (fold structure preserved; nothing crosses a fold boundary)")
    strat = np.full(len(y), np.nan)
    for s in sorted(set(sub)):
        m = sub == s
        sc = probe_auroc(z_pre[m], y[m], folds[m])
        strat[m] = sc
        fmt(f"within-subtype | {s}", auroc_ci(sc, y[m]))

    print("\n=== 4. stratified scores pooled back together ===")
    print("    (per-subtype probes, evaluated over all 126 -- the deployable number)")
    fmt("stratified, all patients", auroc_ci(strat, y))

    print("\n--- interpretation ---")
    print("If (3) > (2) for a subtype, the pooled model is losing signal there and")
    print("the model should be stratified. If (3) ~ (2), pooling is not the problem.")
    print("Published IMPRESS benchmark, for reference: HER2+ 0.8975, TNBC 0.7674")
    print("(H&E + multiplex IHC + clinical; ours is H&E-only, so it is not a")
    print("like-for-like comparison -- see docs/LIMITATIONS.md)")


if __name__ == "__main__":
    main()
