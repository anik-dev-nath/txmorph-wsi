"""16_exp1_definitive.py - the Exp-1 table against a correctly configured baseline.

Supersedes the Exp-1 numbers in docs/repro.md. The original baseline was an
under-regularised logistic on raw 512-d embeddings (sklearn default C=1.0, no
standardisation, ~100 training patients), whose ECE of 0.161 is an artefact of
misconfiguration. The "2x better calibration" claim rested on that comparison.

This regenerates the table with one pre-registered baseline configuration --
standardised, L2 strength chosen by inner CV on the training fold only -- plus
its Platt and isotonic calibrated variants and a constant base-rate reference.
All CIs are patient-level bootstrap (1000 resamples). Nothing is fit outside a
training fold.

Emits T2 (CSV + LaTeX) and a reliability diagram.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def _metrics(p, y, n_boot=1000, seed=0):
    import torch
    from txmorph.eval.classification import auroc, auprc, bootstrap_ci
    from txmorph.eval.calibration import expected_calibration_error, brier_score

    def ece(pp, yy):
        return float(expected_calibration_error(torch.tensor(pp), torch.tensor(yy)))

    def brier(pp, yy):
        return float(brier_score(torch.tensor(pp), torch.tensor(yy)))

    out = {}
    for name, fn in (("auroc", auroc), ("auprc", auprc), ("ece", ece), ("brier", brier)):
        out[name] = bootstrap_ci(fn, p, y, n_boot=n_boot, seed=seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", default="/home/anik-server/data/paired.parquet")
    ap.add_argument("--biomarkers", default="/home/anik-server/runs/samples/biomarkers.json")
    ap.add_argument("--out-dir", default="/home/anik-server/runs/manuscript")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    import pandas as pd
    from txmorph.data.paired_dataset import load_paired
    from txmorph.experiments.exp1 import baseline_zpre_logistic

    df, z, _ = load_paired(args.paired)
    recs = json.loads(Path(args.biomarkers).read_text())
    idx = np.array([r["index"] for r in recs])
    y = np.array([int(r["pcr"]) for r in recs])
    p_gen = np.array([float(r["p_pcr"]) for r in recs])
    folds = df["fold"].to_numpy()[idx]
    zi = np.asarray(z)[idx]
    print(f"n={len(y)}  base rate={y.mean():.4f}  constant Brier={y.mean()*(1-y.mean()):.4f}")

    arms = {"TxMorph (generative)": p_gen}
    for label, kw in (
        ("Logistic (tuned C, uncalibrated)", dict(calibration="none")),
        ("Logistic + Platt", dict(calibration="platt")),
        ("Logistic + isotonic", dict(calibration="isotonic")),
    ):
        print(f"  fitting {label} ...", flush=True)
        arms[label] = baseline_zpre_logistic(zi, y, folds, **kw)
    arms["Constant base rate"] = np.full(len(y), y.mean())

    rows = []
    for name, p in arms.items():
        m = _metrics(p, y, n_boot=args.n_boot)
        rows.append({
            "model": name,
            **{k: f"{v[0]:.3f} ({v[1]:.3f}-{v[2]:.3f})" for k, v in m.items()},
        })
        print(f"  {name:34s} AUROC {m['auroc'][0]:.3f}  ECE {m['ece'][0]:.3f}  "
              f"Brier {m['brier'][0]:.3f}")

    out = Path(args.out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    from txmorph.viz.tables import _write
    paths = _write(pd.DataFrame(rows), str(out / "tables" / "T2_exp1_definitive"))
    print("[T2]", paths)

    # reliability diagram over every arm
    import matplotlib.pyplot as plt
    from txmorph.eval.calibration import reliability_curve
    from txmorph.viz.style import set_style, save_figure
    import torch
    set_style()
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color="#999", label="perfect")
    for name, p in arms.items():
        if name == "Constant base rate":
            continue
        conf, acc = reliability_curve(torch.tensor(p), torch.tensor(y), n_bins=15)[:2]
        ax.plot(np.asarray(conf), np.asarray(acc), marker="o", ms=3, lw=1.2, label=name)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title("Exp 1 reliability (15 bins, out-of-fold)", fontsize=9)
    ax.legend(fontsize=6, loc="upper left")
    print("[fig]", save_figure(fig, str(out / "figures" / "fig4_reliability_definitive")))


if __name__ == "__main__":
    main()
