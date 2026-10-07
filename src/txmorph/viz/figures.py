"""Manuscript figures (CLAUDE.md S15).

Each figure function takes results (dicts from experiments) and writes vector
(PDF+SVG) + PNG via ``style.save_figure``. ``make_all_figures`` regenerates the set
from a results directory. No figure is ever hand-edited - change the code.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from .style import set_style, save_figure, PALETTE, CATEGORICAL


def fig_calibration(exp1_result, out_base: str):
    """Fig 4: reliability diagram + ROC/PR for TxMorph vs baseline."""
    import matplotlib.pyplot as plt
    set_style()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["muted"], label="perfect")
    for name, color in [("txmorph", PALETTE["model"]), ("baseline", PALETTE["muted"])]:
        rel = exp1_result[name]["reliability"]
        conf = np.array(rel["conf"], dtype=float)
        acc = np.array(rel["acc"], dtype=float)
        ok = ~np.isnan(conf)
        ax.plot(conf[ok], acc[ok], "-o", color=color, ms=3, label=name)
    ax.set_xlabel("predicted P(pCR)"); ax.set_ylabel("observed pCR rate")
    ax.set_title("Calibration (reliability)"); ax.legend(loc="best")
    return save_figure(fig, out_base)


def fig_forest(exp2_result, out_base: str):
    """Fig 5b: forest plot of Cox HRs (all covariates)."""
    import matplotlib.pyplot as plt
    set_style()
    cov = exp2_result["cox"]["covariates"]
    names = list(cov.keys())
    hr = [cov[n]["hr"] for n in names]
    lo = [cov[n]["ci_low"] for n in names]
    hi = [cov[n]["ci_high"] for n in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(4, 0.5 * len(names) + 1))
    ax.errorbar(hr, y, xerr=[np.array(hr) - lo, np.array(hi) - np.array(hr)],
                fmt="o", color=PALETTE["model"], capsize=3)
    ax.axvline(1.0, ls="--", color=PALETTE["muted"])
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("hazard ratio (95% CI)"); ax.set_title("Cox proportional hazards")
    return save_figure(fig, out_base)


def fig_ablations(exp5_result, out_base: str, metric: str = "auprc"):
    """Fig 8: grouped bar chart of metric deltas with CIs."""
    import matplotlib.pyplot as plt
    set_style()
    names = list(exp5_result.keys())
    deltas = [exp5_result[n]["delta"] for n in names]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(len(names)), deltas, color=CATEGORICAL[: len(names)])
    ax.axhline(0, color=PALETTE["muted"], lw=0.8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(f"delta {metric} (variant - full)")
    ax.set_title("Ablations")
    return save_figure(fig, out_base)


def fig_drug_scatter(exp4_result, out_base: str):
    """Fig 7: predicted vs observed per-arm pCR rate with Spearman rho."""
    import matplotlib.pyplot as plt
    set_style()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(exp4_result["observed"], exp4_result["pred"], color=PALETTE["output"])
    for a, ox, py in zip(exp4_result["arms"], exp4_result["observed"], exp4_result["pred"]):
        ax.annotate(str(a), (ox, py), fontsize=6)
    ax.set_xlabel("observed pCR rate"); ax.set_ylabel("predicted pCR rate")
    ax.set_title(f"In-silico drug comparison (rho={exp4_result['spearman_rho']:.2f})")
    return save_figure(fig, out_base)




def fig_exp3_clusters(exp3_result, out_base: str, latents=None):
    """Exp 3: residual-disease clusters + their clinical associations (FDR).

    Left: 2-D projection of the residual-disease latents coloured by cluster.
    Right: -log10(FDR-adjusted p) per clinical variable, with the 0.05 line, so a
    reader sees immediately whether the clusters are validated or merely exist.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from .style import set_style, save_figure, PALETTE

    if not exp3_result or "error" in exp3_result:
        return None
    set_style()
    assoc = exp3_result.get("clinical_assoc", []) or []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    labels = np.asarray(exp3_result.get("labels", []))
    ax = axes[0]
    if latents is not None and len(labels):
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2, random_state=0).fit_transform(np.asarray(latents))
        colors = [PALETTE["signal"], PALETTE["model"], PALETTE["output"],
                  PALETTE["muted"]]
        for i, c in enumerate(sorted(set(labels.tolist()))):
            m = labels == c
            ax.scatter(xy[m, 0], xy[m, 1], s=14, alpha=0.8,
                       c=colors[i % len(colors)], label=f"cluster {c} (n={int(m.sum())})")
        ax.legend(loc="best", fontsize=6)
    ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")
    ax.set_title(f"residual-disease clusters (k={exp3_result.get('k','?')}, "
                 f"ARI={exp3_result.get('stability_ari', float('nan')):.2f})")

    ax = axes[1]
    if assoc:
        names = [a["variable"] for a in assoc]
        q = [max(a.get("p_fdr", 1.0), 1e-12) for a in assoc]
        y = -np.log10(q)
        sig = [a.get("significant_fdr", False) for a in assoc]
        ax.barh(range(len(names)), y,
                color=[PALETTE["signal"] if s else PALETTE["muted"] for s in sig])
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
        ax.axvline(-np.log10(0.05), ls="--", lw=0.8, c="k")
        ax.set_xlabel(r"$-\log_{10}$ FDR-adjusted $p$")
        if not any(sig):
            ax.set_title("no variable survives FDR\n(clusters unvalidated)", fontsize=7)
        else:
            ax.set_title("cluster vs clinical association", fontsize=7)
    fig.tight_layout()
    out = save_figure(fig, out_base)
    plt.close(fig)
    return out


def fig_killswitch_summary(gate, out_base: str):
    """Fig 3 companion: the gate's separation AUROC against its GO threshold."""
    import matplotlib.pyplot as plt
    from .style import set_style, save_figure, PALETTE

    if not gate:
        return None
    set_style()
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    auc = float(gate.get("separation_auroc", float("nan")))
    thr = float(gate.get("threshold", 0.65))
    ax.barh([0], [auc], color=PALETTE["signal"] if auc >= thr else PALETTE["muted"])
    ax.axvline(thr, ls="--", lw=0.9, c="k")
    ax.axvline(0.5, ls=":", lw=0.8, c="grey")
    ax.set_xlim(0.4, 1.0); ax.set_yticks([])
    ax.set_xlabel("separation AUROC")
    ax.set_title(f"Week-12 gate: {gate.get('decision','?')} "
                 f"({gate.get('latent','z')}, {auc:.3f})")
    fig.tight_layout()
    out = save_figure(fig, out_base)
    plt.close(fig)
    return out


def fig_external(external, out_base: str):
    """External validation: TxMorph vs baseline vs a constant base-rate predictor.

    The constant predictor is drawn deliberately -- it is perfectly calibrated and
    useless, so a calibration win that does not beat it is not evidence of a
    useful model.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from .style import set_style, save_figure, PALETTE

    if not external:
        return None
    set_style()
    metrics = ["auroc", "auprc", "ece", "brier"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    width = 0.38
    x = np.arange(len(metrics))
    for j, (who, color) in enumerate((("txmorph", PALETTE["model"]),
                                      ("baseline", PALETTE["muted"]))):
        vals = [external.get(who, {}).get(m, {}).get("mean", np.nan) for m in metrics]
        los = [external.get(who, {}).get(m, {}).get("lo", np.nan) for m in metrics]
        his = [external.get(who, {}).get(m, {}).get("hi", np.nan) for m in metrics]
        err = np.abs(np.vstack([np.array(vals) - np.array(los),
                                np.array(his) - np.array(vals)]))
        ax.bar(x + (j - 0.5) * width, vals, width, yerr=err, capsize=2,
               color=color, label=who)
    ax.axhline(0.5, ls=":", lw=0.8, c="grey")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("value (95% CI)")
    ax.set_title(f"external validation, n={external.get('n','?')}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = save_figure(fig, out_base)
    plt.close(fig)
    return out


def make_all_figures(results, out_dir: str):
    """Regenerate available figures from a results dict. Missing results are skipped."""
    out = Path(out_dir)
    written = []
    if results.get("killswitch"):
        written.append(fig_killswitch_summary(results["killswitch"], str(out / "fig3_killswitch")))
    if "exp1" in results:
        written.append(fig_calibration(results["exp1"], str(out / "fig4_calibration")))
    if "exp2" in results:
        written.append(fig_forest(results["exp2"], str(out / "fig5_forest")))
    if results.get("exp3"):
        written.append(fig_exp3_clusters(results["exp3"], str(out / "fig6_exp3_clusters"),
                                         latents=results.get("exp3_latents")))
    if "exp4" in results:
        written.append(fig_drug_scatter(results["exp4"], str(out / "fig7_drug")))
    if "exp5" in results:
        written.append(fig_ablations(results["exp5"], str(out / "fig8_ablations")))
    if results.get("external"):
        written.append(fig_external(results["external"], str(out / "fig9_external")))
    return [w for w in written if w]
