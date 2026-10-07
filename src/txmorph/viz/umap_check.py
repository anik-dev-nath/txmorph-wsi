"""Week-12 kill-switch (CLAUDE.md S10).

UMAP of the slide latent colored by pCR; quantify separation via silhouette
score + a patient-level CV logistic-regression AUROC of pCR from that latent.
GO if separation AUROC >~ 0.65 and there is visible structure; else fix the
ENCODER (Blocks 1-2), not the diffusion. Writes the figure and a killswitch.md
GO/NO-GO record.

The gate tests the **encoder's** representation, so it runs on whichever latent
the cohort provides: ``z_post`` in the paired design, ``z_pre`` in the
single-timepoint design (free cohorts), where ``paired_dataset`` fills ``z_post``
with NaN because there is no post-treatment slide.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np


def separation_metrics(z_post, pcr, folds=None, seed: int = 0):
    """Silhouette (pCR vs non-pCR) + CV logistic AUROC of pCR from z_post."""
    from sklearn.metrics import silhouette_score
    from sklearn.linear_model import LogisticRegression
    from ..eval.classification import auroc

    z = np.asarray(z_post, dtype=float)
    y = np.asarray(pcr).astype(int)
    sil = float(silhouette_score(z, y)) if len(np.unique(y)) > 1 else float("nan")

    if folds is None:                                       # 5-fold if not supplied
        from ..utils.cv import assign_folds
        folds = assign_folds([str(i) for i in range(len(y))], y, n_folds=5, seed=seed)
    folds = np.asarray(folds)
    oof = np.zeros(len(y), dtype=float)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(z[tr], y[tr])
        oof[te] = clf.predict_proba(z[te])[:, 1]
    sep_auroc = auroc(oof, y)
    return {"silhouette": sil, "separation_auroc": float(sep_auroc)}


def select_gate_latent(z_pre, z_post):
    """Pick the latent the kill-switch should separate. Returns ``(z, name)``.

    Prefers ``z_post`` (paired design). In single-timepoint mode there is no
    post-treatment slide, so ``z_post`` arrives as all-NaN and the gate falls
    back to ``z_pre`` — the same frozen encoder is under test either way, which
    is what the gate actually decides on. Requires the chosen latent to be
    finite, since silhouette/logistic/UMAP all reject NaN.
    """
    z_pre = np.asarray(z_pre, dtype=float)
    z_post = np.asarray(z_post, dtype=float)
    if z_post.size and np.isfinite(z_post).all():
        return z_post, "z_post"
    if z_pre.size and np.isfinite(z_pre).all():
        return z_pre, "z_pre"
    raise ValueError(
        "kill-switch needs a finite latent: z_post has non-finite values "
        "(expected in single-timepoint mode) and z_pre does too — rebuild "
        "paired.parquet (script 05).")


def killswitch(paired_parquet: str, out_pdf: str, go_threshold: float = 0.65,
               seed: int = 0):
    """Produce the kill-switch UMAP + metrics + GO/NO-GO record. Returns the decision."""
    import matplotlib.pyplot as plt
    import umap
    from ..data.paired_dataset import load_paired
    from .style import set_style, save_figure, PALETTE

    df, z_pre, z_post = load_paired(paired_parquet)
    y = df["pcr"].astype(int).to_numpy()
    folds = df["fold"].to_numpy() if "fold" in df else None

    z_gate, latent_name = select_gate_latent(z_pre, z_post)
    metrics = separation_metrics(z_gate, y, folds, seed=seed)
    metrics["latent"] = latent_name
    emb = umap.UMAP(random_state=seed).fit_transform(z_gate)

    set_style()
    fig, ax = plt.subplots(figsize=(4, 4))
    for label, color, name in [(0, PALETTE["muted"], "non-pCR"),
                               (1, PALETTE["signal"], "pCR")]:
        m = y == label
        ax.scatter(emb[m, 0], emb[m, 1], s=8, c=color, label=name, alpha=0.7)
    decision = "GO" if metrics["separation_auroc"] >= go_threshold else "NO-GO"
    ax.set_title(f"{latent_name} UMAP - sep AUROC={metrics['separation_auroc']:.3f} "
                 f"[{decision}]")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.legend(loc="best")
    save_figure(fig, str(Path(out_pdf).with_suffix("")))
    plt.close(fig)

    _write_record(out_pdf, metrics, decision, go_threshold, latent_name)
    return {"decision": decision, **metrics}


def _write_record(out_pdf: str, metrics: dict, decision: str, thr: float,
                  latent_name: str = "z_post"):
    md = Path(out_pdf).parent / "killswitch.md"
    md.write_text(
        f"# Week-12 kill-switch: {decision}\n\n"
        f"- latent under test: `{latent_name}`"
        f"{' (single-timepoint: no post-treatment slide)' if latent_name == 'z_pre' else ''}\n"
        f"- separation AUROC (CV logistic of pCR from {latent_name}): "
        f"{metrics['separation_auroc']:.3f} (GO threshold {thr})\n"
        f"- silhouette (pCR vs non-pCR): {metrics['silhouette']:.3f}\n\n"
        f"{'Proceed to Experiments 1-4.' if decision == 'GO' else 'STOP: fix the encoder (Blocks 1-2) - revisit SimCLR augmentations, batch size, MPP, backbone. Do NOT tune the diffusion model; a failed separation means the representation lacks signal.'}\n"
    )
