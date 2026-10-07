"""Experiment 3 - Resistance clusters (CLAUDE.md S11-Exp3).

Residuals r (non-pCR) -> PCA (~90% variance) -> GMM, K by BIC over 2-8. Validate:
(a) correlate cluster membership with TCGA expression signatures (FDR); (b) per-
cluster morphological features; (c) pathologist face-validity on nearest tiles.
Report cluster stability (bootstrap ARI). Clustering alone proves nothing.
"""
from __future__ import annotations

import numpy as np

from ..eval.stats import benjamini_hochberg


def cluster_residuals(residuals, var_keep: float = 0.90, k_range=range(2, 9),
                      seed: int = 0):
    """PCA(retain var_keep) -> GMM selected by BIC. Returns labels + fitted objects."""
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture

    R = np.asarray(residuals, dtype=float)
    pca = PCA(n_components=var_keep, svd_solver="full", random_state=seed)
    X = pca.fit_transform(R)

    best = None
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              random_state=seed, n_init=3)
        gmm.fit(X)
        bic = gmm.bic(X)
        if best is None or bic < best[0]:
            best = (bic, k, gmm)
    _, k_best, gmm = best
    labels = gmm.predict(X)
    return {"labels": labels, "k": k_best, "pca": pca, "gmm": gmm, "X": X}


def cluster_stability(residuals, k: int, n_boot: int = 100, var_keep: float = 0.90,
                      seed: int = 0):
    """Bootstrap ARI: refit on resamples, compare to the full-data labels."""
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import adjusted_rand_score

    R = np.asarray(residuals, dtype=float)
    base = cluster_residuals(R, var_keep=var_keep, k_range=[k], seed=seed)["labels"]
    rng = np.random.default_rng(seed)
    aris = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(R), size=len(R))
        pca = PCA(n_components=var_keep, svd_solver="full", random_state=seed)
        X = pca.fit_transform(R[idx])
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              random_state=seed, n_init=1).fit(X)
        aris.append(adjusted_rand_score(base[idx], gmm.predict(X)))
    return {"ari_mean": float(np.mean(aris)), "ari_std": float(np.std(aris))}


def associate_expression(labels, expression, gene_names=None):
    """One-vs-rest association of each cluster with each gene's expression (FDR).

    labels: (n,) cluster ids. expression: (n, g) matrix. Returns per (cluster,gene)
    p-values, BH q-values, and rejection flags.
    """
    from scipy import stats

    labels = np.asarray(labels)
    E = np.asarray(expression, dtype=float)
    g = E.shape[1]
    names = gene_names or [f"gene{j}" for j in range(g)]
    rows, pvals = [], []
    for c in sorted(np.unique(labels)):
        member = (labels == c).astype(float)
        for j in range(g):
            r, p = stats.pointbiserialr(member, E[:, j])
            rows.append({"cluster": int(c), "gene": names[j], "r": float(r), "p": float(p)})
            pvals.append(p)
    rejected, q = benjamini_hochberg(pvals)
    for row, rj, qq in zip(rows, rejected, q):
        row["q"] = float(qq)
        row["significant"] = bool(rj)
    return rows


def run_exp3(records, expression=None, gene_names=None, seed: int = 0):
    """records from inference.run; use non-pCR residuals only."""
    res = [r["residual"] for r in records if r["pcr"] == 0 and r["residual"] is not None]
    if len(res) < 4:
        return {"error": "too few non-pCR residuals to cluster"}
    R = np.stack(res)
    clus = cluster_residuals(R, seed=seed)
    stab = cluster_stability(R, clus["k"], seed=seed)
    out = {"k": clus["k"], "labels": clus["labels"], "stability": stab}
    if expression is not None:
        out["expression_assoc"] = associate_expression(clus["labels"], expression,
                                                       gene_names)
    return out


def run(cfg):
    from ..data.paired_dataset import load_paired
    from ..inference.run import run_inference
    df, z_pre, z_post = load_paired(cfg.paths.paired)
    records = run_inference(z_pre, z_post, df["drug_id"].to_numpy(),
                            df["pcr"].to_numpy(), df["fold"].to_numpy(),
                            cfg.paths.ckpt, out_dir=f"{cfg.paths.runs}/samples")
    return run_exp3(records)


# ---------------------------------------------------------------- single-timepoint
# Under the free-cohort pivot there is no z_post, so the residual r = z_post_obs -
# z_post_bar that this experiment was built on does not exist and run_exp3 returns
# "too few non-pCR residuals". The pivot's intent (CLAUDE.md data-strategy note,
# data/PROVENANCE.md) is to cluster **residual-disease morphology itself** on
# Post-NAT-BRCA, whose patients are selected on having residual disease and are
# therefore all non-pCR by construction -- the same population the residual was
# meant to describe, observed directly rather than inferred.
#
# Expression validation is unavailable: the morphology->expression bridge failed
# its permutation null (see scripts/10), so clusters are validated against the
# clinical variables Post-NAT actually ships -- residual cancer burden components
# (cellularity, residual size, nodal involvement), response category and receptor
# subtype -- plus bootstrap-ARI stability. Weaker than transcriptomic grounding,
# and it must be reported as such.

_ASSOC_CONTINUOUS = ("cellularity", "residual_size_mm", "tumor_bed_mm", "ln_involved")
_ASSOC_CATEGORICAL = ("response_bc", "receptor_subtype")


def _first_float(v):
    """Parse a possibly multi-focus measurement like '4.5,1' -> 4.5 (largest focus).

    Post-NAT records multifocal residual disease as comma-separated sizes; taking
    the largest focus is the standard RCB convention.
    """
    if v is None:
        return np.nan
    s = str(v).strip()
    if not s:
        return np.nan
    try:
        return max(float(x) for x in s.split(",") if x.strip())
    except ValueError:
        return np.nan


def associate_clinical(labels, meta):
    """Associate cluster membership with Post-NAT clinical variables (BH-FDR).

    Continuous variables use Kruskal-Wallis (no normality assumption, robust at
    this n); categorical use a chi-square test of independence. Returns a list of
    per-variable records with raw p, FDR-adjusted p and the rejection flag.
    """
    from scipy import stats as sps

    labels = np.asarray(labels)
    rows, pvals = [], []
    for col in _ASSOC_CONTINUOUS:
        if col not in meta:
            continue
        vals = np.array([_first_float(v) for v in meta[col]], dtype=float)
        groups = [vals[(labels == c) & np.isfinite(vals)] for c in np.unique(labels)]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue
        p = float(sps.kruskal(*groups).pvalue)
        rows.append({"variable": col, "test": "kruskal", "p": p,
                     "group_medians": [float(np.median(g)) for g in groups]})
        pvals.append(p)
    for col in _ASSOC_CATEGORICAL:
        if col not in meta:
            continue
        vals = np.array([str(v) for v in meta[col]])
        ok = vals != ""
        if ok.sum() < 4:
            continue
        cats = sorted(set(vals[ok]))
        table = np.array([[int(((labels == c) & (vals == k) & ok).sum()) for k in cats]
                          for c in np.unique(labels)])
        if table.shape[0] < 2 or table.shape[1] < 2 or table.sum() == 0:
            continue
        p = float(sps.chi2_contingency(table + 1e-9).pvalue)
        rows.append({"variable": col, "test": "chi2", "p": p,
                     "table": table.tolist(), "categories": cats})
        pvals.append(p)

    if pvals:
        rejected, adj = benjamini_hochberg(pvals)
        for r, q, rej in zip(rows, adj, rejected):
            r["p_fdr"] = float(q)
            r["significant_fdr"] = bool(rej)
    return rows


def run_exp3_residual_disease(Z, meta, seed: int = 0, var_keep: float = 0.90,
                              max_k: int = 6):
    """Cluster residual-disease latents and validate against clinical variables.

    ``Z``    (n_patients, d) slide latents of post-treatment residual-disease WSIs.
    ``meta`` dict of column -> sequence, aligned to Z's rows.

    ``max_k`` is capped relative to n: fitting an 8-component GMM to ~50 patients
    produces components with fewer members than dimensions, which BIC will happily
    select and which mean nothing.
    """
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0]
    if n < 12:
        return {"error": f"only {n} residual-disease cases; too few to cluster"}
    k_hi = int(min(max_k, max(2, n // 12)))
    clus = cluster_residuals(Z, var_keep=var_keep, k_range=range(2, k_hi + 1), seed=seed)
    stab = cluster_stability(Z, clus["k"], var_keep=var_keep, seed=seed)
    out = {"n": int(n), "k": int(clus["k"]), "k_range_searched": [2, k_hi],
           "labels": np.asarray(clus["labels"]).tolist(),
           "stability_ari": float(stab["ari_mean"]),
           "stability_ari_std": float(stab["ari_std"]),
           "cluster_sizes": np.bincount(np.asarray(clus["labels"])).tolist()}
    out["clinical_assoc"] = associate_clinical(clus["labels"], meta)
    out["validation"] = ("clinical only (RCB components, response category, receptor "
                         "subtype); transcriptomic validation unavailable because the "
                         "morphology->expression bridge did not beat its permutation "
                         "null")
    return out
