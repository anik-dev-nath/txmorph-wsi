"""Pathway signature scores for the resistance phenotypes in CLAUDE.md S11-Exp3.

Three phenotypes, with the gene sets named in the plan:

* **stromal barricade** — POSTN, FAP, COL1A1 (desmoplastic stroma walling off the tumour)
* **proliferative** — MKI67, CCND1
* **immune exclusion** — CD274 **low** / FOXP3 **high**

The third is directional: CLAUDE.md specifies CD274-low *and* FOXP3-high, so scoring it
as a plain mean would cancel the two genes against each other and produce a phenotype
that is high for the wrong patients. Signatures therefore carry **signed weights**, and
CD274 enters with -1.

Scoring is the mean z-scored log2(TPM+1) over a signature's genes. log2 tames the heavy
right tail of TPM; the z-score is taken across patients per gene so that a signature is
not dominated by whichever of its genes happens to be most highly expressed
(COL1A1 runs orders of magnitude above FAP, for instance).
"""
from __future__ import annotations

import numpy as np

# {signature: {gene_symbol: weight}}. Weight sign encodes direction (see module docstring).
SIGNATURES: dict[str, dict[str, float]] = {
    "stromal_barricade": {"POSTN": 1.0, "FAP": 1.0, "COL1A1": 1.0},
    "proliferative": {"MKI67": 1.0, "CCND1": 1.0},
    "immune_exclusion": {"CD274": -1.0, "FOXP3": 1.0},
}


def signature_names() -> list[str]:
    """Stable signature order, so score columns line up across runs."""
    return sorted(SIGNATURES)


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-gene z-score across patients (axis 0)."""
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + eps)


def score_signatures(matrix, gene_names, signatures: dict | None = None,
                     return_missing: bool = False):
    """Expression matrix -> (n_cases, n_signatures) signature scores.

    ``matrix`` is (n_cases, n_genes) TPM, ``gene_names`` its columns. Returns scores in
    ``signature_names()`` order. Genes absent from ``gene_names`` are dropped from their
    signature and reported via ``return_missing``; a signature that loses **all** its
    genes raises, because silently emitting zeros there would look like a measured
    phenotype rather than missing data.
    """
    sigs = signatures if signatures is not None else SIGNATURES
    X = np.asarray(matrix, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expression matrix must be 2-D, got shape {X.shape}")
    idx = {g: i for i, g in enumerate(gene_names)}

    Z = _zscore(np.log2(X + 1.0))
    names = sorted(sigs)
    scores = np.zeros((X.shape[0], len(names)), dtype=np.float64)
    missing: dict[str, list[str]] = {}

    for k, name in enumerate(names):
        weights = sigs[name]
        present = [(idx[g], w) for g, w in weights.items() if g in idx]
        absent = [g for g in weights if g not in idx]
        if absent:
            missing[name] = sorted(absent)
        if not present:
            raise KeyError(
                f"signature {name!r} has none of its genes {sorted(weights)} in the "
                "expression matrix; scoring it would fabricate a phenotype")
        cols = np.array([i for i, _ in present])
        w = np.array([wt for _, wt in present], dtype=np.float64)
        # Normalise by |w| so a signature that lost a gene stays on the same scale.
        scores[:, k] = (Z[:, cols] * w).sum(axis=1) / np.abs(w).sum()

    if return_missing:
        return scores, names, missing
    return scores, names
