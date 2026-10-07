"""Read GDC STAR-Counts RNA-seq quantification into a (cases x genes) TPM matrix.

GDC's ``*.rna_seq.augmented_star_gene_counts.tsv`` looks like::

    # gene-model: GENCODE v36
    gene_id	gene_name	gene_type	unstranded	...	tpm_unstranded	fpkm_unstranded	...
    N_unmapped			2731233	...
    N_multimapping		6675542	...
    N_noFeature			6168054	...
    N_ambiguous			7929985	...
    ENSG00000000003.15	TSPAN6	protein_coding	2026	...	21.7171	...

Two things bite if handled naively: the file opens with a ``#`` comment line before the
header, and the first four data rows are STAR *alignment summary* counters, not genes.
Both are skipped here — summing them into an expression matrix would silently corrupt
every downstream normalisation.

TPM is used rather than raw counts because the bridge compares signature scores across
patients, which needs within-sample library-size and gene-length normalisation.
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np

# STAR alignment-summary rows that precede the genes; never expression values.
_SUMMARY_ROWS = {"N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"}
_VALUE_COL = "tpm_unstranded"


def read_star_counts(path: str | Path, value_col: str = _VALUE_COL,
                     gene_types: set[str] | None = None) -> dict[str, float]:
    """One STAR-Counts TSV -> ``{gene_symbol: value}``.

    ``gene_types`` restricts to e.g. ``{"protein_coding"}``. Duplicate symbols (several
    Ensembl ids map to one symbol) are summed, which is the usual convention and keeps
    the matrix keyed by the symbols the signature definitions use.
    """
    out: dict[str, float] = {}
    with open(path, newline="") as fh:
        # Skip leading '#' comment lines, then let DictReader take the header row.
        pos = fh.tell()
        line = fh.readline()
        while line.startswith("#"):
            pos = fh.tell()
            line = fh.readline()
        fh.seek(pos)
        reader = csv.DictReader(fh, delimiter="\t")
        if value_col not in (reader.fieldnames or []):
            raise ValueError(
                f"{path}: no {value_col!r} column; found {reader.fieldnames}")
        for row in reader:
            gid = (row.get("gene_id") or "").strip()
            if not gid or gid in _SUMMARY_ROWS:
                continue
            if gene_types and (row.get("gene_type") or "").strip() not in gene_types:
                continue
            sym = (row.get("gene_name") or "").strip()
            raw = (row.get(value_col) or "").strip()
            if not sym or not raw:
                continue
            try:
                out[sym] = out.get(sym, 0.0) + float(raw)
            except ValueError:                      # non-numeric cell: skip the gene
                continue
    if not out:
        raise ValueError(f"{path}: parsed 0 genes — wrong file layout?")
    return out


def _find_expr_file(expr_root: Path, file_id: str, file_name: str) -> Path | None:
    """gdc-client writes ``<expr_root>/<file_id>/<file_name>``; tolerate flat layouts."""
    cand = expr_root / file_id / file_name
    if cand.exists():
        return cand
    for alt in (expr_root / file_name, ):
        if alt.exists():
            return alt
    hits = list((expr_root / file_id).glob("*.tsv")) if (expr_root / file_id).is_dir() else []
    return hits[0] if hits else None


def build_expression_matrix(pairs_csv: str | Path, expr_root: str | Path,
                            genes: list[str] | None = None,
                            value_col: str = _VALUE_COL):
    """Assemble the per-case expression matrix for the bridge's training pairs.

    ``pairs_csv`` is the join written by ``build_tcga_bridge_manifest.py``
    (case_submitter_id, slide_file_id, slide_file_name, expr_file_id, expr_file_name).

    Returns ``(cases, gene_names, matrix)`` where ``matrix`` is (n_cases, n_genes)
    float32. Cases whose expression file has not been downloaded are skipped and
    reported by the caller via the returned ``cases`` list — the matrix and the case
    list always stay aligned, which is what keeps expression matched to the right
    patient downstream.
    """
    expr_root = Path(expr_root)
    rows = list(csv.DictReader(open(pairs_csv, newline="")))

    per_case: dict[str, dict[str, float]] = {}
    for r in rows:
        case = r["case_submitter_id"]
        f = _find_expr_file(expr_root, r["expr_file_id"], r["expr_file_name"])
        if f is None:
            continue
        per_case[case] = read_star_counts(f, value_col=value_col)

    cases = sorted(per_case)
    if not cases:
        raise FileNotFoundError(
            f"no expression files found under {expr_root} for {len(rows)} pairs")

    if genes is None:                    # genes present in EVERY case, for a dense matrix
        common = set.intersection(*(set(per_case[c]) for c in cases))
        genes = sorted(common)
    mat = np.zeros((len(cases), len(genes)), dtype=np.float32)
    for i, c in enumerate(cases):
        g = per_case[c]
        for j, sym in enumerate(genes):
            mat[i, j] = g.get(sym, 0.0)
    return cases, list(genes), mat
