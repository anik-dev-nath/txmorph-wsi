"""STAR-Counts parsing + signature scoring (morphology->expression bridge).

The TSVs written here are **format fixtures**, not cohort data: they exist to pin the
GDC file layout (leading '#' comment, four STAR summary rows before the genes) and the
scoring maths. No fixture value ever reaches a reported number — real scores come from
the downloaded TCGA-BRCA STAR-Counts files.
"""
import csv

import numpy as np
import pytest

from txmorph.expression.star_counts import read_star_counts, build_expression_matrix
from txmorph.expression.signatures import SIGNATURES, score_signatures, signature_names

HEADER = ["gene_id", "gene_name", "gene_type", "unstranded", "stranded_first",
          "stranded_second", "tpm_unstranded", "fpkm_unstranded", "fpkm_uq_unstranded"]


def _write_star_tsv(path, genes, comment=True):
    """Reproduce GDC's exact layout: comment line, header, STAR summary rows, genes."""
    with open(path, "w", newline="") as f:
        if comment:
            f.write("# gene-model: GENCODE v36\n")
        w = csv.writer(f, delimiter="\t")
        w.writerow(HEADER)
        for n in ("N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"):
            w.writerow([n, "", "", "12345", "12345", "12345", "", "", ""])
        for i, (sym, tpm) in enumerate(genes.items()):
            w.writerow([f"ENSG{i:011d}.1", sym, "protein_coding",
                        "100", "50", "50", f"{tpm:.4f}", "1.0", "1.0"])


def test_skips_comment_and_star_summary_rows(tmp_path):
    """The four N_* counters must never enter the matrix as genes."""
    p = tmp_path / "a.tsv"
    _write_star_tsv(p, {"POSTN": 10.0, "FAP": 5.0, "COL1A1": 2.5})
    got = read_star_counts(p)
    assert set(got) == {"POSTN", "FAP", "COL1A1"}
    assert not any(k.startswith("N_") for k in got)
    assert got["POSTN"] == pytest.approx(10.0)


def test_duplicate_symbols_are_summed(tmp_path):
    """Several Ensembl ids can map to one symbol; they aggregate, not overwrite."""
    p = tmp_path / "dup.tsv"
    with open(p, "w", newline="") as f:
        f.write("# gene-model: GENCODE v36\n")
        w = csv.writer(f, delimiter="\t")
        w.writerow(HEADER)
        w.writerow(["ENSG1.1", "FAP", "protein_coding", "1", "1", "1", "3.0", "1", "1"])
        w.writerow(["ENSG2.1", "FAP", "protein_coding", "1", "1", "1", "4.0", "1", "1"])
    assert read_star_counts(p)["FAP"] == pytest.approx(7.0)


def test_gene_type_filter(tmp_path):
    """Filtering to a type keeps matching genes; filtering everything out is an error."""
    p = tmp_path / "t.tsv"
    _write_star_tsv(p, {"POSTN": 1.0})                 # written as protein_coding
    assert set(read_star_counts(p, gene_types={"protein_coding"})) == {"POSTN"}
    with pytest.raises(ValueError, match="parsed 0 genes"):
        read_star_counts(p, gene_types={"lncRNA"})


def test_missing_value_column_raises(tmp_path):
    p = tmp_path / "bad.tsv"
    with open(p, "w", newline="") as f:
        csv.writer(f, delimiter="\t").writerow(["gene_id", "gene_name"])
        csv.writer(f, delimiter="\t").writerow(["ENSG1.1", "FAP"])
    with pytest.raises(ValueError, match="tpm_unstranded"):
        read_star_counts(p)


def test_build_matrix_keeps_cases_aligned(tmp_path):
    """Matrix rows must correspond to the returned case list, in order."""
    expr = tmp_path / "expr"
    genes = ["POSTN", "FAP", "COL1A1", "MKI67", "CCND1", "CD274", "FOXP3"]
    rows = []
    for i, case in enumerate(["TCGA-AA-0001", "TCGA-BB-0002", "TCGA-CC-0003"]):
        fid = f"file{i}"
        (expr / fid).mkdir(parents=True)
        _write_star_tsv(expr / fid / "q.tsv", {g: float(i + 1) * (j + 1)
                                               for j, g in enumerate(genes)})
        rows.append({"case_submitter_id": case, "slide_file_id": f"s{i}",
                     "slide_file_name": f"s{i}.svs", "expr_file_id": fid,
                     "expr_file_name": "q.tsv"})
    pairs = tmp_path / "pairs.csv"
    with open(pairs, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    cases, gene_names, mat = build_expression_matrix(pairs, expr)
    assert cases == ["TCGA-AA-0001", "TCGA-BB-0002", "TCGA-CC-0003"]
    assert mat.shape == (3, len(gene_names))
    # case i has POSTN = (i+1)*1 by construction
    j = gene_names.index("POSTN")
    assert list(mat[:, j]) == pytest.approx([1.0, 2.0, 3.0])


def test_missing_expression_files_raise(tmp_path):
    pairs = tmp_path / "pairs.csv"
    with open(pairs, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_submitter_id", "slide_file_id",
                                          "slide_file_name", "expr_file_id",
                                          "expr_file_name"])
        w.writeheader()
        w.writerow({"case_submitter_id": "TCGA-AA-0001", "slide_file_id": "s",
                    "slide_file_name": "s.svs", "expr_file_id": "nope",
                    "expr_file_name": "q.tsv"})
    (tmp_path / "expr").mkdir()
    with pytest.raises(FileNotFoundError):
        build_expression_matrix(pairs, tmp_path / "expr")


# ---------------------------------------------------------------- signatures


def test_immune_exclusion_is_directional():
    """CD274-low / FOXP3-high: the two genes must not cancel out.

    Scored as a plain mean, a patient high in both would look immune-excluded. The
    signed weights make CD274 count against the score.
    """
    genes = ["CD274", "FOXP3"]
    # patient 0: CD274 low, FOXP3 high (excluded). patient 1: the reverse.
    mat = np.array([[1.0, 100.0], [100.0, 1.0]])
    scores, names = score_signatures(mat, genes,
                                     {"immune_exclusion": SIGNATURES["immune_exclusion"]})
    assert scores[0, 0] > scores[1, 0]
    assert SIGNATURES["immune_exclusion"]["CD274"] < 0


def test_stromal_signature_tracks_its_genes():
    genes = ["POSTN", "FAP", "COL1A1"]
    mat = np.array([[1.0, 1.0, 1.0], [50.0, 50.0, 50.0], [10.0, 10.0, 10.0]])
    scores, _ = score_signatures(mat, genes,
                                 {"stromal_barricade": SIGNATURES["stromal_barricade"]})
    assert scores[1, 0] > scores[2, 0] > scores[0, 0]


def test_zscore_prevents_one_gene_dominating():
    """COL1A1 runs orders of magnitude above FAP; per-gene z-scoring equalises them."""
    genes = ["POSTN", "FAP", "COL1A1"]
    # FAP separates the patients; COL1A1 is huge but constant, so it carries no signal.
    mat = np.array([[1.0, 1.0, 9e4], [1.0, 500.0, 9e4]])
    scores, _ = score_signatures(mat, genes,
                                 {"stromal_barricade": SIGNATURES["stromal_barricade"]})
    assert scores[1, 0] > scores[0, 0]


def test_signature_with_no_genes_present_raises():
    """Emitting zeros for an absent signature would look like a measured phenotype."""
    with pytest.raises(KeyError, match="fabricate"):
        score_signatures(np.ones((2, 1)), ["ACTB"],
                         {"stromal_barricade": SIGNATURES["stromal_barricade"]})


def test_partial_signature_reports_missing_genes():
    mat = np.array([[1.0, 2.0], [3.0, 4.0]])
    scores, names, missing = score_signatures(
        mat, ["POSTN", "FAP"],
        {"stromal_barricade": SIGNATURES["stromal_barricade"]}, return_missing=True)
    assert missing["stromal_barricade"] == ["COL1A1"]
    assert scores.shape == (2, 1)


def test_all_signatures_score_together():
    genes = ["POSTN", "FAP", "COL1A1", "MKI67", "CCND1", "CD274", "FOXP3"]
    rng = np.random.default_rng(0)
    mat = rng.lognormal(size=(12, len(genes))) * 10
    scores, names = score_signatures(mat, genes)
    assert names == signature_names()
    assert scores.shape == (12, 3)
    assert np.isfinite(scores).all()
