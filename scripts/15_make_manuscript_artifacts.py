"""15_make_manuscript_artifacts.py - CONSORT diagram + cohort/encoder tables.

Closes the CLAUDE.md S15 items that were specified but never generated: the
CONSORT flow diagram, a T1 cohort table that actually reports the subtype
distribution, and a T5 encoder-comparison table for the elimination study.

Every count is **derived from disk** (raw slide files, tile dirs, embedding
caches, manifests, paired.parquet) rather than hardcoded, so the diagram cannot
drift away from the data -- CLAUDE.md S15: "generate programmatically from
paired.parquet provenance", and S13's cohort accounting requirement.

Usage (server):
    PYTHONPATH=src python scripts/15_make_manuscript_artifacts.py \
        --out-dir /home/anik-server/runs/manuscript
"""
import argparse
import json
from pathlib import Path

DATA = "/home/anik-server/data"


def _count_files(pattern_dir: str, suffix: str = ".svs") -> int:
    p = Path(pattern_dir)
    return len(list(p.rglob(f"*{suffix}"))) if p.is_dir() else 0


def _count_dir(d: str) -> int:
    p = Path(d)
    return len([x for x in p.iterdir()]) if p.is_dir() else 0


def gather_provenance(data_root: str) -> dict:
    """Derive every cohort count from what is actually on disk."""
    d = Path(data_root)
    raw = d / "raw_wsi"
    prov = {
        "impress": {
            "raw_all": _count_files(str(raw / "impress")),
            "raw_he": len(list((raw / "impress").rglob("*_HE.svs"))) if (raw / "impress").is_dir() else 0,
            "raw_ihc": len(list((raw / "impress").rglob("*_IHC.svs"))) if (raw / "impress").is_dir() else 0,
            "embedded": _count_dir(str(d / "tile_emb")),
        },
        "postnat": {
            "raw_all": _count_files(str(raw / "post_nat_brca")),
            "embedded": _count_dir(str(d / "tile_emb_postnat")),
        },
        "her2_rois": {
            "raw_all": _count_files(str(raw / "her2_tumor_rois")),
            "embedded": _count_dir(str(d / "tile_emb_external")),
        },
        "tiles_total": _count_dir(str(d / "tiles")),
    }
    return prov


def consort_diagram(prov: dict, paired_df, out_base: str):
    """Proper CONSORT flow: boxes, arrows, and side exclusions."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from txmorph.viz.style import set_style, save_figure

    set_style()
    fig, ax = plt.subplots(figsize=(7.0, 7.2))
    ax.set_xlim(0, 10); ax.set_ylim(1.7, 14.7); ax.axis("off")

    imp, pn, h2 = prov["impress"], prov["postnat"], prov["her2_rois"]
    n_final = len(paired_df)

    main = [
        (13.6, "Open breast-cancer WSI cohorts screened\n"
               f"IMPRESS {imp['raw_all']} · Post-NAT-BRCA {pn['raw_all']} · "
               f"HER2-TUMOR-ROIS {h2['raw_all']} slide files"),
        (11.2, "Primary cohort: IMPRESS\n"
               f"{imp['raw_he']} H&E whole-slide images"),
        (8.8,  f"Tiled and quality-filtered\n{prov['tiles_total']} slide tile-sets "
               f"(all cohorts), 256×256 @ 20×"),
        (6.4,  f"Tile embeddings extracted (frozen encoder)\n{imp['embedded']} IMPRESS slides"),
        (4.0,  f"ANALYSED — primary endpoint pCR\n{n_final} patients"),
    ]
    excl = [
        (12.4, f"Excluded: IHC slides not read by\nthe pipeline (n={imp['raw_ihc']})"),
        (10.0, "Excluded from primary analysis:\n"
               f"Post-NAT-BRCA (n={pn['embedded']}) — residual\n"
               "disease only, non-pCR by construction"),
        (7.6,  f"Held out as external validation:\n"
               f"HER2-TUMOR-ROIS Yale arm (n={h2['embedded']})\n"
               "excluded from encoder pretraining"),
    ]

    for y, txt in main:
        ax.add_patch(FancyBboxPatch((0.4, y - 0.85), 5.2, 1.7, boxstyle="round,pad=0.12",
                                    fc="#EAF4F4", ec="#2A9D8F", lw=1.2))
        ax.text(3.0, y, txt, ha="center", va="center", fontsize=7.4)
    for i in range(len(main) - 1):
        ax.add_patch(FancyArrowPatch((3.0, main[i][0] - 0.9), (3.0, main[i + 1][0] + 0.9),
                                     arrowstyle="-|>", mutation_scale=11, color="#444"))
    for y, txt in excl:
        ax.add_patch(FancyBboxPatch((6.1, y - 0.72), 3.6, 1.44, boxstyle="round,pad=0.1",
                                    fc="#FDF3E7", ec="#E9A23B", lw=1.0))
        ax.text(7.9, y, txt, ha="center", va="center", fontsize=6.6)
        ax.add_patch(FancyArrowPatch((3.0, y), (6.05, y), arrowstyle="-|>",
                                     mutation_scale=9, color="#888"))

    pcr = int(paired_df["pcr"].astype(int).sum())
    ax.text(3.0, 2.35, f"pCR {pcr}/{n_final} ({pcr/n_final:.1%})  ·  "
                       "5-fold patient-level CV", ha="center", fontsize=7,
            style="italic", color="#555")
    ax.set_title("CONSORT cohort flow — TxMorph-WSI", fontsize=9, pad=6)
    return save_figure(fig, out_base)


def table_cohort_full(paired_df, prov: dict, out_base: str):
    """T1 that reports the actual subtype and regimen distribution, per cohort.

    Replaces viz.tables.table_cohort, which emitted a *count* of distinct
    subtypes and a median RFS column that is NaN for every row (IMPRESS carries
    no survival endpoint).
    """
    import pandas as pd
    rows = []
    for cohort, g in paired_df.groupby("cohort"):
        n = len(g)
        sub = g["receptor_subtype"].value_counts()
        drug = g["drug_id"].value_counts()
        rows.append({
            "cohort": cohort,
            "n_patients": n,
            "n_slides": prov["impress"]["embedded"] if cohort == "IMPRESS" else n,
            "pCR n (%)": f"{int(g.pcr.sum())} ({g.pcr.astype(int).mean():.1%})",
            "subtype distribution": "; ".join(f"{k} {v}" for k, v in sub.items()),
            "regimen": "; ".join(f"{k} {v}" for k, v in drug.items()),
            "survival endpoint": "none" if g.get("rfs_time", pd.Series(dtype=float)).isna().all() else "present",
        })
    rows.append({"cohort": "Post-NAT-BRCA (Exp 3)", "n_patients": prov["postnat"]["embedded"],
                 "n_slides": prov["postnat"]["raw_all"], "pCR n (%)": "0 (0.0%) by design",
                 "subtype distribution": "residual disease only",
                 "regimen": "NAC (mixed)", "survival endpoint": "none"})
    rows.append({"cohort": "HER2-ROIS Yale (external)", "n_patients": prov["her2_rois"]["embedded"],
                 "n_slides": prov["her2_rois"]["embedded"],
                 "pCR n (%)": "n/a — endpoint is trastuzumab response",
                 "subtype distribution": "HER2+", "regimen": "trastuzumab",
                 "survival endpoint": "none"})
    from txmorph.viz.tables import _write
    return _write(pd.DataFrame(rows), out_base)


def table_encoder_comparison(out_base: str, results: dict | None = None):
    """T5: the encoder elimination study (docs/STATUS_FOR_REVIEW.md S2)."""
    import pandas as pd
    rows = results or [
        {"encoder": "SimCLR v5 (ResNet-34, from scratch)", "pretrain": "1.33M tiles / ~310 slides",
         "dim": 512, "NCT-CRC probe": "0.9291", "subtype AUROC": "0.958",
         "pCR AUROC (95% CI)": "0.691 (0.598-0.779)"},
        {"encoder": "SimCLR v6 (more training)", "pretrain": "1.33M tiles / ~310 slides",
         "dim": 512, "NCT-CRC probe": "0.9050", "subtype AUROC": "n/a",
         "pCR AUROC (95% CI)": "not extracted (probe regressed)"},
        {"encoder": "Phikon-v2 (ViT-L/16, foundation)", "pretrain": "456M tiles / ~60k slides",
         "dim": 1024, "NCT-CRC probe": "0.9478", "subtype AUROC": "0.976",
         "pCR AUROC (95% CI)": "0.669 (0.575-0.762)"},
    ]
    from txmorph.viz.tables import _write
    return _write(pd.DataFrame(rows), out_base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA)
    ap.add_argument("--paired", default=f"{DATA}/paired.parquet")
    ap.add_argument("--out-dir", default="/home/anik-server/runs/manuscript")
    args = ap.parse_args()

    import pandas as pd
    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    prov = gather_provenance(args.data_root)
    print("[provenance]", json.dumps(prov, indent=2))
    paired = pd.read_parquet(args.paired)

    f = consort_diagram(prov, paired, str(out / "figures" / "consort_flow"))
    print("[consort]", f)
    t1 = table_cohort_full(paired, prov, str(out / "tables" / "T1_cohort"))
    print("[T1]", t1)
    t5 = table_encoder_comparison(str(out / "tables" / "T5_encoder_comparison"))
    print("[T5]", t5)
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print("[done]", out)


if __name__ == "__main__":
    main()
