"""17_supplementary.py - Fig 6 and the supplementary tables (CLAUDE.md S15).

Generates the S15 items that were specified but never produced:
  * Fig 6 -- Exp-3 residual-disease clusters and their clinical associations;
  * S1 -- full hyperparameters, read from configs/ rather than transcribed;
  * S2 -- software versions (pip freeze + GPU + git SHA), for the repro package;
  * S3 -- per-fold results, so readers see the spread behind each pooled CI;
  * S4 -- per-cluster clinical associations with BH-FDR (Exp 3's validation table;
          CLAUDE.md S11 lists it as the result, since clustering alone proves nothing).
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def supp_hyperparameters(config_dir: str, out_base: str):
    """S1: flatten every config file into one table -- no transcription by hand."""
    import pandas as pd
    import yaml

    rows = []
    for p in sorted(Path(config_dir).rglob("*.yaml")):
        try:
            cfg = yaml.safe_load(p.read_text()) or {}
        except Exception as e:
            rows.append({"file": p.name, "parameter": "<unreadable>", "value": str(e)})
            continue

        def walk(d, prefix=""):
            for k, v in (d or {}).items():
                key = f"{prefix}{k}"
                if isinstance(v, dict):
                    walk(v, f"{key}.")
                else:
                    rows.append({"file": str(p.relative_to(config_dir)),
                                 "parameter": key, "value": str(v)})
        walk(cfg)
    from txmorph.viz.tables import _write
    return _write(pd.DataFrame(rows), out_base)


def supp_software(out_base: str):
    """S2: exact environment -- versions, GPU, git SHA."""
    import pandas as pd
    rows = []
    try:
        freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True,
                                timeout=120).stdout.splitlines()
        keep = ("torch", "numpy", "pandas", "scikit-learn", "scipy", "h5py",
                "pyarrow", "matplotlib", "lifelines", "statsmodels", "umap",
                "transformers", "timm", "openslide", "rdkit", "Pillow")
        for line in freeze:
            if any(line.lower().startswith(k.lower()) for k in keep):
                name, _, ver = line.partition("==")
                rows.append({"component": name, "version": ver or "n/a"})
    except Exception as e:
        rows.append({"component": "pip freeze", "version": f"failed: {e}"})
    try:
        import torch
        rows.append({"component": "CUDA (torch)", "version": torch.version.cuda or "n/a"})
        if torch.cuda.is_available():
            rows.append({"component": "GPU", "version": torch.cuda.get_device_name(0)})
    except Exception:
        pass
    for label, cmd in (("git SHA", ["git", "rev-parse", "HEAD"]),
                       ("git branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"])):
        try:
            rows.append({"component": label,
                         "version": subprocess.run(cmd, capture_output=True, text=True,
                                                   timeout=30).stdout.strip()})
        except Exception:
            pass
    from txmorph.viz.tables import _write
    return _write(pd.DataFrame(rows), out_base)


def supp_per_fold(biomarkers_path: str, out_base: str):
    """S3: per-fold AUROC/AUPRC/n, behind the pooled cross-validated estimate."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score, average_precision_score

    recs = json.loads(Path(biomarkers_path).read_text())
    df = pd.DataFrame(recs)
    rows = []
    for k, g in df.groupby("fold"):
        y, p = g.pcr.to_numpy().astype(int), g.p_pcr.to_numpy()
        rows.append({
            "fold": int(k), "n": len(g), "n_pCR": int(y.sum()),
            "base_rate": f"{y.mean():.3f}",
            "AUROC": f"{roc_auc_score(y, p):.3f}" if len(np.unique(y)) > 1 else "n/a",
            "AUPRC": f"{average_precision_score(y, p):.3f}" if len(np.unique(y)) > 1 else "n/a",
        })
    y, p = df.pcr.to_numpy().astype(int), df.p_pcr.to_numpy()
    rows.append({"fold": "pooled", "n": len(df), "n_pCR": int(y.sum()),
                 "base_rate": f"{y.mean():.3f}",
                 "AUROC": f"{roc_auc_score(y, p):.3f}",
                 "AUPRC": f"{average_precision_score(y, p):.3f}"})
    from txmorph.viz.tables import _write
    return _write(pd.DataFrame(rows), out_base)


def supp_exp3_assoc(exp3_result: dict, out_base: str):
    """S4: Exp-3 cluster/clinical associations with BH-FDR -- the validation table."""
    import pandas as pd
    rows = []
    for a in exp3_result.get("clinical_assoc", []) or []:
        rows.append({
            "variable": a.get("variable"), "test": a.get("test"),
            "p_raw": f"{a.get('p', float('nan')):.4f}",
            "p_FDR": f"{a.get('p_fdr', float('nan')):.4f}",
            "significant (FDR<0.05)": "yes" if a.get("significant_fdr") else "no",
            "group medians / table": str(a.get("group_medians") or a.get("categories") or ""),
        })
    from txmorph.viz.tables import _write
    return _write(pd.DataFrame(rows), out_base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="configs")
    ap.add_argument("--biomarkers", default="/home/anik-server/runs/samples/biomarkers.json")
    ap.add_argument("--exp3", default="/home/anik-server/runs/2026-08-14_root_exp3_"
                                      "residual_phenotypes/exp3_residual_phenotypes.json")
    ap.add_argument("--exp3-latents", default="/home/anik-server/runs/2026-08-14_root_exp3_"
                                              "residual_phenotypes/residual_latents.npz")
    ap.add_argument("--out-dir", default="/home/anik-server/runs/manuscript")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    print("[S1]", supp_hyperparameters(args.config_dir, str(out / "tables" / "S1_hyperparameters")))
    print("[S2]", supp_software(str(out / "tables" / "S2_software_versions")))
    print("[S3]", supp_per_fold(args.biomarkers, str(out / "tables" / "S3_per_fold")))

    exp3 = json.loads(Path(args.exp3).read_text()) if Path(args.exp3).exists() else {}
    if exp3:
        print("[S4]", supp_exp3_assoc(exp3, str(out / "tables" / "S4_exp3_associations")))
        lat = None
        if Path(args.exp3_latents).exists():
            z = np.load(args.exp3_latents)
            lat = z[list(z.keys())[0]]
        from txmorph.viz.figures import fig_exp3_clusters
        print("[Fig6]", fig_exp3_clusters(exp3, str(out / "figures" / "fig6_exp3_clusters"),
                                          latents=lat))
    else:
        print("[Fig6] skipped: no exp3 result at", args.exp3)


if __name__ == "__main__":
    main()
