"""Control: is the Phikon latent healthy, or is pCR genuinely absent?

gate_diag.py's logic, applied to the foundation-model latent. The MIL was trained
on subtype, so subtype MUST be highly separable if the features are sound. If
subtype separates well and pCR does not, the features are fine and the pCR signal
is simply not there -- which would mean the encoder is not the bottleneck.
"""
import sys, glob, os; sys.path.insert(0, "src")
sys.path.insert(0, "scripts/analysis")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from fm_killswitch_probe import load_pooled, oof_auroc, boot_ci

paired = pd.read_parquet("/home/anik-server/data/paired.parquet",
                         columns=["patient_id", "pcr", "fold", "receptor_subtype"])
slides = pd.read_csv("/home/anik-server/data/slides.csv")
df = paired.merge(slides[["patient_id", "slide_id"]].drop_duplicates("patient_id"),
                  on="patient_id", how="left")

for label, d in [("SimCLR v5 (512-d)", "/home/anik-server/data/tile_emb"),
                 ("Phikon-v2 (1024-d)", "/home/anik-server/data/tile_emb_phikon2")]:
    have = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(d, "*.h5"))}
    s = df[df.slide_id.isin(have)].reset_index(drop=True)
    X = load_pooled(d, s.slide_id.tolist())
    folds, y = s.fold.to_numpy(), s.pcr.to_numpy().astype(int)
    auc_p, sc = oof_auroc(X, y, folds); lo, hi = boot_ci(y, sc)

    # HER2+ vs TNBC -- the aux task the MIL was trained on
    sub = s.receptor_subtype.astype(str).str.upper()
    her2 = sub.str.contains("HER2\+").to_numpy().astype(int)
    auc_s, _ = oof_auroc(X, her2, folds)

    print(f"\n{label}  n={len(s)}  dim={X.shape[1]}")
    print(f"   pCR      AUROC {auc_p:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"   SUBTYPE  AUROC {auc_s:.3f}   <- health check")
