"""Does label-free batch correction recover external signal?

Standard computational-pathology mitigation: align the external latent
distribution to the training one before applying the model. Uses only *unlabeled*
external features (per-cohort mean/std), so no external labels leak -- but it is
transductive and must be disclosed as such.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from txmorph.data.paired_dataset import load_paired
from txmorph.data.embeddings import read_slide_bag
from txmorph.eval.classification import auroc, auprc

df, z_int, _ = load_paired("/home/anik-server/data/paired.parquet")
y_int = df["pcr"].astype(int).to_numpy()
ext = pd.read_csv("/home/anik-server/data/clinical_her2_yale.csv")
y_ext = ext["pcr"].astype(int).to_numpy()
z_ext = np.stack([read_slide_bag(str(s), "/home/anik-server/data/tile_emb_external")
                  .astype(np.float32).mean(0) for s in ext["pre_slide_id"]])

def fit_apply(Zi, Ze):
    mu, sd = Zi.mean(0), Zi.std(0) + 1e-8
    m = LogisticRegression(max_iter=3000, C=0.01, class_weight="balanced")
    m.fit((Zi - mu) / sd, y_int)
    return m.predict_proba((Ze - mu) / sd)[:, 1]

print("discriminative baseline on the EXTERNAL cohort:")
p = fit_apply(z_int, z_ext)
print(f"  raw (no correction)        AUROC={auroc(p,y_ext):.4f}  AUPRC={auprc(p,y_ext):.4f}")

# Per-cohort standardisation (label-free batch correction).
zi_c = (z_int - z_int.mean(0)) / (z_int.std(0) + 1e-8)
ze_c = (z_ext - z_ext.mean(0)) / (z_ext.std(0) + 1e-8)
m = LogisticRegression(max_iter=3000, C=0.01, class_weight="balanced").fit(zi_c, y_int)
p2 = m.predict_proba(ze_c)[:, 1]
print(f"  per-cohort standardised    AUROC={auroc(p2,y_ext):.4f}  AUPRC={auprc(p2,y_ext):.4f}")

# Mean-centring only (keep scale).
zi_m = z_int - z_int.mean(0); ze_m = z_ext - z_ext.mean(0)
sd = zi_m.std(0) + 1e-8
m = LogisticRegression(max_iter=3000, C=0.01, class_weight="balanced").fit(zi_m/sd, y_int)
p3 = m.predict_proba(ze_m/sd)[:, 1]
print(f"  mean-centred per cohort    AUROC={auroc(p3,y_ext):.4f}  AUPRC={auprc(p3,y_ext):.4f}")

print(f"\n  external positive rate = {y_ext.mean():.3f}  (chance AUROC = 0.5)")
print("  internal 5-fold CV reference (same model class): 0.695")
