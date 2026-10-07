"""Is the external failure domain shift in the encoder, or absent signal?

If IMPRESS and Yale latents are trivially separable, the encoder is encoding
institution/scanner rather than biology, and the model is extrapolating off-
distribution -- a fixable problem. If they overlap, the signal genuinely does not
transfer.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from txmorph.data.paired_dataset import load_paired
from txmorph.data.embeddings import read_slide_bag
from txmorph.eval.classification import auroc

df, z_int, _ = load_paired("/home/anik-server/data/paired.parquet")
ext = pd.read_csv("/home/anik-server/data/clinical_her2_yale.csv")
EMB = "/home/anik-server/data/tile_emb_external"
z_ext = np.stack([read_slide_bag(str(s), EMB).astype(np.float32).mean(0)
                  for s in ext["pre_slide_id"]])

X = np.vstack([z_int, z_ext])
d = np.r_[np.zeros(len(z_int)), np.ones(len(z_ext))]      # cohort label
p = cross_val_predict(LogisticRegression(max_iter=3000, C=0.01,
                                         class_weight="balanced"),
                      (X - X.mean(0)) / (X.std(0) + 1e-8), d, cv=5,
                      method="predict_proba")[:, 1]
print(f"IMPRESS n={len(z_int)}  Yale n={len(z_ext)}")
print(f"COHORT-DISCRIMINATION AUROC = {auroc(p, d):.4f}")
print("  1.0 => latents encode institution/scanner, not shared biology")
print()
print(f"norm  IMPRESS mean={np.linalg.norm(z_int,axis=1).mean():.3f} "
      f"Yale mean={np.linalg.norm(z_ext,axis=1).mean():.3f}")
mu_i, mu_e = z_int.mean(0), z_ext.mean(0)
cos = mu_i @ mu_e / (np.linalg.norm(mu_i) * np.linalg.norm(mu_e))
print(f"cosine(mean_IMPRESS, mean_Yale) = {cos:.4f}")
sd_i = z_int.std(0).mean(); sd_e = z_ext.std(0).mean()
print(f"per-dim spread  IMPRESS={sd_i:.4f}  Yale={sd_e:.4f}")
print(f"centroid distance / avg within-cohort spread = "
      f"{np.linalg.norm(mu_i-mu_e)/((sd_i+sd_e)/2*np.sqrt(len(mu_i))):.3f}")
