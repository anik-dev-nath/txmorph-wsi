"""Is the latent empty of pCR signal, or is the logistic probe just weak?

Control: the MIL was trained on the subtype aux task, so subtype MUST be highly
separable from z_pre. If subtype separates well and pCR does not, the latent is
doing what it was trained to do and simply lacks response signal -- the encoder is
the problem, not the probe.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from txmorph.data.paired_dataset import load_paired
from txmorph.eval.classification import auroc

df, z_pre, _ = load_paired("/home/anik-server/data/paired.parquet")
folds = df["fold"].to_numpy()
pcr = df["pcr"].astype(int).to_numpy()
sub = pd.Categorical(df["receptor_subtype"]).codes

def cv_auroc(y, model_fn, binary=True):
    oof = np.zeros(len(y), dtype=float)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        m = model_fn(); m.fit(z_pre[tr], y[tr])
        p = m.predict_proba(z_pre[te])
        oof[te] = p[:, 1] if p.shape[1] == 2 else p.max(1)
    return auroc(oof, y) if binary else None

print(f"n={len(pcr)}  pCR rate={pcr.mean():.3f}  subtypes={np.bincount(sub)}")
print()
print("--- pCR (the gate) ---")
print(f"  logistic (balanced) : {cv_auroc(pcr, lambda: LogisticRegression(max_iter=1000, class_weight='balanced')):.3f}")
print(f"  logistic C=0.01     : {cv_auroc(pcr, lambda: LogisticRegression(max_iter=2000, C=0.01, class_weight='balanced')):.3f}")
print(f"  random forest       : {cv_auroc(pcr, lambda: RandomForestClassifier(n_estimators=300, random_state=0)):.3f}")
print()
print("--- CONTROL: HER2+ vs TNBC (what the MIL was trained on) ---")
her2 = (df["receptor_subtype"].astype(str).str.contains("HER2\+")).astype(int).to_numpy()
print(f"  logistic (balanced) : {cv_auroc(her2, lambda: LogisticRegression(max_iter=1000, class_weight='balanced')):.3f}")
print()
print("--- pCR within subtype strata (confounding check) ---")
for name, mask in [("HER2+", her2 == 1), ("TNBC", her2 == 0)]:
    y = pcr[mask]
    if len(np.unique(y)) < 2: continue
    z = z_pre[mask]; f = folds[mask]
    oof = np.zeros(len(y))
    for k in np.unique(f):
        tr, te = f != k, f == k
        if len(np.unique(y[tr])) < 2 or te.sum() == 0: continue
        m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(z[tr], y[tr])
        oof[te] = m.predict_proba(z[te])[:, 1]
    print(f"  {name:6s} n={len(y):3d} pCR={y.mean():.2f}  AUROC={auroc(oof, y):.3f}")
