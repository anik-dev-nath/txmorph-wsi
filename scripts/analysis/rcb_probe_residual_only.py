"""Decisive RCB probe: signal AMONG residual-disease patients only.

Removes zero-inflation (no exact zeros in the subset) and the collider induced by
conditioning on non-pCR after fitting on everyone. Trains and tests within the 61
non-pCR patients, out-of-fold, with train-only scaling. Also runs a permutation
null, because at n=61 a nominal p is not trustworthy on its own.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from txmorph.data.paired_dataset import load_paired
from txmorph.utils.cv import train_val_masks

df, z, _ = load_paired("/home/anik-server/data/paired.parquet")
clin = pd.read_csv("/home/anik-server/data/clinical.csv")
df = df.merge(clin[["patient_id", "rcb_value"]], on="patient_id", how="left")
m = (df.pcr.to_numpy().astype(int) == 0) & df.rcb_value.notna().to_numpy()
X, y, folds = z[m], df.rcb_value.to_numpy(float)[m], df.fold.to_numpy()[m]
print("residual-disease subset n = %d | RCB %.2f-%.2f median %.2f"
      % (len(y), y.min(), y.max(), np.median(y)))
print("per-fold n:", [int((folds == k).sum()) for k in range(5)])

def oof(target):
    p = np.zeros(len(target))
    for k in range(5):
        tr, va = train_val_masks(folds, k)
        if tr.sum() < 5 or va.sum() < 1: continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        r = RidgeCV(alphas=np.logspace(-1, 4, 30)).fit((X[tr] - mu) / sd, target[tr])
        p[va] = r.predict((X[va] - mu) / sd)
    return p

pred = oof(y)
rho = spearmanr(pred, y)[0]
print("\nout-of-fold Spearman rho = %.3f" % rho)

rng = np.random.default_rng(0)
null = np.array([spearmanr(oof(rng.permutation(y)), y)[0] for _ in range(200)])
pval = (np.abs(null) >= abs(rho)).mean()
print("permutation null: mean %.3f  sd %.3f  |  p = %.3f (200 perms)"
      % (null.mean(), null.std(), pval))
bs = []
for _ in range(2000):
    i = rng.integers(0, len(y), len(y))
    if len(np.unique(y[i])) > 2: bs.append(spearmanr(pred[i], y[i])[0])
print("bootstrap 95%% CI [%.3f, %.3f]" % (np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
print("\nVERDICT:", "SIGNAL" if pval < 0.05 and abs(rho) > 0.2 else "NO USABLE SIGNAL")
