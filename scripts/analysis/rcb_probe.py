"""RCB feasibility probe — does z_pre carry CONTINUOUS response signal?

The binary-pCR gate answered 0.695. That collapses 61 non-pCR patients into one
class. This asks the question the project should have asked: is there graded
response information in the latent? Out-of-fold, patient-level 5-fold, train-only
scaling. Ridge (not the diffusion model) -- a floor, not a ceiling.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import roc_auc_score
from txmorph.data.paired_dataset import load_paired
from txmorph.utils.cv import train_val_masks

df, z, _ = load_paired("/home/anik-server/data/paired.parquet")
clin = pd.read_csv("/home/anik-server/data/clinical.csv")
df = df.merge(clin[["patient_id", "rcb_value"]], on="patient_id", how="left")

# pCR == RCB 0 by definition; NaN for pcr==1 is structural, not missing
rcb = df.rcb_value.to_numpy(dtype=float).copy()
rcb[df.pcr.to_numpy().astype(bool)] = 0.0
ok = ~np.isnan(rcb)
print("n total %d | RCB defined %d | RCB=0 %d | RCB>0 %d"
      % (len(df), ok.sum(), (rcb[ok] == 0).sum(), (rcb[ok] > 0).sum()))

y, folds = rcb[ok], df.fold.to_numpy()[ok]
X, pcr = z[ok], df.pcr.to_numpy().astype(int)[ok]
pred = np.zeros(len(y))
for k in range(5):
    tr, va = train_val_masks(folds, k)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    m = RidgeCV(alphas=np.logspace(-1, 4, 30)).fit((X[tr] - mu) / sd, y[tr])
    pred[va] = m.predict((X[va] - mu) / sd)

rho, p = spearmanr(pred, y)
print("\n--- CONTINUOUS RCB (all %d) ---" % len(y))
print("out-of-fold Spearman rho = %.3f  (p = %.4g)" % (rho, p))
print("Pearson r = %.3f" % np.corrcoef(pred, y)[0, 1])

# the decisive one: graded signal AMONG non-pCR, which binary pCR cannot see
nz = y > 0
rho_nz, p_nz = spearmanr(pred[nz], y[nz])
print("\n--- WITHIN non-pCR only (n=%d) — invisible to binary pCR ---" % nz.sum())
print("out-of-fold Spearman rho = %.3f  (p = %.4g)" % (rho_nz, p_nz))

# bootstrap CI on the within-non-pCR correlation
rng = np.random.default_rng(0); bs = []
pn, yn = pred[nz], y[nz]
for _ in range(2000):
    i = rng.integers(0, len(yn), len(yn))
    if len(np.unique(yn[i])) > 2: bs.append(spearmanr(pn[i], yn[i])[0])
print("   95%% CI [%.3f, %.3f]" % (np.percentile(bs, 2.5), np.percentile(bs, 97.5)))

# reference: same latent, same folds, binary task
pb = np.zeros(len(pcr))
for k in range(5):
    tr, va = train_val_masks(folds, k)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    lm = LogisticRegression(max_iter=3000, C=0.01, class_weight="balanced")
    lm.fit((X[tr] - mu) / sd, pcr[tr])
    pb[va] = lm.decision_function((X[va] - mu) / sd)
print("\nreference binary pCR AUROC (same latent/folds) = %.3f" % roc_auc_score(pcr, pb))
