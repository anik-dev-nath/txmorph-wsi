"""Is the generative calibration advantage real, or trivially obtainable?

A model that predicts near the base rate is well calibrated and useless, so ECE
alone cannot justify a generative model. The decisive comparison is against a
*properly calibrated discriminative baseline*: if Platt/isotonic scaling -- fit
inside the training fold, standard practice, ~10 lines -- matches the generative
ECE while keeping the discriminative AUROC, the generative machinery buys nothing.

Also reports refinement (Brier decomposition) and the constant-predictor floor,
because Brier = calibration + refinement and only refinement reflects skill.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from txmorph.data.paired_dataset import load_paired
from txmorph.eval.classification import auroc, bootstrap_ci
from txmorph.eval.calibration import expected_calibration_error, brier_score
from txmorph.utils.cv import train_val_masks

df, z, _ = load_paired("/home/anik-server/data/paired.parquet")
y = df["pcr"].astype(int).to_numpy(); folds = df["fold"].to_numpy()

def oof(calibrate=None):
    p = np.zeros(len(y))
    for k in range(5):
        tr, va = train_val_masks(folds, k)
        ti, vi = np.where(tr)[0], np.where(va)[0]
        mu, sd = z[ti].mean(0), z[ti].std(0) + 1e-8
        m = LogisticRegression(max_iter=3000, C=0.01, class_weight="balanced")
        m.fit((z[ti] - mu) / sd, y[ti])
        s_va = m.decision_function((z[vi] - mu) / sd)
        if calibrate is None:
            p[vi] = m.predict_proba((z[vi] - mu) / sd)[:, 1]
            continue
        # inner CV on the TRAINING fold to get honest scores for the calibrator
        inner = np.zeros(len(ti))
        for a, b in StratifiedKFold(4, shuffle=True, random_state=0).split(z[ti], y[ti]):
            mi = LogisticRegression(max_iter=3000, C=0.01, class_weight="balanced")
            mu2, sd2 = z[ti][a].mean(0), z[ti][a].std(0) + 1e-8
            mi.fit((z[ti][a] - mu2) / sd2, y[ti][a])
            inner[b] = mi.decision_function((z[ti][b] - mu2) / sd2)
        if calibrate == "platt":
            c = LogisticRegression(max_iter=1000).fit(inner.reshape(-1, 1), y[ti])
            p[vi] = c.predict_proba(s_va.reshape(-1, 1))[:, 1]
        else:
            c = IsotonicRegression(out_of_bounds="clip").fit(inner, y[ti])
            p[vi] = np.clip(c.predict(s_va), 1e-6, 1 - 1e-6)
    return p

def show(name, p):
    t = torch.tensor
    a = bootstrap_ci(auroc, p, y, n_boot=1000, seed=0)
    e = bootstrap_ci(lambda u, v: expected_calibration_error(t(u), t(v)), p, y, n_boot=1000, seed=0)
    b = float(brier_score(t(p), t(y)))
    print(f"{name:34s} AUROC {a[0]:.3f}[{a[1]:.3f},{a[2]:.3f}]   "
          f"ECE {e[0]:.3f}[{e[1]:.3f},{e[2]:.3f}]   Brier {b:.4f}   pred-std {p.std():.3f}")

base = y.mean()
print(f"n={len(y)} base rate={base:.4f}   constant-predictor Brier={base*(1-base):.4f}\n")
show("discriminative (uncalibrated)", oof(None))
show("discriminative + Platt", oof("platt"))
show("discriminative + isotonic", oof("isotonic"))
show("CONSTANT base rate", np.full(len(y), base))
print("\ngenerative (from Exp 1):        AUROC 0.588   ECE 0.084   Brier 0.2430")
