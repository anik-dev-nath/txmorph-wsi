"""Can generative uncertainty and discriminative ranking be combined?

CLAUDE.md's Exp-1 claim is "match AUROC, beat calibration". The generative LLR
beats the baseline's calibration (ECE 0.084 vs 0.161) but loses discrimination
(0.588 vs 0.695), and that gap is a ceiling of generative classification, not a
tuning gap. A hybrid keeps each component where it is strong: the discriminative
score carries ranking, the LLR carries the calibrated density information.

Everything is fit inside the training fold -- the discriminative model, the LLR
calibrator and the combiner -- so the out-of-fold estimate stays honest.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from txmorph.data.paired_dataset import load_paired
from txmorph.drug.features import build_drug_features
from txmorph.diffusion.schedule import NoiseSchedule
from txmorph.diffusion.gaussian_diffusion import GaussianDiffusion
from txmorph.inference.run import load_fold_module, single_timepoint_scores
from txmorph.utils.cv import train_val_masks
from txmorph.eval.classification import auroc, auprc, bootstrap_ci
from txmorph.eval.calibration import expected_calibration_error, brier_score

dev = "cuda"
df, z, _ = load_paired("/home/anik-server/data/paired.parquet")
y = df["pcr"].astype(int).to_numpy(); folds = df["fold"].to_numpy()
feats, meta = build_drug_features(df, variant="onehot")
sch = NoiseSchedule(T=1000, schedule="cosine", s=0.008)
diff = GaussianDiffusion(sch, lambda_vlb=1e-3).to(dev)

p_gen = np.zeros(len(y)); p_dis = np.zeros(len(y)); p_hyb = np.zeros(len(y))
for k in range(5):
    tr, va = train_val_masks(folds, k)
    ti, vi = np.where(tr)[0], np.where(va)[0]
    mod = load_fold_module(f"/home/anik-server/checkpoints/diffusion_fold{k}.pt",
                           512, 512, "onehot", meta.get("num_drugs", 16), 2048, 3, 512,
                           "film", dev, single_timepoint=True)
    llr_tr, _ = single_timepoint_scores(diff, mod, z[ti], feats[ti], "onehot", dev,
                                        n_mc=50, seed=k)
    llr_va, _ = single_timepoint_scores(diff, mod, z[vi], feats[vi], "onehot", dev,
                                        n_mc=50, seed=k)

    # discriminative component, fit on the training fold only
    mu, sd = z[ti].mean(0), z[ti].std(0) + 1e-8
    disc = LogisticRegression(max_iter=3000, C=0.01, class_weight="balanced")
    disc.fit((z[ti] - mu) / sd, y[ti])
    d_tr = disc.decision_function((z[ti] - mu) / sd)
    d_va = disc.decision_function((z[vi] - mu) / sd)

    def std_fit(a):                       # scale both features on train stats
        return a.mean(), a.std() + 1e-12
    lm, ls = std_fit(llr_tr); dm, ds = std_fit(d_tr)

    gen = LogisticRegression(max_iter=1000).fit(((llr_tr - lm) / ls).reshape(-1, 1), y[ti])
    p_gen[vi] = gen.predict_proba(((llr_va - lm) / ls).reshape(-1, 1))[:, 1]
    dis = LogisticRegression(max_iter=1000).fit(((d_tr - dm) / ds).reshape(-1, 1), y[ti])
    p_dis[vi] = dis.predict_proba(((d_va - dm) / ds).reshape(-1, 1))[:, 1]
    X_tr = np.c_[(d_tr - dm) / ds, (llr_tr - lm) / ls]
    X_va = np.c_[(d_va - dm) / ds, (llr_va - lm) / ls]
    hyb = LogisticRegression(max_iter=1000).fit(X_tr, y[ti])
    p_hyb[vi] = hyb.predict_proba(X_va)[:, 1]

def report(name, p):
    t = torch.tensor
    a = bootstrap_ci(auroc, p, y, n_boot=1000, seed=0)
    pr = bootstrap_ci(auprc, p, y, n_boot=1000, seed=0)
    e = bootstrap_ci(lambda u, v: expected_calibration_error(t(u), t(v)), p, y, n_boot=1000, seed=0)
    b = bootstrap_ci(lambda u, v: brier_score(t(u), t(v)), p, y, n_boot=1000, seed=0)
    print(f"{name:26s} AUROC {a[0]:.3f}[{a[1]:.3f},{a[2]:.3f}]  "
          f"AUPRC {pr[0]:.3f}[{pr[1]:.3f},{pr[2]:.3f}]  "
          f"ECE {e[0]:.3f}[{e[1]:.3f},{e[2]:.3f}]  Brier {b[0]:.3f}")

print(f"n={len(y)} base={y.mean():.3f}\n")
report("generative (LLR only)", p_gen)
report("discriminative only", p_dis)
report("HYBRID (disc + LLR)", p_hyb)
print(f"\nconstant base-rate Brier = {y.mean()*(1-y.mean()):.4f}")
