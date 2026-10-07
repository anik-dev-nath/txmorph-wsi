"""Is the generative biomarker limited by Monte-Carlo noise or by the model?

Sweeps the MC budget. If AUROC climbs with n_mc, the LLR estimator is the
bottleneck and inference just needs a bigger budget (cheap). If it plateaus well
below the discriminative baseline (0.695), the class-conditional density itself
is the limit -- 512-d density estimation from ~30 patients per (drug, pCR) cell.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, torch
from txmorph.data.paired_dataset import load_paired
from txmorph.drug.features import build_drug_features
from txmorph.diffusion.schedule import NoiseSchedule
from txmorph.diffusion.gaussian_diffusion import GaussianDiffusion
from txmorph.inference.run import load_fold_module, single_timepoint_scores, _fit_llr_calibrator
from txmorph.utils.cv import train_val_masks
from txmorph.eval.classification import auroc

dev = "cuda"
df, z_pre, _ = load_paired("/home/anik-server/data/paired.parquet")
folds = df["fold"].to_numpy(); pcr = df["pcr"].astype(int).to_numpy()
feats, meta = build_drug_features(df, variant="onehot")
sch = NoiseSchedule(T=1000, schedule="cosine", s=0.008)
diff = GaussianDiffusion(sch, lambda_vlb=1e-3).to(dev)

mods = {k: load_fold_module(f"/home/anik-server/checkpoints/diffusion_fold{k}.pt",
                            512, 512, "onehot", meta.get("num_drugs", 16), 2048, 3, 512,
                            "film", dev, single_timepoint=True) for k in range(5)}

for n_mc in (50, 200, 800):
    p_all = np.zeros(len(pcr)); llr_sd = []
    for k in range(5):
        tr, va = train_val_masks(folds, k)
        ti, vi = np.where(tr)[0], np.where(va)[0]
        lt, _ = single_timepoint_scores(diff, mods[k], z_pre[ti], feats[ti], "onehot", dev,
                                        n_mc=n_mc, n_reps=5, seed=k)
        lv, se = single_timepoint_scores(diff, mods[k], z_pre[vi], feats[vi], "onehot", dev,
                                         n_mc=n_mc, n_reps=5, seed=k)
        p_all[vi] = _fit_llr_calibrator(lt, pcr[ti])(lv)
        llr_sd.append(lv.std() / max(se.mean(), 1e-12))
    print(f"n_mc={n_mc:4d}  AUROC={auroc(p_all, pcr):.4f}  mean S/N={np.mean(llr_sd):.2f}")
print("\ndiscriminative baseline (logistic on the same z_pre): 0.695")
