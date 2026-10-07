"""Why is the likelihood-ratio P(pCR) constant per fold?

Checks, in order:
  1. does the LLR vary across patients at all?
  2. is its spread larger than the Monte-Carlo noise (mc_se)?
  3. does the denoiser's eps-error respond to flipping the pCR bit?
If (3) is ~0 the model ignored the conditioning and the ratio cannot work.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, torch
from txmorph.data.paired_dataset import load_paired
from txmorph.drug.features import build_drug_features
from txmorph.diffusion.schedule import NoiseSchedule
from txmorph.diffusion.gaussian_diffusion import GaussianDiffusion
from txmorph.inference.run import load_fold_module, single_timepoint_scores
from txmorph.utils.cv import train_val_masks

dev = "cuda"
df, z_pre, z_post = load_paired("/home/anik-server/data/paired.parquet")
folds = df["fold"].to_numpy(); pcr = df["pcr"].astype(int).to_numpy()
feats, meta = build_drug_features(df, variant="onehot")
sch = NoiseSchedule(T=1000, schedule="cosine", s=0.008)
diff = GaussianDiffusion(sch, lambda_vlb=1e-3).to(dev)

k = 0
tr, va = train_val_masks(folds, k)
mod = load_fold_module(f"/home/anik-server/checkpoints/diffusion_fold{k}.pt",
                       512, 512, "onehot", meta.get("num_drugs", 16), 2048, 3, 512,
                       "film", dev, single_timepoint=True)
idx = np.where(va)[0]
llr, se = single_timepoint_scores(diff, mod, z_pre[idx], feats[idx], "onehot", dev,
                                  n_mc=50, n_reps=5, seed=k)
print(f"fold {k}: n={len(idx)}")
print(f"  llr   : mean={llr.mean():+.6f} std={llr.std():.6f} range=[{llr.min():+.5f},{llr.max():+.5f}]")
print(f"  mc_se : mean={se.mean():.6f}  -> signal/noise = {llr.std()/max(se.mean(),1e-12):.3f}")
print("  (docstring requires mc_se SMALL vs the llr spread across patients)")

# Higher MC budget: does the spread survive averaging, or is it pure noise?
llr2, se2 = single_timepoint_scores(diff, mod, z_pre[idx], feats[idx], "onehot", dev,
                                    n_mc=400, n_reps=5, seed=k + 99)
print(f"  n_mc=400: llr std={llr2.std():.6f} mc_se={se2.mean():.6f} "
      f"S/N={llr2.std()/max(se2.mean(),1e-12):.3f}")
print(f"  corr(llr@50, llr@400) = {np.corrcoef(llr, llr2)[0,1]:+.3f}  "
      "(should be ~1 if the LLR is a stable per-patient quantity)")

# Does the model use the pCR bit at all? Compare eps-MSE under pcr=1 vs pcr=0.
zc = torch.from_numpy(z_pre[idx].astype(np.float32)).to(dev)
db = torch.from_numpy(np.asarray(feats)[idx]).long().to(dev)
g = torch.Generator(device=dev); g.manual_seed(0)
with torch.no_grad():
    d = mod.drug(db)
    c1 = mod.context(zc, d, torch.ones(len(idx), 1, device=dev))
    c0 = mod.context(zc, d, torch.zeros(len(idx), 1, device=dev))
    print(f"  ||c1-c0|| mean = {(c1-c0).norm(dim=1).mean().item():.6f}  "
          "(0 => context ignores the pCR bit entirely)")
    t = torch.full((len(idx),), 500, device=dev, dtype=torch.long)
    eps = torch.randn(zc.shape, device=dev, generator=g)
    zt = diff.q_sample(zc, t, eps)
    e1 = mod.denoiser(zt, t, c1); e0 = mod.denoiser(zt, t, c0)
    print(f"  ||eps_theta(c1)-eps_theta(c0)|| mean = {(e1-e0).norm(dim=1).mean().item():.6f}")
    print(f"  ||eps_theta(c1)|| mean               = {e1.norm(dim=1).mean().item():.6f}")
    print(f"  MSE(c1) = {((e1-eps)**2).mean().item():.6f}   MSE(c0) = {((e0-eps)**2).mean().item():.6f}")
