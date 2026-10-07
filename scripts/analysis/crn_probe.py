"""Does the single-timepoint P(pCR) estimator survive a realistic (weak) signal?

The toy test in tests/test_single_timepoint.py separates the two classes by 8
sigma, where any estimator wins. Real slide embeddings will separate weakly. This
scores the SAME trained model two ways:

  independent : estimate_nll called separately per hypothesis (current code)
  common-rng  : same timesteps and same noise for both hypotheses (paired)

If the difference of two independent MC means is dominated by sampling noise,
the independent variant's AUROC collapses toward 0.5 while common-rng holds.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
torch.set_num_threads(2)

from txmorph.diffusion.context import ContextBuilder
from txmorph.diffusion.schedule import NoiseSchedule
from txmorph.diffusion.gaussian_diffusion import GaussianDiffusion
from txmorph.diffusion.denoiser import LatentDenoiser
from txmorph.utils.seed import seed_everything
from sklearn.metrics import roc_auc_score

SEP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35   # class separation in sigma
DIM = 32
N_PER = 120
T = 200
N_MC = 50

seed_everything(0)

# ---- weakly separated classes, correlated dims (embedding-like) ----
mu = torch.zeros(DIM)
mu[:8] = SEP
z1 = torch.randn(N_PER, DIM) + mu
z0 = torch.randn(N_PER, DIM) - mu
z_all = torch.cat([z1, z0])
pcr_all = torch.cat([torch.ones(N_PER), torch.zeros(N_PER)])
drug = torch.zeros(z_all.shape[0], 4)

cond_dim = 32
cb = ContextBuilder(z_dim=DIM, drug_dim=4, pcr_dim=1, out_dim=cond_dim)
den = LatentDenoiser(dim=DIM, width=128, cond_dim=cond_dim, n_blocks=2)
sch = NoiseSchedule(T=T, schedule="cosine")
diff = GaussianDiffusion(sch, lambda_vlb=1e-3)
opt = torch.optim.Adam(list(cb.parameters()) + list(den.parameters()), lr=1e-3)

n = z_all.shape[0]
for epoch in range(400):
    perm = torch.randperm(n)
    for i in range(0, n, 64):
        idx = perm[i:i + 64]
        z0b = z_all[idx]
        c = cb(z0b, drug[idx], pcr_all[idx].unsqueeze(-1))
        out = diff.loss(den, z0b, c)
        opt.zero_grad()
        out["total"].backward()
        opt.step()

den.eval(); cb.eval()


def nll_paired(z, c1, c0, n_mc, shared):
    """Mean eps-MSE under each hypothesis; `shared` reuses t and noise."""
    acc1 = torch.zeros(z.shape[0])
    acc0 = torch.zeros(z.shape[0])
    for _ in range(n_mc):
        t = torch.randint(0, T, (z.shape[0],))
        noise = torch.randn_like(z)
        zt = diff.q_sample(z, t, noise)
        e1, _ = den(zt, t, c1)
        acc1 += F.mse_loss(e1, noise, reduction="none").mean(-1)
        if not shared:                      # independent draw for hypothesis 0
            t = torch.randint(0, T, (z.shape[0],))
            noise = torch.randn_like(z)
            zt = diff.q_sample(z, t, noise)
        e0, _ = den(zt, t, c0)
        acc0 += F.mse_loss(e0, noise, reduction="none").mean(-1)
    return acc1 / n_mc, acc0 / n_mc


with torch.no_grad():
    c1 = cb(z_all, drug, torch.ones(n, 1))
    c0 = cb(z_all, drug, torch.zeros(n, 1))
    y = pcr_all.numpy()
    print(f"separation={SEP} sigma  dim={DIM}  n={n}  n_mc={N_MC}")
    for label, shared in (("independent (current code)", False), ("common-rng (paired)", True)):
        aucs, spreads = [], []
        for rep in range(5):
            a1, a0 = nll_paired(z_all, c1, c0, N_MC, shared)
            delta = (a0 - a1).numpy()          # >0 => favours pCR=1
            aucs.append(roc_auc_score(y, delta))
            spreads.append(float(np.std(delta)))
        print(f"  {label:28s} AUROC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f} "
              f"| std(delta) {np.mean(spreads):.5f}")
