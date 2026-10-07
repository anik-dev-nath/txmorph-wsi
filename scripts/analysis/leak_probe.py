"""Does conditioning on z_pre while denoising z_pre destroy the pCR signal?

Single-timepoint mode sets target z0 = z_pre and builds the context from
[z_pre || drug || pCR]. The denoiser can then recover the noise exactly from the
context alone -- eps = (z_t - sqrt(abar) z_pre)/sqrt(1-abar) -- so it never has
to use the pCR bit, and the likelihood ratio that P(pCR) is built from collapses.

Compares the two context recipes on identical data, both scored with common
random numbers:

  with_zpre : c = f(z_pre, drug, pCR)   <- current code
  clean     : c = f(drug, pCR)          <- true class-conditional density
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "src")
torch.set_num_threads(2)

from txmorph.diffusion.schedule import NoiseSchedule
from txmorph.diffusion.gaussian_diffusion import GaussianDiffusion
from txmorph.diffusion.denoiser import LatentDenoiser
from txmorph.utils.seed import seed_everything
from sklearn.metrics import roc_auc_score

SEP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
DIM, N_PER, T, N_MC = 32, 120, 200, 50
COND = 32


def make_data():
    seed_everything(0)
    mu = torch.zeros(DIM)
    mu[:8] = SEP
    z = torch.cat([torch.randn(N_PER, DIM) + mu, torch.randn(N_PER, DIM) - mu])
    y = torch.cat([torch.ones(N_PER), torch.zeros(N_PER)])
    return z, y


def run(use_zpre: bool, z_all, pcr_all, epochs=400):
    seed_everything(1)
    in_dim = (DIM if use_zpre else 0) + 4 + 1        # z_pre + drug(4) + pcr(1)
    proj = nn.Sequential(nn.Linear(in_dim, COND), nn.LayerNorm(COND))
    den = LatentDenoiser(dim=DIM, width=128, cond_dim=COND, n_blocks=2)
    sch = NoiseSchedule(T=T, schedule="cosine")
    diff = GaussianDiffusion(sch, lambda_vlb=1e-3)
    opt = torch.optim.Adam(list(proj.parameters()) + list(den.parameters()), lr=1e-3)
    drug = torch.zeros(z_all.shape[0], 4)

    def ctx(idx, pcr_vals):
        parts = ([z_all[idx]] if use_zpre else []) + [drug[idx], pcr_vals]
        return proj(torch.cat(parts, dim=-1))

    n = z_all.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            c = ctx(idx, pcr_all[idx].unsqueeze(-1))
            out = diff.loss(den, z_all[idx], c)
            opt.zero_grad()
            out["total"].backward()
            opt.step()

    den.eval(); proj.eval()
    allidx = torch.arange(n)
    with torch.no_grad():
        c1 = ctx(allidx, torch.ones(n, 1))
        c0 = ctx(allidx, torch.zeros(n, 1))
        aucs = []
        for rep in range(5):
            g = torch.Generator(); g.manual_seed(rep)
            nll = diff.nll_contrast(den, z_all, [c1, c0], n_mc=N_MC, generator=g)
            aucs.append(roc_auc_score(pcr_all.numpy(), (nll[1] - nll[0]).numpy()))
        # how well can the model denoise at all (lower = context is doing the work)
        g = torch.Generator(); g.manual_seed(99)
        base = diff.nll_contrast(den, z_all, [c1], n_mc=N_MC, generator=g)[0].mean()
    return float(np.mean(aucs)), float(np.std(aucs)), float(base)


z_all, pcr_all = make_data()
print(f"separation={SEP} sigma  dim={DIM}  n={z_all.shape[0]}")
for label, use_zpre in (("with_zpre (current code)", True), ("clean  c=f(drug,pCR)", False)):
    auc, sd, base = run(use_zpre, z_all, pcr_all)
    print(f"  {label:26s} AUROC {auc:.3f} +/- {sd:.3f} | mean eps-MSE {base:.4f}")
