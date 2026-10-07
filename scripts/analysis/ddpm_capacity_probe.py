"""ddpm_capacity_probe.py - why is the unpaired transport loss pinned at 1.00?

CLAUDE.md S18 debugging protocol, step 2-3: isolate the smallest failing unit.

For epsilon-prediction on standardised latents, loss 1.0 means the network is
emitting zero -- it has learned nothing. The floor is NOT zero: if z_0 were exactly
N(0,I) the optimal predictor is E[eps|z_t] = sqrt(1-abar_t)*z_t, giving loss
E[abar_t] ~ 0.5 under a cosine schedule. So:

    loss ~ 1.0  -> learned nothing (bug)
    loss ~ 0.5  -> learned the Gaussian structure (expected floor for near-N(0,I) data)
    loss < 0.5  -> learned real structure beyond a Gaussian

This trains the SAME denoiser UNCONDITIONALLY on the post latents, at several
dimensions and learning rates, and reports the loss floor reached. Unconditional
removes the coupling, the context builder and the drug encoder from the picture, so
whatever remains is the diffusion stack itself.

Usage (server):
    PYTHONPATH=src python scripts/analysis/ddpm_capacity_probe.py
"""
from __future__ import annotations
import argparse

import numpy as np


def load_post(emb_dir, slides_csv, clinical_csv):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from importlib import import_module
    mod = import_module("21_unpaired_transport")
    return mod.load_post_pool(emb_dir, slides_csv, clinical_csv)


def run(z, dim, lr, epochs, batch_size, device, seed=0):
    """Train unconditionally; return the mean loss over the final 10 epochs."""
    import torch
    from txmorph.diffusion.schedule import NoiseSchedule
    from txmorph.diffusion.gaussian_diffusion import GaussianDiffusion
    from txmorph.diffusion.denoiser import LatentDenoiser

    torch.manual_seed(seed)
    if dim < z.shape[1]:
        from sklearn.decomposition import PCA
        zz = PCA(n_components=dim, random_state=seed).fit_transform(z)
    else:
        zz = z
    zz = (zz - zz.mean(0)) / (zz.std(0) + 1e-6)
    zt = torch.from_numpy(zz.astype(np.float32)).to(device)

    sch = NoiseSchedule(T=1000, schedule="cosine", s=0.008)
    diff = GaussianDiffusion(sch, lambda_vlb=1e-3).to(device)
    net = LatentDenoiser(dim=dim, width=512, cond_dim=512, n_blocks=3,
                         conditioning="film").to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)

    n = len(zt)
    bs = min(batch_size, n)
    tail = []
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        ep_loss = []
        for i in range(0, n, bs):
            zb = zt[perm[i:i + bs]]
            loss = diff.loss(net, zb, None)["total"]   # None -> null context
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss.append(float(loss))
        if ep >= epochs - 10:
            tail.append(float(np.mean(ep_loss)))
    return float(np.mean(tail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post-emb", default="/home/anik-server/data/tile_emb_unlabeled")
    ap.add_argument("--post-slides", default="/home/anik-server/data/slides_postnat.csv")
    ap.add_argument("--post-clinical", default="/home/anik-server/data/clinical_postnat.csv")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    z, pid, sub = load_post(args.post_emb, args.post_slides, args.post_clinical)
    print(f"[probe] post latents {z.shape}, {len(np.unique(pid))} patients")
    print("[probe] reference: 1.0 = learned nothing | ~0.5 = Gaussian floor "
          "| <0.5 = real structure\n")

    print(f"{'dim':>6} {'lr':>8} {'batch':>6} {'final_loss':>11}")
    for dim in (512, 64, 16):
        for lr in (2e-4, 1e-3, 3e-3):
            got = run(z, dim, lr, args.epochs, 64, args.device)
            flag = "  <-- learning" if got < 0.75 else ""
            print(f"{dim:>6} {lr:>8.0e} {64:>6} {got:>11.4f}{flag}", flush=True)

    # a control: pure N(0,I) of the same shape. The stack MUST reach ~0.5 here;
    # if it does not, the bug is in the diffusion stack rather than in the data.
    rng = np.random.default_rng(0)
    ctrl = rng.normal(size=(len(z), 64)).astype(np.float32)
    got = run(ctrl, 64, 1e-3, args.epochs, 64, args.device)
    print(f"\n[control] synthetic N(0,I) 64-d, lr 1e-3 -> {got:.4f} "
          f"(expected ~0.5; >0.9 means the diffusion stack is at fault)")


if __name__ == "__main__":
    main()
