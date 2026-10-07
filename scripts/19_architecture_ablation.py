"""19_architecture_ablation.py - three architectural changes to TxMorph, ablated.

Each change targets a *measured* failure rather than a guess, and each is a switch
so the contribution of each is separable. All use the frozen patient-level folds
from paired.parquet; nothing is fit outside a training fold.

(1) ``--agg pcr_attn`` -- pCR-supervised aggregation.
    The delivered pipeline does **not** use attention_mil.py at all: the latents in
    paired.parquet are byte-identical to a plain mean over the tile bag (verified,
    max abs diff 0.0), because the kill-switch fix switched to mean pooling and the
    table was rebuilt. So pooling has never been optimised for pCR by anything.
    Here a gated-attention pooler is trained on pCR **inside each training fold**
    and applied to the held-out fold.
    Measured: 0.675, against 0.691 for mean pooling -- a negative result, reported.

(2) ``--subspace-dim D`` -- diffuse in a low-dimensional response subspace.
    gen_dim_probe.py found a class-conditional Gaussian peaks at **dim 16** and
    degrades above it, so estimating p(z|c) in 512-d from ~100 training samples is
    the binding statistical problem. PCA is fit on the training fold only and the
    DDPM runs in D dimensions.

(3) ``--semisup`` -- semi-supervised density.
    A generative model can use unlabeled data; a discriminative one cannot. This is
    the structural advantage the project has never exercised. The denoiser is
    pretrained unconditionally (null pCR token) on the training fold **plus the
    unlabeled Post-NAT-BRCA slides**, then fine-tuned conditionally on the labeled
    training fold.
    The Yale external cohort is deliberately NOT used here -- it is the external
    validation set, and putting it in any training stage would contaminate it.

Reports AUROC / AUPRC / ECE / Brier with patient-level bootstrap CIs, against the
current model and the calibrated discriminative baseline.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- bags / pooling
def load_bags(emb_dir: str, slide_ids, max_tiles: int | None = None, seed: int = 0):
    """Return a list of (K, D) float32 tile bags, in ``slide_ids`` order."""
    import h5py
    rng = np.random.default_rng(seed)
    bags = []
    for sid in slide_ids:
        with h5py.File(os.path.join(emb_dir, f"{sid}.h5"), "r") as h:
            e = np.asarray(h["emb"], dtype=np.float32)
        if max_tiles and len(e) > max_tiles:
            e = e[rng.choice(len(e), max_tiles, replace=False)]
        bags.append(e)
    return bags


class GatedAttentionPooler:
    """Gated-attention MIL (Ilse 2018) trained on pCR, fit per training fold.

    a_k = softmax(w^T (tanh(V h_k) * sigmoid(U h_k))) ;  z = sum_k a_k h_k

    Tiles are subsampled per step during training (bags are ~1500 tiles) and the
    full bag is used at inference.
    """

    def __init__(self, dim, hidden=128, device="cuda", seed=0):
        import torch
        import torch.nn as nn
        torch.manual_seed(int(seed))
        self.device = device
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
        self.head = nn.Linear(dim, 1)
        self.net = nn.ModuleList([self.V, self.U, self.w, self.head]).to(device)

    def _pool(self, h):
        import torch
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))   # (K,1)
        a = torch.softmax(a, dim=0)
        return (a * h).sum(0), a

    def fit(self, bags, y, epochs=150, lr=1e-3, sub=256, seed=0):
        import torch
        import torch.nn.functional as F
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        yt = torch.tensor(np.asarray(y, dtype=np.float32), device=self.device)
        pos_w = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)],
                             device=self.device, dtype=torch.float32)
        tb = [torch.from_numpy(b).to(self.device) for b in bags]
        self.net.train()
        for ep in range(epochs):
            order = torch.randperm(len(tb), generator=g).tolist()
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for i in order:
                h = tb[i]
                if len(h) > sub:
                    idx = torch.randint(0, len(h), (sub,), generator=g).to(self.device)
                    h = h[idx]
                z, _ = self._pool(h)
                logit = self.head(z).reshape(1)
                loss = loss + F.binary_cross_entropy_with_logits(
                    logit, yt[i].reshape(1), pos_weight=pos_w)
            (loss / len(tb)).backward()
            opt.step()
        return self

    def transform(self, bags):
        import torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for b in bags:
                z, _ = self._pool(torch.from_numpy(b).to(self.device))
                out.append(z.float().cpu().numpy())
        return np.stack(out)


# ---------------------------------------------------------------- fold machinery
def run_fold(k, z_all, y, folds, drug, cfg, unlabeled=None, device="cuda"):
    """Train one fold under ``cfg`` and return held-out LLR scores."""
    import torch
    from txmorph.diffusion.schedule import NoiseSchedule
    from txmorph.diffusion.gaussian_diffusion import GaussianDiffusion
    from txmorph.training.diffusion import DiffusionModule

    tr, va = folds != k, folds == k
    dim = z_all.shape[1]
    sch = NoiseSchedule(T=cfg["T"], schedule="cosine", s=0.008)
    diff = GaussianDiffusion(sch, lambda_vlb=1e-3).to(device)
    mod = DiffusionModule(dim, cfg["cond_dim"], "onehot", 16, 2048,
                          cfg["n_blocks"], cfg["width"], "film", device,
                          single_timepoint=True)
    opt = torch.optim.AdamW(mod.parameters(), lr=cfg["lr"])

    zt = torch.from_numpy(z_all[tr]).float().to(device)
    dt = torch.from_numpy(drug[tr]).long().to(device)
    yt = torch.from_numpy(y[tr].astype(np.float32)).to(device)

    def steps(z, d, lab, epochs, p_uncond):
        n = len(z)
        bs = min(cfg["batch_size"], n)
        for ep in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                zb, db = z[idx], d[idx]
                pe = lab[idx].unsqueeze(-1) if lab is not None else \
                    torch.full((len(idx), 1), -1.0, device=device)
                if lab is not None and p_uncond > 0:
                    drop = torch.rand(len(idx), device=device) < p_uncond
                    pe = pe.clone(); pe[drop] = -1.0
                c = mod.context(zb, mod.drug(db), pe)
                loss = diff.loss(mod.denoiser, zb, c)["total"]
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    mod.train()
    if cfg["semisup"] and unlabeled is not None and len(unlabeled):
        # (3) unconditional pretraining on labeled-train + unlabeled slides
        zu = torch.from_numpy(np.vstack([z_all[tr], unlabeled])).float().to(device)
        du = torch.cat([dt, torch.zeros(len(unlabeled), dtype=torch.long, device=device)])
        steps(zu, du, None, cfg["pretrain_epochs"], 0.0)
    steps(zt, dt, yt, cfg["epochs"], cfg["p_uncond"])

    # likelihood ratio on held-out patients, common random numbers
    mod.eval()
    zv = torch.from_numpy(z_all[va]).float().to(device)
    dv = torch.from_numpy(drug[va]).long().to(device)
    ztr_s = torch.from_numpy(z_all[tr]).float().to(device)

    def llr(zz, dd):
        with torch.no_grad():
            dg = mod.drug(dd)
            one = torch.ones(len(zz), 1, device=device)
            c1 = mod.context(zz, dg, one)
            c0 = mod.context(zz, dg, one * 0.0)
            g = torch.Generator(device=device).manual_seed(int(cfg["seed"]) + int(k))
            nll = diff.nll_contrast(mod.denoiser, zz, [c1, c0],
                                    n_mc=cfg["n_mc"], generator=g)
            return (nll[1] - nll[0]).float().cpu().numpy()

    return llr(ztr_s, dt), llr(zv, dv), tr, va


def calibrate(llr_tr, y_tr, llr_va):
    """Map LLR -> probability with a logistic fit on train-fold statistics only."""
    from sklearn.linear_model import LogisticRegression
    m, s = llr_tr.mean(), llr_tr.std() + 1e-12
    lr = LogisticRegression(max_iter=1000)
    lr.fit(((llr_tr - m) / s).reshape(-1, 1), y_tr)
    return lr.predict_proba(((llr_va - m) / s).reshape(-1, 1))[:, 1]


def metrics(p, y, n_boot=1000):
    import torch
    from txmorph.eval.classification import auroc, auprc, bootstrap_ci
    from txmorph.eval.calibration import expected_calibration_error, brier_score
    out = {}
    for name, fn in (("auroc", auroc), ("auprc", auprc),
                     ("ece", lambda a, b: float(expected_calibration_error(
                         torch.tensor(a), torch.tensor(b)))),
                     ("brier", lambda a, b: float(brier_score(
                         torch.tensor(a), torch.tensor(b))))):
        out[name] = bootstrap_ci(fn, p, y, n_boot=n_boot, seed=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", default="/home/anik-server/data/paired.parquet")
    ap.add_argument("--slides-csv", default="/home/anik-server/data/slides.csv")
    ap.add_argument("--emb-dir", default="/home/anik-server/data/tile_emb")
    ap.add_argument("--unlabeled-dir", default="/home/anik-server/data/tile_emb_postnat")
    ap.add_argument("--out", default="/home/anik-server/runs/manuscript/architecture_ablation.json")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--pretrain-epochs", type=int, default=300)
    ap.add_argument("--attn-epochs", type=int, default=150)
    ap.add_argument("--n-mc", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import pandas as pd
    from txmorph.data.paired_dataset import load_paired

    df, z_mil, _ = load_paired(args.paired)
    slides = pd.read_csv(args.slides_csv)[["patient_id", "slide_id"]].drop_duplicates("patient_id")
    df = df.merge(slides, on="patient_id", how="left")
    y = df.pcr.to_numpy().astype(int)
    folds = df.fold.to_numpy()
    drug = pd.factorize(df.drug_id)[0].astype(np.int64)
    print(f"n={len(df)} base={y.mean():.3f}")

    print("[load] tile bags ...", flush=True)
    bags = load_bags(args.emb_dir, df.slide_id.tolist())
    z_mean = np.stack([b.mean(0) for b in bags])
    unlab_bags = load_bags(args.unlabeled_dir,
                           [os.path.basename(p)[:-3] for p in
                            sorted(glob.glob(os.path.join(args.unlabeled_dir, "*.h5")))])
    print(f"[load] {len(bags)} labeled bags, {len(unlab_bags)} unlabeled")

    base_cfg = dict(T=1000, cond_dim=512, n_blocks=3, width=512, lr=1e-4,
                    batch_size=64, p_uncond=0.1, seed=0, n_mc=args.n_mc,
                    epochs=args.epochs, pretrain_epochs=args.pretrain_epochs,
                    semisup=False)

    CONFIGS = [
        ("current (mean pooling, 512-d)",    dict(agg="mil",      dim=None, semisup=False)),
        ("(1) pCR-supervised aggregation",   dict(agg="pcr_attn", dim=None, semisup=False)),
        ("(2) PCA-32 subspace",              dict(agg="mil",      dim=32,   semisup=False)),
        ("(2) PCA-16 subspace",              dict(agg="mil",      dim=16,   semisup=False)),
        ("(3) semi-supervised density",      dict(agg="mil",      dim=None, semisup=True)),
        ("(1)+(2) attn + PCA-32",            dict(agg="pcr_attn", dim=32,   semisup=False)),
        ("(2)+(3) PCA-32 + semisup",         dict(agg="mil",      dim=32,   semisup=True)),
        ("(1)+(2)+(3) all",                  dict(agg="pcr_attn", dim=32,   semisup=True)),
    ]

    results = {}
    for label, opt_cfg in CONFIGS:
        print(f"\n=== {label} ===", flush=True)
        p_oof = np.zeros(len(y))
        for k in sorted(np.unique(folds)):
            tr, va = folds != k, folds == k

            # ---- (1) aggregation, fit on the training fold only
            if opt_cfg["agg"] == "pcr_attn":
                pooler = GatedAttentionPooler(bags[0].shape[1], device=args.device, seed=k)
                pooler.fit([bags[i] for i in np.where(tr)[0]], y[tr],
                           epochs=args.attn_epochs, seed=k)
                z_fold = pooler.transform(bags)
                z_un = pooler.transform(unlab_bags) if opt_cfg["semisup"] else None
            else:
                z_fold = z_mil if opt_cfg["agg"] == "mil" else z_mean
                z_un = np.stack([b.mean(0) for b in unlab_bags]) if opt_cfg["semisup"] else None

            # ---- (2) subspace, fit on the training fold only
            if opt_cfg["dim"]:
                from sklearn.decomposition import PCA
                mu, sd = z_fold[tr].mean(0), z_fold[tr].std(0) + 1e-8
                pca = PCA(n_components=opt_cfg["dim"], random_state=0)
                pca.fit((z_fold[tr] - mu) / sd)
                zz = pca.transform((z_fold - mu) / sd).astype(np.float32)
                if z_un is not None:
                    z_un = pca.transform((z_un - mu) / sd).astype(np.float32)
            else:
                zz = np.ascontiguousarray(z_fold, dtype=np.float32)

            cfg = dict(base_cfg)
            cfg["semisup"] = opt_cfg["semisup"]
            cfg["cond_dim"] = min(512, max(64, zz.shape[1] * 4))
            cfg["width"] = min(512, max(64, zz.shape[1] * 8))
            l_tr, l_va, tr_m, va_m = run_fold(k, zz, y, folds, drug, cfg,
                                              unlabeled=z_un, device=args.device)
            p_oof[va_m] = calibrate(l_tr, y[tr_m], l_va)
            print(f"  fold {k} done", flush=True)

        m = metrics(p_oof, y)
        results[label] = {kk: list(vv) for kk, vv in m.items()}
        print(f"  AUROC {m['auroc'][0]:.3f} [{m['auroc'][1]:.3f},{m['auroc'][2]:.3f}]  "
              f"AUPRC {m['auprc'][0]:.3f}  ECE {m['ece'][0]:.3f}  Brier {m['brier'][0]:.3f}",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("\n=== SUMMARY (reference: current 0.588 / baseline ~0.69) ===")
    for lab, m in results.items():
        print(f"  {lab:34s} AUROC {m['auroc'][0]:.3f}  ECE {m['ece'][0]:.3f}  "
              f"Brier {m['brier'][0]:.3f}")
    print("[written]", args.out)


if __name__ == "__main__":
    main()
