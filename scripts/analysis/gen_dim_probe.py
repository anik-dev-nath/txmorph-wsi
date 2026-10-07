"""Can ANY generative classifier work here, and at what dimension?

Fits a class-conditional Gaussian (the simplest possible density model) on
PCA-reduced z_pre, per fold, train-only. This is a 2-minute stand-in for "diffusion
in a reduced latent": if the likelihood ratio of a simple density model recovers
signal at low dimension, rebuilding the diffusion in that space is worth it. If it
is flat everywhere, generative classification is the wrong tool at this n and the
result should be reported as such.
"""
import sys; sys.path.insert(0, "src")
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from txmorph.data.paired_dataset import load_paired
from txmorph.eval.classification import auroc

df, z, _ = load_paired("/home/anik-server/data/paired.parquet")
y = df["pcr"].astype(int).to_numpy(); folds = df["fold"].to_numpy()
her2 = df["receptor_subtype"].astype(str).str.contains(r"HER2\+").astype(int).to_numpy()

def gauss_llr(dim, shrink=0.1, cond_on_drug=True):
    """Out-of-fold LLR from class-conditional Gaussians in PCA space."""
    p_all = np.zeros(len(y))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        pca = PCA(n_components=dim, random_state=0).fit(z[tr])
        Ztr, Zte = pca.transform(z[tr]), pca.transform(z[te])
        ytr = y[tr]
        cells_tr = (her2[tr] if cond_on_drug else np.zeros(tr.sum(), int))
        cells_te = (her2[te] if cond_on_drug else np.zeros(te.sum(), int))
        llr_tr = np.zeros(tr.sum()); llr_te = np.zeros(te.sum())
        for cell in np.unique(cells_tr):
            m = cells_tr == cell
            stats = {}
            for cls in (0, 1):
                sel = m & (ytr == cls)
                if sel.sum() < dim + 2: return None      # too few to estimate
                X = Ztr[sel]
                mu = X.mean(0); S = np.cov(X, rowvar=False)
                S = (1 - shrink) * S + shrink * np.trace(S) / dim * np.eye(dim)
                stats[cls] = (mu, np.linalg.inv(S), np.linalg.slogdet(S)[1])
            def ll(X, cls):
                mu, Si, ld = stats[cls]
                d = X - mu
                return -0.5 * (np.einsum("ij,jk,ik->i", d, Si, d) + ld)
            llr_tr[m] = ll(Ztr[m], 1) - ll(Ztr[m], 0)
            mt = cells_te == cell
            if mt.any(): llr_te[mt] = ll(Zte[mt], 1) - ll(Zte[mt], 0)
        mu_, sd_ = llr_tr.mean(), llr_tr.std() + 1e-12
        lr = LogisticRegression(max_iter=1000).fit(((llr_tr - mu_) / sd_).reshape(-1, 1), ytr)
        p_all[te] = lr.predict_proba(((llr_te - mu_) / sd_).reshape(-1, 1))[:, 1]
    return auroc(p_all, y)

print("class-conditional Gaussian LLR in PCA space (generative, like the diffusion):")
for dim in (2, 4, 8, 16, 32, 64):
    a = gauss_llr(dim)
    print(f"  dim={dim:3d}  AUROC={'n/a (too few per cell)' if a is None else f'{a:.4f}'}")
print("\nreference points:")
print("  diffusion LLR in 512-d      : 0.588")
print("  discriminative logistic     : 0.695")
