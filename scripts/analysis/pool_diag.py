"""Diagnostic 1: is the subtype-trained MIL attention discarding pCR signal?

Compares the MIL slide latent already in paired.parquet against simple pooling of
the same tile bags. If mean/max pooling separates pCR better than MIL does, the
aggregator -- not the encoder -- is the bottleneck, and that is a cheap fix.
Subtype is carried as a control: MIL should win there, since that is what it was
trained on.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from txmorph.data.paired_dataset import load_paired
from txmorph.data.embeddings import read_slide_bag
from txmorph.eval.classification import auroc

EMB = "/home/anik-server/data/tile_emb"
df, z_mil, _ = load_paired("/home/anik-server/data/paired.parquet")
clin = pd.read_csv("/home/anik-server/data/clinical.csv")
sid = clin.set_index("patient_id")["pre_slide_id"].to_dict()

folds = df["fold"].to_numpy()
pcr = df["pcr"].astype(int).to_numpy()
her2 = df["receptor_subtype"].astype(str).str.contains(r"HER2\+").astype(int).to_numpy()

pools = {"mil": z_mil}
mean_v, max_v, std_v = [], [], []
for p in df["patient_id"]:
    bag = read_slide_bag(str(sid[p]), EMB).astype(np.float32)
    mean_v.append(bag.mean(0)); max_v.append(bag.max(0)); std_v.append(bag.std(0))
pools["mean"] = np.stack(mean_v)
pools["max"] = np.stack(max_v)
pools["mean+std"] = np.concatenate([np.stack(mean_v), np.stack(std_v)], axis=1)

def cv_auroc(Z, y, C=1.0):
    oof = np.zeros(len(y), dtype=float)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        m = LogisticRegression(max_iter=3000, C=C, class_weight="balanced")
        m.fit((Z[tr] - mu) / sd, y[tr])
        oof[te] = m.predict_proba((Z[te] - mu) / sd)[:, 1]
    return auroc(oof, y)

print(f"n={len(pcr)} pCR={pcr.mean():.3f}   (GO threshold 0.65)\n")
print(f"{'pooling':12s} {'dim':>6s} | {'pCR C=1':>8s} {'pCR C=.01':>10s} | {'subtype(ctrl)':>13s}")
for name, Z in pools.items():
    Z = np.asarray(Z, dtype=np.float64)
    print(f"{name:12s} {Z.shape[1]:6d} | {cv_auroc(Z,pcr):8.3f} {cv_auroc(Z,pcr,C=0.01):10.3f} | "
          f"{cv_auroc(Z,her2):13.3f}")

print("\n--- HER2+ subgroup only (where signal appeared) ---")
m = her2 == 1
for name, Z in pools.items():
    Z = np.asarray(Z, dtype=np.float64)[m]
    y, f = pcr[m], folds[m]
    oof = np.zeros(len(y))
    for k in np.unique(f):
        tr, te = f != k, f == k
        if len(np.unique(y[tr])) < 2 or te.sum() == 0: continue
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        mdl = LogisticRegression(max_iter=3000, C=0.01, class_weight="balanced").fit((Z[tr]-mu)/sd, y[tr])
        oof[te] = mdl.predict_proba((Z[te]-mu)/sd)[:, 1]
    print(f"  {name:12s} n={len(y)} AUROC={auroc(oof, y):.3f}")
