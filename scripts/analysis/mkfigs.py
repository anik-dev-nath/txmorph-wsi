"""Regenerate every figure from the real result files on disk."""
import sys, json, glob; sys.path.insert(0, "src")
import numpy as np
from txmorph.viz.figures import make_all_figures

res = {}
exp = sorted(glob.glob("/home/anik-server/runs/*experiments*/results.json"))
if exp:
    d = json.load(open(exp[-1]))
    for k in ("exp1", "exp2", "exp4", "exp5"):
        if d.get(k): res[k] = d[k]
ks = "/home/anik-server/runs/killswitch/killswitch.md"
try:
    txt = open(ks).read()
    auc = float(txt.split("from z_pre): ")[1].split(" ")[0])
    res["killswitch"] = {"separation_auroc": auc, "threshold": 0.65,
                         "decision": "GO" if auc >= 0.65 else "NO-GO", "latent": "z_pre"}
except Exception as e:
    print("killswitch parse skipped:", e)
ext = sorted(glob.glob("/home/anik-server/runs/*external*/external_metrics.json"))
if ext: res["external"] = json.load(open(ext[-1]))
e3 = sorted(glob.glob("/home/anik-server/runs/*exp3*/exp3_residual_phenotypes.json"))
if e3:
    res["exp3"] = json.load(open(e3[-1]))
    lat = e3[-1].replace("exp3_residual_phenotypes.json", "residual_latents.npz")
    try: res["exp3_latents"] = np.load(lat)["Z"]
    except Exception: pass

out = "/home/anik-server/runs/figures_final"
w = make_all_figures(res, out)
print("results present:", sorted(k for k in res if not k.endswith("_latents")))
print(f"wrote {len(w)} figures to {out}:")
for f in w: print("  ", f)
