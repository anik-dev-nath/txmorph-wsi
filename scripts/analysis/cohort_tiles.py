import re
from pathlib import Path
t = Path("/home/anik-server/data/tiles")
def cohort(n):
    if n.startswith("TCGA-"): return "TCGA"
    if re.match(r"^\d+_HE$", n): return "IMPRESS_HE"
    if re.match(r"^\d+_IHC$", n): return "IMPRESS_IHC"
    if re.match(r"^\d{5,6}$", n): return "PostNAT"
    return "HER2_ROIS/other"
agg = {}
for d in t.iterdir():
    if not d.is_dir(): continue
    c = cohort(d.name)
    n = sum(1 for _ in d.glob("*.npy"))
    a = agg.setdefault(c, [0, 0])
    a[0] += 1; a[1] += n
print(f"{'cohort':18s} {'slides':>7s} {'tiles':>12s} {'GB (est)':>10s} {'tiles/slide':>12s}")
tot_t = 0
for c, (s, n) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    tot_t += n
    print(f"{c:18s} {s:7d} {n:12,} {n*196608/1e9:10.1f} {n/max(s,1):12.0f}")
print(f"{'TOTAL':18s} {sum(v[0] for v in agg.values()):7d} {tot_t:12,} {tot_t*196608/1e9:10.1f}")
