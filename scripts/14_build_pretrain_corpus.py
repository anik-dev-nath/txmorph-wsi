"""14_build_pretrain_corpus.py - pin the SimCLR pretraining tile list.

The encoder must never see the external validation cohort. That cohort's slides are
tiled and sitting in the same cache as everything else, so "train on all tiles"
would silently pull them into pretraining and invalidate the external result --
which is the one experiment whose whole value is that the model has not seen it.
This script writes an explicit, auditable tile list with those slides excluded, and
refuses to write one if the exclusion cannot be verified.

    python scripts/14_build_pretrain_corpus.py --out checkpoints/simclr_tile_manifest_v6.txt

Prints the per-cohort composition so the methods section can state exactly what the
encoder saw.

Note on label leakage: pretraining is self-supervised, so including a cohort that is
later *clustered* (Post-NAT, Exp 3) uses no labels and is standard practice for SSL
in pathology. Including a cohort that is later *scored against labels* (the Yale
arm) is not, and is what this script prevents.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import _bootstrap


def _cohort_of(slide_dir_name: str) -> str:
    if slide_dir_name.startswith("TCGA-"):
        return "TCGA"
    if re.match(r"^\d+_HE$", slide_dir_name):
        return "IMPRESS_HE"
    if re.match(r"^\d+_IHC$", slide_dir_name):
        return "IMPRESS_IHC"
    if re.match(r"^\d{5,6}$", slide_dir_name):
        return "PostNAT"
    return "other"


def main():
    ap = argparse.ArgumentParser(description="Build the SimCLR pretraining tile list.")
    _bootstrap.common_args(ap)
    ap.add_argument("--out", default=None,
                    help="output tile list (default: <ckpt>/simclr_tile_manifest_v6.txt)")
    ap.add_argument("--exclude-manifest", action="append", default=None,
                    help="CSV with a slide_id column whose slides must NOT be seen "
                         "(repeatable). Defaults to the external HER2 Yale arm.")
    ap.add_argument("--exclude-cohort", default="IMPRESS_IHC",
                    help="comma-separated cohort tags to drop (see _cohort_of)")
    args = ap.parse_args()
    cfg = _bootstrap.load(args)

    tiles_root = Path(cfg.paths.tiles)
    out = Path(args.out or (Path(cfg.paths.ckpt) / "simclr_tile_manifest_v6.txt"))
    excl_manifests = args.exclude_manifest or [
        str(Path(cfg.paths.clinical).parent / "slides_her2_yale.csv")]

    banned: set[str] = set()
    for m in excl_manifests:
        p = Path(m)
        if not p.exists():
            sys.exit(f"[corpus] exclusion manifest missing: {p}. Refusing to build a "
                     "corpus whose exclusions cannot be verified.")
        with p.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or "slide_id" not in rows[0]:
            sys.exit(f"[corpus] {p} has no slide_id column; cannot verify exclusions")
        ids = {str(r["slide_id"]).strip() for r in rows if r.get("slide_id")}
        banned |= ids
        print(f"[corpus] excluding {len(ids)} slides listed in {p.name}")
    drop_cohorts = {c.strip() for c in args.exclude_cohort.split(",") if c.strip()}

    counts: dict[str, int] = {}
    slides_used: dict[str, int] = {}
    n_excluded_slides = 0
    lines: list[str] = []
    for d in sorted(p for p in tiles_root.iterdir() if p.is_dir()):
        if not (d / "done.json").exists():
            continue
        if d.name in banned:
            n_excluded_slides += 1
            continue
        cohort = _cohort_of(d.name)
        if cohort in drop_cohorts:
            continue
        tiles = sorted(str(t) for t in d.glob("*.npy"))
        if not tiles:
            continue
        lines.extend(tiles)
        counts[cohort] = counts.get(cohort, 0) + len(tiles)
        slides_used[cohort] = slides_used.get(cohort, 0) + 1

    if not lines:
        sys.exit("[corpus] no tiles selected — check paths.tiles and the exclusions")

    # Verify, don't assume: no banned slide may appear anywhere in the written list.
    leaked = sorted({b for b in banned if any(f"/{b}/" in ln for ln in lines)})
    if leaked:
        sys.exit(f"[corpus] ABORT: {len(leaked)} excluded slides leaked into the "
                 f"corpus, e.g. {leaked[:3]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")

    total = len(lines)
    print(f"\n[corpus] wrote {out}")
    print(f"[corpus] {total:,} tiles from {sum(slides_used.values())} slides")
    for c in sorted(counts):
        print(f"    {c:14s} {slides_used[c]:4d} slides  {counts[c]:9,d} tiles "
              f"({100*counts[c]/total:.1f}%)")
    print(f"[corpus] held out {n_excluded_slides} external-cohort slides "
          f"(verified absent from the list)")


if __name__ == "__main__":
    main()
