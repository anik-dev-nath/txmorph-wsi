"""01_preprocess_tiles.py - WSI -> tissue mask -> tiles 256x256@20x -> stain norm.

Iterates the raw WSIs under ``paths.raw_wsi`` and writes tissue/blur-filtered,
stain-normalized tiles under ``paths.tiles`` (idempotent per slide).

Tiling is CPU-bound and embarrassingly parallel across slides, so slides are
processed in a process pool (``--workers``, default = all CPU cores). Each slide
is independent and guarded by its ``done.json`` manifest, so the run stays
idempotent/resumable and a single bad slide cannot abort the batch.
"""
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import _bootstrap

from txmorph.data.tile_shards import extract_tiles


def _tile_one(job):
    """Worker: tile one slide. Returns (slide_id, n_tiles, error_or_None)."""
    slide_path, tiles_dir, tile_px, mpp, tissue_thresh, force = job
    try:
        m = extract_tiles(slide_path, tiles_dir, tile_px=tile_px, mpp=mpp,
                          tissue_thresh=tissue_thresh, force=force)
        return (m.slide_id, m.n_tiles, None)
    except Exception as e:  # noqa: BLE001 - report per-slide, keep the batch alive
        return (Path(slide_path).stem, -1, repr(e))


def main():
    ap = argparse.ArgumentParser(description="Preprocess WSIs into tiles.")
    _bootstrap.common_args(ap)
    ap.add_argument("--ext", default="svs,tiff,ndpi", help="comma-separated WSI extensions")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel slide workers (0 = all CPU cores)")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N slides (0 = all); useful for smoke tests")
    ap.add_argument("--subdir", default=None,
                    help="restrict to one cohort directory under paths.raw_wsi "
                         "(e.g. post_nat_brca). Tiles for other cohorts may have "
                         "been reclaimed for disk, and without this they look like "
                         "unfinished work and get re-tiled.")
    ap.add_argument("--exclude", default="_IHC",
                    help="comma-separated substrings; slides whose filename stem "
                         "contains any of them are skipped. Defaults to IMPRESS's "
                         "immunohistochemistry slides, which the pipeline never reads "
                         "(manifests reference _HE only) but which otherwise reappear "
                         "as unfinished work on every run.")
    args = ap.parse_args()
    cfg = _bootstrap.load(args)

    dcfg = cfg.get("preprocess", {})
    tile_px = dcfg.get("tile_px", 256)
    mpp = dcfg.get("mpp", 0.5)
    tissue_thresh = dcfg.get("tissue_thresh", 0.25)

    raw = Path(cfg.paths.raw_wsi)
    if args.subdir:
        raw = raw / args.subdir
        if not raw.is_dir():
            raise SystemExit(f"[preprocess] no such cohort directory: {raw}")
    exts = tuple(f".{e.strip().lstrip('.')}" for e in args.ext.split(","))
    slides = sorted(p for p in raw.rglob("*") if p.suffix.lower() in exts)
    skip = tuple(s for s in (x.strip() for x in args.exclude.split(",")) if s)
    if skip:
        before = len(slides)
        slides = [p for p in slides if not any(s in p.stem for s in skip)]
        if before != len(slides):
            print(f"[preprocess] excluded {before - len(slides)} slides matching {skip}")
    if args.limit:
        slides = slides[: args.limit]
    workers = args.workers or os.cpu_count() or 1
    workers = max(1, min(workers, len(slides) or 1))
    print(f"[preprocess] {len(slides)} slides under {raw} | {workers} workers "
          f"| mpp={mpp} tile_px={tile_px} tissue_thresh={tissue_thresh}")

    jobs = [(str(sp), cfg.paths.tiles, tile_px, mpp, tissue_thresh, args.force)
            for sp in slides]

    done = 0
    if workers == 1:
        for job in jobs:
            sid, n, err = _tile_one(job)
            done += 1
            tag = f"{n} tiles" if err is None else f"ERROR {err}"
            print(f"[preprocess] ({done}/{len(jobs)}) {sid}: {tag}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_tile_one, job) for job in jobs]
            for fut in as_completed(futs):
                sid, n, err = fut.result()
                done += 1
                tag = f"{n} tiles" if err is None else f"ERROR {err}"
                print(f"[preprocess] ({done}/{len(jobs)}) {sid}: {tag}", flush=True)


if __name__ == "__main__":
    main()
