"""03_extract_embeddings.py - freeze encoder; extract tile embeddings -> tile_emb/*.h5.

After this the tiles are not needed; ``tile_emb/`` is IMMUTABLE.

``--slides-csv`` restricts extraction to the slide_ids in a manifest. The tile cache
holds every cohort (plus the TCGA pretraining corpus), but the kill-switch only needs
IMPRESS, so extracting per-cohort reaches the gate without waiting on ~1.9 M tiles.
Extraction is idempotent per slide, so later cohorts simply top the cache up.
"""
import argparse
import csv
from pathlib import Path

import _bootstrap


def _wanted_slides(path: str) -> set[str]:
    """slide_ids from a manifest CSV (slides.csv, or any CSV with a slide_id column)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "slide_id" not in rows[0]:
        raise ValueError(f"{path}: expected a CSV with a 'slide_id' column")
    return {str(r["slide_id"]).strip() for r in rows if r.get("slide_id")}


def main():
    ap = argparse.ArgumentParser(description="Extract frozen tile embeddings.")
    _bootstrap.common_args(ap)
    ap.add_argument("--slides-csv", default=None,
                    help="restrict to slide_ids in this manifest (e.g. paths.slides)")
    ap.add_argument("--encoder", default="simclr",
                    help="'simclr' (loads ckpt/simclr.pt) or a HuggingFace model id "
                         "such as owkin/phikon-v2")
    ap.add_argument("--out-dir", default=None,
                    help="override paths.tile_emb; use a NEW dir per encoder so "
                         "embeddings from different encoders never mix")
    args = ap.parse_args()
    cfg = _bootstrap.load(args)

    from txmorph.encoders.extract import extract_all
    from txmorph.utils import ckpt

    device = cfg.get("device", "cuda")
    ec = cfg.get("simclr", cfg.get("encoder", {}))
    if args.encoder == "simclr":
        from txmorph.encoders.simclr import SimCLREncoder
        encoder = SimCLREncoder(proj_dim=ec.get("proj_dim", 128))
        ckpt.load(str(Path(cfg.paths.ckpt) / "simclr.pt"), encoder,
                  map_location=device, restore_rng=False)
    else:
        from txmorph.encoders.foundation import FoundationEncoder
        encoder = FoundationEncoder(args.encoder, device=device)
        print(f"[extract] {encoder!r}", flush=True)

    # NOTE: keep this a local. cfg.paths returns a fresh DotDict wrapper on every
    # read (utils/config.DotDict wraps nested dicts on access), so assigning to
    # cfg.paths.tile_emb writes to a throwaway object and is silently lost.
    out_dir = args.out_dir or cfg.paths.tile_emb
    if args.out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        print(f"[extract] writing to {out_dir}", flush=True)

    if args.slides_csv:
        from txmorph.encoders.extract import extract_slide_embeddings

        want = _wanted_slides(args.slides_csv)
        tiles_root = Path(cfg.paths.tiles)
        dirs = [d for d in sorted(tiles_root.iterdir())
                if d.is_dir() and d.name in want]
        missing = sorted(want - {d.name for d in dirs})
        print(f"[extract] {len(dirs)}/{len(want)} requested slides have tiles")
        if missing:
            print(f"[extract] no tiles for {len(missing)} slides, e.g. {missing[:5]}")
        paths = []
        for i, d in enumerate(dirs, 1):
            paths.append(extract_slide_embeddings(
                encoder, str(d), out_dir, d.name,
                batch_size=ec.get("extract_batch_size", 256),
                num_workers=ec.get("num_workers", cfg.get("num_workers", 4)),
                device=cfg.get("device", "cuda"), force=args.force))
            if i % 25 == 0:
                print(f"[extract] {i}/{len(dirs)} slides", flush=True)
    else:
        paths = extract_all(encoder, cfg.paths.tiles, out_dir,
                            batch_size=ec.get("extract_batch_size", 256),
                            num_workers=ec.get("num_workers", cfg.get("num_workers", 4)),
                            device=cfg.get("device", "cuda"), force=args.force)
    print(f"[extract] wrote {len(paths)} embedding files to {out_dir}")


if __name__ == "__main__":
    main()
