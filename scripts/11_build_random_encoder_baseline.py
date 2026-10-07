"""11_build_random_encoder_baseline.py - paired table from an UNTRAINED encoder.

Exp 5's `random_encoder` ablation asks what SimCLR pretraining actually bought:
the same architecture with random weights still produces a 512-d embedding, and a
randomly-projected ResNet is a surprisingly strong baseline. If the trained encoder
does not beat it, the pretraining claim (CLAUDE.md S6.1, "from scratch, no
foundation model" as a contribution) is unsupported.

Everything except the encoder weights is held fixed -- same architecture, same
tiles, same pooling, same seeded patient-level folds -- so the delta isolates
pretraining rather than any pipeline difference.

    python scripts/11_build_random_encoder_baseline.py

Writes ``tile_emb_random/`` and ``paired_random.parquet``; point
``paths.paired_random`` at the latter to enable the ablation in script 09.
"""
import argparse
from pathlib import Path

import _bootstrap


def main():
    ap = argparse.ArgumentParser(description="Build the random-encoder baseline table.")
    _bootstrap.common_args(ap)
    ap.add_argument("--emb-dir", default=None,
                    help="where to write random-encoder bags (default: <tile_emb>_random)")
    ap.add_argument("--out", default=None,
                    help="output parquet (default: <paired>.parent/paired_random.parquet)")
    ap.add_argument("--seed", type=int, default=1234,
                    help="init seed for the untrained encoder (kept separate from the "
                         "pipeline seed so the baseline is reproducible on its own)")
    args = ap.parse_args()
    cfg = _bootstrap.load(args)

    import numpy as np
    import pandas as pd
    import torch
    from txmorph.encoders.simclr import SimCLREncoder
    from txmorph.encoders.extract import extract_slide_embeddings
    from txmorph.data.paired_dataset import build_paired_dataset

    emb_dir = args.emb_dir or (str(cfg.paths.tile_emb).rstrip("/") + "_random")
    out = args.out or str(Path(cfg.paths.paired).parent / "paired_random.parquet")
    Path(emb_dir).mkdir(parents=True, exist_ok=True)

    clinical = pd.read_csv(cfg.paths.clinical)
    tiles_root = Path(cfg.paths.tiles)
    ec = cfg.get("simclr", cfg.get("encoder", {}))
    device = cfg.get("device", "cuda")

    # Untrained encoder: same class, same proj_dim, NO checkpoint load.
    torch.manual_seed(args.seed)
    encoder = SimCLREncoder(proj_dim=ec.get("proj_dim", 128)).to(device).eval()
    print(f"[random] untrained {type(encoder).__name__} (seed={args.seed}) -> {emb_dir}")

    slide_ids = [str(s) for s in clinical["pre_slide_id"].dropna().unique()]
    missing = []
    for i, sid in enumerate(slide_ids, 1):
        d = tiles_root / sid
        if not (d / "done.json").exists():
            missing.append(sid)
            continue
        extract_slide_embeddings(
            encoder, str(d), emb_dir, sid,
            batch_size=ec.get("extract_batch_size", 256), device=device,
            num_workers=ec.get("num_workers", cfg.get("num_workers", 3)),
            force=args.force)
        if i % 25 == 0:
            print(f"[random] embedded {i}/{len(slide_ids)}", flush=True)
    if missing:
        print(f"[random] {len(missing)} slides have no tiles, skipped: {missing[:5]}")

    mc = cfg.get("mil", cfg.get("aggregator", {}))
    pooling = mc.get("pooling", "mean")
    if pooling == "mil":
        raise SystemExit(
            "[random] refusing to pool with the SimCLR-trained MIL: its attention was "
            "fit on trained-encoder features, so the ablation would not isolate "
            "pretraining. Use pooling='mean' for this baseline.")

    dc = cfg.get("diffusion", {})
    df = build_paired_dataset(clinical, emb_dir, None, out,
                              n_folds=dc.get("n_folds", 5), seed=cfg.get("seed", 0),
                              device=device, pooling=pooling)
    print(f"[random] wrote {out} ({len(df)} patients, pooling={pooling})")

    # Immediate read-out: how separable is pCR without pretraining?
    from sklearn.linear_model import LogisticRegression
    from txmorph.data.paired_dataset import load_paired
    from txmorph.eval.classification import auroc

    rdf, z, _ = load_paired(out)
    y = rdf["pcr"].astype(int).to_numpy()
    folds = rdf["fold"].to_numpy()
    oof = np.zeros(len(y), dtype=float)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        mu, sd = z[tr].mean(0), z[tr].std(0) + 1e-8
        m = LogisticRegression(max_iter=3000, C=0.01, class_weight="balanced")
        m.fit((z[tr] - mu) / sd, y[tr])
        oof[te] = m.predict_proba((z[te] - mu) / sd)[:, 1]
    print(f"[random] pCR separation from RANDOM encoder: AUROC={auroc(oof, y):.3f}")
    print("[random] compare against the trained encoder's gate value in "
          "runs/killswitch/killswitch.md — that difference is what pretraining bought")


if __name__ == "__main__":
    main()
