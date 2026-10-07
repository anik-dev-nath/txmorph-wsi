"""13_exp3_residual_phenotypes.py - Exp 3 variant on Post-NAT-BRCA residual disease.

The original Exp 3 clusters the residual r = z_post_obs - z_post_bar for non-pCR
patients. Single-timepoint cohorts have no z_post, so that residual does not exist
and run_exp3 correctly reports "too few non-pCR residuals". This runs the pivot's
intended variant instead: cluster the morphology of **residual disease itself** on
Post-NAT-BRCA, whose patients are selected on having residual disease and are all
non-pCR by construction.

Post-NAT is deliberately kept out of the pCR training table (every patient is a
non-responder, so merging it would poison the Exp-1 label); this script reads its
own manifest and never touches paired.parquet.

    python scripts/13_exp3_residual_phenotypes.py
"""
import argparse
import json
from pathlib import Path

import _bootstrap


def main():
    ap = argparse.ArgumentParser(description="Exp 3: residual-disease phenotypes.")
    _bootstrap.common_args(ap)
    ap.add_argument("--clinical", default="/home/anik-server/data/clinical_postnat.csv")
    ap.add_argument("--emb-dir", default=None, help="default: <tile_emb>_postnat")
    ap.add_argument("--name", default="exp3_residual_phenotypes")
    args = ap.parse_args()
    cfg = _bootstrap.load(args)

    import numpy as np
    import pandas as pd
    from txmorph.encoders.simclr import SimCLREncoder
    from txmorph.encoders.extract import extract_slide_embeddings
    from txmorph.data.embeddings import read_slide_bag
    from txmorph.experiments.exp3 import run_exp3_residual_disease
    from txmorph.utils import ckpt
    from txmorph.utils.runlog import new_run_dir

    device = cfg.get("device", "cuda")
    emb_dir = args.emb_dir or (str(cfg.paths.tile_emb).rstrip("/") + "_postnat")
    Path(emb_dir).mkdir(parents=True, exist_ok=True)

    clin = pd.read_csv(args.clinical)
    tiles_root = Path(cfg.paths.tiles)
    ec = cfg.get("simclr", cfg.get("encoder", {}))

    encoder = SimCLREncoder(proj_dim=ec.get("proj_dim", 128))
    ckpt.load(str(Path(cfg.paths.ckpt) / "simclr.pt"), encoder, map_location=device,
              restore_rng=False)
    encoder = encoder.to(device).eval()

    # Residual disease is the POST-treatment resection, so this cohort populates
    # post_slide_id and leaves pre_slide_id empty -- the mirror image of IMPRESS.
    Z, keep = [], []
    for i, r in enumerate(clin.itertuples(), 1):
        sid = str(getattr(r, "post_slide_id", "") or "").strip()
        if not sid or sid.lower() == "nan":
            continue
        d = tiles_root / sid
        if not (d / "done.json").exists():
            continue
        extract_slide_embeddings(encoder, str(d), emb_dir, sid,
                                 batch_size=ec.get("extract_batch_size", 256),
                                 device=device,
                                 num_workers=ec.get("num_workers", cfg.get("num_workers", 3)),
                                 force=args.force)
        Z.append(read_slide_bag(sid, emb_dir).astype(np.float32).mean(axis=0))
        keep.append(r.Index)
        if i % 20 == 0:
            print(f"[exp3] embedded {len(keep)} of {i} scanned", flush=True)

    if not Z:
        raise SystemExit("[exp3] no Post-NAT slides are tiled yet; run script 01 first")
    clin = clin.loc[keep].reset_index(drop=True)
    Z = np.stack(Z)
    print(f"[exp3] {len(clin)} residual-disease cases | latents {Z.shape}")

    meta = {c: clin[c].tolist() for c in clin.columns}
    res = run_exp3_residual_disease(Z, meta, seed=cfg.get("seed", 0))

    run_dir = new_run_dir(cfg.paths.runs, args.name)
    (run_dir / "exp3_residual_phenotypes.json").write_text(
        json.dumps(res, indent=2, default=str))
    np.savez(run_dir / "residual_latents.npz", Z=Z,
             patient_id=clin["patient_id"].to_numpy(),
             labels=np.asarray(res.get("labels", [])))

    if "error" in res:
        print(f"[exp3] {res['error']}")
    else:
        print(f"\n[exp3] k={res['k']} (searched {res['k_range_searched']}) "
              f"sizes={res['cluster_sizes']} bootstrap ARI="
              f"{res['stability_ari']:.3f}+/-{res['stability_ari_std']:.3f}")
        print(f"{'variable':20s} {'test':9s} {'p':>9s} {'p_FDR':>9s}  sig")
        for a in res["clinical_assoc"]:
            print(f"{a['variable']:20s} {a['test']:9s} {a['p']:9.4f} "
                  f"{a.get('p_fdr', float('nan')):9.4f}  "
                  f"{'YES' if a.get('significant_fdr') else 'no'}")
        if not any(a.get("significant_fdr") for a in res["clinical_assoc"]):
            print("\n[exp3] no clinical variable is associated with cluster membership "
                  "after FDR. Clusters exist but are not validated -- report them as "
                  "unvalidated rather than as resistance phenotypes.")
    print(f"[exp3] wrote {run_dir}")


if __name__ == "__main__":
    main()
