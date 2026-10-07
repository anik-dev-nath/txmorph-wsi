"""10_train_expression_bridge.py - learn morphology -> pathway signature on TCGA-BRCA.

Exp 3 clusters resistance phenotypes and validates them against expression, but the
treatment cohorts have no RNA-seq. This trains the predictor that closes that gap, on
TCGA-BRCA cases carrying both a diagnostic slide and STAR-Counts RNA-seq, and reports
**out-of-fold** performance with a permutation null. Exp 3 may only use a signature
whose bridge clears that null.

Slides are matched to expression by *case*, not by file: the already-tiled TCGA slides
were chosen for encoder pretraining, so they are usually a different slide of the same
patient than the one the GDC join picked. Any diagnostic slide of the case is a valid
training pair.

    python scripts/10_train_expression_bridge.py --bridge-dir /home/anik-server/data/tcga_bridge

Writes ``bridge_results.json`` + ``bridge_model.npz`` under the run dir.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import _bootstrap


def _case_of(slide_name: str) -> str:
    """TCGA-5L-AAT1-01Z-00-DX1.<uuid> -> TCGA-5L-AAT1 (the patient)."""
    return "-".join(slide_name.split("-")[:3])


def _is_diagnostic_primary(name: str) -> bool:
    """FFPE diagnostic slide ('DX') of a primary tumour (sample code 01).

    Frozen sections (TS/BS) have different morphology from the FFPE biopsies in
    IMPRESS/Post-NAT, and code 11 is normal tissue -- neither belongs in a predictor
    that will be applied to FFPE tumour slides.
    """
    parts = name.split("-")
    if len(parts) < 4 or parts[3][:2] != "01":
        return False
    return bool(re.search(r"-DX\w*\.", name))


def main():
    ap = argparse.ArgumentParser(description="Train the morphology->expression bridge.")
    _bootstrap.common_args(ap)
    ap.add_argument("--bridge-dir", default="/home/anik-server/data/tcga_bridge")
    ap.add_argument("--aggregate", default="mean", choices=["mean", "mil"],
                    help="slide-latent pooling; 'mil' needs a trained mil.pt")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--name", default="expression_bridge")
    args = ap.parse_args()
    cfg = _bootstrap.load(args)

    import numpy as np
    from txmorph.encoders.simclr import SimCLREncoder
    from txmorph.encoders.extract import extract_slide_embeddings
    from txmorph.data.embeddings import read_slide_bag
    from txmorph.expression.star_counts import build_expression_matrix
    from txmorph.expression.signatures import score_signatures
    from txmorph.expression.bridge import train_bridge_cv, fit_full_bridge
    from txmorph.utils import ckpt
    from txmorph.utils.runlog import new_run_dir

    bridge_dir = Path(args.bridge_dir)
    pairs = list(csv.DictReader((bridge_dir / "pairs.csv").open()))
    paired_cases = {r["case_submitter_id"] for r in pairs}
    print(f"[bridge] {len(paired_cases)} cases with slide + RNA-seq in the GDC join")

    # --- slides available as tiles, matched to those cases -------------------
    tiles_root = Path(cfg.paths.tiles)
    usable: dict[str, Path] = {}
    for d in sorted(p for p in tiles_root.iterdir() if p.is_dir()):
        if not d.name.startswith("TCGA-") or not (d / "done.json").exists():
            continue
        if not _is_diagnostic_primary(d.name):
            continue
        case = _case_of(d.name)
        if case in paired_cases:
            usable.setdefault(case, d)          # one slide per case
    print(f"[bridge] {len(usable)} of those are tiled as FFPE diagnostic primaries")
    if len(usable) < 20:
        print("[bridge] too few tiled cases to fit a bridge; tile more TCGA slides first")
        return

    # --- embed + pool to one latent per case ---------------------------------
    emb_dir = bridge_dir / "tile_emb"
    emb_dir.mkdir(parents=True, exist_ok=True)
    ec = cfg.get("simclr", cfg.get("encoder", {}))
    encoder = SimCLREncoder(proj_dim=ec.get("proj_dim", 128))
    ckpt.load(str(Path(cfg.paths.ckpt) / "simclr.pt"), encoder,
              map_location=cfg.get("device", "cuda"), restore_rng=False)

    aggregator = None
    if args.aggregate == "mil":
        from txmorph.training.mil import load_aggregator
        aggregator = load_aggregator(str(Path(cfg.paths.ckpt) / "mil.pt"),
                                     device=cfg.get("device", "cuda"))

    cases = sorted(usable)
    Z = []
    for i, case in enumerate(cases, 1):
        d = usable[case]
        extract_slide_embeddings(
            encoder, str(d), str(emb_dir), d.name, batch_size=ec.get("extract_batch_size", 256),
            device=cfg.get("device", "cuda"),
            num_workers=ec.get("num_workers", cfg.get("num_workers", 3)),
            force=args.force)
        bag = read_slide_bag(d.name, str(emb_dir))
        if aggregator is None:
            Z.append(bag.astype("float32").mean(axis=0))
        else:
            import torch
            with torch.no_grad():
                z, _, _ = aggregator(torch.from_numpy(bag).float().to(
                    cfg.get("device", "cuda")))
            Z.append(z.squeeze(0).float().cpu().numpy())
        if i % 20 == 0:
            print(f"[bridge] embedded {i}/{len(cases)} slides", flush=True)
    Z = np.stack(Z)
    print(f"[bridge] slide latents: {Z.shape} ({args.aggregate} pooling)")

    # --- measured signature scores for the same cases ------------------------
    expr_cases, genes, mat = build_expression_matrix(
        bridge_dir / "pairs.csv", bridge_dir / "expression")
    have = {c: i for i, c in enumerate(expr_cases)}
    keep = [i for i, c in enumerate(cases) if c in have]
    if len(keep) < len(cases):
        print(f"[bridge] {len(cases)-len(keep)} cases lack a downloaded expression "
              f"file; using {len(keep)}")
    Z = Z[keep]
    cases = [cases[i] for i in keep]
    scores, sig_names = score_signatures(mat[[have[c] for c in cases]], genes)
    print(f"[bridge] signatures: {sig_names} over {len(cases)} cases, "
          f"{len(genes):,} genes")

    # --- fit + honest evaluation ---------------------------------------------
    results = train_bridge_cv(Z, scores, sig_names, n_folds=args.n_folds,
                              seed=cfg.get("seed", 0), n_perm=args.n_perm)
    run_dir = new_run_dir(cfg.paths.runs, args.name)
    print(f"\n[bridge] out-of-fold performance (n={len(cases)} cases)")
    print(f"{'signature':22s} {'spearman':>9s} {'95% CI':>18s} {'pearson':>8s} {'p_perm':>8s}  usable")
    for r in results:
        lo, hi = r.spearman_ci
        print(f"{r.signature:22s} {r.spearman:9.3f} [{lo:6.3f},{hi:6.3f}] "
              f"{r.pearson:8.3f} {r.p_permutation:8.3f}  {'YES' if r.passes else 'no'}")

    usable_sigs = [r.signature for r in results if r.passes]
    print(f"\n[bridge] signatures usable by Exp 3: {usable_sigs or 'NONE'}")
    if not usable_sigs:
        print("[bridge] no signature beat its permutation null -- Exp 3 must NOT report "
              "predicted expression as validation. Say so rather than reporting them.")

    (run_dir / "bridge_results.json").write_text(json.dumps({
        "n_cases": len(cases), "aggregate": args.aggregate,
        "n_genes": len(genes), "cases": cases,
        "signatures": [{"signature": r.signature, "spearman": r.spearman,
                        "spearman_ci": list(r.spearman_ci), "pearson": r.pearson,
                        "p_permutation": r.p_permutation, "n": r.n,
                        "usable": r.passes} for r in results],
        "usable_signatures": usable_sigs,
    }, indent=2))

    if usable_sigs:
        predict = fit_full_bridge(Z, scores)
        np.savez(run_dir / "bridge_model.npz", Z=Z, scores=scores,
                 signature_names=np.array(sig_names), cases=np.array(cases))
        print(f"[bridge] refit on all {len(cases)} cases for application to "
              f"IMPRESS/Post-NAT -> {run_dir/'bridge_model.npz'}")
    print(f"[bridge] wrote {run_dir}")


if __name__ == "__main__":
    main()
