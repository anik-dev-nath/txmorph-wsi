"""07_week12_umap.py - KILL-SWITCH: UMAP separation of z_post (GO / NO-GO).

Prints the GO/NO-GO decision and writes the figure + killswitch.md. Inspect
before continuing to the experiments.
"""
import argparse
from pathlib import Path

import _bootstrap


def main():
    ap = argparse.ArgumentParser(description="Week-12 kill-switch UMAP.")
    _bootstrap.common_args(ap)
    args = ap.parse_args()
    cfg = _bootstrap.load(args)

    from txmorph.viz.umap_check import killswitch
    out_dir = Path(cfg.paths.runs) / "killswitch"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = killswitch(cfg.paths.paired, str(out_dir / "umap_killswitch.pdf"),
                        seed=cfg.get("seed", 0))
    print(f"[killswitch] {result['decision']} "
          f"(separation AUROC={result['separation_auroc']:.3f}, "
          f"silhouette={result['silhouette']:.3f})")


if __name__ == "__main__":
    main()
