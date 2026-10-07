"""run_simclr_v6.py - retrain the encoder on the full in-domain tile corpus.

v5 trained on 580,057 tiles (208 TCGA-BRCA + 103 IMPRESS H&E). Two things changed:

  * The TCGA tiles were reclaimed for disk and no longer exist locally, so that
    exact corpus is not reproducible without re-downloading ~1 TB from GDC.
  * Tiling since then added the full IMPRESS H&E set and Post-NAT-BRCA, so the
    available corpus is now ~1.3 M tiles of **breast NAC tissue** rather than a
    TCGA-heavy mix of primary tumours. Better domain match for this task, and
    ~2.3x more tiles than v5 saw.

Motivation: at the Week-12 gate the v5 latent separated *institution* at 0.94 and
*subtype* at 0.98, but pCR at only 0.59, and it did not transfer externally. An
encoder trained on 17% of the tiles now available, most of them out-of-domain, is
the most plausible cause.

Writes to **simclr_v6.pt**. simclr.pt stays frozen and read-only so every result
already reported remains reproducible against the encoder that produced it.

The tile list comes from scripts/14_build_pretrain_corpus.py, which excludes the
external HER2 Yale arm and verifies the exclusion before writing.
"""
import os
import signal
import sys

if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)

REPO = "/home/anik-server/txmorph-wsi"

if __name__ == "__main__":
    os.chdir(REPO)
    sys.path.insert(0, "src")

    import torch

    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    print(f"CUDA OK: {torch.cuda.get_device_name(0)}", flush=True)
    del _

    from pathlib import Path

    from txmorph.training.simclr import train_simclr_gpu_aug
    from txmorph.utils.config import load_config
    from txmorph.utils.seed import seed_everything

    batch_size = int(os.environ.get("SIMCLR_BATCH", "384"))
    epochs = int(os.environ.get("SIMCLR_EPOCHS", "60"))
    stain_sigma = float(os.environ.get("SIMCLR_STAIN_SIGMA", "0.15"))

    cfg = load_config(overrides=[f"simclr.batch_size={batch_size}"])
    seed_everything(0)

    manifest = Path(cfg.paths.ckpt) / "simclr_tile_manifest_v6.txt"
    if not manifest.exists():
        raise SystemExit(f"missing {manifest}; run scripts/14_build_pretrain_corpus.py")
    n = sum(1 for _ in manifest.open())
    print(f"corpus: {n:,} tiles from {manifest}", flush=True)

    out = Path(cfg.paths.ckpt) / "simclr_v6.pt"
    if out.exists() and not os.environ.get("SIMCLR_RESUME"):
        print(f"NOTE: {out} exists; training will resume from it", flush=True)

    train_simclr_gpu_aug(
        tiles_dir=cfg.paths.tiles,
        ckpt_path=str(out),
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        temperature=0.07,
        proj_dim=128,
        device="cuda",
        num_workers=4,
        seed=0,
        log_path=f"{REPO}/logs/simclr_v6_train.log",
        log_every=25,
        ckpt_every=400,
        stain_jitter="macenko",
        stain_sigma=stain_sigma,
        crop_scale=(0.2, 1.0),
        tile_list_path=str(manifest),
    )
    print("TRAINING COMPLETE", flush=True)
