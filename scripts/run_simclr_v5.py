#!/usr/bin/env python3
"""SimCLR launcher v5: GPU-side augmentation, spec-compliant positive pairs.

Two changes from v4, both measured rather than assumed
(``scripts/probe_augmentation_difficulty.py``):

1. Augmentation moved from numpy+PIL on 4 CPU threads to batched GPU kernels.
   v4 held the A40 at 0-12% utilisation and 72-99 min/epoch; v5 runs the same
   objective and checkpoint format at ~10x the throughput.
2. The positive pair is now the CLAUDE.md S6.2 augmentation: Macenko stain
   jitter (not the cheap RGB stand-in) and random-resized crop over 0.2-1.0
   (not 0.6-1.0). Under v4's settings NT-Xent had collapsed to 0.017 against a
   chance level of 6.64 -- the encoder was solving the task on colour
   statistics, which showed up downstream as 43% stroma accuracy on the
   NCT-CRC probe. The probe scores the same checkpoint at 1.87 under these
   settings, i.e. a genuinely non-trivial task.

Trains from scratch: the v4 weights are adapted to the shortcut and are kept
only as a fallback (checkpoints/simclr_v4_fastjitter_final.pt).

The tile list is PINNED to checkpoints/simclr_tile_manifest.txt. It used to be
rescanned at start-up, which meant a resume during parallel tiling silently grew
the corpus (322,340 -> 592,772 tiles between epoch 0 and epoch 39) and would have
pulled the HER2-TUMOR-ROIS external-validation cohort into pretraining as its
tiles landed. The pinned manifest is 208 TCGA-BRCA slides + 103 IMPRESS H&E
slides = 580,057 tiles; the external cohort and the IHC slides are excluded.
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

    train_simclr_gpu_aug(
        tiles_dir=cfg.paths.tiles,
        ckpt_path=str(Path(cfg.paths.ckpt) / "simclr.pt"),
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        temperature=0.07,
        proj_dim=128,
        device="cuda",
        num_workers=4,
        seed=0,
        log_path=f"{REPO}/logs/simclr_train.log",
        log_every=25,
        ckpt_every=400,
        stain_jitter="macenko",
        stain_sigma=stain_sigma,
        crop_scale=(0.2, 1.0),
        tile_list_path=str(Path(cfg.paths.ckpt) / "simclr_tile_manifest.txt"),
    )
    print("TRAINING COMPLETE", flush=True)
