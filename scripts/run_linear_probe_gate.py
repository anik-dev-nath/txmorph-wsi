"""NCT-CRC-100K linear-probe gate for SimCLR encoder.

Input pipeline note (2026-08-12): the original version did TIFF decode,
float32 conversion and a per-image bilinear resize to 256 px on the main
thread with ``num_workers=0``. Stack sampling during a live run put 12/12
samples in that path and none in the encoder forward -- the A40 sat at 0-1%
util while the gate crawled at ~1 img/s (a 33 h ETA over 114k images).

Two changes, same math:
  * ``num_workers`` > 0 so decode happens in parallel worker processes
    (the DataLoader-hang note in docs/PROGRESS.md no longer reproduces --
    verified 0/2/4 workers before this rewrite).
  * the resize moves to the GPU, batched: __getitem__ now returns the raw
    uint8 tile and ``_Preproc`` does uint8->float/255->bilinear 256 on device.
    Bilinear resize was 10 of the 14 ms/img of CPU work, and shipping uint8
    instead of float32 cuts the host->device copy 4x.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import torch, numpy as np
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from txmorph.encoders.simclr import SimCLREncoder
from txmorph.eval.linear_probe import evaluate_encoder_probe
from txmorph.utils import ckpt

NCT_TRAIN = "/home/anik-server/data/nct_crc/NCT-CRC-HE-100K"
NCT_VAL = "/home/anik-server/data/nct_crc/CRC-VAL-HE-7K"
CLASSES = sorted(["ADI","BACK","DEB","LYM","MUC","MUS","NORM","STR","TUM"])


class NCTDataset(Dataset):
    """Yields the raw uint8 (3,H,W) tile; float conversion + resize happen on GPU."""
    def __init__(self, root):
        self.paths, self.labels = [], []
        for ci, cls in enumerate(CLASSES):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir): continue
            for f in sorted(os.listdir(cls_dir)):
                if f.endswith(".tif") or f.endswith(".png") or f.endswith(".jpg"):
                    self.paths.append(os.path.join(cls_dir, f))
                    self.labels.append(ci)
        print("  %s: %d images, %d classes" % (root, len(self.paths), len(set(self.labels))), flush=True)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(arr)), self.labels[i]


class _Preproc(nn.Module):
    """Wraps the frozen encoder: uint8 -> float/255 -> bilinear 256 -> embed.

    Keeps ``extract_features`` (shared, unit-tested) untouched while moving the
    per-image CPU work onto the GPU in batch.
    """
    def __init__(self, encoder, out_px=256):
        super().__init__()
        self.encoder = encoder
        self.out_px = out_px
    def embed(self, x):
        x = x.float().div_(255.0)
        if x.shape[-1] != self.out_px:
            x = torch.nn.functional.interpolate(
                x, size=self.out_px, mode="bilinear", align_corners=False)
        return self.encoder.embed(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/anik-server/checkpoints/simclr.pt")
    ap.add_argument("--encoder", default="simclr",
                    help="'simclr' (uses --ckpt) or a HuggingFace model id, e.g. "
                         "owkin/phikon-v2. Lets the same probe score any encoder, "
                         "so the from-scratch model stays a reported ablation.")
    ap.add_argument("--pool", default=None, help="foundation models: cls | mean")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=3,
                    help="decode workers; keep <= free cores (tiling may be running)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.encoder == "simclr":
        print("Loading encoder from %s" % args.ckpt, flush=True)
        encoder = SimCLREncoder(proj_dim=128)
        state = ckpt.load(args.ckpt, encoder, map_location=args.device,
                          restore_rng=False)
        encoder = encoder.to(args.device).eval()
        ep = state.get("epoch", "?") if state else "?"
        print("  Loaded (epoch %s)" % str(ep), flush=True)
        out_px = 256
    else:
        from txmorph.encoders.foundation import FoundationEncoder
        print("Loading foundation encoder %s" % args.encoder, flush=True)
        encoder = FoundationEncoder(args.encoder, device=args.device,
                                    pool=args.pool)
        print("  %r" % encoder, flush=True)
        # embed() resizes to the model's own input size; hand it the native tile
        # so we do not pay for two interpolations.
        out_px = encoder.img_size
    model = _Preproc(encoder, out_px=out_px).to(args.device).eval()

    print("Loading NCT-CRC datasets...", flush=True)
    train_ds = NCTDataset(NCT_TRAIN)
    val_ds = NCTDataset(NCT_VAL)
    dl = dict(batch_size=args.batch_size, shuffle=False,
              num_workers=args.num_workers, pin_memory=True,
              persistent_workers=args.num_workers > 0,
              prefetch_factor=4 if args.num_workers > 0 else None)
    train_loader = DataLoader(train_ds, **dl)
    val_loader = DataLoader(val_ds, **dl)

    print("Running linear probe gate (target >= 0.90)...", flush=True)
    result = evaluate_encoder_probe(model, train_loader, val_loader, n_classes=9,
                                    target=0.90, device=args.device, epochs=300,
                                    lr=0.01, seed=42)
    print("\n=== LINEAR PROBE GATE ===", flush=True)
    print("Accuracy: %.4f" % result.accuracy)
    print("Target:   %.4f" % result.target)
    print("Status:   %s" % ("PASS" if result.passed else "FAIL"))
    pca = ["%.3f" % a for a in result.per_class_acc]
    print("Per-class: %s" % str(pca))
    print("Classes:   %s" % str(CLASSES))
    for cls, a in zip(CLASSES, result.per_class_acc):
        print("  %-5s %.3f" % (cls, a))
    if result.passed:
        print("\n>>> GATE PASSED - encoder ready for embedding extraction", flush=True)
    else:
        print("\n>>> GATE FAILED - need more training or hyperparameter changes", flush=True)
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
