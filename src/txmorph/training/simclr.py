"""SimCLR contrastive pretraining loop (CLAUDE.md S6.2, S8 step 3).

Train the ResNet-34 encoder from scratch with NT-Xent (tau=0.07) on two-view
tile augmentations, largest batch the GPU allows, bf16 autocast. Checkpoint +
resume every epoch. Gate on the NCT-CRC-100K linear probe (>=~90%) before the
encoder is frozen for extraction.
"""
from __future__ import annotations
import queue
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..encoders.simclr import SimCLREncoder, nt_xent
from ..data.tile_dataset import list_tiles, SimCLRTransform
from ..utils import ckpt


def _safe_log(msg: str, log_file=None) -> None:
    """Log to file (always) and stdout (best-effort, survives broken pipes)."""
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()
    try:
        print(msg, flush=True)
    except OSError:
        pass


class _ThreadedSimCLRLoader:
    """Threaded batch loader that bypasses PyTorch DataLoader multiprocessing.

    Uses ThreadPoolExecutor to load+augment tiles in parallel. numpy/PIL
    release the GIL so threads actually parallelize the CPU-bound augmentation
    work, without the CUDA fork/spawn issues that plague DataLoader workers
    on vGPU setups.
    """

    def __init__(self, paths: list[str], batch_size: int, transform: SimCLRTransform,
                 num_threads: int = 4, seed: int = 0):
        self.paths = paths
        self.batch_size = batch_size
        self.transform = transform
        self.num_threads = num_threads
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.paths) // self.batch_size

    def _load_one(self, path_and_rng):
        """Load one tile and produce two augmented views."""
        path, seed_val = path_and_rng
        tile = np.load(path)
        # Each thread gets its own RNG for reproducibility
        tfm = SimCLRTransform(rng=np.random.default_rng(seed_val))
        v1, v2 = tfm(tile)
        return v1, v2

    def epoch_iterator(self):
        """Yield (v1_batch, v2_batch) tensors for one epoch."""
        import torch
        indices = self.rng.permutation(len(self.paths))
        n_batches = len(self.paths) // self.batch_size

        with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
            for b in range(n_batches):
                batch_idx = indices[b * self.batch_size:(b + 1) * self.batch_size]
                # Generate per-tile seeds for reproducible augmentation
                seeds = self.rng.integers(0, 2**31, size=len(batch_idx))
                args = [(self.paths[i], s) for i, s in zip(batch_idx, seeds)]
                results = list(pool.map(self._load_one, args))
                v1s = torch.from_numpy(np.stack([r[0] for r in results]))
                v2s = torch.from_numpy(np.stack([r[1] for r in results]))
                yield v1s, v2s


def _load_tile(path: str) -> np.ndarray:
    """Read one uint8 (256,256,3) tile. The only per-tile CPU work in GPU-aug mode."""
    return np.load(path)


class _RawTileLoader:
    """Prefetching loader yielding raw uint8 tile batches for GPU augmentation.

    The counterpart to ``_ThreadedSimCLRLoader``: worker threads do nothing but
    ``np.load``, and jitter/crop/flip/blur run on the GPU (``data.gpu_augment``).
    A producer thread keeps ``queue_size`` batches staged so host reads overlap
    device compute. With 4 cores this is what lets the A40 stay busy.

    Shuffling is seeded per epoch, so a resumed run reproduces the same tile
    order it would have seen without the interruption.
    """

    def __init__(self, paths: list[str], batch_size: int, num_threads: int = 4,
                 seed: int = 0, queue_size: int = 4):
        self.paths = paths
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.seed = seed
        self.queue_size = queue_size

    def __len__(self):
        return len(self.paths) // self.batch_size

    def epoch_iterator(self, epoch: int = 0):
        """Yield (B,256,256,3) uint8 CPU tensors for one epoch."""
        import torch

        rng = np.random.default_rng([self.seed, epoch])
        indices = rng.permutation(len(self.paths))
        n_batches = len(self.paths) // self.batch_size
        q: queue.Queue = queue.Queue(maxsize=self.queue_size)
        _DONE = object()

        def produce():
            try:
                with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
                    for b in range(n_batches):
                        idx = indices[b * self.batch_size:(b + 1) * self.batch_size]
                        arrs = list(pool.map(_load_tile, [self.paths[i] for i in idx]))
                        q.put(torch.from_numpy(np.stack(arrs)))
            except BaseException as exc:            # surface, don't deadlock the consumer
                q.put(exc)
            else:
                q.put(_DONE)

        threading.Thread(target=produce, daemon=True).start()
        while True:
            item = q.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item


def _pinned_tiles(tiles_dir: str, tile_list_path: str | None, log_file=None) -> list[str]:
    """Tile paths for a run, pinned to ``tile_list_path`` when one is given.

    First launch scans ``tiles_dir`` and records the result; later resumes replay
    that record so the corpus cannot drift while tiling runs in parallel.
    """
    if tile_list_path is None:
        return list_tiles(tiles_dir)

    manifest = Path(tile_list_path)
    if manifest.exists():
        recorded = [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip()]
        paths = [p for p in recorded if Path(p).exists()]
        missing = len(recorded) - len(paths)
        _safe_log(f"[simclr] pinned corpus: {len(paths)} tiles from {manifest}"
                  + (f" ({missing} recorded tiles no longer on disk)" if missing else ""),
                  log_file)
        if not paths:
            raise RuntimeError(f"pinned tile list {manifest} matched no files on disk")
        return paths

    paths = list_tiles(tiles_dir)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(paths) + "\n")
    _safe_log(f"[simclr] pinned corpus: scanned {len(paths)} tiles, recorded to {manifest}",
              log_file)
    return paths


def train_simclr_gpu_aug(tiles_dir: str, ckpt_path: str, epochs: int = 100,
                         batch_size: int = 384, lr: float = 1e-3,
                         temperature: float = 0.07, proj_dim: int = 128,
                         num_workers: int = 4, device: str = "cuda", seed: int = 0,
                         log_every: int = 50, ckpt_every: int = 300,
                         log_path: str | None = None, blur_prob: float = 0.5,
                         out_px: int = 256, channels_last: bool = True,
                         cudnn_benchmark: bool = True, stain_jitter: str = "macenko",
                         stain_sigma: float = 0.05,
                         crop_scale: tuple[float, float] = (0.2, 1.0),
                         tile_list_path: str | None = None):
    """SimCLR pretraining with augmentation on the GPU (CLAUDE.md S6.2).

    Same objective, schedule and checkpoint format as :func:`train_simclr` -- the
    difference is purely where the augmentation runs, so an existing checkpoint
    resumes without conversion. ``cudnn_benchmark`` trades the cudnn-deterministic
    setting from ``utils.seed`` for autotuned convolutions; the tile order and all
    augmentation randomness stay seeded either way.

    ``tile_list_path`` pins the pretraining corpus. Without it the tile directory
    is rescanned at every start-up, so a run resumed while tiling is still going
    silently trains on a larger corpus than it began with -- which changes what an
    "epoch" means mid-run and makes the corpus impossible to state in the methods.
    When set, the first launch writes the scanned list to that file and every
    subsequent resume reads it back, so the corpus is fixed for the whole run and
    is a reviewable artifact. Missing entries are dropped with a warning rather
    than raising, so deleting a bad slide's tiles does not strand the run.
    """
    import torch

    from ..data.gpu_augment import two_views

    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    if log_path is None:
        log_path = str(Path(ckpt_path).parent / "simclr_train.log")
    log_file = open(log_path, "a", buffering=1)

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    paths = _pinned_tiles(tiles_dir, tile_list_path, log_file)
    loader = _RawTileLoader(paths, batch_size=batch_size,
                            num_threads=max(num_workers, 4), seed=seed)

    encoder = SimCLREncoder(proj_dim=proj_dim).to(device)
    if channels_last and device.startswith("cuda"):
        encoder = encoder.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    start_epoch = 0
    state = ckpt.maybe_resume(ckpt_path, encoder, opt, map_location=device)
    if state is not None:
        start_epoch = state.get("epoch", 0) + 1
        sched.last_epoch = start_epoch - 1

    gen = torch.Generator(device=device)
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    _safe_log(f"[simclr] gpu-aug resuming at epoch {start_epoch}/{epochs}, "
              f"batch_size={batch_size}, tiles={len(paths)}, "
              f"steps/epoch={len(loader)}, bf16={use_bf16}, "
              f"channels_last={channels_last}, cudnn_benchmark={cudnn_benchmark}, "
              f"stain_jitter={stain_jitter}(sigma={stain_sigma}), "
              f"crop_scale={crop_scale}", log_file)

    encoder.train()
    for epoch in range(start_epoch, epochs):
        gen.manual_seed(seed * 1_000_003 + epoch)     # augmentation RNG, per epoch
        t0, seen = time.time(), 0
        for step, tiles in enumerate(loader.epoch_iterator(epoch)):
            tiles = tiles.to(device, non_blocking=True)
            with torch.no_grad():
                v1, v2 = two_views(tiles, gen, out_px=out_px, blur_prob=blur_prob,
                                   stain_jitter=stain_jitter, stain_sigma=stain_sigma,
                                   crop_scale=crop_scale)
                if channels_last and device.startswith("cuda"):
                    v1 = v1.contiguous(memory_format=torch.channels_last)
                    v2 = v2.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                _, z1 = encoder(v1)
                _, z2 = encoder(v2)
                loss = nt_xent(z1, z2, temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            seen += tiles.shape[0]
            if step % log_every == 0:
                rate = seen / max(time.time() - t0, 1e-9)
                _safe_log(f"[simclr] epoch {epoch} step {step} loss {loss.item():.4f} "
                          f"{rate:.0f} tiles/s", log_file)
            if ckpt_every and step > 0 and step % ckpt_every == 0:
                ckpt.save(ckpt_path, encoder, opt, step=step, epoch=epoch)
        sched.step()
        ckpt.save(ckpt_path, encoder, opt, step=0, epoch=epoch)
        _safe_log(f"[simclr] epoch {epoch} done in {(time.time() - t0) / 60:.1f} min, "
                  f"checkpoint saved", log_file)
    log_file.close()
    return encoder


def train_simclr(tiles_dir: str, ckpt_path: str, epochs: int = 100,
                 batch_size: int = 512, lr: float = 1e-3, temperature: float = 0.07,
                 proj_dim: int = 128, num_workers: int = 8, device: str = "cuda",
                 seed: int = 0, log_every: int = 50, ckpt_every: int = 300,
                 log_path: str | None = None):
    """Contrastively pretrain the tile encoder. Returns the trained encoder."""
    import torch

    # Ignore SIGPIPE so broken stdout pipes don't crash training
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    # Open log file next to checkpoint if not specified
    if log_path is None:
        log_path = str(Path(ckpt_path).parent / "simclr_train.log")
    log_file = open(log_path, "a", buffering=1)

    paths = list_tiles(tiles_dir)
    loader = _ThreadedSimCLRLoader(
        paths, batch_size=batch_size,
        transform=SimCLRTransform(),
        num_threads=max(num_workers, 4),
        seed=seed,
    )

    encoder = SimCLREncoder(proj_dim=proj_dim).to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    start_epoch = 0
    state = ckpt.maybe_resume(ckpt_path, encoder, opt, map_location=device)
    if state is not None:
        start_epoch = state.get("epoch", 0) + 1
        sched.last_epoch = start_epoch - 1

    _safe_log(f"[simclr] resuming from epoch {start_epoch}, "
              f"batch_size={batch_size}, tiles={len(paths)}", log_file)

    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    encoder.train()
    for epoch in range(start_epoch, epochs):
        for step, (v1, v2) in enumerate(loader.epoch_iterator()):
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                _, z1 = encoder(v1)
                _, z2 = encoder(v2)
                loss = nt_xent(z1, z2, temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % log_every == 0:
                _safe_log(f"[simclr] epoch {epoch} step {step} "
                          f"loss {loss.item():.4f}", log_file)
            if ckpt_every and step > 0 and step % ckpt_every == 0:
                ckpt.save(ckpt_path, encoder, opt, step=step, epoch=epoch)
        sched.step()
        ckpt.save(ckpt_path, encoder, opt, step=0, epoch=epoch)
        _safe_log(f"[simclr] epoch {epoch} done, checkpoint saved", log_file)
    log_file.close()
    return encoder
