"""Batched embedding extraction with the frozen SimCLR encoder (CLAUDE.md S6.2).

Run the frozen backbone over every tile of a slide and write the (K, 512) fp16
bag to ``tile_emb/{slide_id}.h5``. This is the one-time freeze step: after it, the
rest of the project reads embeddings, never tiles. Idempotent — an existing
``{slide_id}.h5`` is skipped unless ``force=True``.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from ..data.tile_dataset import make_extract_dataset
from ..data.embeddings import write_tile_embeddings


def extract_slide_embeddings(encoder, slide_tiles_dir: str, out_dir: str,
                             slide_id: str, patient_id: str | None = None,
                             timepoint: str | None = None, batch_size: int = 256,
                             device: str = "cuda", num_workers: int = 4,
                             force: bool = False) -> str:
    """Embed all tiles of one slide -> tile_emb/{slide_id}.h5 (fp16). Returns path."""
    import torch
    from torch.utils.data import DataLoader

    out_path = Path(out_dir) / f"{slide_id}.h5"
    if out_path.exists() and not force:
        return str(out_path)

    ds = make_extract_dataset(slide_tiles_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    encoder = encoder.to(device).eval()
    feats: list[np.ndarray] = []
    tile_ids: list[str] = []
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    with torch.no_grad():
        for x, ids in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_bf16):
                h = encoder.embed(x)
            feats.append(h.float().cpu().numpy())
            tile_ids.extend(list(ids))

    emb = np.concatenate(feats, axis=0) if feats else np.zeros((0, encoder.feat_dim),
                                                               dtype=np.float32)
    return write_tile_embeddings(slide_id, emb, out_dir, tile_ids=tile_ids,
                                 patient_id=patient_id, timepoint=timepoint)


def extract_all(encoder, tiles_root: str, out_dir: str,
                slide_meta: dict[str, dict] | None = None, batch_size: int = 256,
                device: str = "cuda", num_workers: int = 4,
                force: bool = False) -> list[str]:
    """Extract embeddings for every per-slide tile directory under ``tiles_root``.

    slide_meta: optional {slide_id: {"patient_id":..., "timepoint":...}} map.
    Returns the list of written h5 paths.
    """
    written: list[str] = []
    for slide_dir in sorted(p for p in Path(tiles_root).iterdir() if p.is_dir()):
        slide_id = slide_dir.name
        meta = (slide_meta or {}).get(slide_id, {})
        written.append(extract_slide_embeddings(
            encoder, str(slide_dir), out_dir, slide_id,
            patient_id=meta.get("patient_id"), timepoint=meta.get("timepoint"),
            batch_size=batch_size, device=device, num_workers=num_workers,
            force=force,
        ))
    return written
