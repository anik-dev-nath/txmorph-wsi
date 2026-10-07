"""Pathology foundation-model patch encoders (frozen).

Drop-in replacement for :class:`~txmorph.encoders.simclr.SimCLREncoder` in the
frozen-extraction path. Exposes the same duck-typed interface the rest of the
pipeline relies on -- ``.embed(x)`` and ``.feat_dim`` -- so
``encoders/extract.py``, ``eval/linear_probe.py`` and the probe-gate script work
unchanged.

**Why this exists.** The from-scratch SimCLR encoder separates *institution* at
AUROC 0.94 and *tumour biology* at ~0.6 (`scripts/analysis/shift.py`,
`gate_diag.py`), and the route is exhausted: v6 reached a much lower contrastive
loss than v5 (0.088 vs 0.175) and scored *worse* on the NCT-CRC probe (0.9050 vs
0.9291). CLAUDE.md S10's stated response to a representation that lacks signal is
to fix the encoder. Foundation models are pretrained across tens of thousands of
slides and many institutions, which targets exactly the site-confounding failure
mode. See ``docs/STATUS_FOR_REVIEW.md``.

The from-scratch encoder is retained and reported as an ablation, not discarded.

**Inputs.** ``embed`` takes the same tensor the tile loader already produces --
``(B, 3, H, W)`` float in ``[0, 1]`` at the tile's native size (256 px). Resizing
to the model's input size and applying its channel normalisation happen here, on
device, in batch: doing them per-image on the CPU loader is what made the probe
gate run at ~1 image/s before it was fixed.

**Availability.** ``owkin/phikon`` and ``owkin/phikon-v2`` are open. UNI, Virchow,
Prov-GigaPath, H-optimus and Hibou are gated and need an HF account with the
licence accepted plus ``huggingface-cli login``; once a token is present they work
here unchanged via ``model_id``.
"""
from __future__ import annotations

# Registry: model_id -> (feat_dim, input px, how to pool the token sequence).
# feat_dim is verified against the loaded config at construction time, so a wrong
# entry raises rather than silently producing mis-shaped embeddings.
KNOWN_MODELS: dict[str, dict] = {
    "owkin/phikon":    {"feat_dim": 768,  "img_size": 224, "pool": "cls"},
    "owkin/phikon-v2": {"feat_dim": 1024, "img_size": 224, "pool": "cls"},
    "MahmoodLab/UNI":  {"feat_dim": 1024, "img_size": 224, "pool": "cls", "gated": True},
    "MahmoodLab/UNI2-h": {"feat_dim": 1536, "img_size": 224, "pool": "cls", "gated": True},
    "paige-ai/Virchow2": {"feat_dim": 1280, "img_size": 224, "pool": "cls", "gated": True},
    "bioptimus/H-optimus-0": {"feat_dim": 1536, "img_size": 224, "pool": "cls", "gated": True},
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FoundationEncoder:
    """Frozen pathology foundation model with the SimCLREncoder interface.

    Parameters
    ----------
    model_id : HuggingFace repo id (see ``KNOWN_MODELS``).
    device : where to place the model.
    pool : ``"cls"`` (default) or ``"mean"`` over patch tokens. Phikon's released
        benchmarks use the CLS token; ``"mean"`` is offered as an ablation.
    img_size : override the model's expected input size (rarely needed).

    Notes
    -----
    Not an ``nn.Module`` subclass by design -- it wraps one. ``.to()``, ``.eval()``
    and ``.embed()`` are provided because that is the whole surface the extraction
    path uses.
    """

    def __init__(self, model_id: str = "owkin/phikon-v2", device: str = "cuda",
                 pool: str | None = None, img_size: int | None = None):
        import torch
        from transformers import AutoModel

        spec = KNOWN_MODELS.get(model_id, {})
        self.model_id = model_id
        self.pool = pool or spec.get("pool", "cls")
        self.img_size = img_size or spec.get("img_size", 224)
        self.device = device

        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        for p in self.model.parameters():          # frozen: never trained here
            p.requires_grad_(False)

        hidden = getattr(self.model.config, "hidden_size", None)
        expected = spec.get("feat_dim")
        if hidden is None:
            raise ValueError(f"{model_id}: config has no hidden_size; add an "
                             f"explicit entry to KNOWN_MODELS")
        if expected is not None and hidden != expected:
            raise ValueError(
                f"{model_id}: KNOWN_MODELS says feat_dim={expected} but the loaded "
                f"config reports hidden_size={hidden}. Refusing to run -- a wrong "
                f"feature width would silently corrupt every downstream artifact.")
        self.feat_dim = int(hidden)

        mean, std = self._resolve_normalization(model_id)
        self._mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.model.to(device)

    @staticmethod
    def _resolve_normalization(model_id: str):
        """Use the model's own image processor stats; fall back to ImageNet."""
        try:
            from transformers import AutoImageProcessor
            proc = AutoImageProcessor.from_pretrained(model_id)
            mean = tuple(getattr(proc, "image_mean", IMAGENET_MEAN))
            std = tuple(getattr(proc, "image_std", IMAGENET_STD))
            return mean, std
        except Exception:
            return IMAGENET_MEAN, IMAGENET_STD

    def to(self, device):
        self.device = device
        self.model.to(device)
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()

    def embed(self, x):
        """(B, 3, H, W) float in [0,1] -> (B, feat_dim) tile embeddings.

        Resize and normalisation run here, on device, in batch -- see module
        docstring. Accepts uint8 as a convenience and rescales it.
        """
        import torch
        import torch.nn.functional as F

        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = x.to(self.device, non_blocking=True)
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std

        out = self.model(pixel_values=x)
        h = out.last_hidden_state                      # (B, 1 + n_patches, D)
        if self.pool == "cls":
            return h[:, 0, :]
        if self.pool == "mean":
            return h[:, 1:, :].mean(dim=1)
        raise ValueError(f"unknown pool {self.pool!r}")

    def __repr__(self):
        return (f"FoundationEncoder({self.model_id!r}, feat_dim={self.feat_dim}, "
                f"img_size={self.img_size}, pool={self.pool!r})")


def build_encoder(name: str, device: str = "cuda", **kw):
    """Build either the from-scratch SimCLR encoder or a foundation model.

    ``name`` is ``"simclr"`` (requires ``ckpt=``) or a HuggingFace model id. Lets
    the encoder be a config switch so the from-scratch model stays a reported
    ablation rather than being replaced outright.
    """
    if name == "simclr":
        from .simclr import SimCLREncoder
        from ..utils import ckpt as ckpt_utils
        path = kw.pop("ckpt", None)
        if path is None:
            raise ValueError("build_encoder('simclr') needs ckpt=<path>")
        enc = SimCLREncoder(proj_dim=kw.pop("proj_dim", 128))
        ckpt_utils.load(path, enc, map_location=device, restore_rng=False)
        return enc.to(device).eval()
    return FoundationEncoder(name, device=device, **kw)
