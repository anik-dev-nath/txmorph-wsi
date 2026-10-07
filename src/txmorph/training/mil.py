"""Attention-MIL aggregator training loop (CLAUDE.md S6.3, S8 step 5).

Train gated attention pooling with a molecular-subtype auxiliary classifier over
per-slide tile-embedding bags, then freeze. Bags have variable size (biopsy vs
resection), so we train one bag at a time (or grad-accumulate). Checkpoint +
resume every epoch.
"""
from __future__ import annotations

from ..aggregation.attention_mil import GatedAttentionMIL
from ..data.embeddings import read_slide_bag
from ..utils import ckpt


def train_aggregator(slide_ids, subtype_labels, emb_dir: str, ckpt_path: str,
                     n_subtypes: int = 4, epochs: int = 50, lr: float = 1e-4,
                     accum: int = 16, device: str = "cuda", seed: int = 0,
                     log_every: int = 100):
    """Train the MIL aggregator on the subtype aux task. Returns the model.

    slide_ids: sequence of slide_ids with cached embedding bags.
    subtype_labels: parallel sequence of int subtype labels (0..n_subtypes-1).

    ``n_subtypes`` is written into the checkpoint's ``extra`` so reloaders size the
    subtype head from the checkpoint itself rather than re-deriving it from a
    manifest column (see :func:`load_aggregator`).
    """
    import torch
    import torch.nn.functional as F

    if n_subtypes < 2:
        raise ValueError(
            f"subtype aux task needs >=2 classes, got {n_subtypes}. Cross-entropy "
            "over a single class is identically zero, so the aggregator would "
            "train on no signal at all. Check the manifest's subtype column.")

    torch.manual_seed(seed)
    model = GatedAttentionMIL(n_subtypes=n_subtypes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    start_epoch = 0
    state = ckpt.maybe_resume(ckpt_path, model, opt, map_location=device)
    if state is not None:
        start_epoch = state.get("epoch", 0) + 1

    items = list(zip(list(slide_ids), [int(y) for y in subtype_labels]))
    model.train()
    for epoch in range(start_epoch, epochs):
        perm = torch.randperm(len(items))
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for i, idx in enumerate(perm.tolist()):
            sid, y = items[idx]
            bag = torch.from_numpy(read_slide_bag(sid, emb_dir)).float().to(device)
            _, _, logits = model(bag)
            target = torch.tensor([y], device=device)
            loss = F.cross_entropy(logits.unsqueeze(0), target) / accum
            loss.backward()
            running += loss.item() * accum
            if (i + 1) % accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
            if i % log_every == 0:
                print(f"[mil] epoch {epoch} item {i} loss {loss.item()*accum:.4f}")
        opt.step()
        opt.zero_grad(set_to_none=True)
        ckpt.save(ckpt_path, model, opt, step=0, epoch=epoch,
                  extra={"n_subtypes": int(n_subtypes)})
    return model


def load_aggregator(ckpt_path: str, device: str = "cpu", n_subtypes: int | None = None):
    """Rebuild the frozen aggregator from its checkpoint, sized by the checkpoint.

    The subtype head's width is a property of the trained model, not of whatever
    manifest a downstream script happens to read: script 04 sizes it from
    ``slides.csv['subtype']`` while script 05 used to re-derive it from
    ``clinical.csv['receptor_subtype']`` -- a different column in a different file,
    which agreed only by luck and otherwise raised a state_dict size mismatch.
    Reads ``extra['n_subtypes']``, falling back to the checkpoint's actual head
    shape for older checkpoints that predate it.
    """
    import torch

    state = ckpt.load(ckpt_path, map_location=device, restore_rng=False)
    n = state.get("extra", {}).get("n_subtypes")
    if n is None:                      # pre-`extra` checkpoint: read the head shape
        w = state["model"].get("subtype_head.weight")
        if w is None:
            raise KeyError(f"{ckpt_path} has no subtype head to size the model from")
        n = int(w.shape[0])
    if n_subtypes is not None and int(n_subtypes) != int(n):
        print(f"[mil] note: caller expected {n_subtypes} subtypes, checkpoint has "
              f"{n}; using the checkpoint's value")
    model = GatedAttentionMIL(n_subtypes=int(n))
    model.load_state_dict(state["model"])
    return model.to(device).eval()
