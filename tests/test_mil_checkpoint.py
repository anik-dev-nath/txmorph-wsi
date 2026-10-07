"""The MIL checkpoint must carry its own subtype-head width.

Script 04 sizes the head from ``slides.csv['subtype']``; script 05 reloads it.
When 05 re-derived the width from ``clinical.csv['receptor_subtype']`` instead --
a different column in a different file -- loading raised a state_dict size
mismatch whenever the two disagreed, which is the normal case.
"""
import pytest

torch = pytest.importorskip("torch")

from txmorph.aggregation.attention_mil import GatedAttentionMIL
from txmorph.training.mil import load_aggregator
from txmorph.utils import ckpt


def _write_ckpt(path, n_subtypes, with_extra=True):
    model = GatedAttentionMIL(n_subtypes=n_subtypes)
    ckpt.save(path, model, None, step=0, epoch=0,
              extra={"n_subtypes": n_subtypes} if with_extra else None)
    return model


def test_load_aggregator_sizes_head_from_checkpoint(tmp_path):
    """A 3-class checkpoint reloads as 3-class regardless of what the caller guesses."""
    p = tmp_path / "mil.pt"
    _write_ckpt(p, 3)
    loaded = load_aggregator(str(p), device="cpu", n_subtypes=7)   # wrong guess
    assert loaded.subtype_head.weight.shape[0] == 3


def test_load_aggregator_falls_back_to_head_shape(tmp_path):
    """Checkpoints written before `extra` still load, via the head's own shape."""
    p = tmp_path / "mil_legacy.pt"
    _write_ckpt(p, 4, with_extra=False)
    loaded = load_aggregator(str(p), device="cpu")
    assert loaded.subtype_head.weight.shape[0] == 4


def test_loaded_aggregator_matches_saved_weights(tmp_path):
    """Reload is exact and eval-mode, so pooling is deterministic downstream."""
    p = tmp_path / "mil.pt"
    saved = _write_ckpt(p, 2)
    loaded = load_aggregator(str(p), device="cpu")
    assert not loaded.training
    for (k, a), (_, b) in zip(saved.state_dict().items(), loaded.state_dict().items()):
        assert torch.allclose(a, b), f"{k} changed across save/load"


def test_single_class_subtype_task_is_rejected():
    """Cross-entropy over one class is identically 0 -- refuse rather than train on nothing."""
    from txmorph.training.mil import train_aggregator
    with pytest.raises(ValueError, match=r">=2 classes"):
        train_aggregator(["s1"], [0], "unused_dir", "unused.pt", n_subtypes=1)
