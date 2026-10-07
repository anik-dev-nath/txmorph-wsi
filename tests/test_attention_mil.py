import torch
from txmorph.aggregation.attention_mil import GatedAttentionMIL
from txmorph.utils.seed import seed_everything


def test_attention_sums_to_one_and_shapes():
    seed_everything(0)
    mil = GatedAttentionMIL(dim=32, hidden=16, n_subtypes=4)
    bag = torch.randn(120, 32)                       # variable bag size
    z, a, logits = mil(bag)
    assert z.shape == (32,)
    assert a.shape == (120, 1)
    assert abs(a.sum().item() - 1.0) < 1e-5
    assert logits.shape == (4,)


def test_permutation_invariance():
    seed_everything(0)
    mil = GatedAttentionMIL(dim=32, hidden=16)
    bag = torch.randn(50, 32)
    z1, _, _ = mil(bag)
    z2, _, _ = mil(bag[torch.randperm(50)])
    assert torch.allclose(z1, z2, atol=1e-5)         # order must not matter
