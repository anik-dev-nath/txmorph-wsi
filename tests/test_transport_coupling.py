"""Coupling correctness. Pure numpy/scipy — runs on the dev box, no GPU/torch."""
import numpy as np
import pytest

from txmorph.transport.coupling import ot_coupling, coupling_cost


def test_exact_coupling_recovers_a_known_permutation():
    """If y is a shuffled copy of x, OT must undo the shuffle exactly."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 8))
    perm = rng.permutation(40)
    y = x[perm]

    ix, iy = ot_coupling(x, y, mode="exact")
    # x[ix[k]] paired with y[iy[k]]; y[iy] should equal x[ix]
    assert np.allclose(x[ix], y[iy], atol=1e-9)


def test_exact_coupling_beats_independent():
    """The whole point of OT: lower transport cost than an arbitrary pairing."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(64, 16))
    y = rng.normal(size=(64, 16)) + 3.0

    ix_e, iy_e = ot_coupling(x, y, mode="exact")
    ix_i, iy_i = ot_coupling(x, y, mode="independent", seed=1)
    assert coupling_cost(x, y, ix_e, iy_e) < coupling_cost(x, y, ix_i, iy_i)


def test_unequal_batch_sizes_match_the_smaller_side():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(10, 4))
    y = rng.normal(size=(25, 4))
    ix, iy = ot_coupling(x, y, mode="exact")
    assert len(ix) == len(iy) == 10
    assert len(np.unique(iy)) == 10          # no post sample reused


def test_groups_never_pair_across_strata():
    """A HER2+ biopsy must never be coupled to an HR+/HER2- resection."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 5))
    y = rng.normal(size=(30, 5))
    gx = np.array(["HER2+"] * 15 + ["TNBC"] * 15)
    gy = np.array(["HER2+"] * 20 + ["TNBC"] * 10)

    ix, iy = ot_coupling(x, y, groups_x=gx, groups_y=gy, mode="exact")
    assert len(ix) > 0
    assert np.all(gx[ix] == gy[iy])


def test_stratum_present_on_one_side_only_is_dropped():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(12, 3))
    y = rng.normal(size=(12, 3))
    gx = np.array(["A"] * 6 + ["ONLY_PRE"] * 6)
    gy = np.array(["A"] * 12)

    ix, iy = ot_coupling(x, y, groups_x=gx, groups_y=gy, mode="exact")
    assert set(gx[ix]) == {"A"}
    assert len(ix) == 6


def test_mismatched_group_lengths_raise():
    x = np.zeros((4, 2))
    y = np.zeros((4, 2))
    with pytest.raises(ValueError):
        ot_coupling(x, y, groups_x=np.array(["a"]), groups_y=np.array(["a"] * 4))


def test_width_mismatch_raises():
    with pytest.raises(ValueError):
        ot_coupling(np.zeros((4, 2)), np.zeros((4, 3)))
