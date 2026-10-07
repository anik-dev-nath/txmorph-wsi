import torch
from txmorph.eval.calibration import expected_calibration_error, brier_score, auroc


def test_perfectly_calibrated_low_ece():
    torch.manual_seed(0)
    probs = torch.rand(5000)
    labels = (torch.rand(5000) < probs).long()   # labels drawn at the stated prob
    assert expected_calibration_error(probs, labels, n_bins=15) < 0.05


def test_brier_bounds():
    probs = torch.tensor([1.0, 0.0, 1.0, 0.0])
    labels = torch.tensor([1, 0, 1, 0])
    assert brier_score(probs, labels) == 0.0


def test_auroc_perfect_separation():
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([0, 0, 1, 1])
    assert abs(auroc(scores, labels) - 1.0) < 1e-6
