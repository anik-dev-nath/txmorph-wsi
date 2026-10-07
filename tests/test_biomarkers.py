import torch
from txmorph.biomarkers.derive import derive_biomarkers, PCRZone


def test_biomarkers_recover_known_stats():
    torch.manual_seed(0)
    d = 8
    mean = torch.arange(1, d + 1).float()          # mean[0] = 1.0
    samples = mean + 0.1 * torch.randn(5000, d)
    # logistic fires when z[0] > 0.5:  100*z0 - 50 > 0
    w = torch.zeros(d); w[0] = 100.0
    zone = PCRZone(w=w, b=-50.0)
    bm = derive_biomarkers(samples, zone, z_post_obs=mean + 1.0)
    assert torch.allclose(bm.z_post_bar, mean, atol=0.05)
    assert abs(bm.sigma2_tx - 0.01) < 0.005          # var ~ 0.1^2 per dim
    assert bm.p_pcr > 0.99                            # mean[0]=1.0 > 0.5 -> fires
    assert torch.allclose(bm.residual, torch.ones(d), atol=0.05)
