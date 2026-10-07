"""SimCLR tile encoder (Block 1): ResNet-34 + projection head + NT-Xent.

Train from scratch (no foundation model). Use the largest batch the GPU allows
(more negatives -> better encoder); bf16. Validate via NCT-CRC-100K linear probe
(>=~90%) before extracting embeddings, then freeze.

The ResNet-34 backbone is loaded lazily from torchvision so this module imports
without it; tests inject a tiny dummy backbone. NT-Xent is pure torch (tested).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy (Chen et al., 2020).

    z1, z2: (N, d) embeddings of the two augmented views (row i in z1 is the
    positive of row i in z2). Returns a scalar loss.
    """
    N = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)        # (2N, d)
    sim = z @ z.t() / temperature                              # (2N, 2N)
    sim.fill_diagonal_(float("-inf"))                          # mask self-similarity
    # positive of i in [0,N) is i+N; positive of i in [N,2N) is i-N
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)


def make_resnet34_backbone():
    """Lazy torchvision ResNet-34 (no pretrained), fc removed. Returns (module, 512)."""
    try:
        import torchvision
    except ImportError as e:  # pragma: no cover
        raise ImportError("torchvision required for ResNet-34; pip install torchvision") from e
    net = torchvision.models.resnet34(weights=None)
    feat_dim = net.fc.in_features                              # 512 for ResNet-34
    net.fc = nn.Identity()
    return net, feat_dim


class SimCLREncoder(nn.Module):
    """Backbone + 2-layer MLP projection head.

    forward(x) -> (h, z): h = backbone feature (feat_dim, the tile embedding kept
    after training), z = L2-normalized projection (proj_dim, used only for NT-Xent).
    Pass a custom `backbone`/`feat_dim` for testing without torchvision.
    """

    def __init__(self, backbone: nn.Module | None = None, feat_dim: int = 512,
                 proj_dim: int = 128):
        super().__init__()
        if backbone is None:
            backbone, feat_dim = make_resnet34_backbone()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        h = self.backbone(x)
        z = F.normalize(self.proj(h), dim=1)
        return h, z

    @torch.no_grad()
    def embed(self, x):
        """Frozen-encoder tile embedding (backbone features only)."""
        return self.backbone(x)
