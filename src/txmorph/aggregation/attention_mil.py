"""Gated attention-MIL aggregator (Block 2; Ilse et al., 2018).

Pools a bag of tile embeddings into one slide vector via learned attention:
    a_k = softmax_k( w^T ( tanh(V h_k) ⊙ sigmoid(U h_k) ) )
    z   = sum_k a_k h_k
Permutation-invariant; handles variable bag sizes. Trained with a molecular-subtype
auxiliary head to focus attention on relevant tiles, then frozen.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionMIL(nn.Module):
    def __init__(self, dim: int = 512, hidden: int = 256, n_subtypes: int = 4):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
        self.subtype_head = nn.Linear(dim, n_subtypes)

    def attention(self, bag: torch.Tensor) -> torch.Tensor:
        """bag: (K, dim) -> attention weights (K, 1), summing to 1."""
        gated = torch.tanh(self.V(bag)) * torch.sigmoid(self.U(bag))   # (K, hidden)
        scores = self.w(gated)                                         # (K, 1)
        return torch.softmax(scores, dim=0)

    def forward(self, bag: torch.Tensor):
        """bag: (K, dim). Returns (z (dim,), attn (K,1), subtype_logits (n_subtypes,))."""
        a = self.attention(bag)
        z = (a * bag).sum(dim=0)                                       # (dim,)
        return z, a, self.subtype_head(z)
