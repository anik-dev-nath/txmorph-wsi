"""Drug encoder (Block 3) -> d_drug in R^128.

Variant A (one-hot): Embedding(drug_id) -> MLP.  Closed vocabulary.
Variant B (SMILES):  Morgan fingerprint (r=2, 2048 bits) -> MLP(2048->512->128).
                     Enables zero-shot on unseen drugs. Requires RDKit.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class OneHotDrugEncoder(nn.Module):
    def __init__(self, num_drugs: int, emb_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(num_drugs, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, drug_id: torch.LongTensor):
        return self.mlp(self.emb(drug_id))


class SmilesDrugEncoder(nn.Module):
    def __init__(self, n_bits: int = 2048, out_dim: int = 128):
        super().__init__()
        self.n_bits = n_bits
        self.mlp = nn.Sequential(
            nn.Linear(n_bits, 512), nn.SiLU(), nn.Linear(512, out_dim)
        )

    def forward(self, fingerprint: torch.Tensor):
        return self.mlp(fingerprint)


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> torch.Tensor:
    """Compute a Morgan/ECFP fingerprint. Requires RDKit on the lab server.
    For combination regimens, OR/average the per-drug fingerprints (document choice)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as e:  # pragma: no cover
        raise ImportError("RDKit required for SMILES encoding; pip install rdkit") from e
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return torch.tensor([int(b) for b in fp], dtype=torch.float32)
