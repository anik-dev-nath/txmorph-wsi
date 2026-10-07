"""Build model-ready drug features from the paired table (CLAUDE.md S6.4).

Variant A (one-hot): integer drug_id -> a contiguous index vocabulary.
Variant B (SMILES):  Morgan fingerprint per drug; combination regimens are fused
                     by bitwise-OR (default) or averaged bits, documented per run.
"""
from __future__ import annotations

import numpy as np

from .encoder import morgan_fingerprint


def build_onehot_ids(drug_ids):
    """Map raw drug_id values to a contiguous 0..V-1 vocabulary.

    Returns (ids_int array (n,), vocab dict).
    """
    uniq = list(dict.fromkeys(list(drug_ids)))
    vocab = {d: i for i, d in enumerate(uniq)}
    ids = np.array([vocab[d] for d in drug_ids], dtype=np.int64)
    return ids, vocab


def build_smiles_features(smiles_list, radius: int = 2, n_bits: int = 2048,
                          combination: str = "bitwise_or"):
    """Morgan fingerprints (n, n_bits) float32. A ``.``-joined SMILES (a regimen)
    is fused across its component drugs by ``bitwise_or`` or ``average``.
    """
    feats = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        parts = [s for s in str(smi).split(".") if s]
        fps = [morgan_fingerprint(s, radius=radius, n_bits=n_bits).numpy() for s in parts]
        if not fps:
            continue
        stack = np.stack(fps)
        feats[i] = (stack.max(0) if combination == "bitwise_or" else stack.mean(0))
    return feats


def build_drug_features(df, variant: str = "onehot", radius: int = 2,
                        n_bits: int = 2048, combination: str = "bitwise_or"):
    """Return (drug_feats, meta) for the paired df's drug column.

    onehot: drug_feats is int ids; meta has ``num_drugs`` and ``vocab``.
    smiles: drug_feats is (n, n_bits) float; meta has ``n_bits``.
    """
    if variant == "onehot":
        ids, vocab = build_onehot_ids(df["drug_id"].tolist())
        return ids, {"num_drugs": len(vocab), "vocab": vocab}
    if variant == "smiles":
        feats = build_smiles_features(df["drug_smiles"].tolist(), radius, n_bits,
                                      combination)
        return feats, {"n_bits": n_bits}
    raise ValueError(f"unknown drug variant: {variant}")
