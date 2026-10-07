"""Transcriptomic side of the morphology->expression bridge.

Exp 3 validates resistance clusters against expression signatures, but no treatment
cohort in this project carries RNA-seq (IMPRESS and Post-NAT-BRCA are imaging +
clinical only). TCGA-BRCA has both modalities for the *same* patients, so it is used
to learn morphology -> signature and that predictor is applied to the treatment
cohorts. This package holds the transcriptomic half: reading GDC STAR-Counts and
turning it into the signature scores the bridge regresses onto.
"""
from .star_counts import read_star_counts, build_expression_matrix
from .signatures import SIGNATURES, score_signatures, signature_names
from .bridge import train_bridge_cv, fit_full_bridge, BridgeResult

__all__ = ["read_star_counts", "build_expression_matrix", "SIGNATURES",
           "score_signatures", "signature_names", "train_bridge_cv",
           "fit_full_bridge", "BridgeResult"]
