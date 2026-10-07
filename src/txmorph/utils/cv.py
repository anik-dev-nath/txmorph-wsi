"""Patient-level cross-validation folds + leakage guard (CLAUDE.md S5, S9).

THE rule: a patient's rows (both timepoints, all slides/tiles) must live in
exactly ONE fold. Splitting a patient across folds leaks information and
silently inflates every downstream metric. Folds are stratified by label
(e.g. pCR) so each fold has a similar class balance despite the ~30% imbalance.
Pure numpy -> unit-testable without data.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def assign_folds(patient_ids, labels=None, n_folds: int = 5, seed: int = 0):
    """Return a per-row fold index (0..n_folds-1), assigned at the patient level.

    patient_ids: length-N sequence (one per row; a patient may appear many times).
    labels:      optional length-N sequence; if given, folds are stratified so the
                 class balance is preserved across folds (use the patient-level
                 pCR label here).
    """
    patient_ids = list(patient_ids)
    rng = np.random.default_rng(seed)
    uniq = list(dict.fromkeys(patient_ids))                 # unique, order-stable
    pat_fold: dict = {}

    if labels is not None:
        labels = list(labels)
        pat_label = {}
        for p, y in zip(patient_ids, labels):
            pat_label.setdefault(p, int(y))                 # label is patient-level
        by_class = defaultdict(list)
        for p in uniq:
            by_class[pat_label[p]].append(p)
        for _, plist in by_class.items():                   # round-robin per class
            plist = list(plist)
            rng.shuffle(plist)
            for i, p in enumerate(plist):
                pat_fold[p] = i % n_folds
    else:
        plist = list(uniq)
        rng.shuffle(plist)
        for i, p in enumerate(plist):
            pat_fold[p] = i % n_folds

    return np.array([pat_fold[p] for p in patient_ids], dtype=int)


def assert_no_leakage(patient_ids, folds) -> bool:
    """Raise if any patient appears in more than one fold. Call before training."""
    pf = defaultdict(set)
    for p, f in zip(list(patient_ids), list(folds)):
        pf[p].add(int(f))
    bad = [p for p, s in pf.items() if len(s) > 1]
    if bad:
        raise AssertionError(f"LEAKAGE: {len(bad)} patient(s) span multiple folds, "
                             f"e.g. {bad[:5]}")
    return True


def train_val_masks(folds, val_fold: int):
    """Boolean (train_mask, val_mask) for using `val_fold` as validation."""
    folds = np.asarray(folds)
    val = folds == val_fold
    return ~val, val
