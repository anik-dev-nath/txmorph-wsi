# Analysis & diagnostic scripts

These are **not** scratch files. Each one answers a specific falsification question, and
several produced numbers that are quoted in `docs/repro.md` and in the manuscript. They
lived only on the GPU server until 2026-09-07; they are committed here so every reported
number has a traceable producer (CLAUDE.md §13).

Run from the repo root on the server with `PYTHONPATH=src`, inside the venv.

| Script | Question it answers | Number it produced |
|---|---|---|
| `gate_diag.py` | Is the latent empty of pCR signal, or is the probe just weak? | subtype separates 0.978 vs pCR 0.59 → fault is in the MIL attention, not the encoder |
| `pool_diag.py` | Is subtype-trained MIL attention discarding pCR signal? | mean pooling 0.691 vs MIL 0.602 (**withdrawn** — did not replicate end-to-end) |
| `crn_probe.py` | Does the single-timepoint P(pCR) estimator survive a realistic weak signal? | validates the LLR estimator off toy data |
| `leak_probe.py` | Does conditioning on `z_pre` while denoising `z_pre` destroy the pCR signal? | AUROC 0.436 leaked vs 0.864 clean → motivates `z_dim=0` in `context.py` |
| `llr_diag.py` | Why was the likelihood-ratio P(pCR) constant per fold? | found the LLR-calibrator scale bug (AUROC 0.517 → fixed); pinned by `tests/test_llr_calibrator.py` |
| `nmc_sweep.py` | Is the biomarker limited by Monte-Carlo noise or by the model? | n_mc 50→0.588, 200→0.586, 800→0.576 — flat, so it is the model |
| `gen_dim_probe.py` | Can *any* generative classifier work here, and at what dimension? | class-conditional Gaussian peaks 0.611 at dim 16 → generative ceiling |
| `calib_test.py` | Is the calibration advantage real, or trivially obtainable? | Platt ECE 0.059 / isotonic 0.074 vs generative 0.084 → **advantage does not survive** |
| `hybrid.py` | Can generative uncertainty and discriminative ranking be combined? | hybrid 0.685 vs discriminative 0.688 → **no complementary information** |
| `shift.py` | Is the external failure domain shift, or absent signal? | cohort-discrimination AUROC 0.94 |
| `domain_fix.py` | Does label-free batch correction recover external signal? | external 0.552 → 0.553 → shift is not the cause |
| `mkfigs.py` | Regenerate every figure from the result files on disk | `runs/figures_final/` |
| `cohort_tiles.py` | Tile counts per cohort | CONSORT accounting input |

`calib_test.py` and `hybrid.py` were written 2026-08-14 but never executed to a log until
2026-09-07; their outputs are in `docs/results/` and are **new evidence** relative to the
2026-08-16 write-up.
