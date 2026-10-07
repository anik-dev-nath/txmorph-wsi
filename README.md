# TxMorph-WSI

Conditional latent diffusion over whole-slide-image embeddings for predicting
response to neoadjuvant chemotherapy (NAC) in breast cancer from a pre-treatment
H&E biopsy.

This repository contains the **code only**. No patient data, slides, embeddings,
model weights or run outputs are included.

---

## What this is

The study asks whether reframing prediction of pathological complete response (pCR)
as *conditional generative modelling* improves on standard discriminative
classification. A self-supervised patch encoder produces tile embeddings, these are
pooled to one vector per slide, and a conditional denoising diffusion model learns a
class-conditional density over that vector. A patient is scored by the likelihood
ratio between the response-positive and response-negative conditions, calibrated
inside the training fold.

**Headline finding: it does not improve on a calibrated logistic regression.** On 126
patients under five-fold patient-level cross-validation the generative model reached
AUROC 0.588 (95% CI 0.491–0.687) against 0.654–0.669 for logistic regression on the
same embeddings, and its calibration advantage disappeared once the baseline was
itself calibrated (expected calibration error 0.084 against 0.067 with Platt scaling).

The contribution is the set of controls that follow. Each alternative explanation was
tested and ruled out individually, including substituting a pathology foundation model
pretrained on ~456M tiles — which produced a measurably *better* representation
(tissue probe 0.9478 vs 0.9291, receptor subtype 0.976 vs 0.958) and **no** better pCR
prediction (0.669 vs 0.691). Two architecturally unrelated encoders converge on the
same response plateau near 0.67–0.69, which matches the largest published H&E-only
study of this endpoint.

---

## Pipeline

Each stage reads and writes cached artifacts, is seeded, and is idempotent — re-running
a completed stage is a no-op unless `--force` is passed.

```
raw WSIs
 ──01─▶ tiles/                  256×256 px @ 0.5 µm/px, tissue + blur filtered, Macenko normalised
 ──02─▶ simclr.pt               SimCLR ResNet-34 trained from scratch, NT-Xent τ=0.07
 ──  ─▶ linear-probe gate       9-class tissue probe; must clear 0.90 before the encoder is frozen
 ──03─▶ tile_emb/*.h5           frozen encoder over all tiles (write-once)
 ──04─▶ mil.pt                  gated-attention aggregator (trained; see note below)
 ──05─▶ paired.parquet          one row per patient, seeded patient-level folds
 ──07─▶ separability gate       GO/NO-GO on whether pCR is separable from the latents at all
 ──06─▶ diffusion_fold{0..4}.pt 5-fold conditional DDPM
 ──08─▶ biomarkers.json         likelihood ratio → calibrated P(pCR)
 ──09─▶ results/                experiments, figures, tables
```

Scripts `10`–`19` cover the external validation, the morphology-to-expression bridge,
the residual-disease clustering, the manuscript artifacts and the elimination
experiments. `scripts/analysis/` holds 17 focused diagnostic scripts; its `README.md`
maps each one to the number it produced.

## Layout

```
src/txmorph/
  data/         WSI reading, tissue masking, tiling, Macenko stain normalisation, h5 cache
  encoders/     SimCLR encoder, frozen foundation-model encoders, embedding extraction
  aggregation/  gated attention-MIL pooling (Ilse 2018)
  drug/         regimen encoder (one-hot; SMILES/Morgan path also implemented)
  diffusion/    noise schedule, forward/posterior, denoiser, sampler, context builder
  biomarkers/   biomarker derivation and the pCR-zone logistic
  training/     SimCLR, aggregator, diffusion and autoencoder trainers
  inference/    batched reverse sampling and likelihood-ratio scoring
  eval/         calibration, classification, survival, distribution tests, BH-FDR
  experiments/  exp1..exp5 orchestration
  viz/          one shared plotting style; figures and tables
  utils/        seeding, cross-validation and leakage guards, checkpointing, config, run logging
scripts/        thin CLI wrappers, 00..21, plus analysis/ data/ pipeline/
configs/        Hydra-style configs — all paths and hyperparameters live here
tests/          pytest suite
```

## Install

```bash
conda create -n txmorph python=3.11 -y && conda activate txmorph
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match the GPU driver
pip install -e ".[dev]"
pip install openslide-python rdkit lifelines umap-learn h5py pyarrow scikit-learn matplotlib statsmodels
```

OpenSlide also needs the system library (`apt-get install openslide-tools`, or
`conda install -c conda-forge openslide`).

## Tests

```bash
pytest -q
```

The mathematically load-bearing code is tested against closed-form and brute-force
references rather than against saved outputs, so these run **without a GPU and without
any cohort data**:

- `test_schedule.py` — ᾱ_t decreases monotonically 1 → 0; β̃_t ≤ β_t
- `test_diffusion_math.py` — `q_sample` marginal moments match the closed form;
  `q_posterior` matches a brute-force Bayes computation; the reverse mean reproduces the
  posterior mean when the network returns the true noise; a DDPM trained on a 2-D
  Gaussian mixture recovers the distribution
- `test_sampler.py` — batched sampler shapes, determinism under a fixed seed
- `test_biomarkers.py` — recovers known mean and variance
- `test_cv.py` — no patient identifier spans folds; fold-dependent fits exclude test patients
- `test_attention_mil.py` — permutation invariance; attention weights sum to 1
- `test_llr_calibrator.py` — pins a silent failure in which an unstandardised calibrator
  collapsed to an intercept and returned the cohort base rate for every patient

## Data

All cohorts are open access and required no data-use agreement. Nothing is
redistributed here.

| Cohort | Role | Access |
|---|---|---|
| IMPRESS (n=126) | primary training and evaluation, pCR labels | released with Huang et al., npj Precis Oncol 2023 |
| Post-NAT-BRCA (n=53) | residual-disease morphology clustering | TCIA, `10.7937/TCIA.2019.4YIBTJNO` |
| HER2-TUMOR-ROIS, Yale arm (n=85) | external validation (trastuzumab response) | TCIA, `10.7937/E65C-AM96` |
| TCGA-BRCA | encoder pretraining corpus and RNA-seq bridge | NCI GDC |
| NCT-CRC-HE-100K | encoder linear-probe gate only | Zenodo, `10.5281/zenodo.1214456` |

Fetch helpers are in `scripts/data/`. Set every path in `configs/paths.yaml` before
running anything; no path is hard-coded in `src/`.

## Reproducibility

Global seed 0, deterministic cuDNN kernels, bf16 autocast for training and fp32 for
evaluation. Cross-validation folds are assigned at the patient level and **stored** in
`paired.parquet` rather than recomputed, so exact partitions are reusable.
`utils.cv.assert_no_leakage` is called before training, and every fold-dependent fit —
scalers, penalty strengths, calibration maps, PCA, the likelihood-ratio calibrator —
happens inside the training fold. Each run writes its config, a package manifest, the
GPU identifier and the git SHA.

## Two notes an assessor should have

**The delivered system uses mean pooling, not the attention aggregator.** The
aggregator in `src/txmorph/aggregation/` is implemented and was trained, but the slide
latents used downstream are identical to a plain tile-bag mean to machine precision.
Pooling turned out not to be a lever: training the aggregator directly on pCR scored
0.675 against 0.691 for a plain mean. The header comment in
`configs/aggregator/mil.yaml` still asserts a stronger pooling claim that the
end-to-end results withdrew; the code is correct, the comment is stale.

**`scripts/19_architecture_ablation.py` has not been run.** It implements three
extensions — pCR-supervised attention pooling, diffusion in a low-dimensional PCA
response subspace, and semi-supervised pretraining on unlabelled residual-disease
slides. The second is the one most likely to change the headline result, since a
class-conditional Gaussian control peaks at 16 dimensions. No reported number depends
on any of the three.

## Licence

MIT. See `LICENSE`.
