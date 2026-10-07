#!/bin/bash
# Post-SimCLR pipeline: linear probe gate -> extract -> aggregate -> dataset -> diffusion
set -e
cd /home/anik-server/txmorph-wsi
source .venv/bin/activate
export PYTHONPATH=src:$PYTHONPATH

echo "=== STEP 1: Linear Probe Gate ==="
python run_linear_probe_gate.py --ckpt /home/anik-server/checkpoints/simclr.pt
if [ $? -ne 0 ]; then
    echo "GATE FAILED - stopping pipeline"
    exit 1
fi

echo ""
echo "=== STEP 2: Extract Embeddings ==="
python scripts/03_extract_embeddings.py
echo "Embeddings extracted."

echo ""
echo "=== STEP 3: Train Aggregator (MIL) ==="
python scripts/04_train_aggregator.py
echo "Aggregator trained."

echo ""
echo "=== STEP 4: Build Dataset (paired.parquet) ==="
python scripts/05_build_dataset.py --force
echo "Dataset built."

echo ""
echo "=== STEP 5: Train Diffusion (5-fold CV) ==="
python scripts/06_train_diffusion.py
echo "Diffusion models trained."

echo ""
echo "=== STEP 6: Kill-Switch UMAP ==="
python scripts/07_week12_umap.py
echo "Kill-switch evaluated."

echo ""
echo "=== STEP 7: Run Inference ==="
python scripts/08_run_inference.py
echo "Inference complete."

echo ""
echo "=== STEP 8: Run Experiments ==="
python scripts/09_run_experiments.py --experiment all
echo "Experiments complete."

echo ""
echo "=== PIPELINE COMPLETE ==="
date
