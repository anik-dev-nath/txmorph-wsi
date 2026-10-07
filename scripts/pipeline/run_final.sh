#!/usr/bin/env bash
# 06 diffusion -> 08 inference -> 09 experiments. Stops on failure rather than
# feeding the next stage a half-trained model.
cd /home/anik-server/txmorph-wsi
source .venv/bin/activate
export PYTHONPATH=src:$PYTHONPATH
LOG=logs/final_pipeline.log
: > "$LOG"

echo "[final] $(date -Is) 06 train_diffusion (5-fold)" >> "$LOG"
python scripts/06_train_diffusion.py >> "$LOG" 2>&1 || {
    echo "[final] 06 FAILED rc=$?" >> "$LOG"; echo FINAL_RC=1 >> "$LOG"; exit 1; }

echo "[final] $(date -Is) 08 run_inference" >> "$LOG"
python scripts/08_run_inference.py >> "$LOG" 2>&1 || {
    echo "[final] 08 FAILED rc=$?" >> "$LOG"; echo FINAL_RC=2 >> "$LOG"; exit 2; }

echo "[final] $(date -Is) 09 run_experiments (exp1, exp3, exp5)" >> "$LOG"
python scripts/09_run_experiments.py --experiment all >> "$LOG" 2>&1 || {
    echo "[final] 09 FAILED rc=$?" >> "$LOG"; echo FINAL_RC=3 >> "$LOG"; exit 3; }

echo "[final] $(date -Is) DONE" >> "$LOG"
echo FINAL_RC=0 >> "$LOG"
