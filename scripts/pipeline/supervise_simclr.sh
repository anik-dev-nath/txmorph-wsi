#!/bin/bash
# Supervisor: keep SimCLR alive; it resumes from checkpoint on every restart.
# v5 = GPU-side augmentation (see scripts/run_simclr_v5.py).
cd /home/anik-server/txmorph-wsi
source .venv/bin/activate
export PYTHONPATH=src:$PYTHONPATH
LOG=logs/simclr_stdout.log
SUP=logs/simclr_supervisor.log
for i in $(seq 1 200); do
    echo "[supervisor] $(date -Is) attempt $i starting (v5 gpu-aug)" | tee -a "$SUP" >> "$LOG"
    python run_simclr_v5.py >> "$LOG" 2>&1
    rc=$?
    echo "[supervisor] $(date -Is) exited rc=$rc" | tee -a "$SUP" >> "$LOG"
    if grep -q "TRAINING COMPLETE" "$LOG"; then
        echo "[supervisor] $(date -Is) training complete; stopping" | tee -a "$SUP" >> "$LOG"
        break
    fi
    sleep 20
done
