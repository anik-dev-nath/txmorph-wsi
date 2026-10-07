#!/usr/bin/env bash
# Chain 04 -> 05 -> 07 once extraction finishes, so the kill-switch lands without
# babysitting. Each step stops the chain on failure rather than feeding the next
# stage a half-built artifact.
cd /home/anik-server/txmorph-wsi
source .venv/bin/activate
export PYTHONPATH=src:$PYTHONPATH
LOG=logs/after_extract.log

echo "[chain2] $(date -Is) waiting for extraction" >> "$LOG"
while ! grep -q EXTRACT_RC logs/extract_impress.log 2>/dev/null; do sleep 60; done
RC=$(grep EXTRACT_RC logs/extract_impress.log | tail -1)
echo "[chain2] $(date -Is) extraction done: $RC" >> "$LOG"
case "$RC" in *=0) ;; *) echo "[chain2] extraction FAILED; stopping" >> "$LOG"; exit 1;; esac

echo "[chain2] $(date -Is) 04 train_aggregator" >> "$LOG"
python scripts/04_train_aggregator.py >> "$LOG" 2>&1 || {
    echo "[chain2] 04 FAILED rc=$?; stopping" >> "$LOG"; exit 1; }

echo "[chain2] $(date -Is) 05 build_dataset" >> "$LOG"
python scripts/05_build_dataset.py --force >> "$LOG" 2>&1 || {
    echo "[chain2] 05 FAILED rc=$?; stopping" >> "$LOG"; exit 1; }

echo "[chain2] $(date -Is) 07 KILL-SWITCH" >> "$LOG"
python scripts/07_week12_umap.py >> "$LOG" 2>&1
echo "[chain2] 07 rc=$?" >> "$LOG"
echo "[chain2] $(date -Is) CHAIN2_DONE" >> "$LOG"
