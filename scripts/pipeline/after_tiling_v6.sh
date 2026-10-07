#!/usr/bin/env bash
# Wait for Post-NAT tiling, extract its embeddings, run Exp 3, then start the
# encoder retrain. Ordered so the v5-based results are all finished and archived
# before a new encoder exists to confuse them.
cd /home/anik-server/txmorph-wsi
source .venv/bin/activate
export PYTHONPATH=src:$PYTHONPATH
LOG=logs/chain_v6.log
: > "$LOG"

echo "[v6] $(date -Is) waiting for Post-NAT tiling" >> "$LOG"
while ! grep -q TILING_RC logs/tiling_postnat.log 2>/dev/null; do sleep 60; done
echo "[v6] $(date -Is) tiling done: $(grep TILING_RC logs/tiling_postnat.log)" >> "$LOG"

echo "[v6] $(date -Is) Exp 3 residual phenotypes (extract + cluster)" >> "$LOG"
python scripts/13_exp3_residual_phenotypes.py >> "$LOG" 2>&1
echo "[v6] exp3 rc=$?" >> "$LOG"

echo "[v6] $(date -Is) building v6 pretraining corpus (external cohort excluded)" >> "$LOG"
python scripts/14_build_pretrain_corpus.py >> "$LOG" 2>&1 || {
    echo "[v6] corpus build FAILED; NOT starting training" >> "$LOG"
    echo CHAIN_V6_RC=1 >> "$LOG"; exit 1; }

echo "[v6] $(date -Is) starting SimCLR v6" >> "$LOG"
python run_simclr_v6.py >> "$LOG" 2>&1
echo "[v6] simclr rc=$?" >> "$LOG"
echo CHAIN_V6_RC=0 >> "$LOG"
