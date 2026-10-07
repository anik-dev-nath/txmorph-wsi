#!/usr/bin/env bash
# Wait for 01_preprocess_tiles to finish, then rebuild + validate the IMPRESS
# manifests across both cohort sheets. Chained so the manifests are ready the
# moment tiling completes rather than waiting on a human.
cd /home/anik-server/txmorph-wsi
source .venv/bin/activate
export PYTHONPATH=src:$PYTHONPATH
LOG=logs/after_tiling.log
echo "[chain] $(date -Is) waiting for tiling to finish" >> "$LOG"
while ! grep -q TILING_RC logs/tiling_impress.log 2>/dev/null; do sleep 60; done
echo "[chain] $(date -Is) tiling done: $(grep TILING_RC logs/tiling_impress.log)" >> "$LOG"
python scripts/data/build_impress_manifests.py \
    --meta /home/anik-server/data/free_data_meta/IMPRESS/meta/supp_2_cohort_meta.xlsx \
    --data /home/anik-server/data >> "$LOG" 2>&1
echo "[chain] manifest rc=$?" >> "$LOG"
python scripts/data/make_manifests.py validate --data /home/anik-server/data >> "$LOG" 2>&1
echo "[chain] validate rc=$?" >> "$LOG"
echo "[chain] $(date -Is) CHAIN_DONE" >> "$LOG"
