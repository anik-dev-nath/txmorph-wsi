#!/usr/bin/env bash
# 20_build_unlabeled_corpus.sh — grow the unlabeled slide corpus used for
# unconditional pretraining of the conditional density (CLAUDE.md §6.5).
#
# WHY THIS EXISTS
#   scripts/19_architecture_ablation.py measured three architectural changes.
#   Only one helped: semi-supervised density pretraining on unlabeled slides
#   (AUROC 0.617 [0.516,0.720] vs 0.592 for the labeled-only model), and it did
#   so with just 53 unlabeled slides. It is also the only structural advantage a
#   generative model has over the discriminative baseline — a logistic regression
#   cannot consume unlabeled slides at all. This script scales that corpus.
#
# WHAT IS AND IS NOT ELIGIBLE
#   Eligible: slides unrelated to the labeled IMPRESS folds AND to the held-out
#   Yale trastuzumab arm used for external validation.
#     - Post-NAT-BRCA  (TCIA pkg 560) — residual disease, no pCR label, not Yale.
#     - TCGA-BRCA      (GDC)          — many institutions, not Yale, not IMPRESS.
#   NOT eligible, deliberately excluded:
#     - Yale_trastuzumab_response_cohort (85 slides) — the external test set.
#     - Yale_HER2_cohort (192 slides)   — SAME INSTITUTION as the external test
#       set. Pretraining the density on same-institution slides would let it
#       adapt to that scanner/stain domain and inflate external performance
#       without any genuine gain in generalisation. Cohort-discrimination AUROC
#       on this data is 0.94 (docs/LIMITATIONS.md §5), so this is a live risk,
#       not a hypothetical one. Do not add them back for a better external number.
#
# DISK
#   The tile cache is already 276 GB of a 492 GB volume. Each batch is
#   downloaded, tiled, embedded, and then its raw slides and tiles are deleted,
#   so peak additional usage is one batch (~5 GB) rather than the whole corpus.
#   The run stops starting new batches below MIN_FREE_GB.
#
# RESUMABILITY
#   A slide is skipped when its .h5 already exists in the output dir, so the
#   script can be killed and relaunched freely. Idempotent.
#
# USAGE (on the server, inside tmux)
#   tmux new-session -d -s corpus \
#     "cd /home/anik-server/txmorph-wsi && bash scripts/20_build_unlabeled_corpus.sh \
#      >> logs/unlabeled_corpus.log 2>&1; echo CORPUS_RC=\$? >> logs/unlabeled_corpus.log"
set -uo pipefail

REPO=${REPO:-/home/anik-server/txmorph-wsi}
DATA=${DATA:-/home/anik-server/data}
PY=${PY:-$REPO/.venv/bin/python}

RAW="$DATA/raw_wsi"
TILES="$DATA/tiles"
EMB="${EMB:-$DATA/tile_emb_unlabeled}"
STAGE="tcga_unlabeled"                  # subdir of raw_wsi that batches land in
FULL_MANIFEST="${FULL_MANIFEST:-$DATA/gdc_manifest_tcga_brca.txt}"

N_SLIDES=${N_SLIDES:-400}               # seeded subset size drawn from the manifest
BATCH=${BATCH:-8}
CAP_TILES=${CAP_TILES:-600}             # per slide; a slide mean is stable well below this
MIN_FREE_GB=${MIN_FREE_GB:-40}
WORKERS=${WORKERS:-3}                   # 4-core box; leave one core for extraction
SEED=${SEED:-0}

mkdir -p "$EMB" "$RAW/$STAGE"
cd "$REPO" || exit 1
export PYTHONPATH=src

free_gb() { df -BG --output=avail "$DATA" | tail -1 | tr -dc '0-9'; }
log() { echo "[$(date '+%F %T')] $*"; }

log "==================== START 20_build_unlabeled_corpus ===================="
log "emb=$EMB  n=$N_SLIDES  batch=$BATCH  cap=$CAP_TILES  free=$(free_gb)G"

# ---------------------------------------------------------------------------
# Phase 0 — top up cohorts whose raw slides are already on disk (Post-NAT-BRCA).
# 96 slides exist; only 53 were ever embedded. Raw stays (it is a primary cohort
# for Exp 3), so only tiles are reclaimed.
# ---------------------------------------------------------------------------
log "--- phase 0: Post-NAT-BRCA top-up ---"
$PY scripts/01_preprocess_tiles.py --subdir post_nat_brca --workers "$WORKERS" \
  || log "WARN: post-nat tiling returned $?"

$PY - "$RAW/post_nat_brca" "$EMB" <<'PY' > /tmp/postnat_batch.csv
import sys, pathlib
raw, emb = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
print("slide_id")
for p in sorted(raw.rglob("*.svs")):
    if not (emb / f"{p.stem}.h5").exists():
        print(p.stem)
PY
log "post-nat slides needing embeddings: $(($(wc -l < /tmp/postnat_batch.csv) - 1))"
$PY scripts/03_extract_embeddings.py --slides-csv /tmp/postnat_batch.csv --out-dir "$EMB" \
  || log "WARN: post-nat extraction returned $?"

# ---------------------------------------------------------------------------
# Phase 1 — TCGA-BRCA, batched: download -> tile -> cap -> embed -> delete.
# ---------------------------------------------------------------------------
if [ ! -f "$FULL_MANIFEST" ]; then
  log "no manifest at $FULL_MANIFEST — building one"
  $PY scripts/data/build_gdc_manifest.py --project TCGA-BRCA --out "$FULL_MANIFEST" \
    || { log "FATAL: manifest build failed"; exit 1; }
fi

SUB=/tmp/unlabeled_subset.txt
$PY - "$FULL_MANIFEST" "$SUB" "$N_SLIDES" "$SEED" <<'PY'
import random, sys
full, out, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
lines = [l for l in open(full).read().splitlines() if l.strip()]
header, data = lines[0], lines[1:]
random.Random(seed).shuffle(data)          # seeded => the corpus is reproducible
open(out, "w").write("\n".join([header] + data[:n]) + "\n")
print(f"[subset] {min(n, len(data))} of {len(data)} slides -> {out}")
PY

HEADER=$(head -1 "$SUB")
tail -n +2 "$SUB" > /tmp/unlab_rows.txt
rm -f /tmp/unlab_batch_*
split -l "$BATCH" /tmp/unlab_rows.txt /tmp/unlab_batch_

for b in /tmp/unlab_batch_*; do
  avail=$(free_gb)
  log "----- batch $(basename "$b")  free=${avail}G  embedded=$(ls "$EMB"/*.h5 2>/dev/null | wc -l) -----"
  if [ "$avail" -lt "$MIN_FREE_GB" ]; then
    log "[guard] free ${avail}G < ${MIN_FREE_GB}G — stopping. Corpus so far is usable."
    break
  fi

  # skip rows already embedded (resume support)
  : > /tmp/unlab_todo.txt
  while IFS= read -r row; do
    [ -z "$row" ] && continue
    sid=$(printf '%s\n' "$row" | cut -f2); sid=${sid%.svs}
    [ -f "$EMB/$sid.h5" ] || printf '%s\n' "$row" >> /tmp/unlab_todo.txt
  done < "$b"
  if [ ! -s /tmp/unlab_todo.txt ]; then log "all slides in batch already embedded"; continue; fi

  printf '%s\n' "$HEADER" > /tmp/unlab_batch_manifest.txt
  cat /tmp/unlab_todo.txt >> /tmp/unlab_batch_manifest.txt

  log "downloading $(wc -l < /tmp/unlab_todo.txt) slides"
  gdc-client download -m /tmp/unlab_batch_manifest.txt -d "$RAW/$STAGE" --no-file-md5sum \
    >/dev/null 2>&1 || log "WARN: gdc-client returned $?"

  # gdc-client nests each file in a uuid dir; flatten so the tiler sees the slides
  find "$RAW/$STAGE" -mindepth 2 -name "*.svs" -exec mv -t "$RAW/$STAGE" {} + 2>/dev/null
  find "$RAW/$STAGE" -mindepth 1 -type d -empty -delete 2>/dev/null

  n_raw=$(find "$RAW/$STAGE" -maxdepth 1 -name "*.svs" | wc -l)
  if [ "$n_raw" -eq 0 ]; then log "WARN: nothing downloaded, skipping batch"; continue; fi

  log "tiling $n_raw slides"
  $PY scripts/01_preprocess_tiles.py --subdir "$STAGE" --workers "$WORKERS" \
    || log "WARN: tiling returned $?"

  # cap tiles per slide — the slide-level mean is stable far below CAP_TILES,
  # and this bounds both extraction time and peak disk
  while IFS= read -r row; do
    sid=$(printf '%s\n' "$row" | cut -f2); sid=${sid%.svs}
    d="$TILES/$sid"
    [ -d "$d" ] || continue
    n=$(find "$d" -name "*.npy" | wc -l)
    if [ "$n" -gt "$CAP_TILES" ]; then
      find "$d" -name "*.npy" | sort | tail -n +$((CAP_TILES + 1)) | xargs -r rm -f
    fi
  done < /tmp/unlab_todo.txt

  { echo "slide_id"; cut -f2 /tmp/unlab_todo.txt | sed 's/\.svs$//'; } > /tmp/unlab_batch.csv
  log "extracting embeddings"
  $PY scripts/03_extract_embeddings.py --slides-csv /tmp/unlab_batch.csv --out-dir "$EMB" \
    || log "WARN: extraction returned $?"

  # reclaim: raw slides and tiles are both regenerable from the seeded manifest
  while IFS= read -r row; do
    sid=$(printf '%s\n' "$row" | cut -f2); sid=${sid%.svs}
    rm -rf "$TILES/$sid"
  done < /tmp/unlab_todo.txt
  find "$RAW/$STAGE" -maxdepth 1 -name "*.svs" -delete
  find "$RAW/$STAGE" -mindepth 1 -type d -empty -delete 2>/dev/null
done

log "==================== DONE — $(ls "$EMB"/*.h5 2>/dev/null | wc -l) unlabeled embeddings in $EMB ===================="
log "free=$(free_gb)G"
