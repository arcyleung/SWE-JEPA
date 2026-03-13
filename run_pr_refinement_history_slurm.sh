#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

RUN_TAG="${1:-phase4_7_2_slurm}"
PARTITION="${PARTITION:-debug}"
NODES="${NODES:-12}"
WORKERS="${WORKERS:-224}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$WORKERS}"
LIMIT="${LIMIT:-0}"
MIN_COMMITS="${MIN_COMMITS:-2}"
MIN_FILES="${MIN_FILES:-1}"
MAX_FILES="${MAX_FILES:-40}"
MIN_LINES="${MIN_LINES:-10}"
MAX_LINES="${MAX_LINES:-10000}"
ORDER="${ORDER:-newest}"
REQUIRE_REVIEW="${REQUIRE_REVIEW:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
GIT_TIMEOUT_SEC="${GIT_TIMEOUT_SEC:-600}"
MAX_SNAPSHOTS="${MAX_SNAPSHOTS:-5}"
PR_TIMEOUT_SEC="${PR_TIMEOUT_SEC:-300}"
HEAVY_COST_THRESHOLD="${HEAVY_COST_THRESHOLD:-1200}"
HEAVY_COMMITS_THRESHOLD="${HEAVY_COMMITS_THRESHOLD:-80}"
HEAVY_FILES_THRESHOLD="${HEAVY_FILES_THRESHOLD:-30}"
HEAVY_LINES_THRESHOLD="${HEAVY_LINES_THRESHOLD:-6000}"
REPO_CACHE_COPY_CONCURRENCY="${REPO_CACHE_COPY_CONCURRENCY:-4}"
REPO_CACHE_SHM_LOW_WATERMARK_BYTES="${REPO_CACHE_SHM_LOW_WATERMARK_BYTES:-34359738368}"
CONWAY_BLAME_CACHE_MAX_ENTRIES="${CONWAY_BLAME_CACHE_MAX_ENTRIES:-2048}"
REPO_RAMDISK_CACHE_BASE="${REPO_RAMDISK_CACHE_BASE:-/dev/shm/phase4_7_2_repo_cache}"
REPO_OVERLAY_MERGED_BASE="${REPO_OVERLAY_MERGED_BASE:-/dev/shm/phase4_7_2_overlay_merged}"
REPO_OVERLAY_TMP_BASE="${REPO_OVERLAY_TMP_BASE:-/dev/shm/phase4_7_2_overlay_tmp}"
if [[ "$RUN_TAG" == phase4_7_2* ]]; then
  REPORT_OUT="${REPORT_OUT:-docs/phase4_7_2_pr_refinement_history.md}"
else
  REPORT_OUT="${REPORT_OUT:-docs/${RUN_TAG}.md}"
fi

RUN_DIR="data/${RUN_TAG}"
SHARD_DIR="${RUN_DIR}/shards"
SHARD_OUT_DIR="${RUN_DIR}/shard_outputs"
HEAVY_QUEUE_DIR="${RUN_DIR}/heavy_queue"
mkdir -p "$SHARD_DIR" "$SHARD_OUT_DIR" "$HEAVY_QUEUE_DIR"

source .venv/bin/activate

PREP_ARGS=(
  --out-dir "$SHARD_DIR"
  --n-shards "$NODES"
  --limit "$LIMIT"
  --min-commits "$MIN_COMMITS"
  --min-files "$MIN_FILES"
  --max-files "$MAX_FILES"
  --min-lines "$MIN_LINES"
  --max-lines "$MAX_LINES"
  --order "$ORDER"
  --heavy-out-dir "$HEAVY_QUEUE_DIR"
  --heavy-cost-threshold "$HEAVY_COST_THRESHOLD"
  --heavy-commits-threshold "$HEAVY_COMMITS_THRESHOLD"
  --heavy-files-threshold "$HEAVY_FILES_THRESHOLD"
  --heavy-lines-threshold "$HEAVY_LINES_THRESHOLD"
)
if [[ "$REQUIRE_REVIEW" == "1" ]]; then
  PREP_ARGS+=(--require-review-events)
fi
REVIEW_FLAG=""
if [[ "$REQUIRE_REVIEW" == "1" ]]; then
  REVIEW_FLAG="--require-review-events"
fi
export REVIEW_FLAG
export PROGRESS_EVERY
export GIT_TIMEOUT_SEC
export MAX_SNAPSHOTS
export PR_TIMEOUT_SEC
export PR_REFINEMENT_GIT_TIMEOUT_SEC="$GIT_TIMEOUT_SEC"
export CONWAY_GIT_TIMEOUT_SEC="$GIT_TIMEOUT_SEC"
export PR_REFINEMENT_PR_TIMEOUT_SEC="$PR_TIMEOUT_SEC"
export REPO_CACHE_COPY_CONCURRENCY
export REPO_CACHE_SHM_LOW_WATERMARK_BYTES
export CONWAY_BLAME_CACHE_MAX_ENTRIES
export REPO_RAMDISK_CACHE_BASE
export REPO_OVERLAY_MERGED_BASE
export REPO_OVERLAY_TMP_BASE

python prepare_pr_refinement_history_shards.py "${PREP_ARGS[@]}"

srun -N "$NODES" -p "$PARTITION" -n "$NODES" --ntasks-per-node=1 --cpus-per-task="$CPUS_PER_TASK" --cpu-bind=none bash -lc '
  set -euo pipefail
  cd "'"$ROOT"'"
  source .venv/bin/activate
  shard=$(printf "%03d" "${SLURM_PROCID}")
  shard_file="'"$SHARD_DIR"'/shard_${shard}.txt"
  out_jsonl="'"$SHARD_OUT_DIR"'/shard_${shard}.jsonl"
  out_summary="'"$SHARD_OUT_DIR"'/shard_${shard}_summary.json"
  out_report="'"$SHARD_OUT_DIR"'/shard_${shard}.md"
  n=$(wc -l < "$shard_file" | tr -d " ")
  if [[ "$n" == "0" ]]; then
    : > "$out_jsonl"
    printf "{\n  \"n_rows\": 0,\n  \"n_prs\": 0,\n  \"empty_shard\": true\n}\n" > "$out_summary"
    exit 0
  fi
  python extract_pr_refinement_history.py \
    --limit "$n" \
    --source-limit "$n" \
    --workers "${SLURM_CPUS_PER_TASK:-'"$WORKERS"'}" \
    --run-tag "'"$RUN_TAG"'" \
    --progress-every "'"$PROGRESS_EVERY"'" \
    --max-snapshots "'"$MAX_SNAPSHOTS"'" \
    --instance-ids-file "$shard_file" \
    --min-commits "'"$MIN_COMMITS"'" \
    --min-files "'"$MIN_FILES"'" \
    --max-files "'"$MAX_FILES"'" \
    --min-lines "'"$MIN_LINES"'" \
    --max-lines "'"$MAX_LINES"'" \
    --order "'"$ORDER"'" \
    ${REVIEW_FLAG:+$REVIEW_FLAG} \
    --out "$out_jsonl" \
    --summary-out "$out_summary" \
    --report-out "$out_report"
'

python summarize_pr_refinement_history.py \
  --glob "${SHARD_OUT_DIR}/shard_*.jsonl" \
  --out-jsonl "${RUN_DIR}/merged.jsonl" \
  --summary-out "${RUN_DIR}/summary.json" \
  --report-out "${REPORT_OUT}" \
  --limit "$LIMIT"

echo "Run dir -> ${RUN_DIR}"
echo "Deferred heavy queue -> ${HEAVY_QUEUE_DIR}/heavy_queue.txt"
