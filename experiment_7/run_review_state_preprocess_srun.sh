#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_TAG="${RUN_TAG:-phase7_1_review_state_preprocessed_slurm_v1}"
PARTITION="${PARTITION:-debug}"
NODES="${NODES:-22}"
# debug has 10x 192-core bm nodes and 12x 224-core cyber nodes.
# Use 192 by default so a 22-node launch can schedule on the whole partition.
CPUS_PER_TASK="${CPUS_PER_TASK:-192}"
MAX_TOKENS="${MAX_TOKENS:-384}"
MAX_PATCH_CHARS="${MAX_PATCH_CHARS:-0}"
HASH_VOCAB_SIZE="${HASH_VOCAB_SIZE:-32768}"
TAG_BATCH_SIZE="${TAG_BATCH_SIZE:-64}"
FETCH_CHUNK_SIZE="${FETCH_CHUNK_SIZE:-1024}"
PREPROCESS_PROGRESS_EVERY="${PREPROCESS_PROGRESS_EVERY:-1024}"
LIMIT="${LIMIT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_SHARDS="${SKIP_SHARDS:-}"
PROJECTED="${PROJECTED:-}"
SUPER_CLUSTERS="${SUPER_CLUSTERS:-}"
PG_CONFIG="${PG_CONFIG:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"

RUN_DIR="${PROJECT_ROOT}/data/${RUN_TAG}"
SHARD_DIR="${RUN_DIR}/shards"
mkdir -p "$SHARD_DIR"
export RUN_DIR

source "${PROJECT_ROOT}/.venv/bin/activate"

echo "[Exp7.1] Preprocessing review-state cache on ${NODES} nodes x ${CPUS_PER_TASK} CPUs"
echo "  RUN_DIR=${RUN_DIR}"
echo "  PARTITION=${PARTITION}"
echo "  TAG_BATCH_SIZE=${TAG_BATCH_SIZE}"
echo "  FETCH_CHUNK_SIZE=${FETCH_CHUNK_SIZE}"
echo "  PREPROCESS_PROGRESS_EVERY=${PREPROCESS_PROGRESS_EVERY}"
echo "  SKIP_EXISTING=${SKIP_EXISTING}"
echo "  SKIP_SHARDS=${SKIP_SHARDS}"
echo "  PROJECTED=${PROJECTED:-<default>}"
echo "  SUPER_CLUSTERS=${SUPER_CLUSTERS:-<default>}"
echo "  PG_CONFIG=${PG_CONFIG:-<default>}"
echo "  TOKENIZER_PATH=${TOKENIZER_PATH:-<default>}"

srun -N "$NODES" -p "$PARTITION" -n "$NODES" --ntasks-per-node=1 \
     --input=none \
     --cpus-per-task="$CPUS_PER_TASK" --cpu-bind=none \
     ${NODELIST:+--nodelist=$NODELIST} \
     bash -lc '
  set -euo pipefail
  cd "'"${PROJECT_ROOT}"'"
  source .venv/bin/activate
  shard=$(printf "%03d" "${SLURM_PROCID}")
  out_npz="'"${SHARD_DIR}"'/shard_${shard}.npz"
  out_json="'"${SHARD_DIR}"'/shard_${shard}_summary.json"
  out_log="'"${SHARD_DIR}"'/shard_${shard}.log"
  skip_shards=",'"${SKIP_SHARDS}"',"
  if [[ "$skip_shards" == *,"${shard}",* ]]; then
    printf "Skipping shard %s because it is listed in SKIP_SHARDS=%s\n" "$shard" "'"${SKIP_SHARDS}"'" > "$out_log"
    exit 0
  fi
  if [[ "'"${SKIP_EXISTING}"'" == "1" && -s "$out_npz" && -s "$out_json" ]]; then
    printf "Skipping shard %s because outputs already exist:\n%s\n%s\n" "$shard" "$out_npz" "$out_json" > "$out_log"
    exit 0
  fi
  args=(
    --out "$out_npz"
    --summary-out "$out_json"
    --shard-modulus "${SLURM_NTASKS}"
    --shard-remainder "${SLURM_PROCID}"
    --preprocess-workers "${SLURM_CPUS_PER_TASK:-'"${CPUS_PER_TASK}"'}"
    --max-tokens "'"${MAX_TOKENS}"'"
    --max-patch-chars "'"${MAX_PATCH_CHARS}"'"
    --hash-vocab-size "'"${HASH_VOCAB_SIZE}"'"
    --tag-batch-size "'"${TAG_BATCH_SIZE}"'"
    --fetch-chunk-size "'"${FETCH_CHUNK_SIZE}"'"
    --preprocess-progress-every "'"${PREPROCESS_PROGRESS_EVERY}"'"
  )
  if [[ "'"${LIMIT}"'" -gt 0 ]]; then
    args+=(--limit "'"${LIMIT}"'")
  fi
  if [[ -n "'"${PROJECTED}"'" ]]; then
    args+=(--projected "'"${PROJECTED}"'")
  fi
  if [[ -n "'"${SUPER_CLUSTERS}"'" ]]; then
    args+=(--super-clusters "'"${SUPER_CLUSTERS}"'")
  fi
  if [[ -n "'"${PG_CONFIG}"'" ]]; then
    args+=(--pg-config "'"${PG_CONFIG}"'")
  fi
  if [[ -n "'"${TOKENIZER_PATH}"'" ]]; then
    args+=(--tokenizer-path "'"${TOKENIZER_PATH}"'")
  fi
  python experiment_7/preprocess_review_state_shard.py "${args[@]}" > "$out_log" 2>&1
'

python - <<'PY'
import glob
import json
import os

run_dir = os.environ["RUN_DIR"]
shard_dir = os.path.join(run_dir, "shards")
summary_paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*_summary.json")))
npz_paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))

agg = {
    "n_shards": len(summary_paths),
    "n_npz_files": len(npz_paths),
    "n_input_rows": 0,
    "n_joined_rows": 0,
    "n_missing_patch": 0,
    "n_missing_cluster": 0,
    "config": {},
}

for path in summary_paths:
    with open(path) as f:
        row = json.load(f)
    agg["n_input_rows"] += int(row.get("n_input_rows", 0))
    agg["n_joined_rows"] += int(row.get("n_joined_rows", 0))
    agg["n_missing_patch"] += int(row.get("n_missing_patch", 0))
    agg["n_missing_cluster"] = max(agg["n_missing_cluster"], int(row.get("n_missing_cluster", 0)))
    if not agg["config"]:
        agg["config"] = row.get("config", {})

summary_out = os.path.join(run_dir, "preprocess_summary.json")
with open(summary_out, "w") as f:
    json.dump(agg, f, indent=2)

print(json.dumps(agg, indent=2))
print(f"\nShard dir -> {shard_dir}")
print(f"Summary   -> {summary_out}")
PY

echo ""
echo "[Exp7.1] Preprocessing complete."
echo "Train from cache with:"
echo "  python ${PROJECT_ROOT}/experiment_7/train_review_state_student.py --device cuda --preprocessed-dir ${SHARD_DIR}"
