#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_TAG="phase6_1"
PARTITION="${PARTITION:-debug}"
NODES="${NODES:-4}"
TASKS_PER_NODE=4
TOTAL_TASKS=$((NODES * TASKS_PER_NODE))  # 16
GPUS_PER_TASK=$((8 / TASKS_PER_NODE))    # 2
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_TOKENS="${MAX_TOKENS:-512}"
LIMIT="${LIMIT:-0}"

RUN_DIR="${PROJECT_ROOT}/data/${RUN_TAG}"
SHARD_DIR="${RUN_DIR}/shards"
SHARD_OUT_DIR="${RUN_DIR}/shard_outputs"
mkdir -p "$SHARD_DIR" "$SHARD_OUT_DIR"

source "${PROJECT_ROOT}/.venv/bin/activate"

# ── Step 1: Prepare shards ──────────────────────────────────────────────────

PREP_ARGS=(
  --out-dir "$SHARD_DIR"
  --n-shards "$TOTAL_TASKS"
)
if [[ "$LIMIT" -gt 0 ]]; then
  PREP_ARGS+=(--limit "$LIMIT")
fi

python "${SCRIPT_DIR}/prepare_patch_embedding_shards.py" "${PREP_ARGS[@]}"

# ── Step 2: srun (4 nodes × 4 tasks, 2 GPUs each) ───────────────────────────

srun -N "$NODES" -p "$PARTITION" --ntasks-per-node="$TASKS_PER_NODE" \
     ${NODELIST:+--nodelist=$NODELIST} \
     bash -lc '
  set -euo pipefail
  source "'"${PROJECT_ROOT}"'/.venv/bin/activate"
  # Assign 2 GPUs per task based on local task rank (0-3 → GPUs 0-1, 2-3, 4-5, 6-7)
  local_rank="${SLURM_LOCALID:-0}"
  gpu_start=$((local_rank * '"$GPUS_PER_TASK"'))
  gpu_end=$((gpu_start + '"$GPUS_PER_TASK"' - 1))
  gpu_list=$(seq -s, "$gpu_start" "$gpu_end")
  export CUDA_VISIBLE_DEVICES="$gpu_list"
  echo "Task ${SLURM_PROCID} on $(hostname): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  shard=$(printf "%03d" "${SLURM_PROCID}")
  shard_file="'"$SHARD_DIR"'/shard_${shard}.txt"
  n=$(wc -l < "$shard_file" | tr -d " ")
  if [[ "$n" == "0" ]]; then
    echo "Empty shard ${shard}, skipping"
    exit 0
  fi
  python "'"${SCRIPT_DIR}"'/extract_patch_embeddings.py" \
    --instance-ids-file "$shard_file" \
    --out "'"$SHARD_OUT_DIR"'/shard_${shard}.npz" \
    --batch-size '"$BATCH_SIZE"' \
    --max-tokens '"$MAX_TOKENS"' \
    --gpus '"$GPUS_PER_TASK"'
'

# ── Step 3: Merge shards into single NPZ ────────────────────────────────────

python -c "
import numpy as np, glob, os
shards = sorted(glob.glob('$SHARD_OUT_DIR/shard_*.npz'))
assert shards, 'No shard NPZ files found!'
arrays = {}
for s in shards:
    d = np.load(s, allow_pickle=True)
    for k in d.files:
        arrays.setdefault(k, []).append(d[k])
merged = {k: np.concatenate(v) for k, v in arrays.items()}
out = '$RUN_DIR/patch_embeddings_100k.npz'
np.savez_compressed(out, **merged)
z = merged['z_patch']
acc = merged['accepted']
print(f'Merged {len(shards)} shards -> {out}')
print(f'  z_patch: {z.shape}')
print(f'  accepted rate: {acc.mean():.3f}')
print(f'  NaN check: {np.any(np.isnan(z))}')
"

echo "Done. Merged embeddings -> ${RUN_DIR}/patch_embeddings_100k.npz"
