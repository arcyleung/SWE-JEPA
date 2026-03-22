#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

N_GPUS="${N_GPUS:-8}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_TOKENS="${MAX_TOKENS:-512}"
LIMIT="${LIMIT:-0}"

OUT_DIR="${PROJECT_ROOT}/data/phase6_1"
mkdir -p "$OUT_DIR"

source "${PROJECT_ROOT}/.venv/bin/activate"

ARGS=(
  --out "$OUT_DIR/patch_embeddings_100k.npz"
  --n-gpus "$N_GPUS"
  --workers-per-gpu "$WORKERS_PER_GPU"
  --batch-size "$BATCH_SIZE"
  --max-tokens "$MAX_TOKENS"
)
if [[ "$LIMIT" -gt 0 ]]; then
  ARGS+=(--limit "$LIMIT")
fi

echo "Launching: ${N_GPUS} GPUs × ${WORKERS_PER_GPU} workers/GPU = $((N_GPUS * WORKERS_PER_GPU)) workers"
python "${SCRIPT_DIR}/extract_patch_embeddings.py" "${ARGS[@]}"
