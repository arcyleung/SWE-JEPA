#!/usr/bin/env bash
# Backfill unified diffs into prs_copy.file_patches — one shard per srun task.
#
# Launch across 18 nodes:
#   srun --nodes=18 --ntasks=18 --ntasks-per-node=1 --cpus-per-task=32 \
#        --time=6:00:00 --partition=debug \
#        bash experiment_6/run_backfill_patches_srun.sh
#
# SLURM_PROCID (0–17) = shard index.
# Each shard uses ProcessPoolExecutor(max_workers=32) internally.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FOLLOWUP_DIR="/shared_workspace_mfs/arthur/crawled-prs-hf/prs-followup-analysis"

NUM_SHARDS="${NUM_SHARDS:-18}"
SHARD_INDEX="${SLURM_PROCID:-0}"
MAX_WORKERS="${MAX_WORKERS:-32}"

PG_CONFIG="${FOLLOWUP_DIR}/postgres_connection.yaml"
REPOS_BASE="/shared_workspace_mfs/repos"

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"

exec > "${LOG_DIR}/backfill_patches_shard_${SHARD_INDEX}.log" 2>&1

source "${PROJECT_ROOT}/.venv/bin/activate"

echo "============================================================"
echo "Backfill file_patches — Shard ${SHARD_INDEX}/${NUM_SHARDS}"
echo "  Node:        $(hostname)"
echo "  Max workers: ${MAX_WORKERS}"
echo "  Started:     $(date)"
echo "============================================================"

python "${FOLLOWUP_DIR}/backfill_file_patches.py" \
    --pg-config-file "${PG_CONFIG}" \
    --repos-base "${REPOS_BASE}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${SHARD_INDEX}" \
    --max-workers "${MAX_WORKERS}"

echo ""
echo "Shard ${SHARD_INDEX} done at $(date)"
