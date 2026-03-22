#!/usr/bin/env bash
# Followup extraction worker — one shard per srun task.
#
# Launch all 22 shards across 22 nodes (1 task per node, Python does internal parallelism):
#   srun --nodes=22 --ntasks=22 --ntasks-per-node=1 --cpus-per-task=8 \
#        --time=12:00:00 --partition=debug \
#        bash experiment_6/run_followup_extraction_srun.sh
#
# SLURM_PROCID (0–21) = shard index.  Each shard uses ProcessPoolExecutor
# internally for Stage B (max_workers=8), so only 1 srun task per node is needed.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FOLLOWUP_DIR="/shared_workspace_mfs/arthur/crawled-prs-hf/prs-followup-analysis"

NUM_SHARDS="${NUM_SHARDS:-22}"
SHARD_INDEX="${SLURM_PROCID:-0}"
MAX_WORKERS="${MAX_WORKERS:-16}"

PG_CONFIG="${FOLLOWUP_DIR}/postgres_connection.yaml"
TOKENS_FILE="${FOLLOWUP_DIR}/crawl_tokens.yaml"
REPOS_BASE="/shared_workspace_mfs/repos"

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"

# Redirect this shard's output to its own log file
exec > "${LOG_DIR}/followup_shard_${SHARD_INDEX}.log" 2>&1

source "${PROJECT_ROOT}/.venv/bin/activate"

echo "============================================================"
echo "Shard ${SHARD_INDEX}/${NUM_SHARDS} on $(hostname) at $(date)"
echo "  PID: $$  SLURM_PROCID: ${SLURM_PROCID:-unset}"
echo "  Max workers: ${MAX_WORKERS}"
echo "  /dev/shm free: $(df -h /dev/shm | tail -1 | awk '{print $4}')"
echo "============================================================"

# Clean stale overlayfs from prior runs
for mnt in /tmp/repos_tmp_overlayfs/*/; do
    mountpoint -q "$mnt" 2>/dev/null && fusermount3 -u "$mnt" 2>/dev/null || true
done
rm -rf /dev/shm/ovl-upper-* /dev/shm/ovl-work-* 2>/dev/null || true

# ── Stage A: file-level followups ─────────────────────────────────────────
echo ""
echo ">>> Stage A: file-level followup extraction (shard ${SHARD_INDEX})"
python "${FOLLOWUP_DIR}/analyze_feature_followups_v2_pg.py" \
    --pg-config-file "${PG_CONFIG}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${SHARD_INDEX}" \
    --clear-target \
    --insert-batch 500

# ── Stage B: function-level followups ─────────────────────────────────────
echo ""
echo ">>> Stage B: function-level followup extraction (shard ${SHARD_INDEX})"
python "${FOLLOWUP_DIR}/analyze_function_level_pg.py" \
    --pg-config-file "${PG_CONFIG}" \
    --tokens-file "${TOKENS_FILE}" \
    --repos-base "${REPOS_BASE}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${SHARD_INDEX}" \
    --max-workers "${MAX_WORKERS}" \
    --clear-target

# Cleanup
rm -rf /dev/shm/ovl-upper-* /dev/shm/ovl-work-* 2>/dev/null || true

echo ""
echo "============================================================"
echo "Shard ${SHARD_INDEX}/${NUM_SHARDS} done on $(hostname) at $(date)"
echo "============================================================"
