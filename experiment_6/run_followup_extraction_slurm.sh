#!/usr/bin/env bash
#SBATCH --job-name=followup-extract
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-21
#SBATCH --output=logs/followup_extract_%A_%a.log
#
# Slurm job array to extract file-level and function-level followups
# across the full prs_copy table (5,927 repos, ~152k PRs).
#
# 22 nodes × 32 workers per node.  Each array task is one shard.
# Stage A (file-level) runs first, then Stage B (function-level) which
# depends on Stage A's output in followups_file.
#
# Both scripts use stable hash sharding (sum(ord(ch)) % num_shards) so
# shards are independent and can run concurrently without conflicts.
# Each shard DELETEs only its own repos before inserting.
#
# Usage:
#   sbatch experiment_6/run_followup_extraction_slurm.sh
#
# Single-node test (no Slurm):
#   SLURM_ARRAY_TASK_ID=0 bash experiment_6/run_followup_extraction_slurm.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FOLLOWUP_DIR="/shared_workspace_mfs/arthur/crawled-prs-hf/prs-followup-analysis"

NUM_SHARDS=22
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
MAX_WORKERS=8

PG_CONFIG="${FOLLOWUP_DIR}/postgres_connection.yaml"
TOKENS_FILE="${FOLLOWUP_DIR}/crawl_tokens.yaml"
REPOS_BASE="/shared_workspace_mfs/repos"

mkdir -p "${PROJECT_ROOT}/logs"

echo "============================================================"
echo "Followup Extraction — Shard ${SHARD_INDEX}/${NUM_SHARDS}"
echo "  Node:        $(hostname)"
echo "  PG config:   ${PG_CONFIG}"
echo "  Tokens:      ${TOKENS_FILE}"
echo "  Repos base:  ${REPOS_BASE}"
echo "  Max workers: ${MAX_WORKERS}"
echo "  /dev/shm:    $(df -h /dev/shm | tail -1 | awk '{print $4}') free"
echo "============================================================"

source "${PROJECT_ROOT}/.venv/bin/activate"

# Clean up any stale overlayfs mounts from prior runs on this node
echo "Cleaning stale overlayfs mounts..."
for mnt in /tmp/repos_tmp_overlayfs/*/; do
    if mountpoint -q "$mnt" 2>/dev/null; then
        fusermount3 -u "$mnt" 2>/dev/null || true
    fi
done
rm -rf /dev/shm/ovl-upper-* /dev/shm/ovl-work-* 2>/dev/null || true

# ── Stage A: File-level followups (followups_file) ────────────────────────
#
# O(n²) per-repo but no disk/git access needed — pure postgres.
# Each shard clears only its own repos before inserting.
#
echo ""
echo ">>> Stage A: File-level followup extraction (shard ${SHARD_INDEX})"
python "${FOLLOWUP_DIR}/analyze_feature_followups_v2_pg.py" \
    --pg-config-file "${PG_CONFIG}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${SHARD_INDEX}" \
    --clear-target \
    --insert-batch 500

# ── Stage B: Function-level followups (followups_function) ────────────────
#
# Needs overlayfs + git fetch + tree-sitter AST parsing.
# Uses ProcessPoolExecutor internally (max_workers=32).
# Reads from followups_file (written by Stage A above).
# Overlayfs mounts are cleaned up per-repo in the finally block.
#
echo ""
echo ">>> Stage B: Function-level followup extraction (shard ${SHARD_INDEX})"
python "${FOLLOWUP_DIR}/analyze_function_level_pg.py" \
    --pg-config-file "${PG_CONFIG}" \
    --tokens-file "${TOKENS_FILE}" \
    --repos-base "${REPOS_BASE}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${SHARD_INDEX}" \
    --max-workers "${MAX_WORKERS}" \
    --clear-target

# Final cleanup — ensure nothing is left in /dev/shm from this shard
echo ""
echo "Final /dev/shm cleanup..."
rm -rf /dev/shm/ovl-upper-* /dev/shm/ovl-work-* 2>/dev/null || true

echo ""
echo "============================================================"
echo "Shard ${SHARD_INDEX}/${NUM_SHARDS} complete on $(hostname)"
echo "  /dev/shm remaining: $(df -h /dev/shm | tail -1 | awk '{print $4}') free"
echo "============================================================"
