#!/usr/bin/env bash
set -euo pipefail
SHARDS=(0 4 9 16 17)
SHARD_INDEX=${SHARDS[$SLURM_PROCID]}

PROJECT_ROOT="/shared_workspace_mfs/arthur/coder"
FOLLOWUP_DIR="/shared_workspace_mfs/arthur/crawled-prs-hf/prs-followup-analysis"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/followup_func_shard_${SHARD_INDEX}_v3.log" 2>&1

source "${PROJECT_ROOT}/.venv/bin/activate"

echo "Shard ${SHARD_INDEX} on $(hostname) at $(date)"

for mnt in /tmp/repos_tmp_overlayfs/*/; do
    mountpoint -q "$mnt" 2>/dev/null && fusermount3 -u "$mnt" 2>/dev/null || true
done
rm -rf /dev/shm/ovl-upper-* /dev/shm/ovl-work-* 2>/dev/null || true

python "${FOLLOWUP_DIR}/analyze_function_level_pg.py" \
    --pg-config-file "${FOLLOWUP_DIR}/postgres_connection.yaml" \
    --tokens-file "${FOLLOWUP_DIR}/crawl_tokens.yaml" \
    --repos-base /shared_workspace_mfs/repos \
    --num-shards 22 \
    --shard-index "${SHARD_INDEX}" \
    --max-workers 2 \
    --no-clear-target

rm -rf /dev/shm/ovl-upper-* /dev/shm/ovl-work-* 2>/dev/null || true
echo "Shard ${SHARD_INDEX} done at $(date)"
