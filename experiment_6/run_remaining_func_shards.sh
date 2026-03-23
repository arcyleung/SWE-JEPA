#!/usr/bin/env bash
# Run the 7 remaining function-level shards (0,4,8,9,10,16,17) that need fuse-overlayfs.
# Maps SLURM_PROCID 0–6 → actual shard indices.
#
# srun --nodes=7 --ntasks=7 --ntasks-per-node=1 --cpus-per-task=8 \
#      --time=6:00:00 --partition=debug \
#      bash experiment_6/run_remaining_func_shards.sh
set -euo pipefail

SHARDS=(0 4 8 9 10 16 17)
SHARD_INDEX=${SHARDS[$SLURM_PROCID]}

PROJECT_ROOT="/shared_workspace_mfs/arthur/coder"
FOLLOWUP_DIR="/shared_workspace_mfs/arthur/crawled-prs-hf/prs-followup-analysis"

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/followup_func_shard_${SHARD_INDEX}.log" 2>&1

source "${PROJECT_ROOT}/.venv/bin/activate"

echo "Shard ${SHARD_INDEX} on $(hostname) at $(date)"

# Clean stale overlayfs
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
    --max-workers 8 \
    --no-clear-target

rm -rf /dev/shm/ovl-upper-* /dev/shm/ovl-work-* 2>/dev/null || true
echo "Shard ${SHARD_INDEX} done at $(date)"
