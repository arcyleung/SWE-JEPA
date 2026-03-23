#!/usr/bin/env bash
# Run function-level extraction for a single shard, one repo at a time
# with a 300s timeout per repo. Skips repos that hang.
#
# Usage: SHARD=16 bash experiment_6/run_func_shard_serial.sh
set -euo pipefail

PROJECT_ROOT="/shared_workspace_mfs/arthur/coder"
FOLLOWUP_DIR="/shared_workspace_mfs/arthur/crawled-prs-hf/prs-followup-analysis"
PG_CONFIG="${FOLLOWUP_DIR}/postgres_connection.yaml"
TOKENS_FILE="${FOLLOWUP_DIR}/crawl_tokens.yaml"

SHARD="${SHARD:?Set SHARD=N}"
NUM_SHARDS="${NUM_SHARDS:-22}"
TIMEOUT_PER_REPO="${TIMEOUT_PER_REPO:-300}"

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/followup_func_shard_${SHARD}_serial.log"

source "${PROJECT_ROOT}/.venv/bin/activate"

echo "Shard ${SHARD}/${NUM_SHARDS} serial mode, timeout=${TIMEOUT_PER_REPO}s per repo" | tee "$LOG"
echo "Started $(date)" | tee -a "$LOG"

# Get repo list for this shard
REPOS=$(python -c "
import pg8000.native, yaml, os
cfg = yaml.safe_load(open('${PG_CONFIG}'))
conn = pg8000.native.Connection(host=cfg['ip'], port=int(cfg['port']), user=cfg['user'], password=cfg['password'], database=cfg['database'])
rows = conn.run(\"SELECT DISTINCT repo FROM prs_copy WHERE pr_category = 'feature'\")
conn.close()
for r in sorted(r[0] for r in rows):
    if sum(ord(c) for c in r) % ${NUM_SHARDS} == ${SHARD}:
        print(r)
")

TOTAL=$(echo "$REPOS" | wc -l)
echo "Repos: ${TOTAL}" | tee -a "$LOG"

I=0
OK=0
SKIP=0
FAIL=0

for REPO in $REPOS; do
    I=$((I + 1))
    timeout "${TIMEOUT_PER_REPO}" python "${FOLLOWUP_DIR}/analyze_function_level_pg.py" \
        --pg-config-file "${PG_CONFIG}" \
        --tokens-file "${TOKENS_FILE}" \
        --repos-base /shared_workspace_mfs/repos \
        --num-shards "${NUM_SHARDS}" \
        --shard-index "${SHARD}" \
        --max-workers 1 \
        --no-clear-target \
        --repo "${REPO}" >> "$LOG" 2>&1
    RC=$?
    if [ $RC -eq 0 ]; then
        OK=$((OK + 1))
    elif [ $RC -eq 124 ]; then
        echo "[${I}/${TOTAL}] TIMEOUT ${REPO}" | tee -a "$LOG"
        FAIL=$((FAIL + 1))
        # Clean up any stale overlayfs
        for mnt in /tmp/repos_tmp_overlayfs/*/; do
            mountpoint -q "$mnt" 2>/dev/null && fusermount3 -u "$mnt" 2>/dev/null || true
        done
        rm -rf /dev/shm/ovl-upper-* /dev/shm/ovl-work-* 2>/dev/null || true
    else
        FAIL=$((FAIL + 1))
    fi

    if [ $((I % 20)) -eq 0 ]; then
        echo "[${I}/${TOTAL}] ok=${OK} fail=${FAIL}" | tee -a "$LOG"
    fi
done

echo "Done: ${OK} ok, ${FAIL} failed, ${TOTAL} total at $(date)" | tee -a "$LOG"
