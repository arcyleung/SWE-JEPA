#!/usr/bin/env bash
# Run the remaining shards 12–21 that didn't complete.
# Launches them as background processes on the current node.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "${PROJECT_ROOT}/.venv/bin/activate"

PIDS=()
for SHARD in $(seq 12 21); do
    SLURM_PROCID=$SHARD bash "${SCRIPT_DIR}/run_followup_extraction_srun.sh" &
    PIDS+=($!)
    echo "Launched shard ${SHARD} (pid $!)"
done

echo "Waiting for ${#PIDS[@]} shards..."
FAILURES=0
for i in "${!PIDS[@]}"; do
    SHARD=$((i + 12))
    if wait "${PIDS[$i]}"; then
        echo "  Shard ${SHARD} OK"
    else
        echo "  Shard ${SHARD} FAILED"
        FAILURES=$((FAILURES + 1))
    fi
done

echo "Done: $((10 - FAILURES))/10 succeeded"
