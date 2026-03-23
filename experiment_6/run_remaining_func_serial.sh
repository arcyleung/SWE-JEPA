#!/usr/bin/env bash
set -euo pipefail
SHARDS=(0 4 9 16 17)
SHARD=${SHARDS[$SLURM_PROCID]}
export SHARD NUM_SHARDS=22 TIMEOUT_PER_REPO=300
exec bash /shared_workspace_mfs/arthur/coder/experiment_6/run_func_shard_serial.sh
