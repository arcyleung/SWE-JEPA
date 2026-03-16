#!/usr/bin/env bash
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT"

source .venv/bin/activate

WORKERS="${WORKERS:-64}"
LIMIT="${LIMIT:-0}"

python backfill_go_prs_closed_patches.py \
  --table python_js_ts_rust_closed_prs \
  --language python \
  --workers "$WORKERS" \
  --limit "$LIMIT" \
  "$@"
