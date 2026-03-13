#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${1:?run tag required}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$ROOT/data/$RUN_TAG"
OUT_FILE="$ROOT/data/${RUN_TAG}_status_snapshot.txt"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "run_tag=$RUN_TAG"
  echo
  echo "[processes]"
  ps -ef | rg "$RUN_TAG|run_pr_refinement_history_slurm.sh|extract_pr_refinement_history.py" || true
  echo
  echo "[files]"
  find "$RUN_DIR" -maxdepth 2 -type f 2>/dev/null | sort || true
  echo
  echo "[counts]"
  python - <<'PY' "$RUN_DIR"
import json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
shard_dir = run_dir / "shard_outputs"
jsonls = sorted(shard_dir.glob("shard_*.jsonl"))
summaries = sorted(shard_dir.glob("shard_*_summary.json"))
rows = 0
prs = 0
for p in summaries:
    try:
        s = json.load(open(p))
        rows += int(s.get("n_rows", 0))
        prs += int(s.get("n_prs", 0))
    except Exception:
        pass
print(f"jsonl_count={len(jsonls)}")
print(f"summary_count={len(summaries)}")
print(f"rows_done={rows}")
print(f"prs_done={prs}")
PY
  echo
  echo "[disk_usage]"
  du -sh "$RUN_DIR" 2>/dev/null || true
} >"$OUT_FILE" 2>&1

echo "$OUT_FILE"
