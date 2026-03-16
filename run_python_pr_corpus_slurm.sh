#!/usr/bin/env bash
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT"

RUN_TAG="${1:-phase5_1_python_pr_corpus_slurm}"
WORKERS="${WORKERS:-32}"
SUBMIT_BATCH_SIZE="${SUBMIT_BATCH_SIZE:-1024}"

RUN_DIR="data/${RUN_TAG}"
SHARD_OUT_DIR="${RUN_DIR}/shard_outputs"
mkdir -p "$SHARD_OUT_DIR"
export RUN_TAG

source .venv/bin/activate

echo "[Phase 1] Extracting Python PR Conway features on ${SLURM_NTASKS:-1} tasks..."

srun \
  --ntasks="${SLURM_NTASKS:-1}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-$WORKERS}" \
  --cpu-bind=none \
  bash -lc '
    set -euo pipefail
    cd "'"$ROOT"'"
    source .venv/bin/activate
    shard=$(printf "%03d" "${SLURM_PROCID:-0}")
    out_feat="'"$SHARD_OUT_DIR"'/shard_${shard}_features.jsonl"
    out_lbl="'"$SHARD_OUT_DIR"'/shard_${shard}_labels.jsonl"
    out_sum="'"$SHARD_OUT_DIR"'/shard_${shard}_summary.json"
    log_file="'"$SHARD_OUT_DIR"'/shard_${shard}.log"
    python build_go_pr_steerer_corpus.py \
      --merged-table prs_copy \
      --closed-table python_js_ts_rust_closed_prs \
      --language python \
      --workers "${SLURM_CPUS_PER_TASK:-'"$WORKERS"'}" \
      --submit-batch-size "'"$SUBMIT_BATCH_SIZE"'" \
      --shard-modulus "${SLURM_NTASKS:-1}" \
      --shard-remainder "${SLURM_PROCID:-0}" \
      --out "$out_feat" \
      --labels-out "$out_lbl" \
      --summary-out "$out_sum" \
      > "$log_file" 2>&1
  '

python - <<'PY'
import glob
import json
import os
from collections import Counter

run_dir = os.path.join("data", os.environ["RUN_TAG"])
shard_dir = os.path.join(run_dir, "shard_outputs")
feat_out = os.path.join(run_dir, "conway_patch_features_python_merged_closed.jsonl")
lbl_out = os.path.join(run_dir, "python_pr_labels.jsonl")
sum_out = os.path.join(run_dir, "python_pr_corpus_summary.json")

feat_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*_features.jsonl")))
lbl_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*_labels.jsonl")))
sum_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*_summary.json")))


def merge_jsonl(paths, out_path):
    n = 0
    with open(out_path, "w") as fout:
        for path in paths:
            if not os.path.exists(path):
                continue
            with open(path) as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
                        n += 1
    return n


feat_n = merge_jsonl(feat_files, feat_out)
lbl_n = merge_jsonl(lbl_files, lbl_out)
agg = Counter()
source_counts = Counter()
language = None
merged_table = None
closed_table = None

for path in sum_files:
    with open(path) as f:
        row = json.load(f)
    language = row.get("language", language)
    merged_table = row.get("merged_table", merged_table)
    closed_table = row.get("closed_table", closed_table)
    agg["rows_requested"] += int(row.get("rows_requested", 0))
    agg["rows_emitted"] += int(row.get("rows_emitted", 0))
    for key, value in (row.get("by_source") or {}).items():
        source_counts[key] += int(value)
    for key, value in (row.get("errors") or {}).items():
        agg[f"err::{key}"] += int(value)

summary = {
    "run_tag": os.environ["RUN_TAG"],
    "language": language,
    "merged_table": merged_table,
    "closed_table": closed_table,
    "n_shards": len(sum_files),
    "feature_rows_merged": feat_n,
    "label_rows_merged": lbl_n,
    "rows_requested": int(agg["rows_requested"]),
    "rows_emitted": int(agg["rows_emitted"]),
    "by_source": dict(source_counts),
    "errors": {
        key.removeprefix("err::"): int(value)
        for key, value in sorted(agg.items())
        if key.startswith("err::")
    },
}
with open(sum_out, "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PY

echo "[Phase 1] Done. Features: ${RUN_DIR}/conway_patch_features_python_merged_closed.jsonl"
