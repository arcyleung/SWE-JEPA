#!/usr/bin/env bash
# End-to-end pipeline for Go PR steerer:
#   Phase 1: Extract Conway features from Go PRs (srun, multi-node)
#   Phase 2: Train acceptance + refactor steerer heads
#   Phase 3: Train value function V(s_t) from pr_refinement_history_checkpoints
#   Phase 4: Run mini-swe-agent — baseline (no steering) + steered
#   Phase 5: Judge eval (gpt-5-codex A/B comparison)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── Config ──────────────────────────────────────────────────────────────────
RUN_TAG="${1:-phase4_7_3_go_pr_steerer_slurm}"
PARTITION="${PARTITION:-debug}"
NODES="${NODES:-22}"
WORKERS="${WORKERS:-32}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$WORKERS}"
SUBMIT_BATCH_SIZE="${SUBMIT_BATCH_SIZE:-1024}"
AGENT_MODEL="${AGENT_MODEL:-qwen3.5_35b_a3b}"
AGENT_CONCURRENCY="${AGENT_CONCURRENCY:-150}"
AGENT_STEP_LIMIT="${AGENT_STEP_LIMIT:-80}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-1800}"
AGENT_LIMIT="${AGENT_LIMIT:-500}"    # max Go tasks for agentic eval
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-codex}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"
HINT_MODE="${HINT_MODE:-json}"       # json | simple | specific

RUN_DIR="data/${RUN_TAG}"
SHARD_OUT_DIR="${RUN_DIR}/shard_outputs"
mkdir -p "$SHARD_OUT_DIR"
export RUN_TAG

# ── Artifact paths ──────────────────────────────────────────────────────────
MERGED_FEATURES="${RUN_DIR}/conway_patch_features_go_merged_closed.jsonl"
MERGED_LABELS="${RUN_DIR}/phase4_7_3_go_pr_labels.jsonl"
MERGED_SUMMARY="${RUN_DIR}/phase4_7_3_go_pr_corpus_summary.json"
MODEL_OUT="data/phase4_7_3_go_pr_steerer_model.json"
METRICS_OUT="data/phase4_7_3_go_pr_steerer_metrics.json"
VALUE_MODEL_OUT="data/phase4_7_3_go_pr_value_function.json"
VALUE_METRICS_OUT="data/phase4_7_3_go_pr_value_function_metrics.json"
REPORT_OUT="docs/phase4_7_3_go_pr_steerer.md"

BASELINE_TRAJ_DIR="${RUN_DIR}/eval_baseline_trajs"
BASELINE_PATCH_DIR="${RUN_DIR}/eval_baseline_patches"
STEERED_TRAJ_DIR="${RUN_DIR}/eval_steered_trajs"
STEERED_PATCH_DIR="${RUN_DIR}/eval_steered_patches"
JUDGE_SCORES="${RUN_DIR}/judge_scores.jsonl"
JUDGE_SUMMARY="${RUN_DIR}/judge_summary.json"

source .venv/bin/activate

# ════════════════════════════════════════════════════════════════════════════
# Phase 1: Conway feature extraction (distributed over SLURM nodes)
# ════════════════════════════════════════════════════════════════════════════
echo "[Phase 1] Extracting Go PR Conway features on ${NODES} nodes..."

srun -N "$NODES" -p "$PARTITION" -n "$NODES" --ntasks-per-node=1 --cpus-per-task="$CPUS_PER_TASK" --cpu-bind=none bash -lc '
  set -euo pipefail
  cd "'"$ROOT"'"
  source .venv/bin/activate
  shard=$(printf "%03d" "${SLURM_PROCID}")
  out_feat="'"$SHARD_OUT_DIR"'/shard_${shard}_features.jsonl"
  out_lbl="'"$SHARD_OUT_DIR"'/shard_${shard}_labels.jsonl"
  out_sum="'"$SHARD_OUT_DIR"'/shard_${shard}_summary.json"
  log_file="'"$SHARD_OUT_DIR"'/shard_${shard}.log"
  python build_go_pr_steerer_corpus.py \
    --workers "${SLURM_CPUS_PER_TASK:-'"$WORKERS"'}" \
    --submit-batch-size "'"$SUBMIT_BATCH_SIZE"'" \
    --shard-modulus "${SLURM_NTASKS}" \
    --shard-remainder "${SLURM_PROCID}" \
    --out "$out_feat" \
    --labels-out "$out_lbl" \
    --summary-out "$out_sum" \
    > "$log_file" 2>&1
'

# Merge shards
python - <<'PY'
import glob, json, os
from collections import Counter

run_dir = os.path.join("data", os.environ["RUN_TAG"])
shard_dir = os.path.join(run_dir, "shard_outputs")
feat_out = os.path.join(run_dir, "conway_patch_features_go_merged_closed.jsonl")
lbl_out  = os.path.join(run_dir, "phase4_7_3_go_pr_labels.jsonl")
sum_out  = os.path.join(run_dir, "phase4_7_3_go_pr_corpus_summary.json")

feat_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*_features.jsonl")))
lbl_files  = sorted(glob.glob(os.path.join(shard_dir, "shard_*_labels.jsonl")))
sum_files  = sorted(glob.glob(os.path.join(shard_dir, "shard_*_summary.json")))

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
lbl_n  = merge_jsonl(lbl_files,  lbl_out)
agg = Counter()
for path in sum_files:
    with open(path) as f:
        row = json.load(f)
    agg["rows_requested"] += int(row.get("rows_requested", 0))
    agg["rows_emitted"]   += int(row.get("rows_emitted", 0))
    by_source = row.get("by_source", {})
    agg["go_prs"]        += int(by_source.get("go_prs", 0))
    agg["go_prs_closed"] += int(by_source.get("go_prs_closed", 0))
    for k, v in (row.get("errors") or {}).items():
        agg[f"err::{k}"] += int(v)

summary = {
    "run_tag": os.environ["RUN_TAG"],
    "n_shards": len(sum_files),
    "feature_rows_merged": feat_n,
    "label_rows_merged": lbl_n,
    "rows_requested": int(agg["rows_requested"]),
    "rows_emitted":   int(agg["rows_emitted"]),
    "by_source": {
        "go_prs":        int(agg["go_prs"]),
        "go_prs_closed": int(agg["go_prs_closed"]),
    },
    "errors": {
        k.removeprefix("err::"): int(v)
        for k, v in sorted(agg.items()) if k.startswith("err::")
    },
}
with open(sum_out, "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PY

echo "[Phase 1] Done. Features: ${MERGED_FEATURES}  Labels: ${MERGED_LABELS}"

# ════════════════════════════════════════════════════════════════════════════
# Phase 2: Train acceptance + refactor steerer heads
# ════════════════════════════════════════════════════════════════════════════
echo "[Phase 2] Training Go PR steerer (acceptance + refactor heads)..."

python train_pr_steerer.py \
  --data         "$MERGED_FEATURES" \
  --labels-data  "$MERGED_LABELS" \
  --model-out    "$MODEL_OUT" \
  --metrics-out  "$METRICS_OUT" \
  --refactor-target refactor_requested \
  --cv-folds 5

echo "[Phase 2] Done. Model: ${MODEL_OUT}"

# ════════════════════════════════════════════════════════════════════════════
# Phase 3: Train value function V(s_t) from per-checkpoint history
# ════════════════════════════════════════════════════════════════════════════
echo "[Phase 3] Training value function from pr_refinement_history_checkpoints..."

python train_pr_value_function.py \
  --run-tag        v1 \
  --static-model   "$MODEL_OUT" \
  --out-model      "$VALUE_MODEL_OUT" \
  --out-metrics    "$VALUE_METRICS_OUT" \
  --n-splits       5

echo "[Phase 3] Done. Value model: ${VALUE_MODEL_OUT}"

# ════════════════════════════════════════════════════════════════════════════
# Phase 4a: Baseline agentic eval (no steering)
# ════════════════════════════════════════════════════════════════════════════
echo "[Phase 4a] Running baseline eval (no steering) with ${AGENT_MODEL}..."

python run_phase4_7_agentic_eval_steered.py \
  --model-name       "$AGENT_MODEL" \
  --task-keys-jsonl  "$MERGED_LABELS" \
  --limit            "$AGENT_LIMIT" \
  --concurrency      "$AGENT_CONCURRENCY" \
  --step-limit       "$AGENT_STEP_LIMIT" \
  --timeout-sec      "$AGENT_TIMEOUT" \
  --traj-dir         "$BASELINE_TRAJ_DIR" \
  --patch-dir        "$BASELINE_PATCH_DIR" \
  --out-jsonl        "${RUN_DIR}/eval_baseline_results.jsonl" \
  --out-summary      "${RUN_DIR}/eval_baseline_summary.json" \
  --disable-steering

echo "[Phase 4a] Done. Patches: ${BASELINE_PATCH_DIR}"

# ════════════════════════════════════════════════════════════════════════════
# Phase 4b: Steered agentic eval
# ════════════════════════════════════════════════════════════════════════════
echo "[Phase 4b] Running steered eval with ${AGENT_MODEL}, hint_mode=${HINT_MODE}..."

python run_phase4_7_agentic_eval_steered.py \
  --model-name       "$AGENT_MODEL" \
  --task-keys-jsonl  "$MERGED_LABELS" \
  --limit            "$AGENT_LIMIT" \
  --concurrency      "$AGENT_CONCURRENCY" \
  --step-limit       "$AGENT_STEP_LIMIT" \
  --timeout-sec      "$AGENT_TIMEOUT" \
  --traj-dir         "$STEERED_TRAJ_DIR" \
  --patch-dir        "$STEERED_PATCH_DIR" \
  --out-jsonl        "${RUN_DIR}/eval_steered_results.jsonl" \
  --out-summary      "${RUN_DIR}/eval_steered_summary.json" \
  --steerer-model    "$MODEL_OUT" \
  --steer-max-attempts 3 \
  --hint-mode        "$HINT_MODE"

echo "[Phase 4b] Done. Trajs: ${STEERED_TRAJ_DIR}"

# ════════════════════════════════════════════════════════════════════════════
# Phase 5: Judge eval (gpt-5-codex blind A/B)
# ════════════════════════════════════════════════════════════════════════════
echo "[Phase 5] Running judge eval with ${JUDGE_MODEL}..."

python score_patch_judge_4_7.py \
  --judge-model         "$JUDGE_MODEL" \
  --baseline-patch-dir  "$BASELINE_PATCH_DIR" \
  --steered-traj-dir    "$STEERED_TRAJ_DIR" \
  --concurrency         "$JUDGE_CONCURRENCY" \
  --out                 "$JUDGE_SCORES" \
  --summary             "$JUDGE_SUMMARY"

echo "[Phase 5] Done. Scores: ${JUDGE_SCORES}"

# ════════════════════════════════════════════════════════════════════════════
# Final report
# ════════════════════════════════════════════════════════════════════════════
python - <<'PY'
import json, os

run_tag = os.environ["RUN_TAG"]
run_dir = os.path.join("data", run_tag)

corpus  = json.load(open(os.path.join(run_dir, "phase4_7_3_go_pr_corpus_summary.json")))
metrics = json.load(open("data/phase4_7_3_go_pr_steerer_metrics.json"))
judge   = json.load(open(os.path.join(run_dir, "judge_summary.json")))
val_ok = os.path.exists("data/phase4_7_3_go_pr_value_function_metrics.json")
val_metrics = json.load(open("data/phase4_7_3_go_pr_value_function_metrics.json")) if val_ok else {}

lines = [
    "# Experiment 4.7.3 — Go-Only Steerer",
    "",
    f"- Run tag: `{run_tag}`",
    f"- Agent model: `{os.environ.get('AGENT_MODEL', 'qwen3.5_35b_a3b')}`",
    f"- Hint mode: `{os.environ.get('HINT_MODE', 'json')}`",
    f"- Rows emitted: `{corpus['rows_emitted']:,}`",
    f"- Source mix: `go_prs={corpus['by_source']['go_prs']:,}`  `go_prs_closed={corpus['by_source']['go_prs_closed']:,}`",
    f"- Acceptance rate: `{metrics['acceptance_rate']*100:.1f}%`",
    f"- Refactor rate: `{metrics['refactor_rate']*100:.1f}%`",
    "",
    "## Steerer CV metrics",
    "",
]
for head in ("acceptance", "refactor"):
    folds = metrics["cv"].get(head, [])
    if not folds:
        continue
    auroc  = sum(x["auroc"]  for x in folds) / len(folds)
    pr_auc = sum(x["pr_auc"] for x in folds) / len(folds)
    lines.append(f"- `{head}`: mean AUROC `{auroc:.3f}`, mean PR-AUC `{pr_auc:.3f}`")

if val_metrics:
    lines += [
        "",
        "## Value function CV metrics",
        "",
        f"- R²: `{val_metrics.get('value_cv_r2_mean', 0):.3f}` ± `{val_metrics.get('value_cv_r2_std', 0):.3f}`",
        f"- Spearman ρ: `{val_metrics.get('value_cv_spearman_mean', 0):.3f}` ± `{val_metrics.get('value_cv_spearman_std', 0):.3f}`",
    ]
    auroc_ew = val_metrics.get("refactor_early_warning_cv_auroc_mean")
    if auroc_ew:
        lines.append(f"- Refactor early-warning AUROC: `{auroc_ew:.3f}` ± `{val_metrics.get('refactor_early_warning_cv_auroc_std', 0):.3f}`")

if "error" not in judge:
    lines += [
        "",
        "## Judge eval results (gpt-5-codex blind A/B)",
        "",
        f"- Pairs evaluated: `{judge['n_evaluated']}`  (errors: `{judge['n_errors']}`)",
        f"- Steered win rate: `{judge['steered_win_rate']:.1%}` ({judge['steered_wins']}/{judge['n_evaluated']})",
        f"- Overall Δ (steered − baseline): `{judge['delta_overall']:+.3f}`",
        "",
        "| Criterion | Steered | Baseline | Δ |",
        "|-----------|---------|----------|---|",
    ]
    for k, v in judge["per_criterion"].items():
        bar = "▲" if v["delta"] > 0 else ("▼" if v["delta"] < 0 else " ")
        lines.append(f"| {k} | {v['steered']:.3f} | {v['baseline']:.3f} | {bar}{abs(v['delta']):.3f} |")

lines += [
    "",
    "## Artifacts",
    "",
    f"- Features: `{os.path.join(run_dir, 'conway_patch_features_go_merged_closed.jsonl')}`",
    f"- Labels: `{os.path.join(run_dir, 'phase4_7_3_go_pr_labels.jsonl')}`",
    "- Steerer model: `data/phase4_7_3_go_pr_steerer_model.json`",
    "- Value model: `data/phase4_7_3_go_pr_value_function.json`",
    f"- Baseline trajs: `{os.path.join(run_dir, 'eval_baseline_trajs')}`",
    f"- Steered trajs: `{os.path.join(run_dir, 'eval_steered_trajs')}`",
    f"- Judge scores: `{os.path.join(run_dir, 'judge_scores.jsonl')}`",
]
with open("docs/phase4_7_3_go_pr_steerer.md", "w") as f:
    f.write("\n".join(lines) + "\n")
print("Report written -> docs/phase4_7_3_go_pr_steerer.md")
PY

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Run dir    -> ${RUN_DIR}"
echo "Model      -> ${MODEL_OUT}"
echo "Metrics    -> ${METRICS_OUT}"
echo "Judge      -> ${JUDGE_SUMMARY}"
echo "Report     -> ${REPORT_OUT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
