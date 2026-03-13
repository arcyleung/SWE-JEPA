#!/usr/bin/env bash
# Go PR steerer — hint-mode ablation eval
#
# Samples a fixed stratified test set from the Go PR labels, then runs 4
# conditions on the same instances:
#   0. baseline   — no steering
#   1. json       — raw score dict (current default)
#   2. simple     — single-line score summary
#   3. specific   — per-signal actionable hints
#
# All 4 conditions use the same qwen3.5_35b_a3b model and the same test-set
# keys so results are directly comparable.  Judge eval is run once at the end
# comparing each steered condition against the shared baseline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── Config ──────────────────────────────────────────────────────────────────
RUN_TAG="${1:-go_ablation_v1}"
export RUN_TAG
AGENT_MODEL="${AGENT_MODEL:-qwen3.5_35b_a3b}"
AGENT_CONCURRENCY="${AGENT_CONCURRENCY:-150}"
AGENT_STEP_LIMIT="${AGENT_STEP_LIMIT:-80}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-1800}"
TEST_SET_SIZE="${TEST_SET_SIZE:-150}"
TEST_SET_SEED="${TEST_SET_SEED:-42}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-codex}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"

LABELS_FILE="data/phase4_7_3_go_pr_labels.jsonl"
STEERER_MODEL="data/phase4_7_3_go_pr_steerer_model.json"

OUT_DIR="data/${RUN_TAG}"
TEST_KEYS="${OUT_DIR}/test_set_keys.jsonl"
mkdir -p "$OUT_DIR"

source .venv/bin/activate

# ════════════════════════════════════════════════════════════════════════════
# Step 0 — Sample stratified test set (idempotent: skip if already exists)
# ════════════════════════════════════════════════════════════════════════════
if [[ -f "$TEST_KEYS" ]]; then
  n=$(wc -l < "$TEST_KEYS")
  echo "[Step 0] Test set already exists: ${TEST_KEYS} (${n} instances) — skipping"
else
  echo "[Step 0] Sampling ${TEST_SET_SIZE} stratified instances from ${LABELS_FILE}..."
  python3 - <<PY
import json, random, sys

labels_path = "${LABELS_FILE}"
out_path    = "${TEST_KEYS}"
n_total     = int("${TEST_SET_SIZE}")
seed        = int("${TEST_SET_SEED}")

rows = [json.loads(l) for l in open(labels_path) if l.strip()]

# Stratify by (accepted, refactor_requested)
strata: dict[tuple, list] = {}
for r in rows:
    st1 = r.get("s_t1", {})
    key = (int(st1.get("accepted", 0)), int(st1.get("refactor_requested", 0)))
    strata.setdefault(key, []).append(r)

rng = random.Random(seed)
for v in strata.values():
    rng.shuffle(v)

# Round-robin sample from strata until n_total reached
strata_list = sorted(strata.values(), key=lambda v: -len(v))
selected = []
i = 0
while len(selected) < n_total:
    stratum = strata_list[i % len(strata_list)]
    idx = len(selected) // len(strata_list)
    if idx < len(stratum):
        selected.append(stratum[idx])
    i += 1
    if i > n_total * 10:
        break  # safety
selected = selected[:n_total]
rng.shuffle(selected)

# Write as minimal key rows (repo + pull_number — format expected by --task-keys-jsonl)
with open(out_path, "w") as f:
    for r in selected:
        f.write(json.dumps({"repo": r["repo"], "pull_number": r["pull_number"]}) + "\n")

# Print stratum breakdown
from collections import Counter
counts = Counter(
    (int(r.get("s_t1", {}).get("accepted", 0)),
     int(r.get("s_t1", {}).get("refactor_requested", 0)))
    for r in selected
)
print(f"Sampled {len(selected)} instances")
for (acc, ref), cnt in sorted(counts.items()):
    print(f"  accepted={acc} refactor_requested={ref}: {cnt}")
PY
  echo "[Step 0] Done: ${TEST_KEYS}"
fi

export TEST_KEYS  # used by inline python below

# ════════════════════════════════════════════════════════════════════════════
# Helper: run one eval condition
# ════════════════════════════════════════════════════════════════════════════
run_condition() {
  local label="$1"          # e.g. "baseline" or "steered_json"
  local extra_args="$2"     # extra args for run_phase4_7_agentic_eval_steered.py
  local traj_dir="${OUT_DIR}/trajs_${label}"
  local patch_dir="${OUT_DIR}/patches_${label}"
  local results_jsonl="${OUT_DIR}/results_${label}.jsonl"
  local summary_json="${OUT_DIR}/summary_${label}.json"

  if [[ -f "$summary_json" ]]; then
    echo "[${label}] Already done (${summary_json}) — skipping"
    return
  fi

  echo "[${label}] Running..."
  # shellcheck disable=SC2086
  python run_phase4_7_agentic_eval_steered.py \
    --model-name      "$AGENT_MODEL" \
    --task-keys-jsonl "$TEST_KEYS" \
    --source-table    go_prs \
    --ramdisk-worktree \
    --concurrency     "$AGENT_CONCURRENCY" \
    --step-limit      "$AGENT_STEP_LIMIT" \
    --timeout-sec     "$AGENT_TIMEOUT" \
    --traj-dir        "$traj_dir" \
    --patch-dir       "$patch_dir" \
    --out-jsonl       "$results_jsonl" \
    --out-summary     "$summary_json" \
    $extra_args
  echo "[${label}] Done — summary: ${summary_json}"
}

# ════════════════════════════════════════════════════════════════════════════
# Step 1 — Baseline (no steering)
# ════════════════════════════════════════════════════════════════════════════
run_condition "baseline" "--disable-steering"

# ════════════════════════════════════════════════════════════════════════════
# Steps 2-4 — Steered, one run per hint mode
# ════════════════════════════════════════════════════════════════════════════
for MODE in json simple specific; do
  run_condition "steered_${MODE}" \
    "--steerer-model ${STEERER_MODEL} --steer-max-attempts 3 --hint-mode ${MODE}"
done

# ════════════════════════════════════════════════════════════════════════════
# Step 5 — Judge eval: compare each steered condition vs shared baseline
# ════════════════════════════════════════════════════════════════════════════
echo "[Step 5] Running judge eval for each hint mode vs baseline..."
for MODE in json simple specific; do
  judge_scores="${OUT_DIR}/judge_scores_${MODE}.jsonl"
  judge_summary="${OUT_DIR}/judge_summary_${MODE}.json"
  if [[ -f "$judge_summary" ]]; then
    echo "  [judge ${MODE}] Already done — skipping"
    continue
  fi
  echo "  [judge ${MODE}] Scoring..."
  python score_patch_judge_4_7.py \
    --judge-model         "$JUDGE_MODEL" \
    --baseline-patch-dir  "${OUT_DIR}/patches_baseline" \
    --steered-traj-dir    "${OUT_DIR}/trajs_steered_${MODE}" \
    --concurrency         "$JUDGE_CONCURRENCY" \
    --out                 "$judge_scores" \
    --summary             "$judge_summary"
done

# ════════════════════════════════════════════════════════════════════════════
# Step 6 — Comparison summary across all conditions
# ════════════════════════════════════════════════════════════════════════════
echo "[Step 6] Generating comparison report..."
python3 - <<'PY'
import json, os

run_dir = os.path.join("data", os.environ.get("RUN_TAG", "go_ablation_v1"))
modes   = ["json", "simple", "specific"]

# Agentic eval stats per condition
def load_summary(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}

baseline_sum = load_summary(os.path.join(run_dir, "summary_baseline.json"))

lines = [
    "# Go PR Hint-Mode Ablation Results",
    "",
    f"- Run tag: `{os.path.basename(run_dir)}`",
    f"- Test set: `{os.path.join(run_dir, 'test_set_keys.jsonl')}`",
    "",
    "## Agentic eval stats",
    "",
    "| Condition | Tasks OK | Avg changed files | Avg attempts |",
    "|-----------|----------|-------------------|--------------|",
]
for label, spath in [
    ("baseline", os.path.join(run_dir, "summary_baseline.json")),
    *[(f"steered_{m}", os.path.join(run_dir, f"summary_steered_{m}.json")) for m in modes],
]:
    s = load_summary(spath)
    ok    = s.get("status_counts", {}).get("ok", "?")
    total = s.get("n_tasks", "?")
    avg_f = s.get("avg_changed_files_after", 0)
    avg_a = s.get("avg_steer_attempts", 1)
    lines.append(f"| {label} | {ok}/{total} | {avg_f:.1f} | {avg_a:.2f} |")

lines += ["", "## Judge eval (gpt-5-codex blind A/B vs baseline)", ""]
lines += [
    "| Hint mode | Pairs eval | Steered win rate | Δ overall | Best criterion (Δ) | Worst criterion (Δ) |",
    "|-----------|-----------|-----------------|-----------|---------------------|----------------------|",
]
for mode in modes:
    js = load_summary(os.path.join(run_dir, f"judge_summary_{mode}.json"))
    if "error" in js or not js:
        lines.append(f"| {mode} | — | — | — | — | — |")
        continue
    win_rate = js.get("steered_win_rate", 0)
    delta    = js.get("delta_overall", 0)
    n        = js.get("n_evaluated", 0)
    per_c    = js.get("per_criterion", {})
    by_delta = sorted(per_c.items(), key=lambda x: x[1]["delta"])
    worst    = f"{by_delta[0][0]} ({by_delta[0][1]['delta']:+.3f})" if by_delta else "—"
    best     = f"{by_delta[-1][0]} ({by_delta[-1][1]['delta']:+.3f})" if by_delta else "—"
    lines.append(f"| {mode} | {n} | {win_rate:.1%} | {delta:+.3f} | {best} | {worst} |")

report_path = os.path.join(run_dir, "ablation_report.md")
with open(report_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Report written -> {report_path}")

# Also print to stdout
for l in lines:
    print(l)
PY

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Run dir  -> ${OUT_DIR}"
echo "Report   -> ${OUT_DIR}/ablation_report.md"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
