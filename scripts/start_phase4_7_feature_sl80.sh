#!/usr/bin/env bash
set -euo pipefail

ROOT="/shared_workspace_mfs/arthur/coder"
cd "$ROOT"

# Kill lingering mini-swe-agent workers from previous runs.
pkill -9 -f 'minisweagent.run.mini' || true

# Optional: also stop previous evaluator instances to avoid duplicate schedulers.
pkill -9 -f 'run_phase4_7_agentic_eval.py --limit 90840 --source-multiplier 1 --concurrency 200 --step-limit 80' || true

# Start (or resume via append) feature-only run.
exec /usr/bin/python3 run_phase4_7_agentic_eval.py \
  --limit 90840 \
  --source-multiplier 1 \
  --concurrency 200 \
  --step-limit 80 \
  --timeout-sec 3600 \
  --agent-python /usr/bin/python3 \
  --model-name qwen3_coder_30b \
  --pr-category feature \
  --skip-existing-patched-jsonl data/phase4_7_agentic_eval_results_feature_sl80.jsonl \
  --append-out \
  --circuit-breaker-internal-server-errors 10 \
  --out-jsonl data/phase4_7_agentic_eval_results_feature_sl80.jsonl \
  --out-summary data/phase4_7_agentic_eval_summary_feature_sl80.json \
  --traj-dir data/phase4_7_trajectories_feature_sl80 \
  --patch-dir data/phase4_7_patches_feature_sl80
