# Phase 4.7: SWE-JEPA Steerer + mini-swe-agent

## Objective
Build a model-agnostic steering layer that improves PR-level merge readiness in agentic coding runs without changing the base coder model. The steerer should shape behavior toward Conway-aligned edits (ownership-local, low interface stress, low refactor friction risk) while keeping inference cheap.

## High-Level Architecture
1. **Base agent (unchanged):** mini-swe-agent with any LiteLLM-compatible model endpoint.
2. **Small steerer head:** lightweight logistic heads trained on PR evolution signals (`acceptance`, `refactor_requested`) from `prs_copy`-derived data.
3. **Steered policy loop:** for each task, run up to K attempts; score each attempt; keep the best or early-stop when merge-readiness thresholds are met.

## Inputs and Signals
- Static task features:
  - `changed_files`, `additions`, `deletions`
  - `requested_reviewers_count`
  - `has_closing_issue`
- Runtime/attempt features:
  - `changed_files_after` from generated patch
  - scope drift penalty vs expected file-count from ground-truth PR metadata

## Steerer Scoring
For each attempt:
- predict `P(accept)` using acceptance head
- predict `P(refactor_requested)` using refactor head
- compute:

`score = w_accept * P(accept) - w_refactor * P(refactor) - scope_penalty * scope_drift`

Default settings:
- `w_accept=1.0`
- `w_refactor=1.0`
- `scope_penalty=0.15`

## Steering Policy
1. Generate attempt 0 with a focused-scope constraint.
2. Evaluate attempt with steerer score.
3. If thresholds are met, stop early:
   - `P(accept) >= 0.65`
   - `P(refactor) <= 0.35`
4. Else retry with stricter steering hints (reduce scope drift, reduce cross-cutting churn).
5. Keep best-scoring attempt.

This stays transparent to model provider/model family because steering happens in the runner prompt policy and reranking layer, not inside model internals.

## Implementation
Added:
- [`run_phase4_7_agentic_eval_steered.py`](/shared_workspace_mfs/arthur/coder/run_phase4_7_agentic_eval_steered.py)

Core properties:
- mini-swe-agent compatible (`minisweagent.run.mini`)
- model-agnostic via `models.yaml` + optional `--api-base/--litellm-model` overrides
- supports overlayfs/worktree isolation
- de-duplicates PR tasks by `(repo, pull_number)` to avoid workspace collisions
- streams JSONL rows incrementally
- writes per-attempt trajectory files: `instance__prX__aY.traj.json`

## CLI Example
```bash
python run_phase4_7_agentic_eval_steered.py \
  --model-name qwen3_coder_30b \
  --limit 100 \
  --concurrency 100 \
  --step-limit 20 \
  --timeout-sec 1800 \
  --steerer-model data/phase4_7_pr_steerer_model.json \
  --steer-max-attempts 3 \
  --out-jsonl data/phase4_7_agentic_eval_results_steered.jsonl \
  --out-summary data/phase4_7_agentic_eval_summary_steered.json \
  --traj-dir data/phase4_7_trajectories_steered
```

Baseline compatibility run:
```bash
python run_phase4_7_agentic_eval_steered.py --disable-steering ...
```

## Evaluation Plan
1. Generate baseline trajectories (existing unsteered runner).
2. Generate steered trajectories (new runner).
3. Use strong judge model to compare:
   - patch quality vs ground truth
   - merge-readiness preference (baseline vs steered)
   - scope discipline and refactor-churn penalties

## Notes
- This steerer is intentionally small/cheap and can be retrained frequently.
- Next upgrade path: replace hand-crafted runtime features with SWE-JEPA embeddings + lightweight heads for richer PR-state estimation.
