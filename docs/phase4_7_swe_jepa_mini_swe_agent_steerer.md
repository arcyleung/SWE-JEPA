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

---

## Exp 4.7 Judge Evaluation Results

**Date**: 2026-03-11
**Judge model**: `qwen3.5_397b_a17b` (via LiteLLM)
**Scoring script**: `score_patch_judge_4_7.py`
**Results files**: `data/phase4_7_judge_scores.jsonl`, `data/phase4_7_judge_summary.json`

### Setup

Blind A/B comparison: 254 valid pairs of (baseline patch, steered git diff) were extracted from:
- Baseline patches: `data/phase4_7_patches_feature_sl80/` (patch files)
- Steered patches: git diff blocks extracted from tool output messages in `data/phase4_7_trajectories_steered_7k/` traj files

A/B assignment was randomized per instance (seeded by `iid_pr`) so the judge did not know which patch was steered. The judge was asked to score each patch on 10 criteria (1–5 scale) and indicate a preferred patch.

**Note on pair availability**: The steered_7k run used `git_worktree` isolation and did not explicitly save patch files. Only ~16% of traj files contain a `git diff` block in tool output messages; the rest have an empty submission. This yielded 254 valid pairs from the ~1,526 steered trajectories.

### Rubric (10 criteria)

| # | Criterion | Description |
|---|-----------|-------------|
| 1 | correctness | Does the patch address the stated problem? Are all changed files internally consistent? |
| 2 | code_quality | Readability, naming conventions, idiomatic use of the language/framework |
| 3 | scope_discipline | Is the change focused and minimal? Does it avoid unrelated refactoring? |
| 4 | error_handling | Are failure modes, edge cases, and exceptions handled appropriately? |
| 5 | security | Does the change avoid introducing security issues (injection, hardcoded secrets, etc.)? |
| 6 | observability | Where warranted, are logs, metrics, or health signals included or preserved? |
| 7 | test_coverage | Does the patch add or update tests commensurate with the change? |
| 8 | interface_design | Are new functions, APIs, or public interfaces clean and consistent with existing conventions? |
| 9 | documentation | Are complex or non-obvious parts commented or documented? |
| 10 | review_readiness | Overall merge readiness — would you approve this PR as a senior engineer? |

### Results: 239 evaluated, 15 errors

```
Steered win rate:    43.1%  (103/239)
Baseline win rate:   56.9%  (136/239)
Overall mean score:  steered=3.034  baseline=3.167  Δ=−0.133
```

**Per-criterion breakdown:**

| Criterion | Steered | Baseline | Δ |
|-----------|---------|----------|---|
| correctness | 2.912 | 3.192 | **▼ 0.280** |
| code_quality | 3.071 | 3.285 | ▼ 0.214 |
| scope_discipline | 3.494 | 3.686 | ▼ 0.192 |
| error_handling | 3.067 | 3.071 | ▼ 0.004 |
| security | 4.038 | 4.121 | ▼ 0.083 |
| observability | 2.858 | 2.870 | ▼ 0.012 |
| test_coverage | 2.209 | 2.197 | **▲ 0.012** |
| interface_design | 3.163 | 3.322 | ▼ 0.159 |
| documentation | 2.858 | 2.941 | ▼ 0.083 |
| review_readiness | 2.674 | 2.983 | **▼ 0.309** |

### Patch Size Confound Analysis

Steered patches are substantially smaller (mean ~2,865 bytes vs ~15,064 bytes baseline). To check whether the steered loss is a size artefact:

| Subgroup | N | Steered win rate |
|----------|---|-----------------|
| All pairs | 239 | 43.1% |
| Size-matched (within 2×) | ~60 | **39.6%** |
| Baseline ≥5× longer | ~110 | 42.3% |
| Steered patch is longer | ~32 | **46.1%** |

Patch size is not the primary confound. Even when patches are similar in size, steered wins only 39.6% of the time. When the steered patch is longer (the steerer generated a more complete change), win rate rises to 46.1% — closest to 50/50 but still below baseline.

### Interpretation

The Exp 4.7 6-feature steerer (acceptance head + refactor head, features: `changed_files`, `additions`, `deletions`, `requested_reviewers_count`, `has_closing_issue`, `changed_files_after`) applied scope-discipline constraints but **did not improve judge-assessed patch quality**.

Representative judge justification showing the failure mode:
> "Patch A is a complete implementation that updates all necessary files: the interface definition (IModify.ts), the bridge interface (IRoomBridge.ts), the implementation class (Modify.ts), and test stubs. Patch B only modifies the implementation class while missing critical interface updates, which would cause TypeScript compilation errors."

The steerer's scope constraints caused the agent to produce smaller, more focused diffs — but the scope reduction came at the cost of **completeness**. A patch that touches fewer files but misses required interface updates scores worse on `correctness` (−0.280) and `review_readiness` (−0.309), which are the criteria with the largest deficits.

`test_coverage` is the only criterion where steered marginally wins (+0.012) — consistent with the steering hint to include tests — but the improvement is negligible and does not compensate for correctness losses.

### Conclusion

**The Exp 4.7 steerer did not improve patch quality** as evaluated by an independent judge model. The 6-feature model lacks the structural signals needed to distinguish "correctly scoped" from "incorrectly truncated". This motivates the move to **Exp 5.1's 28-feature Conway-enriched steerer**, which includes:
- `trust_boundary_crossings` — catches incomplete interface updates
- `public_api_surface_delta` — detects missing API surface changes
- `file_coupling_score` — measures whether changed files form a coherent unit
- `has_input_validate` / `has_raise_from` — correctness-adjacent signals

The judge eval establishes a baseline: to declare Exp 5.1's steerer a success, the steered win rate must exceed **50%** against the Exp 4.7 steerer (not just against unsteered baseline).
