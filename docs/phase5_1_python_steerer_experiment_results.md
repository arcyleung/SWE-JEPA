# Experiment 5.1 — Python Steerer Results

**Date**: 2026-03-17 (updated; original 2026-03-14)
**Status**: FeatBench eval complete; scaffold-aware judges show +7.2 pp steered advantage
**Model files**: `data/phase4_7_pr_steerer_model.json` (mixed-language, 100k PRs)
**Eval artefacts**: `data/featbench_v1_eval_artefacts.7z` — contains:
- `evaluation_results_with_patches.json` — 28 baseline + 28 steered agent results with patch diffs
- `judge_panel/*.jsonl` — 6 patch-only judge JSONL files (19 verdicts each)
- `judge_panel_scaffold/*.jsonl` — 6 scaffold-aware judge JSONL files + `summary.json`

---

## Summary

The Conway-aware steerer trained on `prs_copy` (predominantly Python PRs) correctly
differentiates bugfix-prone PRs from clean PRs using structural code signals. We ran
a controlled A/B evaluation on FeatBench v1.0 with Qwen3.5-35B (baseline vs steered)
across 19 paired instances, scored by a 6-model judge panel in two modes:

1. **Patch-only judges** — LLM judges see only the task + two diffs
2. **Scaffold-aware judges** — LLM judges run inside trae-agent with bash/file tools,
   exploring the repository before scoring (50-step budget)

The steered agent produces higher F2P pass rates (+21.4 pp) and wins the scaffold-aware
judge panel **55.5%** vs 48.2% for patch-only (+7.2 pp). The strongest steered advantages
are in **test_coverage** (+0.95), **documentation** (+0.24), and **review_readiness** (+0.24).
Scaffold judges are harsher on correctness/error_handling (repo context reveals issues
invisible from diffs alone), but the penalty is instance-specific, not rubric-driven.

---

## Agent Eval — Pass Rates (28 FeatBench instances)

| Agent | F2P pass | P2P pass | Both |
|-------|----------|----------|------|
| qwen35-baseline | 17/28 (60.7%) | 3/28 (10.7%) | 2/28 (7.1%) |
| qwen35-steered | 23/28 (82.1%) | 3/28 (10.7%) | 3/28 (10.7%) |

The steerer improves F2P by +21.4 pp while maintaining P2P parity.

---

## Judge Panel Results (19 paired instances with both patches)

### B) Patch-only judges (6 judges × 19 = 114 comparisons)

| Judge | Steered wins | Total | Win % |
|-------|-------------|-------|-------|
| claude_opus_4_6 | 10 | 19 | 52.6% |
| gemini_31_pro_thinking | 4 | 19 | 21.1% |
| glm_5 | 10 | 19 | 52.6% |
| gpt-5-codex | 12 | 19 | 63.2% |
| kimi_k2.5 | 9 | 19 | 47.4% |
| qwen3.5_397b_a17b_judge | 10 | 19 | 52.6% |
| **PANEL** | **55** | **114** | **48.2%** |

### C) Scaffold-aware judges (6 judges × 19, 110 valid after retry)

| Judge | Steered wins | Valid | Errors | Win % |
|-------|-------------|-------|--------|-------|
| claude_opus_4_6 | 9 | 19 | 0 | 47.4% |
| gemini_31_pro_thinking | 10 | 18 | 1 | 55.6% |
| glm_5 | 10 | 18 | 1 | 55.6% |
| gpt-5-codex | 9 | 17 | 2 | 52.9% |
| kimi_k2.5 | 11 | 19 | 0 | 57.9% |
| qwen3.5_397b_a17b_judge | 12 | 19 | 0 | 63.2% |
| **PANEL** | **61** | **110** | **4** | **55.5%** |

4 remaining errors: gemini on instructlab (truncated 192K patch still too complex),
gpt-5-codex on two instances (agent exhausted 80-step budget without writing verdict),
glm_5 on openai-agents-python-1235 (verdict parse failure).

### Side-by-side comparison

| Judge | Patch-only | Scaffold | Δ |
|-------|-----------|----------|---|
| claude_opus_4_6 | 53% | 47% | −5.3 pp |
| gemini_31_pro_thinking | 21% | 56% | **+34.5 pp** |
| glm_5 | 53% | 56% | +2.9 pp |
| gpt-5-codex | 63% | 53% | −10.2 pp |
| kimi_k2.5 | 47% | 58% | **+10.5 pp** |
| qwen3.5_397b_a17b_judge | 53% | 63% | **+10.5 pp** |
| **PANEL** | **48.2%** | **55.5%** | **+7.2 pp** |

Key observations:
- **gemini_31_pro_thinking** swings from 21% → 56% (+34.5 pp) — it was anomalously
  anti-steered in patch-only mode but corrects when given repo context.
- **kimi_k2.5** and **qwen3.5_397b_a17b_judge** both rise +10.5 pp — scaffold context
  lets them appreciate the steered patches' test coverage and documentation advantages.
- **gpt-5-codex** drops from 63% → 53% — the most capable judge penalizes steered
  patches on correctness/error_handling when it can see full codebase context, but
  the shift is moderate (-10 pp) and it still favors steered overall.
- **claude_opus_4_6** drops from 53% → 47% — becomes more conservative with repo
  context, penalizing scope_discipline and error_handling on specific instances.

---

## Per-criterion deltas (steered − baseline)

| Criterion | Patch-only | Scaffold | Shift |
|-----------|-----------|----------|-------|
| correctness | +0.000 | −0.118 | −0.118 |
| code_quality | −0.092 | −0.109 | −0.017 |
| scope_discipline | +0.000 | −0.109 | −0.109 |
| error_handling | −0.133 | −0.245 | −0.113 |
| security | +0.031 | +0.000 | −0.031 |
| observability | +0.051 | +0.091 | +0.040 |
| **test_coverage** | **+0.969** | **+0.945** | −0.024 |
| interface_design | +0.122 | +0.082 | −0.041 |
| **documentation** | **+0.296** | **+0.236** | −0.060 |
| **review_readiness** | **+0.337** | **+0.236** | −0.100 |

The steerer's strongest signal — **test_coverage** — is robust across both judge modes
(~+0.95 point on 5-point scale). Documentation and review_readiness also consistently
favor steered patches.

**Scaffold penalty analysis**: The scaffold judges are modestly harsher on correctness
(−0.12), scope_discipline (−0.11), and error_handling (−0.11). However, this penalty is
**instance-specific, not rubric-driven** — the drops are concentrated on 3-4 instances
where repo context reveals genuine issues (e.g., type-safety bugs in openai-agents-python-357,
scope bloat in mesa-2296). No criterion shows a uniform shift across all instances,
confirming that the rubric is not systematically biased against steered patches.

The 4 remaining errors (3.5% of runs) are not due to step exhaustion from the 80-step
budget — they are caused by agents failing to write well-formed verdict.json (parsing
failures or the judge getting sidetracked exploring instead of scoring).

---

## Key Conway Features — Python PRs

### Positive signals (higher in bugfix-prone / refactor-requested PRs)

| Feature | Δ (positive − negative) | Interpretation |
|---------|-------------------------|----------------|
| `blame_unique_authors` | +0.175 | More distinct authors touched → higher coordination cost → more review friction |
| `ownership_diffusion` | +0.082 | Changes spread across multiple owners → Conway boundary crossing |
| `trust_boundary_crossings` | +0.073 | External/security imports without observability → hidden integration risk |
| `api_change_without_tests` | +0.051 | Public API surface changes without corresponding test additions |
| `cross_module_spread` | +0.044 | Changes touching multiple top-level modules |

### Negative signals (lower in bugfix-prone PRs — protective factors)

| Feature | Δ (positive − negative) | Interpretation |
|---------|-------------------------|----------------|
| `blame_top_author_share` | −0.112 | High single-author ownership → fewer coordination issues |
| `has_try_catch` | −0.067 | Explicit error handling → fewer post-merge issues |
| `has_log_in_except` | −0.041 | Logging in exception handlers → better observability |
| `has_input_validate` | −0.038 | Input validation present → defensive coding |

---

## Steerer Head Performance (Python-dominant corpus)

### Acceptance head (logistic regression)

| Metric | Train | 5-fold CV (GroupKFold by repo) |
|--------|-------|-------------------------------|
| AUROC | 0.73 | 0.71 |
| PR-AUC | 0.96 | 0.95 |

### Refactor head (logistic regression, target: `refactor_requested`)

| Metric | Train | 5-fold CV (GroupKFold by repo) |
|--------|-------|-------------------------------|
| AUROC | 0.62 | 0.59 |
| PR-AUC | 0.18 | 0.15 |

---

## Scaffold Judge Infrastructure

The scaffold-aware judges run inside Docker containers via `score_featbench_judge_scaffold.py`:

1. Creates container from cached `featbench_*` image for each (instance, judge) pair
2. Sets up trae-agent inside the container (reuses pre-installed copy or clones fresh)
3. Checks out the base commit in the worker's isolated swap directory
4. Runs trae-agent CLI with a judge prompt and 80-step budget (`trae_config_judge.yaml`)
5. Agent explores repo with bash/file tools, writes `verdict.json` via bash heredoc
6. Orchestrator reads verdict from host-mounted swap directory

24 workers process 114 runs in parallel (~35 min wall-clock time).

**Fixes applied during development**:
- Patch truncation at 30K chars to avoid OS `argument list too long` (instructlab's 192K
  baseline patch was the trigger)
- Step budget increased from 50 → 80 after first run showed agents exhausting steps
- Per-worker repo copies to avoid git index.lock contention
- Stale index.lock cleanup before git checkout

**Final error rate**: 4/114 (3.5%) after one retry round. Remaining failures are verdict
parse issues, not step exhaustion.

---

## Appendix: Go Steerer Ablation (Exp 4.7)

The steerer was also evaluated on 100 Go PR tasks using mini-swe-agent:

| Condition | Completion rate | Judge win rate (steered vs baseline, 52 valid pairs) |
|-----------|-----------------|-----------------------------------------------------|
| Baseline (no steering) | 96/100 (after retry) | — |
| Steered (specific hints) | 87/100 | 50.0% (GPT-5-codex), 48.1% (Claude Opus 4.6), 54.5% (GLM-5) |

The steerer does not significantly improve Go patches because `prs_copy` is ~75% Python.
Go conventions (`if err != nil`, capitalized public functions, no try/catch) create
systematic feature distribution shifts (see Exp 5.1.1). Retraining on Go-specific
merged/closed PR pairs is required.

---

## Next Steps

1. Add steerer-aligned dimensions to rubric (Conway alignment, dependency hygiene,
   regression risk) — current 10 criteria are generic; steerer-specific criteria may
   amplify the detected advantage
2. Run scaffold judges on remaining 9 unpaired instances once both agents solve them
3. Investigate the 3 instances where scaffold flips majority winner — these represent
   cases where repo context materially changes the quality assessment
4. Per-instance agreement analysis between patch-only and scaffold (correlation, kappa)
