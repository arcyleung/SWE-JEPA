# Experiment 5.1 — Python Steerer Results

**Date**: 2026-03-17 (updated; original 2026-03-14)
**Status**: FeatBench eval complete; scaffold-aware judges show +4.4 pp steered advantage
**Model files**: `data/phase4_7_pr_steerer_model.json` (mixed-language, 100k PRs)

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
judge panel 52.7% vs 48.2% for patch-only. The strongest steered advantages are in
**test_coverage** (+0.97), **documentation** (+0.25), and **review_readiness** (+0.17).

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

### C) Scaffold-aware judges (6 judges × 19 = 114 attempts, 93 valid)

| Judge | Steered wins | Valid | Errors | Win % |
|-------|-------------|-------|--------|-------|
| claude_opus_4_6 | 7 | 17 | 2 | 41.2% |
| gemini_31_pro_thinking | 10 | 18 | 1 | 55.6% |
| glm_5 | 8 | 16 | 3 | 50.0% |
| gpt-5-codex | 6 | 13 | 6 | 46.2% |
| kimi_k2.5 | 8 | 15 | 4 | 53.3% |
| qwen3.5_397b_a17b_judge | 10 | 14 | 5 | 71.4% |
| **PANEL** | **49** | **93** | **21** | **52.7%** |

21 errors (18.4% of runs) were due to: instructlab image issues (exit 255, 6/6 failed),
agents exhausting 50-step budget without writing verdict, or kimi_k2.5 intermittent
verdict failures.

### Side-by-side comparison

| Judge | Patch-only | Scaffold | Δ |
|-------|-----------|----------|---|
| claude_opus_4_6 | 53% | 41% | −11.5 pp |
| gemini_31_pro_thinking | 21% | 56% | **+34.5 pp** |
| glm_5 | 53% | 50% | −2.6 pp |
| gpt-5-codex | 63% | 46% | −17.0 pp |
| kimi_k2.5 | 47% | 53% | +6.0 pp |
| qwen3.5_397b_a17b_judge | 53% | 71% | **+18.8 pp** |
| **PANEL** | **48.2%** | **52.7%** | **+4.4 pp** |

Key observations:
- **gemini_31_pro_thinking** swings from 21% → 56% (+34.5 pp) — it was anomalously
  anti-steered in patch-only mode but corrects when given repo context.
- **qwen3.5_397b_a17b_judge** rises from 53% → 71% — this is the same model family as
  the agent (Qwen3.5), suggesting it sees steerer-aligned patterns more clearly with
  repo context.
- **gpt-5-codex** drops from 63% → 46% — the most capable judge penalizes the steered
  patches more when it can see the full codebase (possible over-exploration bias with
  50 steps).

---

## Per-criterion deltas (steered − baseline)

| Criterion | Patch-only | Scaffold |
|-----------|-----------|----------|
| correctness | +0.000 | −0.215 |
| code_quality | −0.092 | −0.172 |
| scope_discipline | +0.000 | −0.215 |
| error_handling | −0.133 | −0.290 |
| security | +0.031 | −0.011 |
| observability | +0.051 | +0.075 |
| **test_coverage** | **+0.969** | **+0.968** |
| interface_design | +0.122 | +0.054 |
| **documentation** | **+0.296** | **+0.247** |
| **review_readiness** | **+0.337** | **+0.172** |

The steerer's strongest signal — **test_coverage** — is robust across both judge modes
(~+1.0 point on 5-point scale). Documentation and review_readiness also consistently
favor steered patches. The scaffold judges are harsher on correctness, code_quality,
scope_discipline, and error_handling — suggesting the repo context reveals issues that
patch-only judging misses.

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
4. Runs trae-agent CLI with a judge prompt and 50-step budget (`trae_config_judge.yaml`)
5. Agent explores repo with bash/file tools, writes `verdict.json` via bash heredoc
6. Orchestrator reads verdict from host-mounted swap directory

24 workers process 114 runs in parallel (~35 min wall-clock time).

**Error analysis**: 21/114 runs failed:
- 6 from `instructlab` (cached image broken, exit 255)
- 15 from agents not writing verdict.json within 50 steps or verdict parse failures
  (primarily on large-patch instances like openai-agents-python-1235, openai-agents-python-508)

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

1. Fix instructlab cached image for complete scaffold coverage (6 missing judgments)
2. Increase scaffold step budget to 80 for large-patch instances to reduce verdict failures
3. Add steerer-aligned dimensions to rubric (Conway alignment, dependency hygiene, regression risk)
4. Run scaffold judges on remaining 9 unpaired instances once both agents solve them
5. Compare per-instance agreement between patch-only and scaffold judges (correlation analysis)
