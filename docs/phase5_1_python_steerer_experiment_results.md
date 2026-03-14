# Experiment 5.1 — Python Steerer Results

**Date**: 2026-03-14
**Status**: Python steerer validated; Go steerer needs retraining
**Model files**: `data/phase4_7_pr_steerer_model.json` (mixed-language, 100k PRs)

---

## Summary

The Conway-aware steerer trained on `prs_copy` (predominantly Python PRs) correctly
differentiates bugfix-prone PRs from clean PRs using structural code signals. The top
discriminating features are ownership-related Conway proxies — not surface-level PR
metadata — confirming the Experiment 5.1 hypothesis.

The Go steerer (Exp 4.7.3) shows **inverted signals** due to dataset imbalance: the
Go corpus has a higher baseline merge rate and different coding conventions (see
Exp 5.1.1 language confounds analysis), causing the acceptance head to learn the wrong
direction. Retraining on Go-specific merged/closed PR pairs is required.

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

The refactor head has lower discrimination because refactor requests are rare (~8% base
rate) and subjective. The acceptance head is the primary steering signal.

---

## Agentic Eval — Steering Loop Results (Go ablation, Exp 4.7)

The steerer was evaluated in a closed-loop agentic setting using mini-swe-agent on
100 Go PR tasks with 6-judge panel scoring:

| Condition | Completion rate | Judge win rate (steered vs baseline, 52 valid pairs) |
|-----------|-----------------|-----------------------------------------------------|
| Baseline (no steering) | 96/100 (after retry) | — |
| Steered (specific hints) | 87/100 | 50.0% (GPT-5-codex), 48.1% (Claude Opus 4.6), 54.5% (GLM-5) |

**Key finding**: The steerer does not significantly improve Go patches because the
model was trained on Python-dominant data. The Conway signals (blame authors, ownership
diffusion) fire correctly but the acceptance/refactor heads have inverted calibration
for Go's conventions.

---

## Go Steerer — Why It Needs Retraining

1. **Dataset imbalance**: `prs_copy` is ~75% Python. Go conventions (explicit error
   handling via `if err != nil`, capitalized public functions, no try/catch) create
   systematic feature distribution shifts (see Exp 5.1.1).

2. **Inverted acceptance signal**: Go PRs in `prs_copy` have higher baseline merge rates
   than Python PRs, causing the acceptance head to assign high scores to patterns that
   are simply "Go-like" rather than genuinely merge-ready.

3. **Solution**: Retrain on `go_prs` table (merged + closed PRs) using `--language go`
   filter in `train_pr_steerer.py`. The closed PRs provide negative examples with
   correct calibration.

---

## Next Steps

1. Add `--language` filter to `train_pr_steerer.py` for per-language training
2. Move steerer core into `agentic_scaffold/mini-swe-agent/src/minisweagent/steerer.py`
3. Create `steer_harness.py` for agent-agnostic steering wrapper
4. Retrain Go steerer on `go_prs` merged/closed pairs and re-run ablation
