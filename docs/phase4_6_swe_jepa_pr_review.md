# Experiment 4.6: SWE-JEPA PR Review Reranking

**Date**: 2026-03-05 19:17 UTC
**Packs**: 114 test packs (6 candidates each)
**Coder model**: qwen3_coder_30b
**Coder concurrency**: 500

## Setup

- Ground truth candidate per pack: historically merged PR from `prs_copy`.
- Five synthetic negatives: naming drift, cross-cutting, scope bloat, architecture-boundary, ownership-friction.
- SWE-JEPA-Conway scorer: logistic head on structural + Conway proxy features.
- Baseline: standalone coder model choosing best candidate via prompted review.

## Main Results

| Model | Top-1 | MRR |
|---|---:|---:|
| SWE-JEPA + Conway | 87.7% | 93.9% |
| Coder model | 39.5% | 69.7% |

**Delta (SWE-JEPA - coder)**: Top-1 +48.2 pp, MRR +24.1 pp

## Win Rate by Variant (real PR ranked above synthetic)

| Variant | SWE-JEPA | Coder |
|---|---:|---:|
| arch_boundary | 100.0% | 39.5% |
| cross_cutting | 93.9% | 39.5% |
| naming_drift | 100.0% | 39.5% |
| ownership_friction | 100.0% | 39.5% |
| scope_bloat | 93.9% | 39.5% |

## Robustness

- 5-fold Group CV (by repo): SWE Top-1 86.8% ± 3.8, Coder Top-1 40.7% ± 3.1
- 5-fold Group CV (by repo): SWE MRR 93.3% ± 1.9, Coder MRR 70.3% ± 1.6


## Notes

- `total_comments`/`total_review_threads` are included as pragmatic metadata proxies and may contain partial outcome leakage.
- This benchmark uses synthetic variants; next step is higher-fidelity semantic-preserving transforms on real patch AST/edit scripts.

## Artifacts

- `docs/phase4_6_pr_review_candidates.jsonl`
- `docs/phase4_6_swe_jepa_vs_coder_metrics.json`
