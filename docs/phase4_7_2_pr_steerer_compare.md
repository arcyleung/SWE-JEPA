# Experiment 4.7.2 — Static Patch vs Trajectory-Aware Steerer

This report keeps the original 4.7.2 merged-only comparison and adds the follow-on
terminal-patch retrain on the completed Python merged+closed corpus.

The metric columns are the same across all variants: repo-grouped 5-fold CV AUROC and
PR-AUC for the acceptance head and the refactor head. The label regimes are not the
same, so raw selection-score comparisons across families should be interpreted with
caution:

- `phase4_7_2`: merged-only proxies (`acceptance_proxy = 1 - review_friction`)
- `phase4_7`: mixed-language terminal patches with actual merged/unmerged acceptance
- `phase5_1`: Python terminal patches with actual merged/unmerged acceptance

## 4.7.2 Cross-Validation Summary

| Slice | Variant | Target | Pos rate | CV AUROC | CV PR-AUC |
|---|---|---|---:|---:|---:|
| `all` | `static` | `acceptance_proxy` | 75.4% | 0.960 | 0.983 |
| `all` | `static` | `refactor_requested` | 3.4% | 0.901 | 0.302 |
| `all` | `history` | `acceptance_proxy` | 75.4% | 0.962 | 0.985 |
| `all` | `history` | `refactor_requested` | 3.4% | 0.897 | 0.278 |
| `exclude_merge_rows` | `static` | `acceptance_proxy` | 75.4% | 0.955 | 0.980 |
| `exclude_merge_rows` | `static` | `refactor_requested` | 3.4% | 0.902 | 0.303 |
| `exclude_merge_rows` | `history` | `acceptance_proxy` | 75.4% | 0.957 | 0.982 |
| `exclude_merge_rows` | `history` | `refactor_requested` | 3.4% | 0.895 | 0.280 |

## Cross-Family Comparison

| Family | Variant | Dataset | Acceptance target | Refactor labels | Rows | Repos | Acc pos rate | Acc CV AUROC | Acc CV PR-AUC | Ref pos rate | Ref CV AUROC | Ref CV PR-AUC | Selection score |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `phase4_7_2` | `all/static` | merged PR refinement history | `acceptance_proxy` | review-thread derived | 31,326 | 4,158 | 75.4% | 0.960 | 0.983 | 3.4% | 0.901 | 0.302 | 0.787 |
| `phase4_7_2` | `all/history` | merged PR refinement history | `acceptance_proxy` | review-thread derived | 31,326 | 4,158 | 75.4% | 0.962 | 0.985 | 3.4% | 0.897 | 0.278 | 0.781 |
| `phase4_7_2` | `exclude_merge_rows/static` | merged PR refinement history | `acceptance_proxy` | review-thread derived | 31,305 | 4,158 | 75.4% | 0.955 | 0.980 | 3.4% | 0.902 | 0.303 | 0.785 |
| `phase4_7_2` | `exclude_merge_rows/history` | merged PR refinement history | `acceptance_proxy` | review-thread derived | 31,305 | 4,158 | 75.4% | 0.957 | 0.982 | 3.4% | 0.895 | 0.280 | 0.779 |
| `phase4_7` | `mixed-language static` | mixed-language terminal patch corpus | `accepted` | `llm_refactor_requested` | 99,923 | 4,415 | 92.8% | 0.811 | 0.983 | 17.4% | 0.663 | 0.298 | 0.689 |
| `phase5_1` | `python merged+closed` | Python terminal patch corpus | `accepted` | `refactor_requested` | 74,810 | 1,053 | 72.1% | 0.724 | 0.835 | 13.7% | 0.721 | 0.271 | 0.638 |
| `phase5_1` | `python merged+closed + primary_lang=python` | Python terminal patch corpus | `accepted` | `refactor_requested` | 59,851 | 1,020 | 73.9% | 0.705 | 0.835 | 14.9% | 0.704 | 0.280 | 0.631 |

## Readout

- Within the original 4.7.2 merged-only proxy regime, `all/static` remains the best
  model. The trajectory summary helps the easier `acceptance_proxy` target slightly,
  but still hurts the rarer `refactor_requested` head enough to lose on aggregate.
- The new downstream retrain on combined merged+closed Python PRs is the first variant
  in this line with real negative acceptance examples instead of a merged-only proxy.
  On `74,810` PRs it reaches `0.724` CV AUROC / `0.835` CV PR-AUC on acceptance and
  `0.721` / `0.271` on refactor detection.
- The new Python metrics are lower than the 4.7.2 proxy scores because the task is
  harder: actual merged/unmerged classification on terminal PR states is not directly
  comparable to `acceptance_proxy` on merged-only history rows.
- An extra patch-level `primary_lang=python` filter removes `14,959` rows and lowers
  the aggregate score (`0.638 -> 0.631`), so the selected retrain keeps the full
  source-filtered Python PR corpus.

## Selection Rule

- Keep the history-enabled model only as an offline diagnostic for merged-PR trajectory
  analysis; do not carry the trajectory block into the selected terminal-patch steerer.
- Keep the full source-filtered Python merged+closed corpus; do not add the extra
  patch-primary-language filter.

## Artifacts

- Original 4.7.2 comparison JSON: `data/phase4_7_2_pr_steerer_history_compare.json`
- Cross-family summary JSON: `data/phase4_7_2_pr_steerer_variant_compare_summary.json`
- Mixed-language terminal-patch metrics: `data/phase4_7_pr_steerer_metrics.json`
- Selected Python merged+closed metrics: `data/phase5_1_python_pr_corpus_slurm_v1/pr_steerer_metrics_python_merged_closed_allrows.json`
- Selected Python merged+closed model: `data/phase5_1_python_pr_corpus_slurm_v1/pr_steerer_model_python_merged_closed_allrows.json`
