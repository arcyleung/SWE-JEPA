# Experiment 4.7.2 — PR Refinement History and Conway Drift

## Setup

- Dataset: merged PRs from `prs_copy` with `total_commits >= 2` and stored commit history.
- Snapshot type: cumulative patch `base_sha..commit_i` for each commit in the PR.
- Review alignment: review-thread comments and submitted reviews are attached by timestamp to commit intervals.
- PR limit for this run: `all`.

## Main result

- PRs analyzed: `31326`
- Commit snapshots analyzed: `104994`
- Review-response transitions: `17578`
- Median heuristic Conway risk proxy, first -> final: `3.290` -> `3.370`
- Median heuristic Conway risk proxy, pre-review -> post-review response: `4.500` -> `4.092`

The risk proxy is a heuristic aggregate over Conway-style patch signals. It is used here as a compact summary; the more important evidence is the direction of the individual raw metrics below.

## Metric deltas

| Metric | First->final median delta | First->final improved | Post-review median delta | Post-review improved |
|---|---:|---:|---:|---:|
| `conway_risk_proxy` | 0.000 | 44.9% | 0.000 | 41.4% |
| `conway_risk_flags` | 0.000 | 15.0% | 0.000 | 7.7% |
| `api_change_without_tests` | 0.000 | 6.3% | 0.000 | 2.0% |
| `public_api_without_docs` | 0.000 | 3.9% | 0.000 | 1.5% |
| `shared_change_isolated` | 0.000 | 2.1% | 0.000 | 0.6% |
| `ownership_diffusion` | 0.000 | 31.1% | 0.000 | 26.9% |
| `boundary_density` | 0.000 | 45.8% | 0.000 | 25.0% |
| `operability_score` | 0.000 | 4.9% | 0.000 | 2.5% |

## Interpretation

- Negative deltas are better for risk metrics; positive deltas are better for quality metrics such as `operability_score`.
- If post-review transitions improve more consistently than first->final drift, that is stronger evidence that review rounds are actively shaping the Conway state rather than the effect being only due to PR completion.
- This analysis still only sees the surviving PR commit history in `prs_copy`. Force-pushed-away commits are not recovered here.

## Merge-Commit Sensitivity

Sampled merge commits were not the main reason the aggregate drift signal was noisy.

- Merge-sampled rows in the completed run: `3266`
- PRs with at least one merge-sampled snapshot: `2462`

Re-running the summary after dropping merge snapshots weakened the primary risk signal:

- `conway_risk_proxy` first->final improved fraction: `44.9%` -> `41.6%`
- `conway_risk_proxy` post-review improved fraction: `41.4%` -> `38.6%`

Dropping entire PRs that sampled a merge commit was also not an improvement:

- `conway_risk_proxy` first->final improved fraction: `41.6%`
- `conway_risk_proxy` post-review improved fraction: `38.7%`

One channel did improve slightly under merge exclusion:

- `boundary_density` first->final improved fraction: `45.8%` -> `47.8%`
- `boundary_density` post-review improved fraction: `25.0%` -> `25.9%`

Net readout: excluding merge snapshots removes some obvious noise, but it removes more useful trajectory signal than it adds back for the primary aggregate proxy. The downstream steerer comparison therefore keeps the full `31.3k` PR trajectory set rather than filtering merge rows out.

## Static Patch vs Trajectory-Aware Steerer

The 4.7.2 dataset contains only merged PRs, so raw acceptance is constant and cannot be used as a training target. For offline comparison, the steerer was retrained on two merged-only proxies:

- `acceptance_proxy = 1 - review_friction`
- `refactor_requested = max(refactor_comments_before) > 0`

Two feature sets were compared with repo-grouped 5-fold CV:

- `static`: final visible patch only
- `history`: final patch plus patch-trajectory summary statistics from the sampled commit sequence

### CV summary

| Slice | Variant | Target | CV AUROC | CV PR-AUC |
|---|---|---:|---:|---:|
| `all` | `static` | `acceptance_proxy` | 0.960 | 0.983 |
| `all` | `history` | `acceptance_proxy` | 0.962 | 0.985 |
| `all` | `static` | `refactor_requested` | 0.901 | 0.302 |
| `all` | `history` | `refactor_requested` | 0.897 | 0.278 |
| `exclude_merge_rows` | `static` | `acceptance_proxy` | 0.955 | 0.980 |
| `exclude_merge_rows` | `history` | `acceptance_proxy` | 0.957 | 0.982 |
| `exclude_merge_rows` | `static` | `refactor_requested` | 0.902 | 0.303 |
| `exclude_merge_rows` | `history` | `refactor_requested` | 0.895 | 0.280 |

### Readout

- The trajectory-aware model helps a little on `acceptance_proxy`, but only marginally.
- It hurts the rarer and more important `refactor_requested` target.
- The best overall offline model remains the `static` final-patch model on the full trajectory slice.

This is the opposite of the hoped-for result. The current trajectory summary is not yet extracting a better supervisory signal than the terminal patch alone. A likely explanation is that the trajectory features are still too coarse: first/final deltas and simple volatility summaries do not separate "useful iterative refinement" from "extra churn" well enough, so the history block adds variance without enough directional signal.

## Follow-On Combined Merged+Closed Retrain

To remove the constant-acceptance limitation in the 4.7.2 merged-only dataset, the
terminal-patch steerer was retrained on the completed Python merged+closed corpus:

- merged source: `prs_copy` rows with `primary_language='Python'`
- closed source: `python_js_ts_rust_closed_prs` rows with recovered patches
- corpus size: `74,810` PRs across `1,053` repos

### Selected downstream metrics

| Variant | Acceptance target | Rows | Acc pos rate | Acc CV AUROC | Acc CV PR-AUC | Ref pos rate | Ref CV AUROC | Ref CV PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `python merged+closed` | `accepted` | 74,810 | 72.1% | 0.724 | 0.835 | 13.7% | 0.721 | 0.271 |

### Downstream readout

- This retrain is the preferred terminal-patch steerer for Python PRs because it uses
  real merged/unmerged labels instead of `acceptance_proxy`.
- The old 4.7.2 history block remains useful as analysis infrastructure, but it is not
  the selected downstream model family.
- Applying an extra patch-level `primary_lang=python` filter reduced the corpus to
  `59,851` rows and slightly worsened the aggregate score (`0.638 -> 0.631`), so the
  selected retrain keeps the full source-filtered Python PR corpus.

## Artifacts

- Snapshot rows: `data/phase4_7_2_slurm_ramcache_v1/merged.jsonl`
- Summary JSON: `data/phase4_7_2_slurm_ramcache_v1/summary.json`
- Merge sensitivity JSON: `data/phase4_7_2_merge_sensitivity.json`
- Merge sensitivity report: `docs/phase4_7_2_merge_sensitivity.md`
- Static vs history metrics: `data/phase4_7_2_pr_steerer_history_compare.json`
- Static baseline model: `data/phase4_7_2_pr_steerer_static_model.json`
- Trajectory-aware model: `data/phase4_7_2_pr_steerer_history_model.json`
- Cross-family comparison: `docs/phase4_7_2_pr_steerer_compare.md`
- Cross-family summary JSON: `data/phase4_7_2_pr_steerer_variant_compare_summary.json`
- Selected Python merged+closed metrics: `data/phase5_1_python_pr_corpus_slurm_v1/pr_steerer_metrics_python_merged_closed_allrows.json`
- Selected Python merged+closed model: `data/phase5_1_python_pr_corpus_slurm_v1/pr_steerer_model_python_merged_closed_allrows.json`
