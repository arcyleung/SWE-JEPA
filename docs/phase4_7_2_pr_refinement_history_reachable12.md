# Experiment 4.7.2 — PR Refinement History and Conway Drift

## Setup

- Dataset: merged PRs from `prs_copy` with `total_commits >= 2` and stored commit history.
- Snapshot type: cumulative patch `base_sha..commit_i` for each commit in the PR.
- Review alignment: review-thread comments and submitted reviews are attached by timestamp to commit intervals.
- PR limit for this run: `12`.

## Main result

- PRs analyzed: `11`
- Commit snapshots analyzed: `83`
- Review-response transitions: `0`
- Median heuristic Conway risk proxy, first -> final: `1.100` -> `1.593`
- Median heuristic Conway risk proxy, pre-review -> post-review response: `0.000` -> `0.000`

The risk proxy is a heuristic aggregate over Conway-style patch signals. It is used here as a compact summary; the more important evidence is the direction of the individual raw metrics below.

## Metric deltas

| Metric | First->final median delta | First->final improved | Post-review median delta | Post-review improved |
|---|---:|---:|---:|---:|
| `conway_risk_proxy` | 0.425 | 36.4% | 0.000 | 0.0% |
| `conway_risk_flags` | 0.000 | 0.0% | 0.000 | 0.0% |
| `api_change_without_tests` | 0.000 | 0.0% | 0.000 | 0.0% |
| `public_api_without_docs` | 0.000 | 0.0% | 0.000 | 0.0% |
| `shared_change_isolated` | 0.000 | 0.0% | 0.000 | 0.0% |
| `ownership_diffusion` | 0.000 | 18.2% | 0.000 | 0.0% |
| `boundary_density` | 0.072 | 36.4% | 0.000 | 0.0% |
| `operability_score` | 0.000 | 0.0% | 0.000 | 0.0% |

## Interpretation

- Negative deltas are better for risk metrics; positive deltas are better for quality metrics such as `operability_score`.
- If post-review transitions improve more consistently than first->final drift, that is stronger evidence that review rounds are actively shaping the Conway state rather than the effect being only due to PR completion.
- This analysis still only sees the surviving PR commit history in `prs_copy`. Force-pushed-away commits are not recovered here.

## Artifacts

- Snapshot rows: `data/phase4_7_2_pr_refinement_history_reachable12.jsonl`
- Summary JSON: `data/phase4_7_2_pr_refinement_history_reachable12_summary.json`
