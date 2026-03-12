# Experiment 4.7.2 — PR Refinement History and Conway Drift

## Setup

- Dataset: merged PRs from `prs_copy` with `total_commits >= 2` and stored commit history.
- Snapshot type: cumulative patch `base_sha..commit_i` for each commit in the PR.
- Review alignment: review-thread comments and submitted reviews are attached by timestamp to commit intervals.
- PR limit for this run: `120`.

## Main result

- PRs analyzed: `41`
- Commit snapshots analyzed: `239`
- Review-response transitions: `25`
- Median heuristic Conway risk proxy, first -> final: `3.339` -> `3.425`
- Median heuristic Conway risk proxy, pre-review -> post-review response: `4.320` -> `4.320`

The risk proxy is a heuristic aggregate over Conway-style patch signals. It is used here as a compact summary; the more important evidence is the direction of the individual raw metrics below.

## Metric deltas

| Metric | First->final median delta | First->final improved | Post-review median delta | Post-review improved |
|---|---:|---:|---:|---:|
| `conway_risk_proxy` | 0.000 | 43.9% | 0.000 | 28.0% |
| `conway_risk_flags` | 0.000 | 26.8% | 0.000 | 20.0% |
| `api_change_without_tests` | 0.000 | 12.2% | 0.000 | 8.0% |
| `public_api_without_docs` | 0.000 | 12.2% | 0.000 | 12.0% |
| `shared_change_isolated` | 0.000 | 2.4% | 0.000 | 0.0% |
| `ownership_diffusion` | 0.000 | 22.0% | 0.000 | 8.0% |
| `boundary_density` | -0.048 | 61.0% | 0.000 | 24.0% |
| `operability_score` | 0.000 | 7.3% | 0.000 | 4.0% |

## Interpretation

- Negative deltas are better for risk metrics; positive deltas are better for quality metrics such as `operability_score`.
- If post-review transitions improve more consistently than first->final drift, that is stronger evidence that review rounds are actively shaping the Conway state rather than the effect being only due to PR completion.
- This analysis still only sees the surviving PR commit history in `prs_copy`. Force-pushed-away commits are not recovered here.

## Sampling Note

- The original implementation used every visible commit in a PR, but very long PRs made the run dominated by a small number of commit-heavy or very large repositories.
- The current extractor therefore caps each PR at a small number of cumulative snapshots, sampled at equal intervals across the visible commit sequence. With `max_snapshots=5`, a 9-commit PR is sampled as commit indices `1, 3, 5, 7, 9`.
- This is a lossy approximation: it can miss short-lived oscillations where a risky intermediate revision is introduced and then removed before the next sampled point.
- In practice, the assumption is that most PR evolution is lower-frequency than the raw commit stream. Review-driven refinement usually changes scope, interface shape, tests, or cross-module spread over several commits rather than in a single one-commit spike.
- The Nyquist/Shannon analogy is only partial here. Commits are already discrete author actions rather than a uniformly sampled continuous-time signal, so there is no strict sampling theorem guarantee. The practical question is not alias-free reconstruction of the full trajectory, but whether a sparse set of visible revisions is sufficient to recover the main directional trend of Conway risk.
- For this experiment, the tradeoff is acceptable because the goal is longitudinal trend estimation, not exact replay of every micro-edit. If future analysis suggests meaningful high-frequency review-response patterns, the sampling policy should be revisited or made adaptive around review events.

## Artifacts

- Snapshot rows: `data/phase4_7_2_pr_refinement_history.jsonl`
- Summary JSON: `data/phase4_7_2_pr_refinement_history_summary.json`
