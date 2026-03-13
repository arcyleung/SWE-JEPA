# Experiment 4.7.2 Merge-Commit Sensitivity

This report re-runs the 4.7.2 refinement-history summary on the finished `merged.jsonl`
under three slices:

- `all`: original dataset
- `exclude_merge_rows`: keep PRs but drop sampled merge-commit snapshots
- `exclude_merge_prs`: drop any PR that sampled a merge commit

## Dataset slices

| Slice | PRs | Rows | Review-response transitions | Merge rows present | Merge PRs present |
|---|---:|---:|---:|---:|---:|
| `all` | 31326 | 104994 | 17578 | 3266 | 2462 |
| `exclude_merge_rows` | 31305 | 101728 | 16303 | 0 | 0 |
| `exclude_merge_prs` | 28864 | 95493 | 15376 | 0 | 0 |

## Risk proxy medians

| Slice | First | Final | Pre-review | Post-review |
|---|---:|---:|---:|---:|
| `all` | 3.290 | 3.370 | 4.500 | 4.092 |
| `exclude_merge_rows` | 3.233 | 3.532 | 4.391 | 4.157 |
| `exclude_merge_prs` | 3.108 | 3.425 | 4.286 | 4.068 |

## Delta Comparison vs `all`: `exclude_merge_rows`

| Metric | First->final improved | Change vs all | Post-review improved | Change vs all |
|---|---:|---:|---:|---:|
| `conway_risk_proxy` | 41.6% | -3.3% | 38.6% | -2.9% |
| `conway_risk_flags` | 14.0% | -1.0% | 7.1% | -0.6% |
| `api_change_without_tests` | 6.1% | -0.2% | 1.8% | -0.2% |
| `public_api_without_docs` | 3.9% | -0.0% | 1.5% | -0.0% |
| `shared_change_isolated` | 2.1% | -0.0% | 0.6% | -0.0% |
| `ownership_diffusion` | 28.9% | -2.3% | 25.1% | -1.8% |
| `boundary_density` | 47.8% | 2.0% | 25.9% | 0.9% |
| `operability_score` | 4.7% | -0.3% | 2.4% | -0.1% |

## Delta Comparison vs `all`: `exclude_merge_prs`

| Metric | First->final improved | Change vs all | Post-review improved | Change vs all |
|---|---:|---:|---:|---:|
| `conway_risk_proxy` | 41.6% | -3.3% | 38.7% | -2.7% |
| `conway_risk_flags` | 14.3% | -0.7% | 7.2% | -0.6% |
| `api_change_without_tests` | 6.4% | 0.1% | 1.9% | -0.1% |
| `public_api_without_docs` | 4.0% | 0.1% | 1.5% | 0.0% |
| `shared_change_isolated` | 2.2% | 0.1% | 0.6% | -0.0% |
| `ownership_diffusion` | 28.6% | -2.6% | 25.0% | -1.9% |
| `boundary_density` | 48.7% | 2.9% | 26.2% | 1.3% |
| `operability_score` | 4.7% | -0.3% | 2.4% | -0.1% |

## Readout

- `exclude_merge_rows` does not improve the primary `conway_risk_proxy` drift signal versus `all` if its improved fraction or post-review improved fraction falls below the original.
- `exclude_merge_prs` is a stricter robustness check, but it is not the intended downstream training slice because it discards full PRs rather than only merge snapshots.

The selection rule for downstream training is simple:

- keep merge rows if the primary risk-proxy trend weakens when they are removed
- otherwise train on the merge-excluded row slice
