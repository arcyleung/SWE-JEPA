# Experiment 4.7.2 — Static Patch vs Trajectory-Aware Steerer

This report compares two offline steerer variants on the completed 4.7.2 merged-PR
trajectory dataset:

- `static`: final visible patch only
- `history`: final patch plus patch-trajectory summary statistics

Targets are merged-only proxies because the 4.7.2 dataset contains only merged PRs:

- `acceptance_proxy = 1 - review_friction`
- `refactor_requested = max(refactor_comments_before) > 0`

## Cross-validation summary

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

## Readout

- Selected training slice: `all`
- Selected model variant: `static`
- Selection score: `0.787` (mean of target CV AUROC and PR-AUC)

## Selection rule

- choose the history-enabled model only if it outperforms the static baseline on the mean of `acceptance_proxy` and `refactor_requested` CV metrics
- choose the merge-excluded row slice only if it beats the full row slice under the same rule
