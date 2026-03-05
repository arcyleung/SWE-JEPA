# Experiment 4.5: Conway Signals for Region Localization

**Date**: 2026-03-05
**Data**: 6,651 function anchors, 135 repos
**Split**: 5,258 train / 789 val / 604 test (repo split)
**Labels**: has_bugfix=3,036 (45.6%), has_feature=3,691 (55.5%)
**min_overlap**: 0.10
**Models**: linear, fusion, mlp, pairwise
**Seeds**: [42, 43, 44]
**Static CC coverage**: 6,258/6,651 (94.09%)
**Org metrics coverage**: 6,651/6,651 (100.00%)
**Uses org metrics**: yes
**SE feature channels**: teacher_logit, student_logit, tfidf_logit, loc_logit, cyclomatic_complexity, churn_proxy, org_commits_touching_file, org_distinct_authors, org_top_author_fraction, org_author_entropy, org_ownership_friction, org_cochange_weighted_degree, org_cochange_unique_neighbors, org_cochange_cross_module_ratio, org_interface_stress

## Reframing summary

Frozen teacher/student embeddings are treated as a representation substrate.
Downstream SE behavior is learned via cheap task heads optimized for within-PR ranking.
No full-model SFT/RL is used.

## Task A: Bugfix Localization (`has_bugfix`)

| Method | R@1 | R@3 | R@5 | R@10 | MRR | n_prs |
|--------|-----|-----|-----|-----|-----|------|
| Random | 16.6% | 39.8% | 44.8% | 60.0% | 62.0% | 13 |
| LOC | 8.5% | 29.1% | 31.7% | 61.1% | 50.6% | 13 |
| TF-IDF | 7.9% | 25.4% | 29.4% | 56.2% | 41.2% | 13 |
| Teacher | 11.6% | 42.0% | 50.6% | 56.1% | 69.5% | 13 |
| Student | 1.3% | 22.3% | 43.1% | 46.2% | 49.4% | 13 |
| SE-Linear | 3.9% | 39.3% | 49.8% | 58.2% | 52.9% | 13 |
| SE-Fusion | 3.9% | 40.0% | 51.4% | 58.9% | 55.1% | 13 |
| SE-MLP | 11.0% | 30.5% | 50.6% | 59.0% | 51.6% | 13 |
| SE-Pairwise | 3.3% | 38.8% | 49.2% | 58.1% | 50.8% | 13 |

## Task B: Feature Localization (`has_feature`)

| Method | R@1 | R@3 | R@5 | R@10 | MRR | n_prs |
|--------|-----|-----|-----|-----|-----|------|
| Random | 7.6% | 16.4% | 17.9% | 33.5% | 63.4% | 10 |
| LOC | 4.1% | 12.3% | 28.2% | 42.2% | 71.6% | 10 |
| TF-IDF | 1.1% | 16.0% | 21.1% | 34.2% | 47.8% | 10 |
| Teacher | 1.8% | 11.4% | 16.9% | 31.7% | 51.4% | 10 |
| Student | 1.8% | 11.9% | 21.7% | 36.5% | 53.2% | 10 |
| SE-Linear | 3.6% | 14.2% | 17.1% | 38.2% | 64.3% | 10 |
| SE-Fusion | 0.9% | 12.6% | 18.7% | 37.4% | 54.1% | 10 |
| SE-MLP | 6.0% | 15.9% | 18.5% | 38.8% | 81.0% | 10 |
| SE-Pairwise | 3.6% | 15.8% | 17.9% | 37.7% | 67.2% | 10 |

## Key deltas

- Pairwise vs Teacher (bugfix R@10): **+2.0 pp**
- Pairwise vs Random (feature R@10): **+4.2 pp**

## Artifacts

- `docs/phase4_5_conway_metrics.json`
- `docs/phase4_5_topk_predictions.jsonl`
