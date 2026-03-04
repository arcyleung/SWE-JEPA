# Experiment 4.4: SE-Head Stack for Region Localization

**Date**: 2026-03-04
**Data**: 6,651 function anchors, 135 repos
**Split**: 5,258 train / 789 val / 604 test (repo split)
**Labels**: has_bugfix=3,036 (45.6%), has_feature=3,691 (55.5%)
**min_overlap**: 0.10
**Models**: linear, fusion, mlp, pairwise
**Seeds**: [42, 43, 44]
**Static CC coverage**: 6,258/6,651 (94.09%)
**SE feature channels**: teacher_logit, student_logit, tfidf_logit, loc_logit, cyclomatic_complexity, churn_proxy

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
| SE-Linear | 3.9% | 39.4% | 51.4% | 57.7% | 58.0% | 13 |
| SE-Fusion | 3.9% | 40.0% | 51.4% | 58.9% | 55.1% | 13 |
| SE-MLP | 11.0% | 40.3% | 49.6% | 58.4% | 54.6% | 13 |
| SE-Pairwise | 3.3% | 39.4% | 50.5% | 58.8% | 50.0% | 13 |

## Task B: Feature Localization (`has_feature`)

| Method | R@1 | R@3 | R@5 | R@10 | MRR | n_prs |
|--------|-----|-----|-----|-----|-----|------|
| Random | 7.6% | 16.4% | 17.9% | 33.5% | 63.4% | 10 |
| LOC | 4.1% | 12.3% | 28.2% | 42.2% | 71.6% | 10 |
| TF-IDF | 1.1% | 16.0% | 21.1% | 34.2% | 47.8% | 10 |
| Teacher | 1.8% | 11.4% | 16.9% | 31.7% | 51.4% | 10 |
| Student | 1.8% | 11.9% | 21.7% | 36.5% | 53.2% | 10 |
| SE-Linear | 1.9% | 11.6% | 19.6% | 38.1% | 59.1% | 10 |
| SE-Fusion | 0.9% | 12.6% | 18.7% | 37.4% | 54.1% | 10 |
| SE-MLP | 4.2% | 13.6% | 19.3% | 39.1% | 73.5% | 10 |
| SE-Pairwise | 0.9% | 13.2% | 18.6% | 35.6% | 55.6% | 10 |

## Key deltas

- Pairwise vs Teacher (bugfix R@10): **+2.7 pp**
- Pairwise vs Random (feature R@10): **+2.1 pp**

## Artifacts

- `docs/phase4_4_se_heads_metrics.json`
- `docs/phase4_4_topk_predictions.jsonl`
