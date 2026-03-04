# Experiment 4.3: Region-Level Defect & Feature Localization

**Date**: 2026-03-04
**Extends**: Exp 4.1 (PR-level binary classification → within-PR function ranking)
**Data**: followups_function — 6,651 function anchors, 135 repos
**Labels**: has_bugfix=3,036 pos (45.6%) | has_feature=3,691 pos (55.5%)
**Split**: 5,258 train / 789 val / 604 test (by repo, 80/10/10)
**Bugfix min overlap**: ≥ 0.10

## Task description

For each test-set feature PR, all function anchors from that PR are ranked by
their predicted probability of being a bugfix/feature-extension target.

- **Predicted region**: top-K ranked functions and their line ranges
  (`feature_function_start`–`feature_function_end`) at the feature-PR commit.
- **Ground-truth region**: functions where `has_bugfix = 1` (or `has_feature = 1`).
- **Metric**: Mean Recall@K over PRs with ≥1 positive function.

Recall@K = (# true-positive functions ranked in top-K) / (# total true positives
in the PR), averaged over PRs. K is capped at the PR's function count, so
Recall@K = 1.0 when K ≥ n_functions.

## Setup

| Parameter | Value |
|-----------|-------|
| Teacher | Qwen3-8B-base, layer 18 (frozen) |
| Student ckpt | student_3_0_ckpt.pt |
| Embeddings | 4096-dim, mean-pool over sig tokens |
| Probe | LogisticRegression(C=1.0, class_weight='balanced') fitted on train+val |
| Ranking score | predict_proba[:, 1] |
| TF-IDF | 5000 features, word-level, sublinear_tf |
| Bugfix min hunk overlap | 0.10 |

---

## Task A: Bugfix Localization (`has_bugfix`)

Ranking functions within a feature PR by predicted bugfix-proneness.
Ground truth: functions touched in a later bugfix PR with hunk_overlap ≥ 0.10.

- PRs with ≥1 bugfix-positive function (test set): **13**
- Average functions per such PR: **46.4**

| Method | R@1 | R@3 | R@5 | R@10 | GPU-min | R@10/1k GPU-min |
|--------|-----|-----|-----|-----|---------|---------------------|
| Random | 16.6% | 39.8% | 44.8% | 60.0% | 0 | — |
| LOC (sig lines) | 8.5% | 29.1% | 31.7% | 61.1% | 0 | — |
| TF-IDF (5k) | 7.8% | 25.9% | 37.6% | 50.2% | 0 | — |
| Teacher (frozen) | 11.6% | 42.0% | 50.6% | 56.1% | 2 | 28047.63 |
| **Student JEPA** | 1.3% | 22.3% | 43.1% | 46.2% | 2,018 | 22.91 |

Student vs Teacher  Δ R@10: **-9.9 pp**
Student vs TF-IDF   Δ R@10: **-3.9 pp**

---

## Task B: Feature-Extension Localization (`has_feature`)

Ranking functions by predicted likelihood of being refactored or extended in a
later feature PR.

- PRs with ≥1 feature-positive function (test set): **10**
- Average functions per such PR: **58.8**

| Method | R@1 | R@3 | R@5 | R@10 | GPU-min | R@10/1k GPU-min |
|--------|-----|-----|-----|-----|---------|---------------------|
| Random | 1.6% | 9.6% | 19.5% | 41.2% | 0 | — |
| LOC (sig lines) | 4.1% | 12.3% | 28.2% | 42.2% | 0 | — |
| TF-IDF (5k) | 1.7% | 16.0% | 18.7% | 43.9% | 0 | — |
| Teacher (frozen) | 1.8% | 11.4% | 16.9% | 31.7% | 2 | 15833.31 |
| **Student JEPA** | 1.8% | 11.9% | 21.7% | 36.5% | 2,018 | 18.06 |

Student vs Teacher  Δ R@10: **+4.8 pp**
Student vs TF-IDF   Δ R@10: **-7.4 pp**

---

## GPU-hours comparison (Exp 4.2 methodology)

Training cost is fixed per method; efficiency = R@10(bugfix) per 1 000 GPU-min.

| Method | Training GPU-min | R@10 (bugfix) | R@10/1k GPU-min |
|--------|-----------------|-----------------|---------------------|
| Random (no model) | 0 | 60.0% | — |
| TF-IDF (no model) | 0 | 50.2% | — |
| Teacher frozen | 2 (inf only) | 56.1% | 28047.63 |
| **Student JEPA** | **2,018 (train+inf)** | **46.2%** | **22.91** |

*JEPA training GPU-min: 2016 (Exp 3.0, 33 epochs, 32 GPUs × 64 batch).*
*Teacher inference GPU-min: ~2 (Qwen3-8B, 1 GPU, batch=32, 6,651 signatures).*

---

## Interpretation

### Bugfix localization

The localization task tests whether JEPA representations encode spatial information
about *which* functions within a PR are defect-prone — not just whether *any*
function in the PR will need a bugfix followup (Exp 4.1 binary task).

Exp 4.1 result: Student AUROC 0.6003 vs Teacher 0.5673 (+3.3 pp) on PR-level
binary classification.

The within-PR ranking (Recall@K) is a harder evaluation: the probe must correctly
*order* functions relative to each other, not just output a global threshold.
A probe that uniformly shifts all scores upward gains nothing in recall@K; the
signal must come from relative differences between functions in the same PR.

### Feature-extension localization

Feature-extension localization tests whether the JEPA body-structure prediction
objective captures extensibility signals — functions whose signature design will
require future refactoring or API extension — distinct from bugfix-proneness.

Feature followups have different characteristics from bugfixes: they touch
deliberately designed extension points (abstract methods, configurable parameters,
plugin interfaces). A model that only learns defect complexity proxies (cyclomatic
complexity, LOC) should score lower on feature localization than on bugfix
localization.

### Ground-truth region format

For each predicted function, the matched line range in the feature-PR source file is:
```
feature_file : feature_function_start – feature_function_end
```
These line ranges are stored in `followups_function.feature_function_start/end`
and enriched onto each record at runtime (see `--skip-enrich` flag to bypass).

---

## Files

| File | Contents |
|------|----------|
| `probe_region_localization.py` | This experiment script |
| `probe_defect_prediction.py`   | Exp 4.1 (reused for encode_all, load_sigs) |
| `followup_sigs.jsonl`          | Input: per-function anchors with labels |
| `followup_embs.npz`            | Cached embeddings (teacher + student) |
| `phase4_3_topk_predictions.jsonl` | Per-PR top-K ranked functions with line ranges |
| `docs/phase4_3_region_localization.md` | This report |
