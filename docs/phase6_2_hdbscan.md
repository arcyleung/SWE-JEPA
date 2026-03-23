# Experiment 6.2 — HDBSCAN Clustering Results

## Overview

Two clustering analyses on frozen teacher embeddings, profiled post-hoc with
Conway features and organizational metrics.  Both use UMAP reduction → clustering
→ agglomerative merge into interpretable super-clusters.

---

## 1. Patch-Level Super-Clusters (159k patches, 20 groups)

**Source:** `data/phase6_2/super_clusters.json`
**Embeddings:** Qwen2.5-Coder-3B layer-18 → contrastive projection (256-dim)
**Pipeline:** UMAP(256→15) → HDBSCAN(min_cluster=200) → Ward merge → 20 super-clusters
**Profiling:** All 113 Conway patch features (full feature set, not just 12 compound axes)

### Key Groups

| Super | n | Acc% | Label | Steering implication |
|-------|-------|------|-------|---------------------|
| 0 | 3,923 | 98.4 | try-catch↑ / log-warn-err↑ / error-contract↑ | Well-defended code — relax error handling hints |
| 3 | 14,209 | 99.9 | pub-func↑ / api-change-no-tests↑ / try-catch↓ | Public API additions without tests — flag for test coverage |
| 4 | 9,788 | 95.9 | docs-file-ratio↑ / test-file-ratio↑ | Well-documented + tested code — minimal steering needed |
| 5 | 3,331 | 75.9 | deprecated↑ / startup-file↑ / boundary-density↑ | Legacy infra with deprecation — flag for migration risk |
| 6 | 2,858 | 85.2 | thread-spawn↑ / infra-file↑ / log-fatal↑ | Concurrent infrastructure — flag for race conditions |
| 11 | 3,506 | 69.6 | ownership-diffusion↑ / blame-unique-authors↑ | Many-hands code — flag for coordination |
| 13 | 11,192 | **18.6** | shared-change-isolated↑ / inheritance↑ | **Highest risk**: isolated shared util changes — very low acceptance |
| 15 | 3,376 | 75.2 | shared-file-ratio↑ / shared-change-isolated↑ | Shared utility refactors — flag for downstream impact |

### Observations

- 201 HDBSCAN clusters collapsed to 20 super-clusters with 113-feature profiling
- 32.5% of patches are HDBSCAN noise (use per-axis heads only for those)
- Super 13 is the clearest risk signal: 11k patches with 18.6% acceptance — isolated
  changes to shared/inherited code are the most rejected patch pattern
- The 113-feature labels are much more discriminative than the 12-axis labels
  (54 distinct combos from 12 axes vs. 20 fully distinct labels from 113 features)

---

## 2. Function-Level Followup Clusters (6,651 anchors, 15 groups)

**Source:** `data/phase6_2/followup_clusters/super_clusters.json`
**Embeddings:** Qwen3-8B-base layer-18 teacher embeddings (4096-dim)
**Pipeline:** UMAP(4096→15) → KMeans(k=15) on UMAP space
**Profiling:** 13 features (4 from followup_sigs + 9 org metrics)
**Labels:** has_bugfix, n_bugfix_prs, has_feature, max_bugfix_overlap,
commits_touching_file, distinct_authors, top_author_fraction, author_entropy,
ownership_friction, cochange_weighted_degree, cochange_unique_neighbors,
cochange_cross_module_ratio, interface_stress

### Key Groups

| Super | n | Bugfix% | Risk | Label | Steering implication |
|-------|------|---------|------|-------|---------------------|
| 10 | 39 | **71.8** | HIGH | co-change-neighbors↑ / author-entropy↑ / co-change-degree↑ | Heavily coupled hotspot — many files co-changed, many authors → flag for careful testing |
| 11 | 19 | **10.5** | LOW | author-entropy↓ / ownership-friction↓ / co-change-neighbors↓ | Single-owner isolated code — rarely needs bugfixes |
| 6 | 36 | 41.7 | LOW | top-author-share↑ / ownership-friction↓ | Well-owned by one person — lower bugfix risk |
| 5 | 81 | 42.0 | LOW | interface-stress↓ / cross-module-ratio↓ | Self-contained, no boundary crossings — safe |
| 8 | 40 | 45.0 | LOW | author-entropy↓ / co-change-neighbors↓ | Isolated, low-complexity code |
| 0 | 2,044 | 56.1 | MED | cross-module-ratio↑ / interface-stress↑ | Baseline risk — generic cross-module code |
| 4 | 1,680 | 58.5 | MED | n-bugfix-prs↑ / bugfix-overlap↓ | Slightly elevated risk, many small bugfix touches |

### Observations

- Global bugfix rate: 56.0% (high baseline — these are functions introduced in feature PRs)
- Bugfix rate spread: 10.5% – 71.8% (Δ=61.3pp) — significant separation
- **Key predictor pattern**: high coupling (co-change degree, cross-module ratio,
  interface stress) + diffuse ownership → high bugfix probability
- **Safe pattern**: concentrated ownership + low co-change coupling → low bugfix rate
- Only 6,651 anchors from 148 repos (of 5,927 in prs_copy) — massive room to scale

---

## 2b. Function-Level Followup Clusters v2 (3,338 PRs, 20 groups)

**Source:** `data/phase6_2/function_followup_clusters/super_clusters.json`
**Embeddings:** Qwen2.5-Coder-3B layer-18 patch embeddings (2048-dim) from phase6_1
**Pipeline:** UMAP(2048→15, cosine) → KMeans(k=20)
**Profiling:** 121 features (8 followup aggregates from postgres + 113 Conway patch features)
**Join:** 3,338 PRs with both patch embeddings and function-level followup data

Compared to v1 (6,651 anchors from `followup_sigs.jsonl` with only 13 org metrics),
v2 uses patch embeddings for geometry and the full Conway feature set for profiling,
giving much richer cluster labels.

### Key Groups

| Super | n | Bugfix% | Acc% | Risk | Label | Steering implication |
|-------|------|---------|------|------|-------|---------------------|
| 13 | 7 | **100** | **0.0** | HIGH | n_docs↑ / n_maint↑ / n_total↑ | Extreme churn hotspot — all rejected, all get bugfixes |
| 14 | 210 | **66.2** | 87.1 | HIGH | n_bugfix↑ / n_functions↑ | Many functions with bugfix followups — test heavily |
| 2 | 160 | **64.4** | 88.7 | HIGH | max_overlap↑ / n_functions↑ | High-overlap followups — code gets revisited |
| 7 | 47 | **14.9** | 95.7 | LOW | has_bugfix↓ / max_overlap↓ | Safe isolated code — rarely needs fixes |
| 15 | 56 | **25.0** | 78.6 | LOW | dependency_change↑ / dep_file↑ | Dependency changes — low bugfix risk |
| 3 | 119 | **39.5** | 92.4 | LOW | max_overlap↓ / has_bugfix↓ | Low-overlap, self-contained code |
| 12 | 525 | 50.3 | 86.5 | MED | try_catch↑ / blame_top_author↑ | Error-handling code owned by one author |
| 19 | 177 | 59.9 | **73.4** | MED | n_docs↑ / n_functions↑ / n_total↑ | High-churn with many functions — lowest acceptance |

### Observations

- Global bugfix rate: 51.0%, acceptance rate: 87.1%
- **Bugfix spread: 14.9% – 100% (Δ=85.1pp)** — much better than v1 (Δ=61.3pp)
- 7 HIGH-risk clusters, 4 LOW-risk clusters
- Conway features (try_catch, dependency_change, blame_top_author) add structural
  signal on top of the followup counts
- Key pattern: **more distinct functions touched + higher overlap fractions → more bugfixes**
- Acceptance and bugfix risk are partially independent — Super 19 has low acceptance
  (73.4%) but only medium bugfix risk (59.9%)

---

## 3. Steering Strategy

The steerer can combine both signals:

1. **Patch-level cluster** (from 159k-patch HDBSCAN): assign incoming patch to nearest
   super-cluster centroid → select appropriate hint category
2. **Function-level bugfix risk** (from followup clusters): if the touched functions
   fall in a HIGH-risk cluster, prepend a "careful testing" advisory
3. **Per-axis heads** (12 LogReg from Stage 2): for noise points or fine-grained
   per-axis signals, fall back to individual axis predictions

### Risk Tiers for Hint Generation

| Condition | Action |
|-----------|--------|
| Patch in Super 13 (shared-isolated, 18.6% acc) | Strong warning: isolated shared util change |
| Function in followup Super 10 (71.8% bugfix) | Flag: heavily coupled code, add thorough tests |
| Patch in Super 3 (pub-func↑, no tests) | Hint: add integration tests for new API |
| Function in followup Super 11 (10.5% bugfix) | Relax: well-owned isolated code |
| HDBSCAN noise point | Use per-axis heads only |

---

## 4. Coverage Gap: Function Followups

After Slurm-parallel extraction (22 shards × 18 nodes) + backfilling unified diffs
for bare-string repos:

| Table | Rows | Distinct repos | Change |
|-------|------|---------------|--------|
| followups_file | 2,363,517 | 3,692 | 18× repos (from 202) |
| followups_function | 72,341+ | 148 | +56% rows (from 46k) |

File-level coverage expanded massively via `backfill_file_patches.py` which extracted
unified diffs from cached repo clones for ~1,400 PRs that previously only had
bare-string filenames. Function-level is bottlenecked by repos needing fuse-overlayfs
+ tree-sitter AST parsing.

---

## 5. Contrastive Projection Head (Stage 1) — Negative Result

The InfoNCE contrastive projection (2048→256) **did not improve** over raw embeddings:

| Metric | Projected | Raw | Verdict |
|--------|-----------|-----|---------|
| Retrieval P@10 (val) | 0.944 | 0.948 | FAIL |
| Acceptance AUROC | 0.671 | 0.686 | FAIL |
| Proj wins on axes | 2/12 | 10/12 | FAIL |

**Root cause:** Severe overfitting — val loss diverged from epoch ~4 (best=6.39)
while train loss kept dropping to 0.96.  The projection destroyed axis-specific
information rather than concentrating it.  Raw embeddings remain the better
feature space for probing heads.

The HDBSCAN clustering still works because it operates on the projected space's
geometric structure (which preserves some useful grouping even if linear separability
is degraded).
