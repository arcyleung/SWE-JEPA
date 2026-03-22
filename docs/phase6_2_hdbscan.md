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

Current followup data covers only **148/5,927 repos** (2.5%) and **46k function-level
rows**. The full `prs_copy` table has 152k PRs across 5,926 repos with cached clones
on disk. Both extraction scripts already support `--num-shards`/`--shard-index` for
parallel execution.

See `experiment_6/run_followup_extraction_slurm.sh` for the Slurm job array to
scale extraction to all repos.

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
