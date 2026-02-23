# Experiment 0.2: Nearest-Neighbour Retrieval Results

**Date**: 2026-02-21
**Index**: FAISS `IndexFlatIP` (cosine via L2-normalised inner product), one per model × layer
**Models**: Qwen2.5-Coder-3B, Qwen2.5-Coder-3B-Instruct, Qwen3-8B, Qwen3-8B-base
**Layer**: 18 (mid-layer, best from Experiment 0.1)
**Corpus**: 28 400 function embeddings from real PRs (postgres `function_embeddings` table)

---

## Per-function Results (Qwen2.5-Coder-3B, k=5)

| Query id | Function | Pattern observed | Top-5 sims |
|---|---|---|---|
| 10691 | `stock_em_yjbb` | **Structural template** — same HTTP-fetch-and-parse skeleton, different East-Money endpoints | 0.993, 0.992, 0.991, 0.985, 0.985 |
| 19774 | `createfunc` | Exact copies at 1.0; clean drop to related-but-distinct `_do_orm_execute` | 1.000, 1.000, 0.868, 0.868, 0.868 |
| 12162 | `OnReceive` | Exact copies (1.0), then sibling event handler `HandleChar` | 1.000, 1.000, 1.000, 0.960, 0.960 |
| 23057 | `_media_size_to_long` | All 5 are exact copies across PR versions | 0.9999 × 5 |
| 140 | `_get_permissions` | All 5 are exact copies across strawberry PR history | 1.000 × 5 |
| 17448 | `is_arn_match` | Copies (1.0), then earlier version history (0.9996-0.9997) | 1.000, 1.000, 0.9997, 0.9996, 0.9996 |
| 25332 | `handle_flow_dict` | **Structural family** — `handle_dns_dict` (0.97), `handle_device_dict` (0.95) | 1.000, 0.973, 0.973, 0.952, 0.952 |
| 4716 | `get_absolute_mbr_path` | Cross-repo match `_GetPil` (0.90) — needs LLM judge investigation | 0.901, 0.901, 0.901, 0.901, 0.898 |

---

## Key Finding: Structural Chain in `iot-inspector-client`

The most informative result is the graded similarity hierarchy in `nyu-mlab/iot-inspector-client`:

```
handle_flow_dict   ←→  handle_dns_dict     sim ≈ 0.97
handle_flow_dict   ←→  handle_device_dict  sim ≈ 0.95
handle_dns_dict    ←→  handle_device_dict  (inferred ≈ 0.95)
```

These three functions share a common **upsert-into-local-SQLite template** — they all:
1. Accept a `dict` payload from a network capture
2. Extract typed fields with `.get()` defaults
3. Upsert into a local table via `INSERT OR REPLACE`
4. Emit a Redis pub/sub event

They differ only in the schema (flow 5-tuple vs DNS query/response vs device MAC/IP). The embedding similarity precisely captures the shared algorithmic skeleton while discriminating on schema differences.

### Time-evolution and software maintenance angle

The PR history for this repo reveals the **copy-paste-then-diverge** lifecycle common in real software maintenance:

- `handle_flow_dict` appears first — the canonical implementation
- `handle_dns_dict` was added in a subsequent PR, copied from the flow handler and adapted
- `handle_device_dict` followed the same pattern

The similarity scores decay monotonically with the degree of adaptation: the most recently-derived handler has drifted most from the original. This is precisely the kind of structural temporal signal SWE-JEPA should learn to encode: **latent representations that capture both the shared template and the direction of divergence**.

From a software engineering standpoint, this pattern is a refactoring opportunity — the three functions could be collapsed into a single `handle_dict(table, fields)` abstraction. A student model that can predict the latent of `handle_device_dict` from the signature + surrounding context alone would effectively need to understand this abstraction without being told about it.

---

## Cross-model Comparison: `handle_flow_dict`

| Model | Rank 2 | Rank 3 | Rank 2 sim | Rank 6 |
|---|---|---|---|---|
| Qwen2.5-Coder-3B | `handle_dns_dict` | `handle_dns_dict` (other PR) | 0.9725 | `fill_in_counterparties` (0.946) |
| Qwen2.5-Coder-3B-Instruct | `handle_dns_dict` | `handle_dns_dict` (other PR) | 0.9705 | `_prepare_upload_data` (0.944) |
| Qwen3-8B-base | `handle_dns_dict` | `handle_dns_dict` (other PR) | **0.9779** | `_process_dhcp` (0.960) |
| **Qwen3-8B (instruct)** | `_process_arp` | cross-repo `get_sorted_table_and_fkc_nam` | 0.9628 | — |

**Observation**: The three base/smaller models agree closely on the structural grouping. Qwen3-8B instruct diverges sharply — it retrieves `_process_arp` (same repo but different abstraction layer) and a completely unrelated SQLAlchemy reflection function from a different repository at the same similarity score. This suggests instruction fine-tuning at 8B scale suppresses the syntactic structural signal in favour of a more abstract semantic space — but in this case it misfires, grouping structurally dissimilar functions that happen to share some high-level concept (network data processing, table reflection).

For SWE-JEPA, the 3B base model is the better teacher for structural masking targets; the 8B instruct embedding captures something different (and potentially useful for a second probing dimension).

---

## Conclusions

1. **PR-history dominates retrieval**: The most common pattern is exact copies of the same function across multiple PR versions. This needs filtering or deduplication for clean evaluation.

2. **Structural template detection works** at 0.95–0.99 within a codebase. The `handle_*_dict` family is the clearest example.

3. **Sharp discrimination boundary**: The `createfunc` example shows a clean step from sim=1.0 (copies) to sim=0.87 (related but algorithmically different). This is the geometric property needed for JEPA: dense, discriminative latent space.

4. **Cross-repo matches at ~0.90** require investigation — could be genuine structural similarity or noise. LLM judge needed.

5. **Model recommendation for JEPA teacher**: Qwen2.5-Coder-3B (base) or Qwen3-8B-base. The instruct variants appear to encode a different geometry less suited to structural masking prediction.

---

## Next Steps

- **Experiment 0.3**: Linear probing for static structural properties (cyclomatic complexity, LOC, side effects, return type, branch/loop count, API call count) → validates that the embedding space encodes what the student will need to predict
- Run LLM judge on the `handle_flow_dict` family triplet to quantify the score_a / score_b disconnect
- Deduplicate exact copies before downstream training (keep one representative per function identity)

## References
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/kim-tse-2014.pdf
https://www.melconway.com/Home/pdf/committees.pdf

---

## Full-Scale Evaluation (Feb 22 2026)

**Scale**: 4 models × 28,400 functions × k=10 = **1,136,000 pairs**
**LLM judge**: Qwen3-Coder-30B
**Output**: `docs/phase0_2_knn_results_full.jsonl` (1.1 GB)

### Pipeline Summary

| Step | Result |
|------|--------|
| KNN retrieval | 1,136,000 pairs (4 models) |
| Cross-repo pairs | 31,068 (2.7%) |
| Selected for judging | 808,594 (cross-repo ≥0.80 + same-repo [0.80, 0.999)) |
| Actually judged | 599,250 (74%; 209,344 skipped — source unavailable) |
| Disconnects (\|a−b\| ≥ 2) | 53,769 (9.0% of judged) |

### Cross-Repo Score Distributions (23,032 pairs)

| Prompt | Avg | 0 | 1 | 2 | 3 |
|--------|-----|---|---|---|---|
| a — signature only | 0.65 | 54.3% | 29.9% | 12.7% | 3.2% |
| b — body (names redacted) | 0.55 | 62.7% | 22.4% | 11.7% | 3.2% |
| c — overall | 0.57 | 58.3% | 27.3% | 13.5% | **0.9%** |

**Precision@score_c≥2**: 14.4% of cross-repo pairs at cosine_sim ≥ 0.80 have meaningful functional similarity. LLM judging is essential for filtering training data.

### Key Findings

#### 1. High false-positive rate at cosine sim ≥ 0.80

58.3% of cross-repo pairs are functionally unrelated (score_c=0) despite cosine similarity ≥ 0.80. The embedding space captures syntactic surface patterns (indentation depth, token distribution, call patterns) that transfer across unrelated functions. Threshold-based filtering alone is insufficient for training data curation.

#### 2. 209 gold cross-repo equivalents (score_c=3)

The most valuable subset — same algorithm implemented independently in different repositories:

| Query | Neighbour | Sim | a | b | c | Pattern |
|-------|-----------|-----|---|---|---|---------|
| `is_forward_ref` (strawberry-graphql) | `is_fwd_ref` (sqlalchemy) | 0.964 | 2 | 3 | 3 | same type-check, abbreviated name |
| `__str__` (strawberry-graphql) | `__str__` (EventGhost) | 0.943 | 2 | 2 | 3 | same repr pattern |
| `GetItemData` (EventGhost) | `_get_by_int_impl` (sqlalchemy) | 0.914 | 1 | 3 | 3 | same index-lookup body, different names |
| `__call__` (strawberry-graphql) | `_util_async_run` (sqlalchemy) | 0.899 | 1 | 3 | 3 | same async wrapper, completely different names |

The `is_forward_ref` ↔ `is_fwd_ref` pair is the clearest case: two teams independently implemented the same forward-reference type check with the same algorithm. The embedding captures this semantic equivalence at sim=0.964 despite the functions living in different codebases and carrying different (but semantically linked) names.

The `__call__` ↔ `_util_async_run` pair (a=1, b=3) is the canonical **name_misleads** case: the signatures suggest no relationship, but both implement the same async runner wrapper pattern. A model that correctly clusters these functions must have learned the algorithm, not the name.

#### 3. Score asymmetry: signatures score higher than bodies

Average signature score (0.65) > average body score (0.55) across cross-repo pairs. The embedding is more sensitive to naming conventions and argument patterns than to algorithmic content. Functions sharing naming conventions (`get_`, `_handle_`, `_process_`) get pulled together even when their implementations differ — a dominant source of false positives.

#### 4. Disconnect cases: same algorithm, different names (53,769 pairs)

Top same-repo disconnects at sim≈0.999 include `init`, `pairs`, and `merge` functions in googleapis and sqlalchemy where the body is implementation-equivalent (b=3) but the signature context diverges (a=0–1). These arise from overloaded methods across class hierarchies where the implementation pattern is copied but the constructor signature changes.

### Training Data Tier Structure

| Tier | Count | Criteria | Description |
|------|-------|----------|-------------|
| 1 — Gold equivalents | 209 | score_c=3, cross-repo | Independent implementations of same algorithm |
| 2 — Strong cross-repo | 3,102 | score_c=2, cross-repo | Same purpose, notable implementation differences |
| 3 — Same-repo structural variants | ~40,000 | same-repo, sim 0.80–0.999 | Copy-paste-then-diverge patterns |
| 4 — PR-version copies | ~1,050,000 | sim ≥ 0.999 or score_c≥3 same-repo | Deduplicate to one representative per function |

Tier 1 pairs are the highest-quality JEPA targets: the student that correctly predicts the body latent of `_util_async_run` from its signature alone must have learned the async wrapper abstraction rather than memorising a specific codebase's conventions. Tier 3 provides rich data for learning the direction of divergence in copy-adapt chains (see `handle_*_dict` section above).