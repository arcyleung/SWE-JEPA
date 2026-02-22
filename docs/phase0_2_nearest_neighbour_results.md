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