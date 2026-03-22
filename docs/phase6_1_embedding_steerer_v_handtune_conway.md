# Experiment 6.1 — Patch Embedding Steerer vs Hand-Tuned Conway Features

## Summary

Frozen Qwen2.5-Coder-3B layer-18 mean-pooled embeddings (2048-dim) replace 61 hand-crafted Conway patch features for PR acceptance and refactor steering. All three embedding-based models decisively outperform the Phase 5.1 baseline.

**Gate**: Acceptance AUROC 0.91 vs threshold 0.71 — **PASS**

## Results

| Model | Acceptance AUROC | Acceptance PR-AUC | Refactor AUROC | Refactor PR-AUC |
|-------|:----------------:|:-----------------:|:--------------:|:---------------:|
| Phase 5.1 baseline (61 Conway features) | 0.71 | — | 0.59 | — |
| LogReg-raw (2048-dim) | 0.901 (±0.007) | 0.982 (±0.002) | 0.744 (±0.017) | 0.170 (±0.024) |
| LogReg-PCA256 (whiten) | 0.901 (±0.010) | 0.982 (±0.002) | 0.752 (±0.027) | 0.168 (±0.027) |
| **MLP (2048→512→1)** | **0.913 (±0.007)** | **0.984 (±0.002)** | **0.763 (±0.025)** | **0.178 (±0.030)** |

Improvement over Phase 5.1: **+20 pp acceptance AUROC, +17 pp refactor AUROC** (MLP).

## Dataset

- **Source**: `prs_copy` (merged) + `python_js_ts_rust_closed_prs` (closed)
- **N**: 159,256 patches
- **Repos**: 6,303 unique
- **Accepted rate**: 86.0%
- **Refactor labels matched**: 55,476 / 159,256 (from `conway_patch_features_v2_100k.jsonl`)
- **Refactor rate**: 8.2%

## Embedding details

- **Teacher**: Qwen2.5-Coder-3B (`/home/original_models/Qwen2.5-Coder-3B`), frozen
- **Layer**: 18 (mid-layer; avoids last-layer collapse per Phase 0 findings)
- **Pooling**: Attention-mask mean-pool → (N, 2048) float32
- **Tokenization**: max_length=512, truncation + padding
- **Cosine sim** (merged centroid vs closed centroid): 0.9911

## Evaluation

- GroupKFold cross-validation, 5 splits, grouped by repo
- Metrics: AUROC, PR-AUC
- Acceptance head: binary (merged vs closed)
- Refactor head: binary (`review_friction` from Conway features)

## Model details

| Model | Input | Method |
|-------|-------|--------|
| LogReg-raw | z_patch (2048) | StandardScaler + LogisticRegression(C=1.0, balanced) |
| LogReg-PCA256 | PCA(z_patch, 256, whiten) | PCA explains 91.0% variance + LogisticRegression |
| MLP | z_patch (2048) | 2048→512→1, ReLU, BCEWithLogits, Adam, 50 epochs |

## Observations

1. **Embeddings dominate features**: A single forward pass through the frozen teacher extracts more steering signal than 61 engineered features combining file-level, import-graph, Conway, and code-structure analysis.
2. **Linear separability**: LogReg matches MLP within ~1 pp — the embedding space is already well-structured for these tasks. Nonlinearity helps marginally.
3. **PCA efficiency**: 256 dims capture 91% of variance and match full-rank performance, suggesting downstream heads can use compressed representations.
4. **Refactor PR-AUC is low** (~0.17) despite decent AUROC — expected given the 8.2% refactor rate (heavy class imbalance). The ranking quality (AUROC 0.76) is the actionable metric.

## Infrastructure

- **Extraction**: 24 workers (3 per GPU x 8 GPUs, A100 80GB), 159k patches in ~20 min (136 patches/s)
- **Training**: LogReg ~3 min, PCA ~12s, MLP ~3 min
- **Code**: `experiment_6/extract_patch_embeddings.py`, `experiment_6/train_embedding_steerer.py`
- **Metrics**: `data/phase6_1/embedding_steerer_metrics.json`
- **Embeddings**: `data/phase6_1/patch_embeddings_100k.npz`

## Next steps

Gate passed — proceed to Experiment 6.2/6.3 (JEPA self-supervised training on patch embeddings).
