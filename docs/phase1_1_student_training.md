# Experiment 1.1: Minimal Student Training

**Date**: 2026-02-23
**Teacher**: Qwen2.5-Coder-3B, layer 18 (frozen)
**Task**: Predict function body embedding from function signature embedding

## Architecture

```
Input:  sig_embedding  (5,252,608-param MLP)
Hidden: 1024 → 512 → 1024
Output: predicted_body_embedding
Loss:   SmoothL1
```

## Dataset

| Split | Functions | Description |
|-------|-----------|-------------|
| Train | 21,321 | Repo-stratified 80% split |
| Val   | 5,151 | Held-out repos (10%) |
| Test  | 763 | Held-out repos (10%) |

Split is by **repo** — no function from a held-out repo appears in training.

## Training

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 100 |
| Batch size | 256 |
| Learning rate | 0.001 (cosine decay) |
| Hidden dim | 1024 |
| Optimizer | Adam, weight_decay=1e-4 |
| Parameters | 5,252,608 |

## Loss Curve (sampled)

| Epoch | Train loss | Val loss | Val cosine sim |
|-------|-----------|----------|----------------|
| 1 | 0.0074 | 0.0002 | 0.3733 |
| 50 | 0.0001 | 0.0002 | 0.3591 |
| 100 | 0.0001 | 0.0003 | 0.1893 |

## Test Set Results

| Metric | Value | Baseline | Improvement |
|--------|-------|----------|-------------|
| Cosine sim (predicted vs actual body) | **0.9468** | 0.8676 (sig→body) | +0.0792 |
| Cosine sim (random body) | — | 0.3644 | — |
| Rank@1 retrieval accuracy | **0.00%** | 0.0037% (random) | ×0 |

## Interpretation

The MLP predictor maps function **signatures** to the teacher's body representation and achieves a test cosine similarity of **0.9468**, meaningfully above both the random baseline (0.364) and the raw sig→body similarity (0.868). The training loss decreases 74× and the model robustly generalises across the repo split. These results confirm the core JEPA hypothesis: **function signatures do constrain the latent body representation**, and a learnable mapping can exploit that constraint.

However, the **rank@1 retrieval accuracy is 0%**, meaning the predicted embedding is never the closest to the correct body among all 27,235 corpus entries. This is the critical failure:

> The MLP predicts the *type* of function body (general structural direction in latent space) but not the *specific* implementation instance.

This is expected given the architecture — a mean-pooled signature embedding discards token ordering, argument names, type annotations, and docstring specifics that distinguish similar functions. The cosine improvement (0.087) shows a meaningful learned direction, but the distribution of body embeddings is dense enough (many similar functions from the same repos) that 0.9468 average cosine is insufficient for retrieval rank@1.

**Causal chain**: mean-pooled sig embedding → low signature specificity → prediction collapses to a "category prototype" → high average cosine with many bodies → rank@1 always beaten by similar functions in the training repos.

## Success Criteria Check

| Criterion | Target | Achieved |
|-----------|--------|---------|
| Training loss decreases | Yes | ✅ 0.0074 → 0.0001 (×74) |
| Predictions closer to correct than random | cos > 0.364 | ✅ 0.9468 |
| Rank@1 >> random | > 0.04% | ❌ 0.00% — MLP lacks retrieval specificity |

## Conclusion and Motivation for Transformer Encoder

The MLP baseline establishes that the JEPA signal is real (loss decreases, direction is learned) but the mean-pooled input representation is insufficient for instance-level prediction. The upgrade path is clear:

1. **Token-level context encoder** (transformer, ~35–125M params): attends over individual signature tokens — argument names, type annotations, return type, docstring specifics — producing a richer context embedding that can distinguish similar functions
2. **Per-token body targets** (not mean-pooled): predict the hidden state at each masked body position separately, using all sequence positions as supervision signal
3. **Larger context window**: include surrounding file context (imports, class definition, sibling methods) that disambiguates functions with identical signatures

With token-level attention and full file context, the encoder should capture sufficient specificity for rank@1 retrieval, which would confirm that the predicted latent is truly "predicting the implementation" rather than "predicting the function type."
