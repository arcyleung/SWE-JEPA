# Experiment 2.2: Token-Level Body Prediction + InfoNCE

**Date**: 2026-02-24
**Teacher**: Qwen2.5-Coder-3B, layer 18 (frozen)
**Task**: Predict function body embedding from function signature token hidden states

## Architecture

```
Frozen:   Qwen2.5-Coder-3B (all 19 transformer layers, bf16, no grad)
          ├─ hook at layer 18 on sig tokens  → sig_hs  [B, T_sig, 2048]
          └─ hook at layer 18 on body tokens → body_hs [B, T_body, 2048] (train only)

Trainable SigPredictorV2 (15,892,992 params):
  Encoder:
    Linear(2048 → 512)
    2× TransformerBlock(d_model=512, nhead=8, ffn=2048, dropout=0.1)
    mean-pool → Linear(512 → 2048)  ← retrieval head (InfoNCE)
  Decoder (training only):
    body_pos_enc: Embedding(256, 512)
    2× CrossAttentionBlock(d_model=512, nhead=8)
    Linear(512 → 2048)  ← per-token prediction head (SmoothL1)
```

## Dataset

| Split | Functions | Description |
|-------|-----------|-------------|
| Train | 42,047 | Repo-stratified 80% split |
| Val   | 4,615 | Held-out repos (10%) |
| Test  | 2,969 | Held-out repos (10%) |

Corpus size (for retrieval): **49,631** functions.

## Training

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 97 (early stopped; patience=15 from best at epoch 82) |
| Batch/GPU | 32 × 8 GPUs = 256 effective |
| Learning rate | 0.0001 (cosine, warmup=5 epochs) |
| d_model | 512 |
| Transformer layers | 2 |
| Attention heads | 8 |
| Dropout | 0.1 |
| Token loss weight λ | 0.1 |
| Max body tokens | 256 |
| Precision | BF16 (frozen Qwen), FP32 (predictor) |

## Results

### Val Set (in-distribution — primary metric)

| Metric | Value | Notes |
|--------|-------|-------|
| Cosine sim | **0.2255** | Held-out repos, same domain as training |
| Rank@1 retrieval | **1.13%** | random=0.0020% |
| Rank@5 retrieval | **2.58%** | — |
| Rank@10 retrieval | **4.20%** | — |

### Test Set

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| Cosine sim | **0.2260** | 0.3650 (sig→body) | ↓0.1390 |
| Cosine sim (random) | — | 0.2603 | Anisotropy baseline |
| Rank@1 retrieval | **1.62%** | 0.0020% (random) | improvement over MLP (0.00%) |
| Rank@5 retrieval | **3.97%** | — | — |
| Rank@10 retrieval | **5.52%** | — | — |

## Comparison with Prior Experiments

| Experiment | Architecture | Val Rank@10 | Test Rank@10 | Best epoch |
|------------|-------------|-------------|--------------|------------|
| 1.2 Transformer encoder | SigPredictor (8.4M, no contrastive) | ~0% | ~0% | — |
| 1.4 InfoNCE | SigPredictor (8.4M, InfoNCE only) | ~4.03% | — | epoch 24 |
| **2.2 InfoNCE + token SmoothL1** | SigPredictorV2 (15.9M, encoder+decoder) | **4.20%** | **5.52%** | epoch 82 |

## Interpretation

**Token-level supervision did not substantially accelerate convergence.** Exp 1.4 (plain InfoNCE) reached 4.03% by epoch 24; Exp 2.2 needed 82 epochs for a marginal 4.20%. The cross-attention decoder adds ~7.5M parameters and a SmoothL1 gradient path that back-propagates through `enc_tokens` (the encoder output used as cross-attention keys/values), creating gradient conflict with the InfoNCE objective — both losses compete for the encoder's representation.

**Test Rank@10 (5.52%) exceeds val (4.20%).** Unlike Exp 1.2 where the test repo (akshare) was OOD, here the test repos happen to be stylistically closer to the training distribution, so the held-out metric is meaningful but slightly optimistic.

**Convergence was LR-schedule limited, not data limited.** The cosine LR schedule was exhausted by epoch ~91 (LR dropped to 3% of peak); the model continued fitting training loss (1.88 at early stop) but val noise dominated. Running 200 epochs with the same peak LR would give substantially more useful gradient steps.

**Bottleneck diagnosis:** With only 9 repos (7/1/1 split), the val set is a single repo — high variance. The teacher (Qwen2.5-Coder-3B, 3B params) may also be too small to produce sufficiently differentiated representations across the corpus. **Next step: Exp 3.0 — upgrade to Qwen3-8B-base teacher (4096-dim, 64 GPUs across 8 nodes).**

## Training Details

| Setting | Value |
|---------|-------|
| Loss | InfoNCE (primary) + λ·SmoothL1 token-level (auxiliary) |
| Final temperature τ | 0.0273 |
| Negatives per step | 8 GPUs × 32 = 256 |
| Best epoch | 82–83 (4.20% tied) |
| Best val Rank@10 (during training) | 4.20% |
| LR at early stop | 1.24e-06 (<2% of peak) |

## Success Criteria

| Criterion | Target | Achieved |
|-----------|--------|---------|
| Val Rank@1 > 3% | > 3% | ❌ 1.13% |
| Val Rank@10 > 10% | > 10% | ❌ 4.20% |
| Test Rank@1 > 3% | > 3% | ❌ 1.62% |
| Test Rank@10 > 10% | > 10% | ❌ 5.52% |
