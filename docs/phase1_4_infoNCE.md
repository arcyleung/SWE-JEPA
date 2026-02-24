# Experiment 1.4: InfoNCE Contrastive Loss

**Date**: 2026-02-24
**Teacher**: Qwen2.5-Coder-3B, layer 18 (frozen)
**Task**: Predict function body embedding from function signature token hidden states

## Architecture

```
Frozen:   Qwen2.5-Coder-3B (all 19 transformer layers, bf16, no grad)
          ↓ forward hook → token hidden states [B, T, 2048]

Trainable SigPredictor (8,404,480 params):
  Linear(2048 → 512)
  2× TransformerEncoderLayer(d_model=512, nhead=8, ffn=2048, dropout=0.1)
  mean-pool over non-padding tokens
  Linear(512 → 2048)
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
| Epochs | 100 (early stopping patience=15) |
| Batch/GPU | 64 × 8 GPUs = 512 effective |
| Learning rate | 0.0001 (cosine, warmup=5 epochs) |
| d_model | 512 |
| Transformer layers | 2 |
| Attention heads | 8 |
| Dropout | 0.1 |
| Precision | BF16 (frozen Qwen), FP32 (predictor) |
| GPUs | 8× H100 80GB HBM3 |

## Results

### Val Set (in-distribution — primary metric)

| Metric | Value | Notes |
|--------|-------|-------|
| Cosine sim | **0.2292** | Held-out repos, same domain as training |
| Rank@1 retrieval | **1.06%** | random=0.0020% |
| Rank@5 retrieval | **2.45%** | — |
| Rank@10 retrieval | **4.03%** | — |

### Test Set

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| Cosine sim | **0.2254** | 0.3650 (sig→body) | ↓0.1396 |
| Cosine sim (random) | — | 0.2603 | Anisotropy baseline |
| Rank@1 retrieval | **1.38%** | 0.0020% (random) | improvement over MLP (0.00%) |
| Rank@5 retrieval | **2.90%** | — | — |
| Rank@10 retrieval | **4.14%** | — | — |

## Interpretation

InfoNCE directly addressed the discrimination failure from Exp 1.3. Switching from SmoothL1 to the contrastive objective achieved a **46× improvement in Rank@1** (0.03% → 1.38% test) and **17× improvement in Rank@10** (0.24% → 4.14% test) using the same architecture and dataset.

As expected, cosine similarity **decreased** (0.589 → 0.225 test). InfoNCE does not penalise predictions for being far from the target in L2/cosine distance — it only requires ranking the correct body above all negatives. The lower cosine sim is not a regression; the retrieval metrics are the true signal.

The val_r10 plateau at ~3.5–4% from epoch 16 onwards is consistent with the hard negative saturation predicted in the design doc: with 512 in-batch negatives drawn randomly from 50k functions, easy negatives dominate the InfoNCE denominator and gradient signal weakens before reaching 5%. The early-stopping criterion (patience=15) correctly identified this plateau.

### What the plateau tells us

The model correctly ranks the positive above random negatives (reaching Rank@10 ~4%), but is stopped by hard negatives — pairs that are semantically distinct but share syntactic structure (same repo style, similar function signatures). These are almost never sampled by chance from 50k candidates with 512 negatives/step.

**Path to >5% Rank@10**: hard negative mining (Exp 2.1). At inference the FAISS index already identifies the top-K nearest neighbours. Using those as explicit negatives in the InfoNCE denominator would give the model exactly the cases it fails on.

## Training Details

| Setting | Value |
|---------|-------|
| Loss | InfoNCE with learnable temperature |
| Final temperature τ | 0.0499 |
| Negatives per step | 8 GPUs × 64 = 512 |
| Best val Rank@10 (during training) | 4.03% |

## Success Criteria

| Criterion | Target | Achieved |
|-----------|--------|---------|
| Val Rank@1 > 1% | > 1% | ✅ 1.06% |
| Val Rank@10 > 5% | > 5% | ❌ 4.03% |
| Test Rank@1 > 1% | > 1% | ✅ 1.38% |
| Test Rank@10 > 5% | > 5% | ❌ 4.14% |
