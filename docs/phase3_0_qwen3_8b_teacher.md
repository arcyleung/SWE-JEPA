# Experiment 3.0: Qwen3-8B-base Teacher + InfoNCE

**Date**: 2026-02-25
**Teacher**: Qwen3-8B-base, layer 18 (frozen)
**Task**: Predict function body embedding from function signature token hidden states

## Architecture

```
Frozen:   Qwen3-8B-base (36 transformer layers, bf16, no grad)
          ├─ hook at layer 18 on sig tokens  → sig_hs  [B, T_sig, 4096]
          └─ hook at layer 18 on sig+body tokens → body_hs [B, T_body, 4096] (train only)
            (body tokens extracted from concatenated sig+body forward pass)

Trainable SigPredictorV2 (63,251,456 params):
  Encoder:
    LayerNorm(4096)              ← removes input anisotropy
    Linear(4096 → 1024)
    2× TransformerBlock(d=1024, nhead=8, ffn=4096)
    mean-pool → Linear(1024 → 4096)  ← retrieval head (InfoNCE)
  Decoder (training only, λ=0.0 so inactive):
    body_pos_enc: Embedding(256, 1024)
    2× CrossAttentionBlock(d=1024, nhead=8)
    Linear(1024 → 4096)  ← per-token prediction head (SmoothL1)
```

## Dataset

| Split | Functions |
|-------|-----------|
| Train | 42,047 |
| Val   | 4,615 |
| Test  | 2,969 |

Corpus (retrieval): **49,631** functions from 150 repos.

## Training

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 33 (early stopping patience=15, best at epoch ~18) |
| Batch/GPU | 64 × 32 GPUs = 2048 effective |
| Learning rate | 0.0001 (cosine, warmup=5 epochs) |
| d_model | 1024 |
| Transformer layers | 2 enc + 2 dec |
| Attention heads | 8 |
| Dropout | 0.1 |
| Token loss weight λ | 0.0 (decoder present but inactive) |
| Max body tokens | 256 |
| FSDP teacher | False |
| Precision | BF16 (frozen Qwen), FP32 (predictor) |

## Results

### Val Set (in-distribution — primary metric)

| Metric | Value |
|--------|-------|
| Cosine sim | **0.4748** |
| Rank@1 | **19.11%** |
| Rank@5 | **36.77%** |
| Rank@10 | **47.52%** |

### Test Set

| Metric | Value | Baseline |
|--------|-------|----------|
| Cosine sim | **0.4768** | random: 0.0133 |
| Rank@1 | **28.09%** | 0.0020% (random) |
| Rank@5 | **45.47%** | — |
| Rank@10 | **51.77%** | — |

## Comparison with Prior Experiments

| Experiment | Teacher | Dataset | Val Rank@1 | Val Rank@10 | Test Rank@10 | Best epoch |
|------------|---------|---------|-----------|------------|--------------|------------|
| 1.2 Transformer enc | 3B, 2048-dim | 9 repos | ~0% | ~0% | OOD failure | — |
| 1.4 InfoNCE | 3B, 2048-dim | 150 repos | 1.06% | 4.03% | 4.14% | 24 |
| 2.1 Hard neg mining | 3B, 2048-dim | 150 repos | 1.04% | 3.64% | 4.31% | 51 |
| 2.2 Token-level JEPA | 3B, 2048-dim | 150 repos | 1.13% | 4.20% | 5.52% | 82 |
| 3.0 (body-only targets) | 8B, 4096-dim | 9 repos | — | 1.38% | — | 72 |
| 3.0 (body-only targets, 150 repos) | 8B, 4096-dim | 150 repos | — | 2.82% | 3.20% | 54 |
| **3.0 (sig+body targets, 150 repos)** | **8B, 4096-dim** | **150 repos** | **19.11%** | **47.52%** | **51.77%** | **~18** |

## Training Details

| Setting | Value |
|---------|-------|
| Loss | InfoNCE only (λ=0.0, SmoothL1 inactive) |
| Final temperature τ | 0.0654 |
| Negatives per step | 32 GPUs × 64 = 2048 |
| Best val Rank@10 | 47.52% |
| Multi-node launch | `torchrun --nnodes=4 --nproc-per-node=8 --rdzv-backend=c10d --rdzv-endpoint=10.10.110.20:29500 train_student_3_0.py` |

## Interpretation

### The critical insight: target representation quality

The 17× improvement over Exp 2.2 (47.52% vs 4.20% val Rank@10) came almost entirely from one change: **how body embeddings are computed for the InfoNCE target**.

Previous runs fed only body tokens to the teacher, producing body-only hidden states. This caused severe **representation anisotropy**: random cosine similarity of body-only targets was **0.37** (after mean-centering), vs **0.016** for full-function embeddings stored in the DB. The reason: body text alone lacks the function signature for context. Common opening patterns (`self`, `return`, `if not`, etc.) dominate all body representations, pulling them into a tight cluster. With targets clustered near a centroid, InfoNCE cannot build a discriminative embedding space — the temperature τ barely moved from its initial value (0.07 → 0.065 instead of the usual 0.03–0.05 collapse), confirming the signal was too weak to compress the space.

**Fix**: feed `[sig_tokens] + [body_tokens]` concatenated to the teacher, then extract only the body token positions from layer 18. This gives the teacher full function context — body representations are anchored by the unique signature — reducing anisotropy from 0.37 → ~0.016 and aligning targets with the full-function embeddings stored in postgres.

This also makes the JEPA objective more faithful: the target is `f(body | sig_context)`, while the predictor maps `g(sig_only) → f(body | sig_context)`. The predictor must learn to hallucinate the full-context body representation from the signature alone — a true predictive task.

### Why 8B outperforms 3B by 10×

1. **Richer teacher representations**: 4096-dim (vs 2048) with 36-layer model. Layer-18 representations of an 8B model are meaningfully more differentiated than those of a 3B model — information that was shared across functions at 3B is now separated at 8B.

2. **More negatives per step**: 32 GPUs × 64 batch = 2048 negatives vs 8 GPUs × 32 = 256. 8× more negatives per InfoNCE step dramatically improves the quality of the contrastive signal, especially for discriminating within-repo similar functions.

3. **Properly isotropic targets**: The sig+body fix alone accounts for most of the gain. The 3B experiments never suffered body-only anisotropy because they used precomputed body embeddings from the postgres DB (which were computed with full-function context).

### Failed intermediate runs (3.0 diagnostic history)

**Run 1 — 9 repos, body-only targets** (val Rank@10 = 1.38%):
- SQL join required 8B rows in `function_embeddings` → limited to 9 repos where 8B embeddings existed
- torch_dtype → dtype fix needed (fp32 loading = 32GB vs 16GB)
- d_model=512 too tight (8:1 bottleneck vs 3B's 4:1); increased to 1024
- λ=0.1 SmoothL1 over 4096 dims dominates InfoNCE (409 vs 7.62 loss scale); set to 0.0

**Run 2 — 150 repos, body-only targets** (val Rank@10 = 2.82%):
- Simplified SQL to 2-way join → 150 repos restored
- All hyperparameter fixes applied
- Still poor: τ=0.062 barely moved, random cosine=0.37 for body-only targets
- Diagnosis: body-only targets are anisotropic (missing signature context)

**Run 3 — 150 repos, sig+body targets** (val Rank@10 = 47.52%):
- precompute_body_embs feeds concatenated sig+body; extracts body positions
- Random cosine of targets: 0.013 (near-isotropic)
- Training converges by epoch 18; early stopped at epoch 33

### Infrastructure fixes during this experiment

- `dtype=torch.bfloat16` in `from_pretrained` (torch_dtype deprecated in transformers ≥5.x)
- `faiss_index = None` before conditional block (UnboundLocalError on non-rank-0)
- Each rank independently computes all body embeddings at startup (no `all_gather_object`) — eliminates 10+ minute NCCL init hang during precompute
- InfiniBand env vars required: `NCCL_SOCKET_IFNAME=bond0`, `NCCL_IB_HCA=mlx5`, `NCCL_GID_INDEX=3`

## Success Criteria

| Criterion | Target | Achieved |
|-----------|--------|---------|
| Val Rank@1 > 5% | > 5% | ✅ 19.11% |
| Val Rank@10 > 20% | > 20% | ✅ 47.52% |
| Test Rank@1 > 5% | > 5% | ✅ 28.09% |
| Test Rank@10 > 20% | > 20% | ✅ 51.77% |
