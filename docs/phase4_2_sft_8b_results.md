# Experiment 4.2b SFT Baseline Results (8B)

**Date**: 2026-03-01
**Model**: Qwen3-8B-base full fine-tune + Linear(4096→4096) projection head
**Training input**: sig_text + body_text (full function, up to 512 tokens)
**Eval input**: sig_text only (deliberate train-test mismatch — favours JEPA)
**Target**: Qwen3-8B-base layer-18 body embeddings (same as JEPA Exp 3.0)
**Optimizer**: Adafactor (lr=5e-05, relative_step=False) — reduces VRAM vs AdamW

## Architecture

| Component | Detail |
|-----------|--------|
| SFT model | Qwen3-8B-base (8,207,512,576 total params incl. proj) |
| Projection head | Linear(4096 → 4096, no bias) |
| Eval layer | Layer 18 mean-pool (sig-only at inference) |
| Loss | InfoNCE (τ learnable, same as JEPA) |
| Gradient checkpointing | Enabled |
| Teacher (body targets only) | Qwen3-8B-base layer 18 (loaded from body_embs.npy cache) |

## Dataset

| Split | Functions |
|-------|-----------|
| Train | 42,047 |
| Val   | 4,615 |
| Test  | 2,969 |
| Corpus (retrieval) | 49,631 |

## Training

| Hyperparameter | Value |
|----------------|-------|
| Epochs trained | 32 (patience=15) |
| Batch/GPU | 4 |
| LR | 5e-05 (cosine, warmup=3) |
| Total GPU-minutes | 4587 |

## Results

| Split | Rank@1 | Rank@5 | Rank@10 |
|-------|--------|--------|---------|
| Val   | 1.80% | 6.07% | 9.71% |
| Test  | 2.59% | 6.43% | 9.60% |

## Comparison with JEPA and 3B SFT

| Method | Val Rank@10 | Params | Training input | GPU-minutes |
|--------|------------|--------|----------------|-------------|
| JEPA (Exp 3.0) | 47.52% | 63M trainable | sig-only (no mismatch) | (see JEPA log) |
| SFT 3B (Exp 4.2a) | 4.05% | ~3B | sig+body → sig-only eval | 962 |
| SFT 8B (this run) | 9.71% | ~8B | sig+body → sig-only eval | 4587 |

See `docs/sft_8b_compute_log.jsonl` for the quality-vs-GPU-minutes curve.
