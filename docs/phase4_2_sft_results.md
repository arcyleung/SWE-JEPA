# Experiment 4.2 SFT Baseline Results

**Date**: 2026-02-28
**Model**: Qwen2.5-Coder-3B full fine-tune + Linear(2048→4096) projection head
**Training input**: sig_text + body_text (full function, up to 512 tokens)
**Eval input**: sig_text only (deliberate train-test mismatch — favours JEPA)
**Target**: Qwen3-8B-base layer-18 body embeddings (same as JEPA Exp 3.0)

## Architecture

| Component | Detail |
|-----------|--------|
| SFT model | Qwen2.5-Coder-3B (3,094,327,296 total params incl. proj) |
| Projection head | Linear(2048 → 4096, no bias) |
| Eval layer | Layer 18 mean-pool (sig-only at inference) |
| Loss | InfoNCE (τ learnable, same as JEPA) |
| Teacher (body targets only) | Qwen3-8B-base layer 18 (freed after precompute) |

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
| Epochs trained | 29 (patience=15) |
| Batch/GPU | 8 |
| LR | 5e-05 (cosine, warmup=3) |
| Total GPU-minutes | 962 |

## Results

| Split | Rank@1 | Rank@5 | Rank@10 |
|-------|--------|--------|---------|
| Val   | 0.59% | 2.45% | 4.05% |
| Test  | 0.98% | 2.56% | 4.08% |

## Comparison with JEPA (Exp 3.0)

| Method | Val Rank@10 | Training input | GPU-minutes |
|--------|------------|----------------|-------------|
| JEPA (Exp 3.0) | 47.52% | sig-only (no mismatch) | (see JEPA log) |
| SFT Baseline   | 4.05% | sig+body → sig-only eval | 962 |

See `docs/sft_compute_log.jsonl` for the quality-vs-GPU-minutes curve.
