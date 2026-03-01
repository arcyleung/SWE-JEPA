# Phase 4 Results: SWE-JEPA Evaluation

**Date**: 2026-03-01

This document synthesises all Phase 4 experiments evaluating SWE-JEPA (Exp 3.0) against
baseline approaches: defect prediction probing (Exp 4.1) and SFT ablations (Exp 4.2a/b).

---

## Summary Table

| Method | Trainable params | Val R@10 | GPU-minutes | Training input |
|--------|-----------------|----------|-------------|----------------|
| **JEPA (Exp 3.0)** | **63M** | **47.52%** | **~2,016*** | **sig-only** |
| SFT 8B (Exp 4.2b) | ~8,200M | 9.71% | 4,587 | sig+body → sig-only eval |
| SFT 3B (Exp 4.2a) | ~3,094M | 4.05% | 962 | sig+body → sig-only eval |

*JEPA GPU-minutes estimated from 33 epochs × 32 GPUs × 64 batch on 4-node cluster; not directly logged in this format.

---

## Exp 4.1: Defect Prediction (Zero-shot Probe)

**Task**: Predict `has_bugfix` (binary) from function signature embeddings alone.
**Data**: 6,651 function anchors, 144 repos; 56% positive (has bugfix PR).

| Probe | AUROC | Balanced Acc |
|-------|-------|-------------|
| Majority baseline | 0.500 | 0.500 |
| LOC (line count) | 0.517 | 0.509 |
| TF-IDF (5k features) | 0.551 | 0.551 |
| Teacher emb (frozen 8B) | 0.567 | 0.549 |
| **Student emb (JEPA)** | **0.600** | **0.556** |

JEPA beats the frozen teacher by +0.033 AUROC zero-shot — the student learned something
about implementation quality from signatures that the teacher's frozen representations did
not directly encode. See `docs/phase4_1_defect_prediction.md` for full analysis.

---

## Exp 4.2: SFT Baselines — Quality vs Compute

### Architecture comparison

Both SFT baselines share the same InfoNCE objective and body embedding targets as JEPA
(Qwen3-8B-base layer-18, sig+body context). The critical difference is the input at
**eval time**: JEPA is trained and evaluated on sig-only (no train/test mismatch); SFT
models are trained on sig+body but evaluated on sig-only.

| | JEPA | SFT 3B | SFT 8B |
|-|------|--------|--------|
| Student model | 63M custom Transformer | Qwen2.5-Coder-3B | Qwen3-8B-base |
| Teacher (frozen) | Qwen3-8B-base | Qwen3-8B-base | Qwen3-8B-base |
| Train input | sig-only | sig + body | sig + body |
| Eval input | sig-only | sig-only | sig-only |
| Train/test mismatch | None | Yes | Yes |
| Optimizer | AdamW | AdamW | Adafactor |
| Grad checkpointing | No | Optional | Always |

### Final results

| Model | Val R@1 | Val R@5 | Val R@10 | Test R@10 | Epochs | GPU-min |
|-------|---------|---------|----------|-----------|--------|---------|
| JEPA | 19.11% | 36.77% | 47.52% | 51.77% | 33 | ~2,016* |
| SFT 8B | 1.80% | 6.07% | **9.71%** | 9.60% | 32 | 4,587 |
| SFT 3B | 0.59% | 2.45% | **4.05%** | 4.08% | 29 | 962 |

### Quality-vs-compute curves

The tables below show val R@10 at equal GPU-minute checkpoints, drawn from the
per-epoch compute logs (`sft_compute_log.jsonl`, `sft_8b_compute_log.jsonl`).

**SFT 3B** (33 GPU-min/epoch, 8 GPUs):

| GPU-min | Epoch | Val R@10 |
|---------|-------|----------|
| 32 | 1 | 1.52% |
| 99 | 3 | 2.73% |
| 199 | 6 | 3.03% |
| 330 | 10 | 3.47% |
| 461 | 14 | 4.05% ← best |
| 787 | 24 | 3.19% |
| 948 | 29 | 3.01% ← stopped |

**SFT 8B** (143 GPU-min/epoch, 8 GPUs):

| GPU-min | Epoch | Val R@10 |
|---------|-------|----------|
| 141 | 1 | 2.36% |
| 428 | 3 | 5.27% |
| 717 | 5 | 6.98% |
| 1,429 | 10 | 9.62% |
| 2,427 | 17 | 9.71% ← best |
| 3,710 | 26 | 7.89% |
| 4,587 | 32 | 9.64% ← stopped |

---

## Conclusions

### 1. JEPA has a large architectural advantage, not just a scale advantage

Scaling from 3B to 8B SFT improves val R@10 from 4.05% → 9.71% (+5.7pp, 2.4×). This is
a real gain, confirming that parameter count matters for this task. However, JEPA at 63M
trainable parameters achieves 47.52% — **4.9× better than 8B SFT** and at roughly **half
the GPU-minutes**. The gap is architectural, not parametric.

The key structural advantages of JEPA over SFT:

- **No train/test mismatch**: JEPA trains and evaluates on sig-only. Both SFT models
  memorise sig+body→embedding during training but must generalise from sig-only at eval.
  This mismatch alone likely accounts for a large fraction of the 38pp gap.

- **Frozen, richer teacher representations**: JEPA's student sees 4096-dim layer-18
  targets from a 36-layer frozen 8B. The SFT 3B model must both understand the code
  _and_ project its 2048-dim representations into the 4096-dim teacher space via a linear
  head — a harder task. Even the SFT 8B has the same architecture as the teacher, making
  the projection trivial, yet still falls far short.

- **InfoNCE operates on better signal**: With 32 GPUs × 64 batch = 2,048 negatives per
  step, JEPA's contrastive signal is far stronger than either SFT run (8 GPUs × 4–8
  batch = 32–64 negatives/step).

### 2. 8B SFT scaling: real but insufficient

At matched GPU-minutes (~960), the 8B SFT reaches ~6.18% (epoch 6) vs 3B's final 4.05%.
Even with 4.8× more compute, 8B SFT only reaches 9.71%. The quality-vs-compute curves
show both models plateau quickly (3B by epoch 14, 8B by epoch 17) and then degrade
slightly — consistent with overfitting of the projection head to a fixed set of body
embeddings once the backbone saturates.

### 3. The train/test mismatch is the primary bottleneck for SFT

The SFT objective directly optimises matching the concatenated sig+body representation to
the teacher body embedding. But retrieval at eval uses sig-only. This structural mismatch
means the fine-tuned backbone is optimised for inputs it will never see at inference. JEPA
avoids this by design: the predictor is trained exclusively on sig-only inputs.

A fairer SFT baseline would fine-tune on sig-only inputs (discarding body at train time),
but this loses the body embedding target quality advantage. The fundamental difficulty is
that without seeing the body during training, SFT has no signal to learn from; JEPA
sidesteps this via the latent prediction objective.

### 4. Defect prediction (Exp 4.1) validates JEPA representations beyond retrieval

JEPA representations transfer zero-shot to a held-out downstream task (defect prediction)
not in the InfoNCE training distribution, beating the frozen 8B teacher by +3.3pp AUROC.
Neither SFT baseline was evaluated on defect prediction, but the retrieval gap suggests
JEPA's representations are more semantically structured.

---

## Files

| File | Contents |
|------|----------|
| `docs/phase4_1_defect_prediction.md` | Exp 4.1 full results |
| `docs/phase4_2_sft_results.md` | Exp 4.2a (3B SFT) full results |
| `docs/phase4_2_sft_8b_results.md` | Exp 4.2b (8B SFT) full results |
| `docs/sft_compute_log.jsonl` | 3B SFT per-epoch {gpu_minutes, val_r10} |
| `docs/sft_8b_compute_log.jsonl` | 8B SFT per-epoch {gpu_minutes, val_r10} |
| `docs/phase3_0_qwen3_8b_teacher.md` | JEPA (Exp 3.0) full results |
