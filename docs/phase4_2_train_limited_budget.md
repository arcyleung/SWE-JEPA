# Experiment 4.2: JEPA Efficiency vs. SFT Baseline (Fixed Compute Budget)

**Date**: 2026-02-27
**Goal**: Determine whether SWE-JEPA achieves better code retrieval and defect prediction
per GPU-minute than a straight supervised fine-tune of a 3B model.

## Motivation

Exp 3.0 validated the retrieval objective (val Rank@10 = 47.52%). Exp 4.1 showed the
representations transfer zero-shot to defect prediction. The remaining open question from the
research proposal (Point 3) is: *is the JEPA architecture actually compute-efficient?*

The claim is that freezing the 8B teacher and only training a small 63M student amortises the
cost of the large model over many cheap student gradient steps — and that the resulting
representations are higher quality per GPU-minute than end-to-end fine-tuning of a 3B model.

## Comparison Setup

### JEPA (Experiment 3.0 — already trained)

| Property | Value |
|----------|-------|
| Trainable model | SigPredictorV2 — **63M params** |
| Frozen component | Qwen3-8B-base teacher (8.2B params) |
| Training input | sig_text only (up to 256 tokens) |
| Target | 8B body embeddings (mean-pool body positions, full-function context) |
| Loss | InfoNCE |
| Eval | sig-only → student retrieval head → Rank@10 against 8B body corpus |
| Train-test mismatch | None (sig-only both) |

### SFT Baseline (Experiment 4.2 — Scenario B)

| Property | Value |
|----------|-------|
| Trainable model | Qwen2.5-Coder-3B full fine-tune — **3.1B params** + Linear(2048→4096) head |
| Frozen component | Qwen3-8B-base teacher (body emb precompute only, then freed) |
| Training input | sig_text + '\n' + body_text (full function, up to 512 tokens) |
| Target | Same 8B body embeddings as JEPA |
| Loss | InfoNCE (identical) |
| Eval | sig-only → 3B layer-18 mean-pool → proj_head → Rank@10 |
| Train-test mismatch | **Yes** — sees body during training, queries sig-only at eval |

The eval protocol is identical: retrieve from the same corpus using sig-only queries.

## Why Scenario B (full context SFT, sig-only eval)?

This is the most favourable SFT framing:
- SFT sees the full function during training → maximises information available per step
- Must retrieve from sig-only at eval → deliberate train-test mismatch
- If SFT wins despite this mismatch, the efficiency claim fails (a legitimate finding)
- If JEPA wins, it means the JEPA architecture provides a structural advantage

Any alternative SFT framing (sig-only SFT, or contrastive 3B without body) is *worse* for SFT,
so Scenario B is the strongest possible SFT baseline.

The 30B student alternative was considered and rejected: it would lose the 50× parameter-count
advantage (63M vs 3B) that is the core of the efficiency argument, and a 30B SFT baseline would
be 4-5× harder to beat within the same compute budget.

## Compute Accounting (per gradient step, 8 GPUs)

| Method | Model compute | Tokens per step | Approx token-param ops | Notes |
|--------|--------------|-----------------|------------------------|-------|
| JEPA | 8B no-grad (sig) + 63M with-grad (sig) | ~100 | ≈ 8.06B × 100 ≈ 806B | Teacher no-grad ~1/3 cost of grad |
| SFT | 3B with-grad (sig+body) | ~400 | ≈ 3B × 400 × 3 ≈ 3,600B | ×3 for backprop |

**SFT is ~4.5× more expensive per step** in token-parameter operations.

Additionally:
- JEPA student (63M) holds ~0.4 GB VRAM vs SFT ~42 GB (model + Adam states)
- JEPA effective batch: 64/GPU (shorter sequences); SFT: 16/GPU (512-token full functions)
- JEPA steps/hour ≫ SFT steps/hour at equal GPU-minute budget

**Primary metric**: quality-vs-GPU-minutes curve. Both methods log
`{step, epoch, gpu_minutes, val_r10, train_loss}` to JSONL at every epoch. The JEPA training
log is backfilled from its training run; SFT is logged in real time.

## Expected Outcomes

| Outcome | Interpretation |
|---------|---------------|
| JEPA Rank@10 > SFT at equal GPU-minutes | JEPA architecture is more compute-efficient; central claim validated |
| SFT ≈ JEPA at equal GPU-minutes | Teacher overhead cancels efficiency gain; neutral result |
| SFT >> JEPA at equal GPU-minutes | Train-test mismatch matters less than expected; JEPA efficiency claim fails |

The prior (from the proposal's reasoning): JEPA wins at short-to-medium budgets because
(a) the 8B teacher provides high-quality body embedding targets (better signal than 3B
self-supervised), (b) the 63M student trains ~4.5× more steps per GPU-hour, and (c) JEPA has
zero train-test mismatch at eval. At very large budgets, SFT may catch up as the 3B model
saturates the available training signal.

## Defect Prediction Transfer (secondary metric)

After SFT training, re-run the Exp 4.1 probe on SFT sig embeddings (sig-only encode, same
protocol):

```
random (0.50) < LOC (0.52) < TF-IDF (0.55) < Teacher (0.57) < JEPA (0.60) < SFT (??)
```

If SFT AUROC > JEPA despite the train-test mismatch on retrieval, it suggests the 3B model
inherently encodes more defect-relevant structure from signatures alone than the 63M student —
a useful negative result that would motivate scaling the student.

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| SFT converges faster (train-test mismatch small) | Medium | Scenario B is already the hardest framing; failure is a valid finding |
| SFT defect AUROC > JEPA | High | Expected; JEPA student was never trained for defect prediction |
| OOM with 3B fine-tune + Adam | Low | 3B bf16 + Adam ≈ 42 GB/card |
| SFT overfits 50k corpus | Medium | InfoNCE + gradient clipping should regularise |

## Implementation

| File | Purpose |
|------|---------|
| `train_sft_baseline.py` | Fine-tune 3B with InfoNCE, log compute curve, save checkpoint |
| `docs/sft_compute_log.jsonl` | Per-epoch `{step, epoch, gpu_minutes, val_r10, train_loss}` |
| `docs/phase4_2_sft_results.md` | Final results auto-generated at end of training |
| `probe_defect_prediction.py` | Reuse with `--sft-ckpt sft_baseline_ckpt.pt` for defect AUROC |

## Running

```bash
# Single node, 8 GPUs
source .venv/bin/activate && torchrun --nproc-per-node=8 train_sft_baseline.py

# If OOM
torchrun --nproc-per-node=8 train_sft_baseline.py --batch-size 8 --gradient-checkpointing
```
