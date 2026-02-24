# Experiment 2.1: Hard Negative Mining — Results and Analysis

**Date**: 2026-02-24
**Baseline**: Exp 1.4 (InfoNCE, random in-batch negatives)
**Change**: Append LLM-judged hard negative body embeddings to InfoNCE denominator each step

---

## Results vs Exp 1.4

| Metric | Exp 1.4 (random negs) | Exp 2.1 (+ hard negs) | Δ |
|--------|-----------------------|----------------------|---|
| **Val Rank@1** | 1.06% | 1.04% | −0.02pp |
| **Val Rank@5** | 2.45% | 2.38% | −0.07pp |
| **Val Rank@10** | 4.03% | 3.64% | **−0.39pp** |
| **Test Rank@1** | 1.38% | 1.31% | −0.07pp |
| **Test Rank@5** | 2.90% | 3.03% | +0.13pp |
| **Test Rank@10** | 4.14% | 4.31% | **+0.17pp** |
| Val cosine sim | 0.229 | 0.223 | −0.006 |
| Test cosine sim | 0.225 | 0.229 | +0.004 |
| Best val Rank@10 epoch | 24 | 36 | +12 ep |
| Early stop epoch | 39 | 51 | +12 ep |
| Final τ | 0.0499 | 0.0466 | — |

**Summary**: Hard negatives did not improve overall retrieval. Val Rank@10 dropped from 4.03% → 3.64%. Test Rank@10 improved marginally (4.14% → 4.31%), but this is within noise and was not the early-stopping metric. The primary signal is a regression on val.

---

## Training Curve Comparison

| Epoch | Exp 1.4 val_r10 | Exp 2.1 val_r10 |
|-------|----------------|----------------|
| 5 | 1.50% | 1.26% |
| 10 | 2.58% | 2.28% |
| 15 | 3.12% | 3.03% |
| 20 | 3.16% | 3.01% |
| 24/25 | **4.03%** (best) | 3.25% |
| 36 | 3.86% | **3.64%** (best) |
| 39 | 3.81% (stopped) | 3.36% |
| 51 | — | 3.45% (stopped) |

Exp 2.1 is slower at every checkpoint: at epoch 10 it trails by 0.30pp, at epoch 24 by 0.78pp. It catches up slightly later (best at epoch 36 vs 24) but never surpasses Exp 1.4 on val.

---

## Why Hard Negatives Underperformed

### 1. Coverage asymmetry (most important factor)

The hard negative pairs were extracted from the original phase0_2 KNN analysis, which only covered the 28,400-function corpus. The training set now has 42,047 functions — the 22,435 functions added by `extract_from_heads.py` have **no hard negatives**. Only 17,540 of 42,047 training anchors (~42%) received hard negatives.

This creates an uneven loss landscape: some anchors face both random in-batch negatives AND their known confusable pairs; others face only random negatives. The gradient signal per batch is inconsistent depending on which functions happen to be sampled.

### 2. Hard negatives are harder than the model is ready for

The phase0_2 hard negatives have cosine similarity ≥ 0.80 in **teacher embedding space**. The teacher puts these pairs close together. Asking the student to distinguish them is asking it to be more discriminative than the signal it is trained to predict — the teacher itself cannot cleanly separate them.

Evidence: training loss starts higher (6.24 vs 6.05 in 1.4) and the val_r10 ramp is slower at every early epoch. The model is spending gradient budget on the hardest cases before it has mastered the easy structure.

### 3. score_c ≤ 1 negatives include ambiguous pairs

score_c=1 means "slightly related" in the LLM judge's assessment. 38,979 of the 85,125 pairs fall in this category. For some of these, the body implementations share enough structure that treating them as strict negatives gives a contradictory signal: the teacher says the embeddings are close (cosim ≥ 0.80), and the student is being told the body is a *negative* while the body is also close to the anchor's body in the teacher's space.

### 4. Union approach dilutes the hard negative signal

Per step, we collect the union of all hard negative IDs for the entire 64-item batch. The typical batch has ~17 anchors with hard negatives (42% × 64 ≈ 27 anchors), each contributing 4–5 hard negatives, many overlapping. The effective number of unique hard negative rows appended is much smaller than expected, reducing the benefit.

---

## What This Tells Us About the Retrieval Bottleneck

The null result is informative. The plateau in Exp 1.4 at ~4% Rank@10 is **not** primarily caused by the model failing to distinguish specific known-confusable pairs — if it were, adding those pairs as explicit negatives would have helped. Instead the plateau reflects:

- **Structural ambiguity in the teacher's own embedding space**: the functions that confuse the student also confuse the teacher. The student cannot surpass the teacher's discriminative power by training on the teacher's targets.
- **A representation ceiling at this architecture/data scale**: SigPredictor with 2 transformer layers and 8.4M params may have reached its capacity given 42k training functions and a 2048-dim mean-pooled body target.

The implication: to break through ~4% Rank@10, the training signal itself needs to be richer — not just harder negatives applied to the same signal.

---

## Next Steps

### Higher priority: richer supervision signal

**Exp 2.2 — Token-level body prediction (I-JEPA style)**

Rather than predicting one mean-pooled body vector, predict the hidden state at each individual body token. This gives 50–200× more gradient signal per function and forces the model to recover specific token-level content (variable names, control flow) from the signature alone. InfoNCE is kept on the pooled output as the retrieval objective; per-token SmoothL1 is added as an auxiliary reconstruction loss.

This directly addresses the representation ceiling: a model trained to recover token-level body structure will produce body embeddings that are inherently more discriminative than a model trained only to match a mean-pooled direction.

**Expected gain over Exp 1.4**: Rank@1 5–10% (vs 1.38% now), Rank@10 10–20%.

### Lower priority: smarter hard negative scheduling

If revisiting hard negatives, two changes would address the identified failure modes:

1. **Expand hard negative coverage** to the full 50k corpus: re-run the phase0_2 KNN analysis on all 50k functions (not just the original 28,400), then re-extract hard negatives. This eliminates the 58% coverage gap.

2. **Curriculum scheduling**: train for 10–15 epochs with random negatives only (as in Exp 1.4), then introduce hard negatives once the model has learned coarse structure. This avoids spending gradient budget on fine-grained distinctions before coarse structure is learned.

3. **score_c=0 only**: use only the 46,146 pairs where the LLM judge scored the functions as definitely unrelated. Excluding score_c=1 removes the ambiguous signal from marginally-related pairs.

---

## Hard Negative Index Summary

| Statistic | Value |
|-----------|-------|
| Source | `docs/phase0_2_knn_results_full.jsonl` |
| Model | Qwen2.5-Coder-3B |
| Filter | cosim ≥ 0.80, score_c ≤ 1, both IDs in training corpus |
| Total pairs | 85,125 |
| Unique anchors | 17,540 (42% of train split) |
| score_c=0 pairs | 46,146 |
| score_c=1 pairs | 38,979 |
| Avg hard negs per anchor | 4.8 |
| File | `hard_negatives.json` (1.0 MB) |
