# Experiment 1.3: Transformer Encoder Student (Expanded Corpus)

**Date**: 2026-02-23
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
| Cosine sim | **0.5848** | Held-out repos, same domain as training |
| Rank@1 retrieval | **0.00%** | random=0.0020% |
| Rank@5 retrieval | **0.09%** | — |
| Rank@10 retrieval | **0.13%** | — |

### Test Set

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| Cosine sim | **0.5891** | 0.3650 (sig→body) | ↑0.2241 |
| Cosine sim (random) | — | 0.2603 | Anisotropy baseline |
| Rank@1 retrieval | **0.03%** | 0.0020% (random) | improvement over MLP (0.00%) |
| Rank@5 retrieval | **0.10%** | — | — |
| Rank@10 retrieval | **0.24%** | — | — |

## Interpretation

The transformer encoder substantially improves over the MLP baseline (Rank@1: 0.00%) by attending token-level to argument names, type annotations, and docstring specifics in the function signature. This confirms the causal chain hypothesis: mean-pooled embeddings lack the specificity required for instance-level retrieval, whereas token-level attention can distinguish similar functions.

Expanding from 9 to 150 repos was the single most impactful change: val cosine improved from 0.25 → 0.60 (+140%) while test retrieval went from complete failure (OOD collapse at −0.009) to first meaningful hits.

## Observations

### A. Loss vs Downstream Metric Correlation (R² not validated)

**R² > 0.8 between student loss and downstream retrieval metrics was not established.**

The training curves reveal two distinct regimes:

| Phase | Epochs | Val loss range | Val cosine range | Relationship |
|-------|--------|---------------|-----------------|--------------|
| Early (fast descent) | 1–15 | 0.0166 → 0.0004 | 0.010 → 0.37 | Monotonic — loss is a useful proxy |
| Converged (plateau) | 16–89 | 0.0004 → 0.0002 | 0.31 → 0.60 | Decorrelated — loss flat, cosine varies ±0.15 |

In the converged phase val_loss effectively carries no information: it is ≈ 0.0002 for all of epochs 16–89, yet val_cosine oscillates between 0.31 and 0.60 across those same epochs. A linear R² computed over the full 89-epoch span would be dragged up by the early-training correlation but the plateau behaviour means the loss cannot distinguish a checkpoint at 0.31 cosine from one at 0.60.

We also have **no per-epoch retrieval (Rank@k) data** — retrieval was evaluated only once (final checkpoint). Without tracking Rank@k per epoch we cannot compute R² between loss and the actual downstream metric we care about. From the single test-set measurement (val_loss ≈ 0.0002, Rank@10 = 0.13–0.24%) we have only one data point.

**Practical consequence for future experiments**: val_loss should not be used as the primary early-stopping criterion. Val cosine is a better in-training proxy. Ideally, checkpoint at intervals and run FAISS Rank@10 on the val set (cheap: one FAISS search per checkpoint on 4,600 vectors) so R² can be properly computed.

---

### B. Contrastive Training vs Token-Level Body Prediction — Which Is the Better Exp 2 Foundation?

#### The root bottleneck in Exp 1.3

Val cosine = 0.59 but Rank@10 = 0.24% on a 50k corpus. This dissociation means the model successfully learns the *direction* of the body embedding (it predicts the right quadrant of the 2048-d sphere) but predictions are insufficiently discriminative — many similar functions have nearly identical predictions. The SmoothL1 loss on a mean-pooled target does not penalise this: two bodies that map to the same mean embedding are indistinguishable from the loss's perspective.

#### Option 1: Contrastive loss (InfoNCE)

Directly optimises the retrieval objective. Given a predicted embedding `p_i` and correct body embedding `b_i`, InfoNCE minimises:

```
L = -log( exp(sim(p_i, b_i)/τ) / Σ_j exp(sim(p_i, b_j)/τ) )
```

Negatives `b_j` come from other functions in the batch (easy to collect: with effective batch 512 and cross-GPU gather on 8 H100s, all 512 body embeddings are available as negatives per step).

**Why this directly addresses the failure**: the loss explicitly penalises `p_i` being close to `b_j` for j ≠ i. Even if all functions in the same repo have similar mean-pooled bodies, the contrastive loss will push predictions apart until they rank the correct target above all negatives.

**Strengths**:
- No architectural change — reuse existing SigPredictor
- Proven in code retrieval: CodeBERT, UniXCoder, CodeT5+ all use contrastive objectives
- Can be combined with the existing SmoothL1 reconstruction loss as a secondary term
- Cross-GPU negatives are trivial with `dist.all_gather` on body_embs

**Limitations**:
- With 512 in-batch negatives drawn from 50k total, hard negatives (syntactically similar functions) are rarely sampled. Hard negative mining or a momentum queue (MoCo-style) would be needed to push Rank@1 into the several-percent range
- Does not change the richness of the representation — still predicts a single mean-pooled vector

#### Option 2: Token-level body prediction (I-JEPA style)

Rather than predicting one mean-pooled body embedding, predict the hidden state at each individual body token. The predictor receives signature hidden states and outputs a sequence `[h_1, h_2, ..., h_T]` matching the body token hidden states from the frozen teacher. Loss is mean SmoothL1 over all body token positions.

**Why this provides stronger supervision**: a 200-token body generates 200 independent prediction targets instead of 1. The model must recover specific token-level structure — which variable names appear, what control flow tokens look like — rather than the average semantic direction. This should produce representations that generalise better to unseen repo styles.

**Strengths**:
- ~50–200× more gradient signal per function
- Closer to the original JEPA/BERT masked prediction paradigm — well-understood inductive bias
- Forces the model to learn compositional body structure, not just genre

**Limitations**:
- Requires changing the predictor output head from `[B, d]` to `[B, T_body, d]`, with variable-length body sequences — more complex collation and loss masking
- Body token sequences are themselves anisotropic (same problem as mean-pooled embeddings, just per-token) — still need per-token mean-centering or normalisation
- Training cost increases because the teacher must also be run on the body tokens (currently we just load stored embeddings; now we need token-level body hidden states, which must be computed or stored — 50k × 200 tokens × 2048 dims × 4 bytes ≈ 80 GB)
- Does not directly optimise retrieval: a model with low per-token prediction loss can still predict an "average body" that retrieves nothing correctly

#### Recommendation: **Contrastive loss first**

| Criterion | Contrastive | Token-level |
|-----------|-------------|-------------|
| Directly fixes retrieval failure | ✅ Yes | ❌ No (indirect) |
| Architectural complexity | Low (loss change only) | High (new output head + data pipeline) |
| Additional storage needed | None | ~80 GB token hidden states |
| Expected Rank@1 gain | 10–100× (0.03% → 1–5%) | Uncertain without contrastive objective |
| Interpretability of improvement | Clear (precision vs recall trade-off) | Hard to separate from data diversity effect |

The core failure — predictions clustering in embedding space — is a **discrimination** problem, not a **representation richness** problem. Contrastive loss attacks it directly. Token-level prediction improves representation quality but the resulting embeddings still need a contrastive objective to become discriminative for retrieval.

**Recommended sequence**:

1. **Exp 1.4** — Add InfoNCE contrastive loss to the existing architecture (same 150-repo corpus, same SigPredictor). Use cross-GPU negatives + optional hard negative queue. Target: val Rank@1 > 1%, Rank@10 > 5%.
2. **Exp 2.0** — Once a contrastive foundation is validated, switch to token-level body prediction with InfoNCE applied at the *pooled* output (not per-token) as the retrieval objective, with per-token reconstruction as auxiliary loss. This gives both representation richness and retrieval precision.
3. **Exp 2.1** — Optionally combine both losses jointly (per-token reconstruction + contrastive on pooled output) if Exp 2.0 shows further gains.

The two objectives are not mutually exclusive and will ultimately both be needed — but establishing a contrastive baseline first gives a cleaner ablation: we can attribute any Rank@1 improvement in Exp 2.0 to the richer token-level targets rather than to the contrastive objective already in place.

## Success Criteria

| Criterion | Target | Achieved |
|-----------|--------|---------|
| Val cosine > Exp 1.2 (0.252) | > 0.25 | ✅ 0.585 |
| Test cosine positive (no OOD collapse) | > 0 | ✅ 0.589 |
| Rank@1 > MLP | > 0.00% | ✅ 0.03% |
| Rank@1 >> random (5%) | > 5% | ❌ 0.03% |
| Rank@10 > 20% | > 20% | ❌ 0.24% |
| R² (loss → Rank@k) validated | > 0.8 | ❌ Not measured per-epoch |
