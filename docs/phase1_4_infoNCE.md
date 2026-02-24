# Experiment 1.4: InfoNCE Contrastive Loss

**Date**: 2026-02-24
**Teacher**: Qwen2.5-Coder-3B, layer 18 (frozen)
**Task**: Predict function body embedding from signature token hidden states — with a
discriminative contrastive objective rather than a reconstruction objective.

---

## Motivation

Exp 1.3 (SmoothL1 on mean-pooled body embedding, 150 repos) reached val cosine = 0.59 but
Rank@1 = 0.03% on a 50k corpus.  The failure mode is **not** a weak signal — the predicted
direction is largely correct — it is **insufficient discrimination**.  The SmoothL1 loss treats
two predictions as equally good as long as both land near the target; it does not penalise
a prediction that is *also* near 200 other functions' body embeddings.

InfoNCE (Noise Contrastive Estimation) fixes this directly: for each anchor prediction `p_i`
it requires `p_i` to be closer to `b_i` than to every other body embedding in the batch.
This is the exact geometry needed for retrieval.

---

## Why InfoNCE and not cross-entropy?

This is a subtle but important question. **InfoNCE is cross-entropy** — specifically, it is
softmax cross-entropy applied to in-batch cosine similarity logits:

```
L_i = -log  exp(sim(p_i, b_i) / τ)
             ─────────────────────────────
             Σ_j exp(sim(p_i, b_j) / τ)
```

This is identical to `F.cross_entropy(logits, arange(N))` where
`logits[i,j] = sim(p_i, b_j) / τ`.

### What "cross-entropy" would mean without this formulation

Naively applying cross-entropy to function retrieval requires choosing *what the classes are*:

| Formulation | Classes | Problem |
|-------------|---------|---------|
| Multi-class CE over corpus | 50k fixed classes, linear head [D→50k] | 50k-dim head, can't generalise to new functions, memorises identity not geometry |
| Binary CE per pair | (anchor, candidate) → {0, 1} | Requires explicit negative sampling ratio, poor gradient utilisation (only one negative per step) |
| **InfoNCE** | N in-batch items as classes (dynamic) | **This is the right formulation** |

The critical distinction is what the "weight matrix" is:
- Standard CE: fixed learnable W ∈ ℝ^{D×K}. The model learns class boundaries in input space.
- InfoNCE: the weight matrix at each step is the batch's own body embeddings B ∈ ℝ^{N×D}.
  No learned class parameters. The model learns a *metric space* where cosine similarity is
  directly the retrieval score.

At inference time, retrieval over any unseen corpus is just `argmax_j sim(p_i, b_j)` — which
works because the embedding space is meaningful, not because we trained on those specific
classes. This is why standard CE with a fixed head cannot work here: it would need to be
retrained for every new function added to the corpus.

### Temperature τ — why not just cosine CE?

τ controls the "hardness" of the distribution:
- **τ → 0**: CE becomes a max-margin loss (only the hardest negative matters). High gradient
  variance, risk of instability early in training.
- **τ → ∞**: logits all approach 0, loss approaches log(N), gradients vanish. Too soft.
- **τ ≈ 0.07** (CLIP default): well-calibrated — all N negatives contribute gradient, but
  the loss is dominated by the hardest ones that are close to the anchor.

We use a **learnable** log_τ (initialised to log(0.07) ≈ −2.66), which lets the model adapt
the sharpness to its own training stage. A temperature that is too cold at initialisation would
collapse gradients before the encoder has learned anything useful.

---

## Architecture

Identical to Exp 1.3. No structural change needed — only the loss changes.

```
Frozen:   Qwen2.5-Coder-3B (all 19 transformer layers, bf16, no grad)
          ↓ forward hook → token hidden states [B, T, 2048]

Trainable SigPredictor (8,404,480 params) — same as Exp 1.3:
  Linear(2048 → 512)
  2× TransformerEncoderLayer(d_model=512, nhead=8, ffn=2048, dropout=0.1)
  mean-pool over non-padding tokens
  Linear(512 → 2048)
  ↓ L2-normalise          ← new: unit-sphere projection before InfoNCE

InfoNCELoss (1 param — log_temperature, initialised to log(0.07))
```

Normalising the output to the unit sphere before computing similarities is standard for
contrastive learning (CLIP, SimCLR).  It also means the loss is directly in cosine-similarity
space, consistent with how retrieval is evaluated.

---

## Dataset

Same as Exp 1.3 — no new data needed.

| Split | Functions | Repos | Notes |
|-------|-----------|-------|-------|
| Train | 42,047 | ~120 | Same repo-stratified split (seed=42) |
| Val | 4,615 | ~15 | Used for Rank@k early stopping |
| Test | 2,969 | ~15 | Held out until final evaluation |

Corpus (retrieval): **49,631** functions.

---

## Loss Function

```python
class InfoNCELoss(nn.Module):
    """
    In-batch InfoNCE with cross-GPU negative gathering and learnable temperature.

    With world_size=8, batch_size=64:  512 negatives per anchor per step.
    """
    def __init__(self, init_temp: float = 0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))

    def forward(
        self,
        pred: torch.Tensor,      # [B, D]  L2-normalised predictions
        targets: torch.Tensor,   # [N, D]  L2-normalised body embs (N = world_size * B)
        rank_offset: int = 0,    # starting index of this rank's rows in targets
    ) -> torch.Tensor:
        τ = self.log_temp.exp().clamp(min=1e-4)
        logits = (pred @ targets.T) / τ          # [B, N]
        labels = torch.arange(
            rank_offset, rank_offset + len(pred), device=pred.device)
        return F.cross_entropy(logits, labels)
```

Cross-GPU body embedding gathering in the training loop:

```python
# After frozen Qwen forward and SigPredictor forward:
pred_n      = F.normalize(pred, dim=-1)        # [B, D]
body_n_gpu  = F.normalize(body_emb, dim=-1)    # [B, D]

# All-gather body embeddings from all ranks (no grad needed through gathered)
gathered_body = [torch.zeros_like(body_n_gpu) for _ in range(world_size)]
dist.all_gather(gathered_body, body_n_gpu)
all_body = torch.cat(gathered_body, dim=0)      # [world_size*B, D]

loss = criterion(pred_n, all_body, rank_offset=rank * batch_size)
```

This gives **512 negatives per anchor** per step (8 GPUs × 64 batch).  All gathered body
embeddings are treated as negatives; the diagonal at position `rank_offset + i` is the
positive for anchor `i`.

---

## Training

| Hyperparameter | Value | Change from Exp 1.3 |
|----------------|-------|---------------------|
| Loss | InfoNCE (learnable τ, init=0.07) | ✦ SmoothL1 → InfoNCE |
| Epochs | 100 (patience=15 on val Rank@10) | ✦ early-stop metric changed |
| Batch/GPU | 64 × 8 GPUs = **512 effective** | Same |
| Learning rate | 1e-4 (cosine decay, warmup=5 ep) | Same |
| d_model | 512 | Same |
| Transformer layers | 2 | Same |
| Val Rank@k frequency | Every epoch | ✦ new — fixes R² blindspot from Exp 1.3 |
| Precision | BF16 (Qwen), FP32 (predictor) | Same |
| GPUs | 8× H100 80GB HBM3 | Same |

Early stopping on **val Rank@10** (not val loss, not val cosine sim) — directly optimises the
downstream retrieval metric.  Val Rank@10 on 4,615 functions with a 49,631-corpus FAISS search
takes < 1 s per epoch.

---

## Implementation Steps

1. **Copy** `train_student_1_3.py` → `train_student_1_4.py`
2. **Add** `InfoNCELoss` class with learnable `log_temp`
3. **Replace** `nn.SmoothL1Loss()` with `InfoNCELoss(init_temp=0.07)`
4. **Add L2 normalisation** of predictor output before loss (and of body_emb target)
5. **Add `dist.all_gather`** to collect body embeddings across GPUs in training loop
6. **Add per-epoch FAISS Rank@k** in val evaluation (full val set, full corpus)
7. **Change early-stopping criterion** from val loss to val Rank@10
8. **Update** results file → `docs/phase1_4_infoNCE.md`, checkpoint → `student_1_4_ckpt.pt`

---

## Expected Results and Success Criteria

The contrastive objective directly optimises the ranking, so we expect large gains in Rank@k
relative to the 2.4× gain in cosine similarity achieved by simply expanding the corpus.

| Metric | Exp 1.3 | Target (Exp 1.4) | Notes |
|--------|---------|-----------------|-------|
| Val cosine sim | 0.585 | 0.60–0.70 | May decrease — InfoNCE optimises ranking not reconstruction |
| Val Rank@1 | 0.00% | **> 1%** | Primary target |
| Val Rank@10 | 0.13% | **> 5%** | Early-stopping criterion |
| Test Rank@1 | 0.03% | **> 1%** | Final evaluation |
| Test Rank@10 | 0.24% | **> 5%** | Minimum threshold for Exp 2 |

> Note: cosine similarity may not improve or might even decrease slightly under InfoNCE — the
> loss does not require predictions to be close to targets in L2 distance, only to rank above
> all negatives.  The real signal is Rank@k.

---

## Path Forward After Exp 1.4

If Rank@1 > 1% is achieved, the contrastive foundation is validated and Exp 2.0 can begin:

- **Exp 2.0**: Token-level body prediction (I-JEPA style) — predict per-token body hidden
  states rather than the mean-pooled body embedding, with InfoNCE on the pooled output as a
  secondary loss. Expected to improve Rank@10 further by providing richer supervision.
- **Exp 2.1**: Hard negative mining — use the FAISS index to surface the K most similar
  body embeddings for each anchor and include them as explicit negatives in the InfoNCE
  denominator. This is especially important for code, where syntactically similar functions
  with different semantics are the hardest retrieval cases.
