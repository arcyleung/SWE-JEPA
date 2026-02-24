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

---

## Plain-Language Summary of Experiments 0.1 → 1.4

### What we're trying to prove

The core bet of SWE-JEPA is that a large language model's *internal reasoning* — the intermediate representations it builds up while processing code — is a richer training signal than its final output. If we freeze a model and eavesdrop on what it's thinking halfway through, can those "thoughts" teach a separate, smaller network to understand code well enough to do useful things?

The inspiration is a paper called SALT (arxiv:2509.24317), which demonstrated exactly this for video: a cheap, frozen teacher model provides surprisingly good latent targets for training a student, and the student can end up outperforming its teacher. We test whether the same principle holds for code.

### The setup

**The teacher — Qwen2.5-Coder-3B**: A 3-billion-parameter LLM trained on code. We freeze it entirely — its weights are never updated during our experiments. When it processes a piece of code, it builds up an internal representation for each token across 36 successive transformer layers.

We don't freeze "the first 18 layers" — we freeze *all 36*. Layer 18 is where we *intercept* the signal: after the 18th layer's computation, each token in the input is represented as a 2,048-dimensional vector encoding what the model "understands" about that token in context. This is the JEPA target — what we ask our student to predict.

**Why layer 18 specifically?** Exp 0.1 tested this. The final layer (layer 35) is too specialised for next-token prediction — its representations collapse, meaning very different functions get nearly identical embeddings. Mid-layer representations (layer 18 ≈ halfway) retain more structural information about code form and semantics without being contaminated by the model's specific output task.

**The student — SigPredictor (8.4M parameters)**: This is *not* a single linear layer. It's a small two-layer transformer encoder with linear projections on each end. The transformer architecture is important: attention lets the student selectively weight different parts of the function signature — argument names, type annotations, docstring — rather than averaging them all together. The full stack is:

```
Linear projection (2048 → 512)
2× transformer attention layers
Mean-pool over tokens
Linear projection (512 → 2048)
```

**The task**: Given only a function's *signature* (its name, arguments, and docstring — no body), the student must predict where the function *body* would land in the teacher's embedding space. This is the JEPA objective: predict representations in latent space, not raw tokens.

If the student learns this well, it means it has inferred what the body must contain from the signature alone — understanding the semantic contract between a function's interface and its implementation.

### How we measure success: the retrieval test

We evaluate on retrieval: given a query signature, can the student rank the correct body embedding above 49,630 other function bodies in the corpus?

A random guesser achieves Rank@1 ≈ 0.002% (you would need ~50,000 guesses to find the right one by chance). The question is how much better a trained student can do.

### What the four phases of experiments found

**Exp 0.1** — Found the best teacher configuration: Qwen2.5-Coder-3B at layer 18. Base (non-instruction-tuned) models are better teachers than instruction-tuned ones — the latter's representations are skewed by RLHF fine-tuning.

**Exp 1.1 (MLP baseline)** — Confirmed the signal exists: a simple MLP trained to predict mean-pooled body embeddings using standard L2 loss achieved high cosine similarity (0.95). But Rank@1 = 0%. The model predicted the *average* code embedding, not the specific instance — it learned the "shape" of code in general but couldn't distinguish individual functions.

**Exp 1.2 (Transformer student, 9 repos)** — Replaced the MLP with a two-layer transformer student. The transformer can attend over signature tokens and learned a genuine JEPA signal (val cosine 0.25 on training repos). But with only 9 repos total, the test set was a single Chinese financial library — completely out-of-distribution relative to everything seen during training. Retrieval completely failed on test. Key lesson: need far more repos.

**Exp 1.3 (150 repos, 50k functions)** — Expanded from 9 repos to 150, adding 22,435 new functions by walking each repository's file system. Val cosine improved 2.4× (0.25 → 0.59) and test no longer collapses. But Rank@10 remained at 0.24% — still barely above random. The problem: the L2 loss doesn't penalise the student for making a prediction that is close to *hundreds* of other functions simultaneously. The student learned the right *direction* in embedding space but predictions clustered too tightly to rank correctly.

**Exp 1.4 (InfoNCE, this experiment)** — Changed only the loss function. Result: Rank@1 = **1.38%** on test, Rank@10 = **4.14%** — a **46× improvement** in Rank@1 over Exp 1.3.

### Why InfoNCE, and is it really different from cross-entropy?

InfoNCE is cross-entropy — this is a common source of confusion. The difference is what the "classes" are:

**Standard cross-entropy** assigns each example to one of a fixed list of classes (e.g. cat / dog / fish). For function retrieval, there are 50,000 functions and the set changes when new functions are added — a fixed-class approach can't work here and can't generalise to new functions it was never trained on.

**InfoNCE** reformulates it so the "classes" at each training step are the other functions currently in the batch. For each function, the student must identify which of the 512 body embeddings it sees this step is the correct match for its signature. The loss explicitly penalises predictions that are close to the *wrong* bodies, not just predictions that are far from the *right* one. With 8 GPUs each holding a batch of 64, all 512 body embeddings from across all GPUs are pooled together as negative examples — more negatives means a harder, more informative learning problem.

This is the contrastive objective: push the correct match up in the ranking, push all incorrect matches down. At inference, the student's predictions live in a shared embedding space where cosine similarity directly equals the retrieval score — no fixed class boundaries needed.

### Where we are

The primary target — Rank@1 > 1% — was hit (1.38% test). The 5% Rank@10 threshold was narrowly missed at 4.1%. The model plateaued at ~3.5–4% from epoch 16 onwards because it ran out of easy negatives: once it correctly separates the target from random other functions, the only remaining challenge is distinguishing syntactically similar functions that differ in subtle semantics — and those are almost never drawn by chance from 50,000 candidates.

The fix is hard negative mining (Exp 2.1): instead of random batch negatives, actively find the top-K most similar functions per query (using the FAISS index already built) and force the student to distinguish those. This directly targets the failure mode and is the logical next step.
