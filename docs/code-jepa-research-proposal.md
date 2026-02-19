# SWE-JEPA: Latent-Space Predictive Architecture for Software Engineering

## Executive Summary

Current state-of-the-art approaches to AI-assisted code generation rely on large reasoning models (70B+ parameters) producing extremely long chain-of-thought trajectories, or multi-agent systems with expensive environment interactions. These approaches are computationally wasteful: the model simultaneously reasons about high-level software design (architecture, patterns, contracts) and low-level token generation (syntax, variable names, formatting) in a single autoregressive pass, exhausting context windows and compute budgets.

We propose **SWE-JEPA**, a two-stage training framework inspired by SALT (Static-teacher Asymmetric Latent Training), which decouples abstract code understanding from token-level generation. Drawing from recent advances in video self-supervised learning — where V-JEPA learns video representations by predicting masked regions in latent space rather than pixel space — we apply the same principle to code: train models to predict *latent representations* of masked code regions (function bodies, class methods, error handlers) rather than reconstructing exact tokens.

The core hypothesis is that forcing a student model to predict dense latent representations of masked code from only surrounding context (signatures, docstrings, imports, tests, call sites) will compel the emergence of abstract software engineering reasoning — understanding of design patterns, data flow, API contracts, complexity trade-offs — without the cost of explicit chain-of-thought generation. A key finding from SALT is that **small, sub-optimal teachers produce surprisingly strong students**, suggesting we can bootstrap this system cheaply using existing small code models (Qwen-2.5 3B or Qwen3 8B) as frozen teachers.

If successful, SWE-JEPA would enable:

- **Compressed reasoning**: Software design decisions encoded in dense latent vectors rather than sprawled across thousands of reasoning tokens
- **Decoupled generation**: Cheap decoders translate latent plans into actual code, separating "what to build" from "how to write it"
- **Efficient scaling**: Student models that exceed their teacher's capability, trained at a fraction of the compute cost of end-to-end approaches

---

## Background & Motivation

### The SALT Insight (arxiv:2509.24317)

SALT demonstrates that for video representation learning:

1. A frozen teacher trained with simple pixel reconstruction provides sufficient latent targets
2. A student trained to predict those frozen targets in latent space develops *stronger* semantic understanding than the teacher
3. Teacher quality matters far less than expected — small, undertrained teachers produce near-optimal students
4. The information asymmetry (student sees sparse context, must predict dense latents) is the primary driver of representation quality
5. Training loss directly correlates with downstream performance (R² > 0.95), unlike EMA-based methods

### Mapping to Code

| Vision (SALT)                  | Code (SWE-JEPA)                                      |
| ------------------------------ | ----------------------------------------------------- |
| Video frames                   | Source code files / repositories                       |
| Spatiotemporal patch masking   | AST-structural masking (function bodies, methods, etc) |
| Pixel reconstruction (Stage 1) | Token reconstruction / fill-in-middle (Stage 1)        |
| Latent prediction (Stage 2)    | Latent code structure prediction (Stage 2)             |
| Appearance + motion semantics  | Design patterns, data flow, complexity, contracts      |
| Linear probe evaluation        | Retrieval, linear probe, conditional code generation   |

### Key Differences from Standard LLM Distillation

- **Latent targets, not output distributions**: The student predicts the teacher's internal hidden states at masked positions, not token-level logits. This operates in continuous space with richer geometric structure.
- **Small teacher, large student**: Unlike standard distillation (large teacher → small student), SALT shows the reverse works — a cheap frozen teacher bootstraps a stronger student.
- **No collapse risk**: Because the teacher is frozen, the target space is fixed. No EMA scheduling, no stop-gradient tricks, no representation collapse.
- **Structural masking**: Masking is defined by code semantics (AST boundaries), not arbitrary token spans.

---

## Architecture Overview

### Stage 1: Train the Teacher (Pixel Reconstruction)

A standard code language model is trained (or used off-the-shelf) with a fill-in-middle / masked token reconstruction objective. The teacher learns to reconstruct exact code tokens at masked positions. This produces hidden states that encode local code structure: control flow, API calls, variable usage, syntactic patterns.

For the Level 0 prototype, we skip Stage 1 entirely and use a pretrained Qwen-2.5-Coder as the frozen teacher.

### Stage 2: Train the Student (Latent Prediction)

The teacher's weights are frozen. For each code sample:

1. **Mask** structurally meaningful regions (function bodies, class methods, error handlers) at the AST level
2. **Teacher forward pass**: Run the full (unmasked) code through the frozen teacher. Extract hidden states at masked positions → these are the **latent targets**
3. **Student forward pass**: The student sees only the *context* (everything except the masked regions) plus positional information about where masks are. It predicts latent vectors at masked positions.
4. **Loss**: L1 or L2 distance between student predictions and teacher hidden states at masked positions

The student must bridge the gap between sparse context (function signature, imports, tests) and dense latent targets (what the implementation structurally looks like). This forces abstract software engineering reasoning to emerge.

---

## Experiment Plan

### Phase 0: Teacher Hidden State Extraction & Validation

**Goal**: Verify that pretrained code model hidden states contain useful structural information about code, and determine optimal extraction parameters.

#### Experiment 0.1: Extract hidden states across model scales

| Parameter       | Values to test                                     |
| --------------- | -------------------------------------------------- |
| Teacher model   | Qwen-2.5-Coder-3B, Qwen-2.5-Coder-7B, Qwen-3-Coder-8B, Qwen-3-Coder-8B-Base           |
| Extraction layer| Layers 1, L/4, L/2, 3L/4, L (last)               |
| Pool strategy   | Per-token (none), mean, last-token                 |
| Masking level   | Function bodies (regex), AST subtrees (tree-sitter)|

**Steps**:

1. Prepare a corpus of ~10K Python files with well-defined function boundaries (e.g., from The Stack, or your agent conversation dataset)
2. Run `extract_hidden_states.py` on the corpus with each teacher model
3. For each layer and pooling strategy, compute:
   - Pairwise cosine similarity distribution across all extracted regions
   - Variance of representations (are they differentiated or collapsed?)

**Success criteria**: Representations from different functions should be meaningfully separated (cosine sim distribution not concentrated near 1.0). At least one layer should show clear differentiation.

**Deliverable**: Table of (model_size × layer × pool_strategy) → distribution statistics. Identifies the best extraction configuration for subsequent experiments.

#### Experiment 0.2: Nearest-neighbor retrieval validation

**Goal**: Test whether teacher latent space captures functional similarity.

**Steps**:

1. From the extracted hidden states (mean-pooled per function), build a FAISS index
2. For a set of query functions, retrieve the K nearest neighbors
3. Manually evaluate: are neighbors functionally similar (same algorithmic pattern, similar purpose) despite surface differences (different variable names, different libraries)?
4. Quantitative variant: for functions with known functional equivalences (e.g., sort implementations, cache implementations from your agent data), measure recall@K

**Success criteria**: Nearest neighbors should be semantically related, not just syntactically similar. Functions implementing the same spec (from your agent conversation data) should cluster.

#### Experiment 0.3: Linear probing for structural properties

**Goal**: Test whether hidden states encode software engineering-relevant attributes.

**Steps**:

1. For each function in the corpus, extract static properties:
   - Cyclomatic complexity
   - Number of branches / loops
   - Return type category (None, primitive, collection, custom object)
   - Has side effects (I/O, mutation) vs pure
   - Number of external API calls
   - Lines of code
2. Train a linear layer on frozen teacher hidden states (mean-pooled) to predict each property
3. Report accuracy / R² for each property across model sizes and layers

**Success criteria**: At least some structural properties should be linearly decodable from the hidden states with reasonable accuracy. This validates that the representations carry the kind of information the student would need to learn.

**Deliverable**: Table of (property × model_size × layer) → linear probe accuracy.

---

### Phase 1: Student Architecture & Training Loop

**Goal**: Implement and validate the core SWE-JEPA student training pipeline.

#### Experiment 1.1: Minimal student training

**Steps**:

1. Define the student architecture:
   - Encoder: small transformer (e.g., 125M–350M params) that processes unmasked context
   - Predictor: lightweight MLP or small transformer that takes encoder output + mask position embeddings and predicts latent vectors at masked positions
2. Training data: code files from Phase 0 corpus with precomputed teacher targets
3. Train the student to minimize L1 loss between predicted and teacher hidden states at masked function body positions
4. Track training loss curve

**Key design decisions to resolve**:

- How to represent mask positions to the student (learned position embeddings? copy from teacher's positional encoding?)
- Whether the student sees the function signature or just surrounding file context
- Predictor architecture: per-position MLP vs cross-attention over context

**Success criteria**: Training loss decreases consistently. Student predictions at masked positions are closer to teacher targets than to random hidden states from other positions.

#### Experiment 1.2: Student loss vs downstream correlation

**Goal**: Replicate SALT's key finding that student loss correlates with downstream quality.

**Steps**:

1. Save student checkpoints at regular intervals during training
2. For each checkpoint, evaluate on the nearest-neighbor retrieval task and linear probing tasks from Phase 0 (but using the *student's* representations)
3. Plot student training loss vs downstream metrics

**Success criteria**: Strong correlation (R² > 0.8) between student loss and at least one downstream metric. This would validate that the SWE-JEPA training signal is meaningful, analogous to SALT's finding.

---

### Phase 2: Teacher Robustness & Compute Allocation

**Goal**: Test whether SALT's "small teachers suffice" finding transfers to code.

#### Experiment 2.1: Teacher size ablation

**Steps**:

1. Using the pipeline from Phase 1, train identical students using teacher targets from:
   - Qwen2.5-Coder-3B
   - Qwen2.5-Coder-3B-Instruct
   - Qwen3-8B
   - Qwen3-8B-base
2. Evaluate all students on the same downstream tasks
3. Measure the marginal gain from larger teachers

**Success criteria**: The gap between 3B-teacher and 8B-teacher students should be small relative to the gap between an untrained baseline and any teacher. This would confirm the SALT finding transfers.

#### Experiment 2.2: Compute allocation study

**Steps**:

1. Fix a total compute budget (measured in GPU-hours)
2. Vary the split between teacher training (if fine-tuning) and student training
3. Measure downstream quality as a function of the allocation split

**Expected finding** (based on SALT): The optimal allocation overwhelmingly favors the student. Spending more compute on the teacher yields diminishing returns.

---

### Phase 3: Conditional Code Generation (Stretch Goal)

**Goal**: Test whether student latent representations can condition actual code generation.

#### Experiment 3.1: Latent-conditioned decoder

**Steps**:

1. Train a small decoder (GPT-2 scale) that takes the student's predicted latent at a masked position and generates the corresponding code tokens autoregressively
2. The decoder receives: (a) the student's latent vector, (b) the function signature, (c) surrounding context
3. Evaluate generated code with: pass@k on unit tests, functional correctness, comparison to teacher-only generation

**Success criteria**: The latent-conditioned decoder should produce functionally correct code at a higher rate or lower compute cost than a baseline model of equivalent total parameter count generating code from scratch.

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Hidden states too uniform across functions (no differentiation) | Medium | Try middle layers instead of last; add contrastive fine-tuning to teacher (Level 1) |
| Code latent geometry doesn't support functional similarity | Medium | Fine-tune teacher with contrastive objective on (spec, implementation) pairs from agent data |
| Long-range code dependencies exceed context window | High | Start with self-contained functions; later add retrieval-augmented context |
| Student doesn't learn abstract properties (just memorizes surface patterns) | Medium | Evaluate with held-out repos / languages; test invariance to refactoring |
| Latent-to-token decoding loses critical details | Medium | Hybrid approach: latent provides high-level plan, decoder has access to full context |

---

## Resource Requirements

| Phase | Compute (est.) | Data | Timeline |
|-------|----------------|------|----------|
| Phase 0: Extraction & validation | 1-2 GPU-days (inference only) | 10K Python files | 1-2 weeks |
| Phase 1: Student training | 1-4 GPU-days (training small models) | Same corpus + teacher targets | 2-3 weeks |
| Phase 2: Ablations | 2-8 GPU-days | Same | 2-3 weeks |
| Phase 3: Decoder | 4-16 GPU-days | + unit test data | 3-4 weeks |

## Key References

- **SALT**: Li et al., "Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers" (arxiv:2509.24317)
- **V-JEPA**: Bardes et al., "Revisiting Feature Prediction for Learning Visual Representations from Video" (2024)
- **I-JEPA**: Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (2023)
- **JEPA concept**: LeCun, "A Path Towards Autonomous Machine Intelligence" (2022)
