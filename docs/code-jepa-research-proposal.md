# SWE-JEPA: Latent-Space Predictive Architecture for Software Engineering

## Executive Summary

Current state-of-the-art approaches to AI-assisted code generation rely on large reasoning models (70B+ parameters) producing extremely long chain-of-thought trajectories, or multi-agent systems with expensive environment interactions. These approaches are computationally wasteful: the model simultaneously reasons about high-level software design (architecture, patterns, contracts) and low-level token generation (syntax, variable names, formatting) in a single autoregressive pass, exhausting context windows and compute budgets.

More fundamentally, these approaches optimize for a narrow objective — generating code that satisfies an immediate specification — while ignoring the broader software engineering context in which code exists. Real software quality is multi-dimensional: Kim et al.'s empirical study of refactoring at Microsoft (2014) demonstrates that even measuring the impact of structural improvements requires tracking inter-module dependencies, defect rates, complexity metrics, churn patterns, test adequacy, and organizational ownership simultaneously. Conway's Law (1968) establishes that system structure inevitably mirrors the communication structure of the organization that produced it. Together, these findings suggest that "good code" is not a property of individual functions in isolation but an emergent property of how code relates to its surrounding system, its history of evolution, and the organizational structure behind it.

We propose **SWE-JEPA**, a two-stage training framework inspired by SALT (Static-teacher Asymmetric Latent Training), which decouples abstract code understanding from token-level generation. Drawing from recent advances in video self-supervised learning — where V-JEPA learns video representations by predicting masked regions in latent space rather than pixel space — we apply the same principle to code: train models to predict *latent representations* of masked code regions (function bodies, class methods, error handlers) rather than reconstructing exact tokens.

The core hypothesis is that forcing a student model to predict dense latent representations of masked code from only surrounding context (signatures, docstrings, imports, tests, call sites) will compel the emergence of abstract software engineering reasoning — understanding of design patterns, data flow, API contracts, complexity trade-offs — without the cost of explicit chain-of-thought generation. A key finding from SALT is that **small, sub-optimal teachers produce surprisingly strong students**, suggesting we can bootstrap this system cheaply using existing small code models (Qwen-2.5-Coder 1.5B–7B) as frozen teachers.

If successful, SWE-JEPA would enable:

- **Compressed reasoning**: Software design decisions encoded in dense latent vectors rather than sprawled across thousands of reasoning tokens
- **Decoupled generation**: Cheap decoders translate latent plans into actual code, separating "what to build" from "how to write it"
- **Efficient scaling**: Student models that exceed their teacher's capability, trained at a fraction of the compute cost of end-to-end approaches
- **Multi-dimensional code understanding**: Latent representations that capture non-functional requirements (maintainability, modularity, testability) that current token-prediction models are blind to

---

## Background & Motivation

### The Gap Between Code Generation and Software Engineering

Current AI coding assistants excel at the equivalent of Fowler's low-level refactoring catalog — extracting methods, renaming variables, generating function bodies from specifications. But Kim et al.'s survey of 328 Microsoft engineers found that 71% consider these atomic transformations to be merely components of "larger, higher-level effort to improve existing software." One developer stated directly: these low-level refactorings "form the minimum granular unit of any refactoring effort, but none are worthy of being called refactoring in and of themselves." The refactorings developers actually care about are "larger efforts aimed at interfaces and contracts to reduce software complexity."

This is the gap SWE-JEPA targets. The latent space should encode not just "what code goes here" but the structural properties that determine whether that code integrates well with its surroundings: does it reduce inter-module dependencies? Does it maintain the layered architecture? Is it testable? Does it align with the organizational ownership structure?

### Conway's Law and the Organizational Dimension

Conway (1968) proved that "organizations which design systems are constrained to produce designs which are copies of the communication structures of these organizations." This homomorphism between org charts and system architectures means that code structure carries organizational information — and vice versa. The Windows refactoring team's experience confirms this directly: their primary objective was reducing undesirable inter-module dependencies, and they achieved this through centralized architectural analysis using tools like MaX that mapped the de-facto dependency structure against intended layered architecture.

For SWE-JEPA, this implies that the latent space should implicitly encode organizational structure. A function in a core low-layer module that many teams depend on has different structural properties (higher test coverage, lower churn, tighter ownership) than a function in a high-layer application module. Kim et al.'s Table 4 quantifies these correlations precisely: preferentially refactored modules had 13% higher test block coverage, were touched by fewer developers, and had lower organizational cohesiveness scores. These are not properties of the code text — they are properties of the code's *position in a system*, and they are exactly the kind of latent structure that a JEPA-style model should learn to predict.

### The Refactoring Quality Signal

Kim et al.'s multivariate regression analysis (Tables 5 and 6) reveals which factors actually predict where refactoring investment is directed and what outcomes it produces. The strongest predictors for refactoring investment are: locality of changes (files per check-in), number of dependent modules, number of defects, and developer count. The strongest predictors for dependency reduction after refactoring are the number of refactoring commits and prior defect count.

This provides a concrete set of measurable attributes that SWE-JEPA's latent space should encode. If a student model, seeing only a function signature and its surrounding context, can predict a latent vector from which these properties are linearly decodable — inter-module dependency count, churn profile, organizational ownership concentration, test coverage level — then it has learned to reason about software engineering quality in a way that goes beyond token-level code generation.

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

## Preliminary Results

### Experiment 0.1: Layer Selection (Completed)

Extracted hidden states from three models on a reference LRU cache implementation with 5 methods. Measured cosine similarity between `__init__` and `get` function representations across layers.

| Layer | Coder-3B (base) | Coder-3B (instruct) | Qwen3-8B (base) |
|-------|:---:|:---:|:---:|
| 1 | 0.937 | 0.923 | 0.983 |
| 9 (L/4) | 0.896 | 0.877 | 0.944 |
| 18 (L/2) | **0.864** | 0.863 | **0.877** |
| 27 (3L/4) | 0.883 | 0.858 | 0.880 |
| 36 (last) | 0.984 | **0.856** | 0.906 |

**Key finding**: Base models show a U-shaped differentiation curve — best separation at mid-layers (L/2), with last-layer collapse toward uniform representations due to next-token prediction pressure. Instruct models show monotonic improvement through to the last layer. For teacher target extraction, base model at layer L/2 is the recommended configuration.

### Experiment 0.2: Nearest-Neighbour Retrieval (Completed)

Built FAISS index over 28,400 function embeddings from real PR data. Validated with 8 query functions (k=5), then scaled to all 4 models × 28,400 × k=10 = **1,136,000 pairs** with LLM judging on 808,594 selected pairs (see `phase0_2_nearest_neighbour_results.md`).

**Small-scale findings:**
- **Structural template detection works**: The `handle_flow_dict` / `handle_dns_dict` / `handle_device_dict` family in `iot-inspector-client` shows graded similarity (0.97 → 0.95) that precisely captures shared algorithmic skeleton while discriminating on schema differences.
- **Sharp discrimination boundary**: `createfunc` example shows clean step from sim=1.0 (copies) to sim=0.87 (related but algorithmically different).
- **Copy-paste-then-diverge lifecycle detected**: Similarity scores decay monotonically with the degree of adaptation from the original template.
- **Cross-model agreement**: Three base/smaller models agree on structural groupings. Qwen3-8B instruct diverges — wrong geometry for structural masking targets.

**Full-scale findings (1.136M pairs, LLM-judged):**
- **14.4% precision at sim ≥ 0.80** for cross-repo pairs (score_c ≥ 2). Cosine similarity alone is a weak filter; LLM judging is necessary.
- **209 gold cross-repo equivalents** (score_c=3): independent implementations of the same algorithm across different repos (e.g., `is_forward_ref` ↔ `is_fwd_ref`, `GetItemData` ↔ `_get_by_int_impl`). These are the highest-quality JEPA targets.
- **Score asymmetry** (sig avg 0.65 > body avg 0.55): embedding similarity is more driven by naming conventions than algorithmic content.
- **53,769 disconnect pairs** (9%): same algorithm, different name — valuable negative signal for disentangling syntactic form from semantic content.
- **Model recommendation confirmed**: Qwen2.5-Coder-3B (base) at layer 18 for teacher targets.

### Experiment 0.3: Linear Probing for Structural Properties (Completed)

Extracted 11 AST-based static properties for all 28,400 functions into `function_static_props` table. Trained Ridge/LogisticRegression linear probes on frozen embeddings with 5-fold CV. Results (see `phase0_3_linear_probing.md`):

| Property | Coder-3B | Coder-3B-Inst | Q3-8B | Q3-8B-base |
|----------|:---:|:---:|:---:|:---:|
| LOC (R²) | 0.757 | 0.757 | 0.787 | **0.789** |
| Cyclomatic complexity (R²) | 0.698 | 0.697 | 0.729 | **0.735** |
| # API calls (R²) | 0.723 | 0.722 | 0.751 | **0.764** |
| PR churn (R²) | 0.718 | 0.714 | 0.759 | **0.762** |
| Return type cat (BAcc) | 0.934 | 0.934 | 0.941 | **0.944** |
| Has side effects (BAcc) | 0.968 | 0.965 | 0.968 | **0.972** |
| Has docstring (BAcc) | 0.975 | 0.976 | **0.980** | 0.979 |

**Key findings**: All structural properties are linearly decodable from frozen hidden states. PR churn (R²=0.72–0.76) — the Kim et al. organizational signal — is as predictable as cyclomatic complexity. Base models outperform instruct variants for structural prediction. **Phase 0 success criteria fully met.**

---

## Experiment Plan

### Phase 0: Teacher Hidden State Extraction & Validation

**Goal**: Verify that pretrained code model hidden states contain useful structural information about code, and determine optimal extraction parameters.

#### Experiment 0.1: Extract hidden states across model scales ✅

| Parameter       | Values tested                                     |
| --------------- | -------------------------------------------------- |
| Teacher model   | Qwen-2.5-Coder-3B, Qwen-2.5-Coder-3B-Instruct, Qwen3-8B, Qwen3-8B-base |
| Extraction layer| Layers 1, L/4, L/2, 3L/4, L (last)               |
| Pool strategy   | Per-token (none), mean, last-token                 |
| Masking level   | Function bodies (regex), AST subtrees (tree-sitter)|

**Result**: Mid-layer extraction (layer 18 for 36-layer models) provides best differentiation for base models. Base models preferred over instruct for structural target extraction.

#### Experiment 0.2: Nearest-neighbor retrieval validation ✅

**Result**: Structural template families cluster at 0.95–0.99 cosine similarity. Full-scale (1.136M pairs): 14.4% precision@score_c≥2 for cross-repo pairs; 209 gold equivalents (score_c=3). See `phase0_2_nearest_neighbour_results.md`.

#### Experiment 0.3: Linear probing for structural properties ✅

**Result**: All 11 AST-based properties linearly decodable from frozen hidden states (LOC R²=0.76–0.79, CC R²=0.70–0.73, PR churn R²=0.72–0.76, return type BAcc=0.93–0.94). Kim et al. organizational signal (PR churn) is as decodable as cyclomatic complexity. **Phase 0 success criteria met.** See `phase0_3_linear_probing.md`.

---

### Phase 1: Student Architecture & Training Loop

**Goal**: Implement and validate the core SWE-JEPA student training pipeline.

#### Experiment 1.1: Minimal student training (Completed)

**Approach**: Start with the simplest testable form of the JEPA hypothesis: can a predictor trained on function **signatures** (context) predict the teacher's latent for the function **body** (masked region)?

This is simpler than full per-token prediction but directly tests the core claim: the signature should constrain what kind of implementation makes sense, and the teacher's hidden state encodes that implementation structure as a dense vector.

**Data preparation** (`extract_student_targets.py`):
1. For each of the 28,400 functions (model='Qwen2.5-Coder-3B'):
   - Fetch source via overlayfs (same pipeline as Phase 0)
   - Split source into `sig_text` (def line + docstring) and `body_text` (remaining statements)
   - Run frozen teacher (Qwen2.5-Coder-3B, layer 18) on each part
   - Mean-pool hidden states → `sig_embedding` (2048-dim) and `body_embedding` (2048-dim)
2. Store in `function_student_targets` postgres table

**Student architecture** (`train_student.py`):
```
sig_embedding (2048)
      │
  [Linear 2048→1024, GELU]
  [Linear 1024→512,  GELU]
  [Linear 512→1024,  GELU]
  [Linear 1024→2048       ]
      │
predicted_body_embedding (2048)

Loss: SmoothL1(predicted_body_embedding, actual_body_embedding)
```

**Training**:
- 80/10/10 split by **repo** (no function from a held-out repo appears in training) to prevent memorisation
- Adam optimizer, lr=1e-3, cosine decay, batch size=256
- Track: train/val loss, cosine similarity between predicted and actual body embeddings, rank@1 retrieval accuracy on FAISS index

**Key design decisions resolved for this minimal experiment**:
- Context = function signature only (not full file context) — cleanest test of the hypothesis
- No per-token prediction: mean-pooled body embedding as target — minimal engineering, maximal signal
- MLP predictor, not cross-attention — confirms the geometric structure is sufficient before adding complexity

**Results** (see `phase1_1_student_training.md`):

| Metric | Result | Baseline |
|--------|--------|----------|
| Training loss | 0.0074 → 0.0001 (×74 reduction) | — |
| Test cosine sim | **0.9468** | sig→body: 0.868 / random: 0.364 |
| Rank@1 retrieval | **0.00%** | random: 0.004% |

**Key finding**: The JEPA signal is real — loss converges and predictions are directionally correct (cosine 0.9468 > random 0.364). However, rank@1 = 0% reveals the fundamental limitation: the MLP with mean-pooled signature input predicts a *type prototype* (general structural direction) not a *specific instance*. The prediction is too diffuse to beat the many similar functions in a dense corpus. This cleanly motivates the transformer encoder upgrade.

#### Experiment 1.2: Transformer encoder student

**Goal**: Replace the mean-pooled MLP with a token-level transformer encoder that attends over individual signature tokens, producing predictions specific enough for rank@1 retrieval.

**Architecture**:
- **Context encoder**: 6-layer transformer, 512 hidden, 8 heads, ~35M params — processes tokenised function signature (up to 256 tokens)
- **Predictor**: 2-layer cross-attention transformer, ~10M params — attends over encoder output to predict body embedding
- **Target**: mean-pooled body embedding (same as Exp 1.1) — upgrade to per-token in Exp 1.3
- Total: ~45M params, DDP across 8 GPUs

**Training**:
- DDP across all 8 GPUs, batch=256 per GPU (2048 effective)
- Mixed precision (bf16), gradient checkpointing
- Train on 27,235 function pairs from 9 repos, repo-stratified split

**Hypothesis**: Token-level attention over argument names, type annotations, return types, and docstring wording will provide the specificity needed to exceed 5% rank@1 accuracy on held-out repos.

**Success criteria**:
1. Rank@1 > 5% on held-out test repo (vs 0% for MLP baseline)
2. Rank@10 > 20%
3. Training loss curve shows meaningful improvement over MLP baseline

#### Experiment 1.2: Student loss vs downstream correlation

**Goal**: Replicate SALT's key finding that student loss correlates with downstream quality.

**Steps**:

1. Save student checkpoints at regular intervals during training
2. For each checkpoint, evaluate on the nearest-neighbor retrieval task and linear probing tasks from Phase 0 (but using the *student's* representations)
3. Plot student training loss vs downstream metrics

**Success criteria**: Strong correlation (R² > 0.8) between student loss and at least one downstream metric. This would validate that the SWE-JEPA training signal is meaningful, analogous to SALT's finding.

#### Experiment 1.3: Data diversity/ volume study
- Repeat Experiment 1.2 with increased diversity of function sampled across different repos, increase train set size and observe how retrieval performance improvements scale.

#### Experiment 1.4: InfoNCE vs default Cross Entropy Loss
- Standard CE requires fixed classes upfront (which doesn'twork for an open-ended corpus), while InfoNCE uses the other functions in the current batch as dynamic "classes" each step. Repeat Experiment 1.3 with

---

### Phase 2: Teacher Robustness & Compute Allocation

**Goal**: Test whether SALT's "small teachers suffice" finding transfers to code.

#### Experiment 2.1: Teacher size ablation

**Steps**:

1. Using the pipeline from Phase 1, train identical students using teacher targets from:
   - Qwen-2.5-Coder-0.5B
   - Qwen-2.5-Coder-1.5B
   - Qwen-2.5-Coder-7B
2. Evaluate all students on the same downstream tasks
3. Measure the marginal gain from larger teachers

**Success criteria**: The gap between 1.5B-teacher and 7B-teacher students should be small relative to the gap between an untrained baseline and any teacher. This would confirm the SALT finding transfers.

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
| Hidden states too uniform across functions (no differentiation) | Low (disproved by Phase 0) | Try middle layers instead of last; add contrastive fine-tuning to teacher |
| Code latent geometry doesn't support functional similarity | Low (disproved by Phase 0) | Fine-tune teacher with contrastive objective on (spec, implementation) pairs from agent data |
| Long-range code dependencies exceed context window | High | Start with self-contained functions; later add retrieval-augmented context |
| Student doesn't learn abstract properties (just memorizes surface patterns) | Medium | Evaluate with held-out repos / languages; test invariance to refactoring |
| Latent-to-token decoding loses critical details | Medium | Hybrid approach: latent provides high-level plan, decoder has access to full context |
| Organizational/churn signals not present in code text alone | Medium | Enrich training data with git metadata; use PR-level context not just file-level |

---

## Resource Requirements

| Phase | Compute (est.) | Data | Timeline |
|-------|----------------|------|----------|
| Phase 0: Extraction & validation | 1-2 GPU-days (inference only) | 28K function embeddings from real PRs | 1-2 weeks |
| Phase 1: Student training | 1-4 GPU-days (training small models) | Same corpus + teacher targets | 2-3 weeks |
| Phase 2: Ablations | 2-8 GPU-days | Same | 2-3 weeks |
| Phase 3: Decoder | 4-16 GPU-days | + unit test data | 3-4 weeks |

---

## Key References

- **SALT**: Li et al., "Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers" (arxiv:2509.24317, 2025)
- **V-JEPA**: Bardes et al., "Revisiting Feature Prediction for Learning Visual Representations from Video" (2024)
- **I-JEPA**: Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (2023)
- **JEPA concept**: LeCun, "A Path Towards Autonomous Machine Intelligence" (2022)
- **Kim et al.**: Kim, Zimmermann, Nagappan, "An Empirical Study of Refactoring Challenges and Benefits at Microsoft" (IEEE TSE, 2014) — Empirical evidence that code quality is multi-dimensional; Table 4 metrics inform linear probing targets
- **Conway's Law**: Conway, "How Do Committees Invent?" (Datamation, 1968) — System structure mirrors organizational communication structure; implies latent code representations should encode organizational/architectural context