# SWE-JEPA

**Latent-space predictive architecture for code understanding, inspired by SALT and V-JEPA.**

Full research proposal: [`docs/code-jepa-research-proposal.md`](docs/code-jepa-research-proposal.md)

---

## Motivation

Current AI coding assistants are good at token-level generation but blind to the structural properties of code that determine long-term quality: modularity, volatility, organizational ownership, testability. These are not properties of individual functions in isolation — they emerge from how code relates to its surrounding system, authoring organization, and its history of change.

We take inspiration from **SALT** (arxiv:2509.24317), which showed that for video representation learning: (1) a cheap frozen teacher provides sufficient latent targets, (2) students trained on those targets can outperform their teachers, and (3) teacher quality matters far less than the information asymmetry built into the learning task.

**SWE-JEPA** applies the same principle to code: freeze a pretrained code LLM as a teacher, extract its mid-layer hidden states at function body positions, and train a student to predict those latent targets from only the function signature and context. The hypothesis is that forcing the student to bridge this gap — from sparse interface to dense latent — compels the emergence of abstract software engineering reasoning without explicit chain-of-thought.

---

## Results Summary

### Phase 0: Teacher Validation

**Goal**: Verify that frozen teacher hidden states encode useful structural information.

| Experiment | Finding |
|---|---|
| **0.1 — Layer selection** | Mid-layer (L/2 = layer 18) gives best inter-function differentiation. Base models show U-shaped curve; instruct models show monotonic improvement through to the last layer. Base at L/2 is the recommended teacher config. |
| **0.2 — KNN retrieval** | FAISS over 28,400 function embeddings recovers structural template families (e.g., `handle_flow_dict` ↔ `handle_dns_dict` at sim=0.97). Full-scale 1.136M pairs (4 models × 28,400 × k=10): 14.4% precision@score_c≥2 for cross-repo pairs; 209 gold cross-repo equivalents (independent implementations of the same algorithm). |
| **0.3 — Linear probing** | All structural properties are linearly decodable from frozen layer-18 embeddings: LOC R²=0.79, cyclomatic complexity R²=0.73, PR churn R²=0.76, return type BAcc=0.94. PR churn — Kim et al.'s organizational signal — encodes as well as cyclomatic complexity. **Phase 0 success criteria fully met.** |

Best teacher: **Qwen2.5-Coder-3B base, layer 18** (dim=2048).

---

### Phase 1: Student Training

**Task**: Given a function signature, predict the teacher's mean-pooled body embedding. Evaluated on retrieval rank in a corpus of 49,631 functions (random baseline Rank@1 = 0.002%).

| Experiment | Architecture | Key Change | Val Rank@1 | Test Rank@1 | Test Rank@10 |
|---|---|---|---|---|---|
| **1.1 — MLP baseline** | 5M-param MLP, mean-pooled sig | — | — | 0.00% | 0.00% |
| **1.2 — Transformer, 9 repos** | 8.4M SigPredictor (2-layer transformer) | Token-level sig encoder | ~0% | ~0% (OOD test set) | — |
| **1.3 — Transformer, 150 repos** | Same | 49,631 functions from 150 repos | — | 0.03% | 0.24% |
| **1.4 — InfoNCE loss** | Same | SmoothL1 → InfoNCE contrastive | **1.06%** | **1.38%** | **4.14%** |

Key milestones:
- **Exp 1.1**: Confirmed the JEPA signal is real (cosine 0.95 vs random 0.36) but mean-pooled MLP predicts a type prototype, not a specific instance. Zero retrieval accuracy.
- **Exp 1.2→1.3**: Token-level transformer encoder breaks the MLP ceiling; scale (9→150 repos) is critical — val cosine improved 2.4× (0.25→0.59) and OOD failure disappeared.
- **Exp 1.4**: Switching loss from SmoothL1 to InfoNCE yielded **46× Rank@1 improvement** (0.03%→1.38% test). L2 loss pushes toward the average body direction; InfoNCE explicitly penalises wrong bodies in the batch.

---

### Phase 2: Pushing the Ceiling

| Experiment | Change | Val Rank@10 | Test Rank@10 | Verdict |
|---|---|---|---|---|
| **2.1 — Hard negative mining** | Appended LLM-judged hard negative bodies to InfoNCE denominator | 3.64% | 4.31% | **Null result** — worse on val. Plateau is not a hard-negative problem. |
| **2.2 — Token-level targets** | Cross-attention decoder predicts per-token body hidden states (+ InfoNCE on pooled) | 4.20% | 5.52% | Marginal improvement. Teacher-capacity ceiling. |

**Diagnosis**: The ~4–5% Rank@10 plateau is a **representation ceiling** set by the teacher. Functions that confuse the student also cluster tightly in the teacher's own embedding space. Richer training signals (more negatives, per-token supervision) don't help when the teacher itself cannot cleanly separate the confusable pairs.

**Next**: Exp 3.0 — upgrade teacher to **Qwen3-8B-base** (4096-dim, 36 layers). Script: `train_student_3_0.py`. Success target: Val Rank@10 > 20%, Rank@1 > 5%.

---

## Architecture

```
Frozen teacher: Qwen2.5-Coder-3B (or Qwen3-8B-base)
  → hook at layer 18 → token hidden states [B, T, D]

Student SigPredictor (8.4M params):
  Linear(D → 512)
  2× TransformerEncoderLayer(d_model=512, nhead=8, ffn=2048)
  mean-pool → Linear(512 → D)
  → predicted body embedding

Loss: InfoNCE (cross-GPU, ~512 negatives/step)
Eval: Rank@1 / Rank@10 retrieval in 49,631-function FAISS corpus
```

---

## Infrastructure

- **Postgres**: `function_embeddings` (4 models × 50k functions), `function_student_targets` (sig_emb + body_emb, 50k rows), `function_static_props` (AST properties)
- **FAISS**: `faiss_indices/function_embeddings_Qwen2.5-Coder-3B_L18.faiss`
- **Checkpoints**: `student_mlp_ckpt.pt` (1.1), `student_transformer_ckpt.pt` (1.2), `student_transformer_1_3_ckpt.pt` (1.3), `student_1_4_ckpt.pt` (1.4), `student_2_1_ckpt.pt` (2.1), `student_2_2_ckpt.pt` (2.2)
- **GPU**: 8 GPU, DDP training, bf16 frozen teacher + fp32 student

---

## Key References

- **SALT**: Li et al., "Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers" (arxiv:2509.24317, 2025)
- **V-JEPA**: Bardes et al., "Revisiting Feature Prediction for Learning Visual Representations from Video" (2024)
- **Kim et al.**: "An Empirical Study of Refactoring Challenges and Benefits at Microsoft" (IEEE TSE, 2014)
- **Conway's Law**: Conway, "How Do Committees Invent?" (Datamation, 1968)
