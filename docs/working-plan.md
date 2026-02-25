# SWE-JEPA Working Plan

## Status as of 2026-02-25

### Completed experiments

| Exp | Description | Key result |
|-----|-------------|------------|
| 0.1 | Layer selection | Qwen3-8B-base, layer 18; base > instruct |
| 0.2 | KNN retrieval | 14.4% precision@score_c≥2; 209 gold cross-repo equivalents |
| 0.3 | Linear probing | LOC R²=0.79, PR churn R²=0.76, return type BAcc=0.94 |
| 1.1 | MLP student | Cosine 0.95 but Rank@1=0% — predicts type prototype, not instance |
| 1.2 | Transformer encoder | OOD test failure (9 repos); val cosine 0.25 meaningful |
| 1.3 | Expanded corpus | 150 repos; val cosine 0.601, Rank@1=0.03% |
| 1.4 | InfoNCE loss | Rank@1=1.38%; plateau at ~4% Rank@10 |
| 2.1 | Hard negative mining | Null result; plateau is representation ceiling not neg quality |
| 2.2 | Token-level body pred | Rank@10=4.20% val; gradient conflict slows convergence |
| 3.0 | Qwen3-8B-base teacher | **Rank@1=19.11%, Rank@10=47.52% val** — 17× over best 3B result |

### Key insight from 3.0
The breakthrough came from computing body targets with full sig+body context (not body-only),
reducing target anisotropy from 0.37 → 0.013 random cosine. This aligns with the JEPA
objective: target = f(body | full context), predictor maps g(sig only) → f(body | full context).

---

## Planned: Experiment 4.1 — Defect Prediction Probe

### Motivation

With Exp 3.0 achieving 47.52% Rank@10 (up from 4.2%), the retrieval representations are strong
enough to test the central SWE-JEPA hypothesis: do the learned representations encode
**non-functional software engineering properties** beyond syntactic similarity?

Defect-proneness is the canonical such property: it is not visible in the function signature
alone, yet expert engineers use signature-level cues (complexity signals, naming patterns,
interface design) to identify high-risk code. If the SWE-JEPA student's representations predict
future bugfix followups better than frozen teacher representations, it means the JEPA training
objective — predict body structure from signature — caused the model to learn something about
implementation quality from signatures alone.

### Data: `followups_function` table

- **62,071 rows** linking feature PR functions to followup PRs
- **15,296 distinct (instance_id, file, function) anchors** from 144 repos
- `followup_category`: feature / maintenance / bugfix / docs
  - **maintenance excluded** (often dependency bumps, whitespace, not defect signal)
- Labels per anchor (after excluding maintenance):
  - `has_bugfix`: binary, 7,378 (48%) negative / 7,918 (52%) positive — naturally balanced
  - `n_bugfix_prs`: integer count (0, 1, 2, 3, 4, 5+) for regression

### Experiment design

```
For each of 15,296 function anchors:
  - Fetch sig_text (from followups_function → overlayfs at feature PR commit)
  - Tokenize with Qwen3-8B-base tokenizer
  - Encode with:
    A) Frozen teacher  (Qwen3-8B-base, layer 18, mean-pool sig tokens) → 4096-dim
    B) SWE-JEPA student (student_3_0_ckpt.pt, _encode method)          → 4096-dim

Train/val/test split: by repo (same 144 repos)
  - ~115 train repos, ~15 val, ~14 test  (80/10/10 by repo count)

Linear probes (sklearn):
  - LogisticRegression(C=1.0, max_iter=1000) on teacher_emb → has_bugfix
  - LogisticRegression(C=1.0, max_iter=1000) on student_emb → has_bugfix
  - Ridge(alpha=1.0) on teacher_emb → n_bugfix_prs
  - Ridge(alpha=1.0) on student_emb → n_bugfix_prs

Baselines:
  - Random / majority class
  - TF-IDF on sig_text (BoW baseline)
  - LOC + cyclomatic complexity (structural baseline; extract via AST for these 15k functions)

Metrics:
  - has_bugfix:   balanced accuracy, ROC-AUC, F1
  - n_bugfix_prs: R², Spearman ρ

Sensitivity check:
  - Repeat with hunk_overlap_fraction > 0.1 as stricter bugfix threshold
    (filters out PRs that barely touched the function)
```

### Expected outcomes

| Outcome | Interpretation | Next step |
|---------|---------------|-----------|
| student >> teacher (ΔAUROc > 0.05) | JEPA training encodes defect-proneness | Option 1: efficiency comparison (Exp 4.2) |
| student ≈ teacher | Retrieval gains didn't transfer; representations still useful for retrieval only | Diagnose: try with body context too; add codebase-level features |
| teacher >> baseline | Structural props already encode risk (consistent with Exp 0.3 churn R²=0.76) | Expected; establishes why teacher matters |

### Implementation plan

1. `extract_followup_sigs.py` — fetch sig_text for 15,296 anchors at feature PR commit via overlayfs
2. `probe_defect_prediction.py` — encode with teacher + student, fit probes, report metrics
3. Results → `docs/phase4_1_defect_prediction.md`

### Success criteria

| Criterion | Target |
|-----------|--------|
| Student AUROC (has_bugfix) > teacher AUROC | Student learns something beyond teacher |
| Student AUROC > TF-IDF baseline | Representations beat bag-of-words |
| Student AUROC > 0.60 | Practically useful signal |

---

## Future directions (post-4.1)

### Experiment 4.2 — Efficiency comparison vs SFT (Option 1 from proposal point 3)

Compare on a fixed compute budget:
- SWE-JEPA student (frozen teacher + cheap student) trained for N GPU-hours
- Supervised fine-tuned model of equivalent size trained for same N GPU-hours
- Task: code retrieval (Rank@10) or defect prediction (AUROC)

Natural baseline: contrastive fine-tune of Qwen3-8B-base on (sig, body) pairs directly,
without the JEPA architecture. Cost of fine-tuning vs cost of SWE-JEPA training.

### Experiment 4.3 — Latent-conditioned decoder (Phase 3 in proposal)

Train a small decoder that takes the student's 4096-dim predicted latent and generates
function bodies autoregressively. Evaluate with pass@k on unit tests.

### Experiment 4.4 — Conway's law / architectural fit (Option 2.5)

Requires: developer ownership data, module-level dependency graph. Predict whether a function
implementation will require cross-team followups (Conway's law violation signal). Currently
underspecified; revisit after Exp 4.1 establishes defect-proneness baseline.
