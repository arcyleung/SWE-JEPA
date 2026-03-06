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

---

## Planned: Experiment 4.3 — Region-Level Defect & Feature Localization

### Motivation

Exp 4.1 answers "does this PR have a bugfix followup?" at the PR level.  Exp 4.3
sharpens the question to "which specific function (and line range) within this PR
is the risky one?" — a harder and more actionable task for code review tooling.

Two sub-tasks:
- **Bugfix localization**: rank functions within a PR by predicted bugfix-proneness.
- **Feature-extension localization**: rank functions by predicted likelihood of
  being refactored or extended in a later feature PR.

### Data: `followups_function` table (same as Exp 4.1)

- `feature_function_start` / `feature_function_end`: line range of the function
  in the feature-PR source file — the ground-truth "region" being predicted.
- `has_bugfix` / `has_feature`: binary labels aggregated per anchor (same as 4.1).

### Experiment design

```
For each test-set feature PR:
  - Collect all function anchors from that PR
  - Score each function with each method (teacher emb probe, student emb probe,
    TF-IDF probe, LOC probe, random baseline)
  - Rank functions descending by score
  - Recall@K = # true positives in top-K / # total true positives (per PR)

Evaluation:
  - Mean Recall@K (K = 1, 3, 5, 10) over test PRs with ≥1 positive
  - Both labels: has_bugfix and has_feature
  - GPU-hours efficiency table (Exp 4.2 methodology):
      Recall@10 / GPU-min × 1000
```

### Implementation plan

1. `extract_followup_sigs.py` — add `feature_function_start/end` to SQL + JSONL
   (already done; future re-runs will include line ranges)
2. `probe_region_localization.py` — main experiment; imports `encode_all`,
   `load_sigs`, `repo_split` from `probe_defect_prediction.py`; adds
   `enrich_with_line_ranges` (DB query), `eval_localization`, `write_report`
3. Results → `docs/phase4_3_region_localization.md`

### Reuse check

`extract_expanded_targets.py` pipeline (overlayfs → GPU inference → postgres) is
**not needed** for this experiment — the existing `followup_sigs.jsonl` and
cached `followup_embs.npz` (from Exp 4.1) are sufficient. Running:

```bash
# With cached embeddings (no GPU needed):
python probe_region_localization.py --use-cache

# First run (encode on GPU 0):
python probe_region_localization.py --gpu 0
```

### Success criteria

| Criterion | Target |
|-----------|--------|
| Student R@5(bugfix) > Teacher R@5(bugfix) | JEPA ranks buggy functions higher |
| Student R@5(bugfix) > TF-IDF R@5(bugfix) | Representations beat bag-of-words |
| Student R@5(feature) > random baseline | Some extensibility signal learned |

---

## Future directions (post-4.1)

### Reframing after Exp 4.3

Exp 4.3 indicates the current JEPA student objective (global retrieval geometry from
signature-only inputs) does not reliably transfer to **within-PR ranking** tasks.
Teacher and student signals are complementary, but neither alone is consistently best
across bugfix and feature localization settings.

Updated framing:
- Treat frozen embeddings as a reusable representation substrate, not a full task solution.
- Move downstream adaptation to cheap, task-specific heads and rankers.
- Optimise directly for software-engineering outcomes (within-PR ranking, co-change,
  refactor likelihood), not only global nearest-neighbour retrieval.

Practical consequence: avoid expensive end-to-end SFT/RL; keep the backbone frozen and
train lightweight heads/projectors with ranking-aware losses.

### Experiment 4.2 — Efficiency comparison vs SFT (Option 1 from proposal point 3)

Compare on a fixed compute budget:
- SWE-JEPA student (frozen teacher + cheap student) trained for N GPU-hours
- Supervised fine-tuned model of equivalent size trained for same N GPU-hours
- Task: code retrieval (Rank@10) or defect prediction (AUROC)

Natural baseline: contrastive fine-tune of Qwen3-8B-base on (sig, body) pairs directly,
without the JEPA architecture. Cost of fine-tuning vs cost of SWE-JEPA training.

### Experiment 4.4 — SE-Head Stack for Region Localization (Cheap Adaptation)

#### Motivation

Exp 4.1 showed PR-level transfer (student AUROC > teacher), while Exp 4.3 showed unstable
region-level ranking (teacher often stronger on strict bugfix localization).
The likely mismatch is objective-level:

- JEPA student training: global embedding alignment for retrieval.
- Localization eval: relative ranking *within the same PR*.

Exp 4.4 closes this gap by keeping embeddings frozen and training cheap heads directly on
within-PR ranking objectives.

#### Core hypothesis

A lightweight ranking head trained on frozen representations with in-PR pairwise/listwise
loss will outperform standalone linear probes for Recall@K localization, at negligible
compute cost compared with SFT.

#### Methods

Train/evaluate the following on the same repo split as Exp 4.3:

1. Pairwise ranker (primary):
   - Inputs per function: `[teacher_emb, student_emb, tfidf_logit, loc, cc, churn_proxy]`
   - Construct training pairs only within each feature PR:
     - positive = `has_bugfix=1` (or `has_feature=1`)
     - negative = `0` from same PR
   - Loss: logistic pairwise ranking (`score_pos - score_neg`)
2. Fusion baseline:
   - Calibrated weighted sum of independent heads:
     - teacher probe + student probe + TF-IDF + LOC
   - Fit weights on val only (ridge/logistic calibration).
3. Small MLP rank head:
   - 2-layer MLP (e.g., 128 hidden) on concatenated frozen features.
   - BCE + optional pairwise auxiliary term.

#### Data and labels

- Source table: `followups_function` (same anchor set as 4.3).
- Tasks:
  - Bugfix localization (`has_bugfix`) with `min_overlap` in `{0.0, 0.1}`.
  - Feature-extension localization (`has_feature`).
- Groups: feature PR (`feature_instance_id`) as ranking unit.

#### Evaluation

Primary:
- Mean Recall@K over PRs with at least one positive (`K = 1, 3, 5, 10`).
- Mean Reciprocal Rank (MRR) per PR.

Secondary:
- PR-AUC for pooled instance scores.
- Stability across random seeds (3 seeds).
- Efficiency metric: `R@10 / GPU-min × 1000` (Exp 4.2 convention).

#### Implementation plan

1. `probe_region_localization.py`
   - add `--model pairwise|fusion|mlp|linear`
   - add pairwise training data builder grouped by `feature_instance_id`
   - add MRR and seed-averaged reporting
2. `train_region_ranker.py` (new)
   - reusable ranker training (pairwise/listwise) on cached embeddings/features
   - writes checkpoint and JSON metrics
3. `docs/phase4_4_se_heads_localization.md`
   - table: all methods × labels × overlap thresholds
   - ablation: teacher-only, student-only, fusion

#### Success criteria

| Criterion | Target |
|-----------|--------|
| Pairwise ranker R@5(bugfix, overlap=0.1) > Teacher linear probe | +3 pp absolute |
| Fusion R@10(bugfix) ≥ max(Teacher, Student, TF-IDF, LOC) | Strictly best or tied-best |
| Pairwise ranker R@10(feature) > random baseline | +5 pp absolute |
| Training cost (all heads) | < 60 GPU-min total |

#### Why this is an "easy win"

- Reuses existing `followup_embs.npz` and `followup_sigs.jsonl`; no re-encoding required.
- No full-model fine-tuning; only tiny heads/rankers are trained.
- Objective is aligned with deployment need (rank functions within a PR).

### Experiment 4.5 — Conway's law / architectural fit (Option 2.5)

Requires: developer ownership data, module-level dependency graph. Predict whether a function
implementation will require cross-team followups (Conway's law violation signal). Currently
underspecified; revisit after Exp 4.4 establishes stable localization heads.

#### Implementation update (2026-03-05)

Conway proxy metrics are now implemented and materialized per anchor:
- `ownership_friction`
- `interface_stress`
- supporting ownership/co-change channels (`distinct_authors`, weighted co-change degree, etc.)

Collection pipeline:
- `extract_org_metrics.py` computes git-history metrics with DB-proxy fallback
- Output: `followup_org_metrics.jsonl` and Postgres `followup_org_metrics`
- Coverage on active 4.x set: `6,651 / 6,651` anchors (0 missing)

#### Preliminary localization ablation (Exp 4.5 pilot)

Using `train_region_ranker.py --use-org-metrics` with same split/seeds as Exp 4.4:

- Bugfix localization (`min_overlap=0.1`)
  - Best R@10 moved from **58.9%** (Exp 4.4 fusion) to **59.0%** (Conway-MLP), small +0.1 pp.
  - MRR dropped for some heads; org features mostly affect deeper ranks.
- Feature localization
  - Pairwise head improved R@10 by **+2.1 pp** (35.6% → 37.7%) and MRR by **+11.7 pp**.
  - Best absolute R@10 remained near baseline (MLP 39.1% → 38.8%).

Interpretation: Conway signals are learnable by cheap heads and can improve ranking quality
for specific objectives (notably pairwise feature localization), but are not yet a universal
lift across all heads/labels.

### Experiment 4.6 — PR Acceptance Reranking with Conway-Aware SWE-JEPA

#### Motivation

We want to test SWE-JEPA as an **SE review/reranking model** (not a generator): can it pick
the PR variant that is most likely to be accepted upstream when all candidates are
functionally equivalent but differ in engineering quality and organizational fit?

This directly targets the practical use-case: rerank candidate patches/PRs before submission.

#### Core hypothesis

A Conway-aware SWE-JEPA ranker (frozen code reps + cheap ranking head + org proxies) will
outperform a standalone coder model at identifying the variant that mirrors real upstream
acceptance behavior.

#### Dataset design: 6-way PR candidate packs

For each historically merged PR (ground truth positive), construct 5 synthetic negatives:

1. Naming drift variant
   - less consistent variable/function/class naming conventions
2. Cross-cutting noise variant
   - unnecessary edits across unrelated files/modules
3. Scope-bloat variant
   - combines multiple concerns into one PR rather than localized change
4. Architectural-boundary variant
   - introduces changes that cross likely module ownership boundaries
5. Ownership-friction variant
   - edit pattern likely to increase coordination burden

Each pack has 6 candidates total (1 real merged + 5 synthetic). All variants must preserve
functional behavior (tests/lint/syntax checks where possible).

#### Labels and task

- Group/ranking unit: `(repo, source_pr_id, variant_pack_id)`
- Target: rank candidates by merge-likelihood; the true merged PR is label `1`, variants `0`
- Primary framing: listwise/pairwise reranking

#### Feature sets

Base representation channels:
- teacher/student frozen embedding signals
- textual/structural channels from Exp 4.4 (`tfidf`, LOC, CC, churn proxy)
- Conway proxies from Exp 4.5 (`ownership_friction`, `interface_stress`, co-change stats)

Additional metadata from `prs_copy` (candidate pool to evaluate):
- `requested_reviewers`
- `total_review_threads`, `review_threads`
- `total_comments`, `comments`
- `closing_issue_id`
- `created_at`, `merged_at`

Important leakage policy:
- **Allowed at inference-time**: fields observable by review time (e.g., requested reviewers,
  linked issues present on PR, early thread/comment counts if we define a fixed snapshot cutoff).
- **Not allowed as direct features** for merge prediction: post-outcome or target-adjacent fields
  such as final `merged_at`, full/final review-thread totals after decision, or any signal that
  directly encodes acceptance outcome timing.
- `created_at`/`merged_at` should be used for analysis-derived targets (e.g., time-to-merge) or
  stratification, not naive predictive inputs.

#### Baselines

1. Standalone coder model scorer
   - Prompted to score each candidate PR's merge-likelihood from diff + context
2. Non-neural/classical baseline
   - handcrafted + metadata features with logistic/pairwise ranker
3. SWE-JEPA without Conway channels (ablation)
4. SWE-JEPA with Conway + allowed `prs_copy` signals (full model)

#### Evaluation

Primary:
- Top-1 accuracy (select true merged PR among 6)
- MRR
- Pairwise AUC within each pack

Secondary:
- Win-rate by deficiency type (naming/cross-cutting/scope/architecture/ownership)
- Calibration (Brier / reliability bins)
- Robustness by repo/language/PR-size buckets

#### Implementation plan

1. `build_pr_variant_benchmark.py`
   - sample merged PRs, generate 5 controlled variants per PR, run validity checks
2. `extract_prs_copy_signals.py`
   - parse/normalize `prs_copy` signals; enforce inference-time feature mask
3. `train_pr_acceptance_reranker.py`
   - pairwise/listwise ranker on frozen SWE-JEPA + metadata channels
4. `score_pr_candidates_coder_baseline.py`
   - standalone coder-model scoring pipeline
5. `eval_pr_acceptance_reranking.py`
   - aggregate metrics, per-deficiency breakdown, significance tests
6. Report
   - `docs/phase4_6_pr_acceptance_reranking.md`

#### Success criteria

| Criterion | Target |
|-----------|--------|
| SWE-JEPA+Conway Top-1 > coder baseline Top-1 | +8 pp absolute |
| SWE-JEPA+Conway MRR > SWE-JEPA (no Conway) | +5 pp absolute |
| Win-rate vs at least 4/5 deficiency types | > 55% each |
| Compute cost | no full-model SFT/RL; lightweight heads only |

### Experiment 4.7 — Agentic PR Evolution Steering (Small Head + Large Coder)

#### Motivation

Exp 4.6 validates static PR-candidate reranking. The next step is fair, agentic comparison:
can a **small trained steerer** guide a large coder model through iterative PR evolution
(implement → review feedback → revision) better than prompt-only coding agents?

This keeps SWE-JEPA’s core value proposition:
- frozen representation substrate
- cheap downstream adaptation
- no full-model SFT/RL for the large coder backbone

#### Core hypothesis

A small steerer trained on PR-state transitions and review feedback can improve acceptance-rate
and reduce review churn when used to guide a large coder model inside a shared agent scaffold.
SWE-JEPA latents should make this steerer cheaper to train and more sample-efficient.

#### Environment and fairness

- Single agentic scaffold for all methods (OpenHands-style loop).
- Same tools, context window policy, retrieval budget, and step budget.
- Same repo/time holdouts.
- Same task set and stopping criteria.

Compared systems:
1. Prompt-only coder agent (baseline).
2. Coder + small steerer (trained on task/review trajectories).
3. Coder + small steerer on SWE-JEPA latent state (efficiency variant).

#### PR evolution state model

Model each trajectory as state/action transitions over PR lifecycle:
- States (example): `drafting`, `ready_for_review`, `changes_requested_risk`,
  `likely_mergeable`, `stalled`.
- Actions: inspect files, edit patch, run tests, re-scope changes, submit update, etc.

Steerer outputs:
- action-value / rank score for next action
- merge-likelihood estimate for current trajectory prefix
- optional “refactor-risk” score for changed regions

#### Refactor-demand signal (new)

Use `review_threads` and `comments` column of prs_copy table to derive supervision for “likely refactor requested”.
Potential labels/features:
- presence of refactor-style language:
  - “refactor”, “split this”, “extract”, “too large”, “naming”, “architecture”, “cleanup”
- thread resolution latency and reopen patterns
- comment density on specific files/hunks
- repeated reviewer concern categories across updates
- review outcome proxies: `changes requested` rounds before approval

These signals provide a direct learning target for which edits tend to trigger senior-reviewer
refactor demands.

#### Data pipeline

Build transition dataset from `prs_copy` + review artifacts:
- `(state_t, action_t, state_{t+1}, reward_t)`
- rewards/proxies:
  - positive: merged quickly, low churn, few revision rounds
  - negative: multiple change-request cycles, high refactor-demand score, non-merge

Include both code/context features and review interaction features.

#### Evaluation

Primary:
- acceptance/merge rate under fixed rollout budget
- review rounds to acceptance
- time-to-acceptable patch

Secondary:
- refactor-demand incidence after agent submission
- scope discipline (unnecessary file touches)
- architecture/ownership stress delta

Efficiency:
- GPU-hours to train steerer
- rollouts needed to reach target acceptance rate

#### Implementation plan

1. `build_pr_mdp_dataset.py`
   - construct PR evolution transitions and refactor-demand labels
2. `train_pr_steerer.py`
   - train compact steerer head (pairwise/listwise + value prediction)
3. `run_agentic_eval.py`
   - shared scaffold evaluation for baseline vs steered agents
4. `docs/phase4_7_agentic_pr_steering.md`
   - results, fairness config, ablations, efficiency table

#### Success criteria

| Criterion | Target |
|-----------|--------|
| Steered coder acceptance rate > prompt-only coder | +8 pp absolute |
| Median review rounds to acceptance | reduced |
| Refactor-demand incidence after submission | reduced |
| Steerer training cost | substantially below full coder SFT |
