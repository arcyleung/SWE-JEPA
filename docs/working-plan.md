# SWE-JEPA Working Plan

## Status as of 2026-03-23

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
| 5.1 | FeatBench Conway steerer | +21.4pp F2P, +7.2pp scaffold judge advantage |
| 6.1 | Embedding steerer | Frozen layer-18 embeddings: acceptance AUROC 0.91 (+20pp), refactor 0.76 (+17pp) |
| 6.2 | Multi-axis + HDBSCAN | Contrastive head overfits (negative); HDBSCAN 20 super-clusters with 113 features; bugfix followup clusters Δ=85pp |
| 6.2b | Followup extraction at scale | followups_file 720k→2.4M (3,692 repos); followups_function 46k→72k; backfilled 1,391 PR diffs |
| 6.2c | Combined steerer | 121-feature model (113 Conway + 8 followup) + cluster hints; acceptance AUROC 0.897 |

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

- Single agentic scaffold for all methods (Mini-SWE-agent/ OpenHands-style loop).
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

### Experiment 4.7.1 — Paired Cohort Eval + Belief-State Analog

#### Motivation

Exp 4.7 judge results showed steered < baseline (43.1% win rate), with largest deficits in
correctness and review-readiness. A key confound was asynchronous cohort generation:
baseline and steered trajectories were produced in separate runs and only partially paired
post hoc. Another issue was over-constraining scope without forcing repo-level integration
reasoning.

Exp 4.7.1 addresses this with:
1. **paired execution** (both arms per same instance in one worker run), and
2. a lightweight **belief-state analog** via `belief.MD` updates in the repo.

Result note:
- The `belief.MD` surrogate did **not** help in mini-swe-agent.
- In practice it reduced patch yield and appeared to distract the agent from code-edit completion.
- Follow-up runs therefore removed the explicit belief file and kept only structured steer hints.

#### Core changes

1. **Paired runner**
   - New runner executes `non-steered` then `steered` for each `(repo, pull_number)` task.
   - Produces one JSONL row per pair with both arm outputs and patch paths.
   - Ensures judge-ready pairs are collected together (no cross-run matching drift).

2. **Belief-state analog in prompt**
   - Agent maintains `belief.MD` with:
     - architecture assumptions
     - touched module hypotheses
     - interface invariants
     - uncertainty / risks
   - Agent updates `belief.MD` during exploration, then removes it from final diff unless
     explicitly required.

3. **Patch capture parity**
   - Save `git diff --binary` patch files for both baseline and steered arms.
   - Avoid reliance on `submission` field in mini trajectories.

#### Expanded feature list for 4.7.1 steerer (from 6 → 28 channels)

Feature groups:
1. **Scope & churn**
   - changed files, additions/deletions, net churn ratio, file-type dispersion
2. **Interface stress**
   - public API symbol deltas, signature-change count, callsite update ratio
3. **Ownership friction**
   - owner-module boundary crossings, reviewer-owner mismatch prior, co-change unfamiliarity
4. **Architectural risk**
   - import-graph spread, layer-boundary crossings, dependency fan-out touched
5. **Review friction proxies**
   - naming churn ratio, non-functional edit ratio, historically contentious path priors
6. **Task grounding / urgency**
   - closing-issue linkage, requested-reviewer load, PR narrative specificity
7. **Belief-quality proxies**
   - invariant mentions, architecture rationale completeness, uncertainty flagging

#### Evaluation protocol

- Fixed cohort of feature PRs.
- For each instance: collect baseline+steered pair in same run.
- Judge compares paired patches blind A/B with same rubric as Exp 4.7.
- Report:
  - overall steered win rate
  - size-matched win rate
  - correctness/review-readiness deltas (primary)
  - test-coverage and scope metrics (secondary)

#### Success criterion (4.7.1)

| Criterion | Target |
|-----------|--------|
| Steered win rate vs Exp 4.7 steerer baseline | >50% |
| Correctness delta | non-negative |
| Review-readiness delta | non-negative |

### Experiment 4.7.2 — PR Refinement History and Conway Drift

#### Motivation

Exp 4.7/4.7.1 focused on final generated patches, but that collapses the real PR lifecycle
into a single end state. To validate the Conway-state framing directly, we should inspect
**intermediate PR revisions** and ask whether the cumulative patch becomes more
merge-ready after review feedback.

This is the closest real-data analog to the steerer objective:
- PR state is observed at successive revisions
- reviewers provide friction / refactor signals
- the next revision may reduce or amplify architectural stress

#### Core hypothesis

Within real merged PRs, Conway-style risk signals should improve over time:
1. from the first visible PR commit to the final visible PR commit, and
2. more sharply on **post-review response revisions** than on arbitrary commit-to-commit updates.

#### Data source

Use `prs_copy` fields already present in the database:
- `base_sha`, `head_sha`
- `commits` (ordered commit list with hashes and timestamps)
- `review_threads` (comment timestamps, file paths, often `commit_hash`)
- `submitted_reviews` (approval / changes-requested events)

#### Method

For each merged PR with `total_commits >= 2`:
1. Reconstruct cumulative snapshots `git diff base_sha..<commit_i>`.
2. Run the Conway patch extractor on each cumulative patch.
3. Align review comments / submitted reviews to commit intervals by timestamp.
4. Compare:
   - first visible commit vs final visible commit
   - latest pre-review commit vs first post-review commit

Practical sampling note:
- Full per-commit traversal is the highest-fidelity version of the experiment, but it is too expensive on commit-heavy PRs and very large repositories.
- The implementation therefore allows capped equal-interval snapshot sampling, with `max_snapshots=5` as the default fast path. For example, a 9-commit PR is sampled at commit indices `1, 3, 5, 7, 9`.
- This can miss short-lived intermediate states, but the working assumption is that review-driven PR refinement is usually low-frequency relative to the raw commit stream. The goal here is to capture directional Conway drift, not reconstruct every micro-edit.
- The Nyquist/Shannon analogy is only approximate because commit history is already a sequence of discrete author actions, not uniform samples from a continuous signal. The relevant empirical question is whether sparse visible revisions preserve the main trajectory well enough for downstream analysis.

#### Metrics

Primary raw metrics:
- `api_change_without_tests`
- `public_api_without_docs`
- `shared_change_isolated`
- `ownership_diffusion`
- `boundary_density`
- `cross_module_spread`
- `security_risk_score`
- `operability_score`

Derived summary:
- heuristic `conway_risk_proxy` over the raw channels above
- per-metric median delta and improved fraction

#### Implementation plan

1. `extract_pr_refinement_history.py`
   - emits one JSONL row per cumulative commit snapshot
   - aligns review events to adjacent commit intervals
2. `data/phase4_7_2_pr_refinement_history_summary.json`
   - aggregate first->final and post-review trend statistics
3. `docs/phase4_7_2_pr_refinement_history.md`
   - write up whether Conway risk declines through review-driven refinement

#### Success criteria

| Criterion | Target |
|-----------|--------|
| Median `conway_risk_proxy` first -> final | decreases |
| Post-review transitions improve more often than random commit transitions | yes |
| `api_change_without_tests` / `public_api_without_docs` improved fraction | >50% |

Initial readout (2-node Slurm smoke, 41 PRs / 239 snapshots / 25 review-response transitions):
- Mixed result, not yet a clean confirmation.
- `boundary_density` improved first->final on 61.0% of PRs, but the aggregate `conway_risk_proxy`
  did not improve consistently enough yet.
- Post-review transitions were especially noisy; surviving visible PR commit history is likely an
  incomplete proxy when authors squash or force-push away earlier revisions.

### Experiment 4.7.3 — Closed PR Corpus Ingestion (`python_js_ts_rust_closed_prs`)

#### Motivation

Exp 4.7.2 only used merged PRs from `prs_copy`, which makes `accepted` constant and weakens any
attempt to train a true merge-likelihood head from refinement history. The next step is to add
**closed/unmerged PRs** into a parallel table with a `prs_copy`-compatible schema so the same
feature extraction and steerer training code can operate on positive and negative outcomes.

#### Data source

Primary corpus:
- `/shared_workspace_mfs/akki/scratch_mfs/arthur-task/enriched-all-unmerged`

This is a repo-sharded JSONL export of closed PRs with:
- REST PR metadata (`title`, `body`, `base`, `head`, `patch_url`, `comments_url`, etc.)
- GraphQL enrichment (`pull_request.comments`, `reviews`, `reviewThreads`, `commits`,
  `closingIssuesReferences`, `changedFiles`, `additions`, `deletions`)
- top-level lightweight classifier outputs (`pr_type`, `is_feature`, `confidence`)

#### Core question

How much of the `prs_copy` schema can be:
1. recovered directly from the local JSONL,
2. recovered by light GitHub API fetches using `crawl_tokens.yaml`, and
3. not recovered at all from this corpus?

#### Target table

Create `python_js_ts_rust_closed_prs` with the same columns as `prs_copy` and upsert one row per closed PR.

Schema compatibility target:
- exact column names matching `prs_copy`
- same JSONB field shapes where feasible (`commits`, `comments`, `review_threads`, `file_patches`)
- nullable placeholders for unavailable SWE-bench-style fields

#### Recovery plan

Direct/derived from local JSONL:
- `repo`, `pull_number`, `repo_id`, `base_sha`, `head_sha`, `base_branch`, `head_branch`
- `pr_title`, `pr_body`, `pr_url`, `pr_state`, `pr_is_draft`, `pr_author`, `pr_labels`
- `created_at`, `updated_at`, `total_commits`, `commits`
- `total_comments`, `comments`
- `total_review_threads`, `review_threads`
- `requested_reviewers`, `submitted_reviews`
- `additions`, `deletions`, `changed_files`
- `linked_issues`, `closing_issue_id`
- `stars`, `forks`, `primary_language`
- `pr_category`, `pr_category_confidence`, `pr_category_reasoning`

Recovered by fetch using `crawl_tokens.yaml`:
- `patch` via `patch_url`
- `file_patches` via `GET /repos/{owner}/{repo}/pulls/{number}/files`
- repair partial `commits` payloads via `commits_url`
- repair partial `comments` payloads via `comments_url`

Expected null/unavailable in 4.7.3:
- `problem_statement`
- `hints_text`
- `pass_to_pass`
- `fail_to_pass`
- `merged_by`

#### Implementation plan

1. `ingest_python_js_ts_rust_closed_prs.py`
   - normalize the unmerged JSONL corpus to `prs_copy` columns
   - optionally fetch missing patch / file / comment artifacts
   - create `python_js_ts_rust_closed_prs` if absent
   - upsert rows idempotently on `(repo, pull_number)`

2. `docs/phase4_7_3_closed_pr_ingestion.md`
   - report per-column recoverability:
     - direct
     - fetchable
     - unavailable
   - quantify partial GraphQL tails (`commits`, `comments`, `reviewThreads`)

3. Re-run downstream patch-feature and steerer pipelines against:
   - `prs_copy` (merged)
   - `python_js_ts_rust_closed_prs` (closed)
   - combined merged + closed dataset

#### Success criteria

| Criterion | Target |
|-----------|--------|
| Closed PR rows inserted into `python_js_ts_rust_closed_prs` | >20k |
| Direct + fetchable coverage for core modeling columns (`patch`, `file_patches`, `commits`, `comments`) | >95% |
| Same downstream feature extractor can run on `python_js_ts_rust_closed_prs` without schema fork | yes |
| Acceptance head retrained on merged + closed PRs | enabled |

### Experiment 5.1 — RL on Steerer (Conway-Aware State Policy)

#### Motivation

Shift optimization from expensive token-level RL on the large coder to cheap RL on a
small steerer that predicts/improves **software-engineering state transitions**:
mergeability, review friction, refactor-demand risk, and architecture/ownership stress.

This tests the core thesis: high-level state-policy learning can deliver practical SE gains
without brute-force RL + sandboxing on the full code model.

**RAIM complementarity**: RAIM [arXiv:2603.01814] generates architecture-aware multi-design
candidates for repo-level feature additions by selecting among proposals using current module
structure and dependency graphs. This is powerful for local architectural consistency but is
fundamentally **static**: it ranks candidates by fit to the current codebase snapshot and
cannot predict which design choice will accumulate followup debt — repeated bugfixes, refactor
churn, review friction months later. The Exp 5.1 steerer is the longitudinal complement: it
ranks candidate trajectories by predicted followup-debt risk, providing the temporal dimension
static architecture analysis cannot reach. Together RAIM handles design diversity; SWE-JEPA
handles longitudinal quality selection.

**Why this matters for SPE**: An agent without architectural/organizational context is
analogous to an agent with no debugging tools in its scaffold — it can produce syntactically
valid code but has no mechanism to detect that the design will cause recurring failures
downstream. Just as removing tool-call access (bash, test runners) cripples an agent's ability
to validate correctness step by step, removing arch/org context cripples its ability to plan
for long-horizon SE tasks: full-feature implementation, refactoring at module boundaries, API
evolution. Having this context is the difference between a patch that closes an issue once and
a design that prevents the class of issue from recurring — the central concern of **Software
Performance Engineering (SPE)**.

#### Core hypothesis

Training a compact steerer with RL over PR/review state transitions yields better
agentic outcomes than static supervised steering alone, while retaining low training cost.

#### State/action/reward design

State channels (per trajectory step):
- PR/code scope: changed files, churn, cross-module spread
- Review signals: thread count, unresolved threads, refactor-request likelihood
- Conway proxies: ownership friction, interface stress, socio-technical alignment
- Runtime progress: tests attempted/passed, regression indicators
- **Longitudinal followup risk**: predicted followup-debt score from the Exp 4.3 region
  localization probe — probability that functions modified in this PR will be revisited in
  bugfix or feature-extension PRs, based on their JEPA signature embeddings. This is the
  signal RAIM lacks and the unique SWE-JEPA contribution to the state space. Note: quantifying
  this signal precisely is itself part of the research question; the Exp 4.3 probe provides a
  practical operational proxy while the reward design for dense, step-level steering remains
  open.

Steerer actions (policy outputs):
- steer hints and retry policy (scope tighten / split / naming/API consistency / test focus)
- attempt acceptance thresholding and reranking among candidate attempts

Reward (dense + terminal):
- positive: merge-likelihood increase, friction decrease, patch completion
- negative: rising refactor-risk, excessive scope drift, repeated stalled loops
- terminal: accepted-like proxy score, low-friction completion, or timeout/failure penalty

#### Signals to test (ablation matrix)

Train multiple steerer variants and compare:
1. `review-only` signals
2. `conway-only` signals (ownership friction + interface stress)
3. `review + conway` combined
4. `review + conway + jepa_latent` (full)

Goal: identify which proxies carry the strongest causal steering utility.

#### ThinkLogit-informed evaluation protocol

**Paradigm-level connection**: ThinkLogit (arXiv:2510.09354) proves that a small guider's
*reasoning delta* — `logit_guider − logit_guider_base` — can steer a large target model
without retraining the target, improving pass@1 by 26-29% using a 1.5B guider on a 32B
target. Exp 5.1 extends this principle one level of abstraction higher: a small steerer's
*SE-quality delta* guides the large coder's action distribution without SFT. The coder knows
how to write code; the steerer knows which design choices accumulate longitudinal debt. The
open methodological question this creates — what is the steerer-equivalent of ThinkLogit's
base-logit subtraction that isolates the SE-quality signal from the coder's unsteered
distribution? — is itself a contribution of this experiment.

Following ThinkLogit methodology, adapt success criteria to PR-agent runs:

1. Multi-sample metrics (`Avg@N` / `Pass@k` style):
   - For each PR task, run N attempts (e.g., N=8).
   - Report:
     - `Avg@8`: mean judge/proxy score across 8 attempts
     - `Patch@k`: whether any of top-k attempts produces non-empty valid patch
     - `AcceptProxy@k`: whether any of top-k reaches acceptance threshold

2. Sample efficiency curves:
   - x-axis: number of rollouts/attempts
   - y-axis: `AcceptProxy@k` or judge win-rate
   - Compare baseline vs steered to quantify efficiency gain.

3. Stability sweeps:
   - sweep steerer strength / threshold / warmup ratio (analogous to ThinkLogit control sweeps)
   - report mean ± std across seeds.

4. Pair-quality ablation for offline preference tuning:
   - Build preference pairs from target-only vs steered attempts.
   - Compare:
     - target-only pairs
     - steerer-only pairs
     - mixed-strength pairs (expected strongest, consistent with ThinkLogit finding)

#### Scope clarification (ORM-only)

Exp 5.1 uses the steerer as an **outcome reward model (ORM)** only — it scores the
terminal PR state after each attempt and uses those scores for attempt reranking and
early stopping. Step-level process rewards are out of scope for 5.1; that is addressed
in Exp 5.2.

#### Trajectory scaffold reuse

Trajectories from `data/phase4_7_trajectories_feature_sl80/` (16,002 files, format
`instance__prX__aY.traj.json`) were collected using `run_phase4_7_agentic_eval_steered.py`.
The same mini-swe-agent scaffold and trajectory format is used for Exp 5.1 rollouts.
The existing scaffold supports `--disable-steering` for the no-steerer baseline and
`--steerer-model` for the Conway ORM.

#### Experimental setup

- Fixed cohort: current 7k baseline subset (same PR keys as Exp 4.7).
- Agent scaffold: mini-swe-agent, same tools/model/timeouts as Exp 4.7.
- Compare:
  1. baseline prompt-only (`--disable-steering`)
  2. supervised steerer Exp 4.7 (6-feature PR-metadata model)
  3. Conway ORM steerer Exp 5.1 (28-feature, 100k-trained)
- Ablations: `review-only`, `conway-only`, `review+conway` feature sets via zeroing

#### Implementation plan

1. ✅ `build_pr_mdp_dataset_v51.py` — Conway proxy channels + transition rewards
2. ✅ `train_pr_steerer_rl_v51.py` — 4 trained heads (acceptance, refactor, value, pairwise)
3. ✅ `extract_conway_patch_features.py` — tree-sitter + regex, 28 features, 100k PRs
4. `run_phase5_1_agentic_eval.py` — multi-attempt eval, `Avg@8`/`Patch@k`/`AcceptProxy@k`
5. `analyze_phase5_1_signals.py` — proxy/signal ablations and sensitivity plots
6. ✅ report: `docs/phase5_1_rl_steerer_conway.md`

#### Success criteria

| Criterion | Target |
|-----------|--------|
| Conway ORM steerer > no-steerer baseline on `AcceptProxy@1` | +5 pp absolute |
| Conway ORM steerer > Exp 4.7 steerer on `AcceptProxy@k` | consistent improvement |
| `review + conway` > `review-only` | validates Conway signal adds value beyond metadata |
| Training cost | far below full-model RL/SFT |

#### References

- ThinkLogit: https://arxiv.org/abs/2510.09354

---

### Experiment 5.1.1 — Language/Framework-Level Conway Signal Robustness (Side Excursion)

**Not part of the mainline 5.1 → 5.2 experiment sequence.** Triggered when new
languages (Go, Kotlin) are added to `prs_copy`, or when per-language AUROC stratification
shows degradation for a specific language slice.

#### Motivation

Several Exp 5.1 signals are confounded by language-level syntax conventions:
- `has_pub_func`: fires on 65% of Rust PRs (`pub fn` is standard, not a deliberate
  API expansion decision) — different semantics from Go capitalized functions or Java `public`
- `has_try_catch`: structurally zero for all Go PRs (Go uses `if err != nil` not try/catch)
- `has_bare_except`: Python-only; no equivalent signal for Go/Java error swallowing
- `imp_external` / `trust_boundary_crossings`: Go intra-org imports are misclassified
  as external without `go.mod` prefix detection

Additionally, Design by Contract mechanisms (`assert`, `debug_assert!`, `require()`)
are confounded by build-mode semantics: Python `assert` is disabled with `-O`,
Rust `debug_assert!` is stripped in release builds, Java `assert` requires `-ea` flag.
A naive `has_assert` feature would conflate debugging aids with enforced runtime contracts.

#### Plan

1. Audit per-feature prevalence on language-stratified samples
2. Add Go-specific error handling signals (`has_go_err_wrap`, `has_go_err_discard`)
3. Add Go `go.mod` prefix scan for intra-org import reclassification
4. Add language dummies to steerer training or train per-language submodels
5. Establish language onboarding checklist for future prs_copy expansions
6. Evaluate DbC feature feasibility if build configuration data becomes available

Full catalogue of known confounds and proposed fixes:
`docs/phase5_1_1_conway_language_confounds.md`

---

### Experiment 5.2 — Process Reward Model with Potential-Based Shaping

#### Motivation

Exp 5.1 scores the terminal PR state after each attempt (ORM). This gives no guidance
*during* the agent's trajectory — the agent cannot learn that exploring architecture
files before editing, or scoping down cross-module changes mid-attempt, produces better
outcomes than the same terminal state reached blindly.

Exp 5.2 adds **step-level process reward** using potential-based reward shaping, which
is theoretically guaranteed not to corrupt the terminal reward signal.

#### Core design

Following Ng, Harada & Russell (ICML 1999), the process reward at each step is:

```
r_dense(t) = γ · Φ(s_{t+1}) − Φ(s_t)
```

where `Φ(s_t) = steerer.reward_estimate(belief_state_at_step_t)` — the Exp 5.1 ORM
applied to the agent's current best estimate of the PR's Conway features.

**Key property (Ng et al. 1999 Theorem 1)**: Any reward shaping of this potential-based
form leaves the set of optimal policies invariant. Adding `r_dense` to the terminal
reward accelerates learning without introducing spurious optima. No other additive
reward shaping function has this guarantee.

**Belief state**: At each step, the agent's observable state is the set of files read,
imports seen, and module paths touched so far. The Conway feature estimates are updated
incrementally as new tool outputs arrive (same logic as `extract_conway_patch_features.py`,
applied to partial observations). The belief state also carries an uncertainty term:
`Φ(belief_t) = steerer_score(belief_t) − λ · uncertainty(belief_t)`, incentivising
the agent to resolve architectural ambiguity rather than act under it.

#### Anti-hacking properties

The design explicitly rejects action-type rewards (calling exploration tools earns no
reward by itself) and reasoning-text rewards (writing "I am analyzing module boundaries"
earns no reward). Only verifiable state transitions — actual reductions in estimated
Conway risk — contribute to `r_dense`. An additional cap `Σ_t r_dense(t) ≤ |R_terminal|`
ensures bad terminal outcomes cannot be offset by high process scores.

#### Infrastructure delta from Exp 5.1

1. **Trajectory logger extension**: add per-step `(tool_name, tool_input_summary,
   belief_state_delta, Φ_before, Φ_after)` to the existing `.traj.json` format
2. **Belief state tracker**: incremental Conway feature estimation from partial
   tool outputs, reusing `extract_conway_patch_features.py` parsing logic
3. **Φ evaluator**: the trained Exp 5.1 ORM called at each step — cheap (linear model)

#### Comparison plan

- Fixed cohort: same 7k task set as Exp 5.1
- Compare:
  1. no-steerer baseline (from Exp 5.1 eval)
  2. Exp 5.1 ORM steerer (terminal scoring only)
  3. Exp 5.2 PRM steerer (terminal + step-level potential shaping)
- Primary metric: `AcceptProxy@k` and sample efficiency curve
- Secondary: trajectory analysis — does the PRM agent explore more architecture files
  early in its trajectory?

#### Success criteria

| Criterion | Target |
|-----------|--------|
| PRM > ORM on `AcceptProxy@1` | +3 pp absolute |
| PRM > ORM on sample efficiency (same AcceptProxy with fewer attempts) | clear curve separation |
| PRM agent explores more arch-relevant files early in trajectory vs ORM agent | measurable |
| No reward hacking evidence (repeated identical tool calls, keyword stuffing) | confirmed by trajectory audit |

#### References

- Ng, Harada & Russell (1999): *Policy Invariance Under Reward Transformations:
  Theory and Application to Reward Shaping*. ICML 1999.
  https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
- ThinkLogit: https://arxiv.org/abs/2510.09354 (paradigm-level connection: small
  guider's delta steers large model without retraining)

---

## Phase 6: Self-Supervised Latent Steerer — Implementation Plan

### Status as of 2026-03-20

Phase 5.1 validated that Conway-aware steering works (+21.4 pp F2P, +7.2 pp scaffold
judge advantage). Phase 6 replaces the 61 hand-crafted features with self-supervised
latent representations from the frozen teacher. See `swe-jepa-research-proposal.md`
Phase 6 section for full rationale.

### Experiment 6.1: Embedding-Based Steerer

**Objective**: Confirm frozen teacher embeddings carry ≥ steering signal as hand-crafted features.

#### Implementation steps

1. **Extract patch embeddings** — `extract_patch_embeddings.py`
   - Input: `prs_copy` + `python_js_ts_rust_closed_prs` (100k+ PRs with patch diffs)
   - For each PR:
     - Tokenize patch diff text with Qwen2.5-Coder-3B tokenizer
     - Run frozen teacher (layer 18), mean-pool → `z_patch` (2048-dim)
     - Optionally: tokenize 500-token surrounding context window, run teacher
       → `z_context` (2048-dim)
   - Output: `data/patch_embeddings_100k.npz` (z_patch, z_context, instance_ids)
   - Batched inference on 8 GPUs, ~4h for 100k patches

2. **Train acceptance/refactor heads** — `train_embedding_steerer.py`
   - Load `patch_embeddings_100k.npz`
   - Same train/val split as Phase 5.1 (GroupKFold by repo, 5 folds)
   - Models:
     - Logistic regression on `z_patch` (2048-dim)
     - Logistic regression on `[z_patch; z_context]` (4096-dim)
     - 2-layer MLP (2048 → 512 → 1) on `z_patch`
   - Targets: `accepted` (merged=1, closed=0), `refactor_requested`
   - Report: AUROC, PR-AUC per fold; compare to Phase 5.1 baselines (0.71 / 0.59)

3. **Ablation: PCA whitening**
   - Apply PCA to 256/512 dims before linear head (address potential isotropy issue)
   - Compare AUROC with/without whitening

4. **FeatBench steering eval** (if AUROC ≥ 0.71)
   - Replace 61-feature extraction in `steered_trae_agent.py` with embedding extraction
   - Run same 19-pair FeatBench eval with scaffold judges
   - Compare F2P and judge panel scores to Phase 5.1

#### Files to create/modify

| File | Purpose |
|------|---------|
| `extract_patch_embeddings.py` | Batch extraction of z_patch, z_context from frozen teacher |
| `train_embedding_steerer.py` | Train acceptance/refactor heads on embeddings |
| `data/patch_embeddings_100k.npz` | Cached embeddings (output) |
| `docs/phase6_1_embedding_steerer.md` | Results writeup |

#### Dependencies

- Frozen Qwen2.5-Coder-3B checkpoint (same as Phase 0–3)
- `prs_copy` + `python_js_ts_rust_closed_prs` tables with patch diffs
- Phase 5.1 feature baselines for comparison

---

### Experiment 6.2: Contrastive Projection

**Objective**: Learn a compact merge-readiness subspace via contrastive learning on
merged/closed pairs.

#### Implementation steps

1. **Build contrastive pairs** — `build_contrastive_pairs.py`
   - For each repo with both merged and closed PRs:
     - Positive pairs: (merged_patch_i, merged_patch_j) from same repo
     - Negative pairs: (merged_patch_i, closed_patch_k) from same repo
     - Hard negatives: closed patches touching same files as merged patches
   - Output: `data/contrastive_pairs_100k.jsonl`

2. **Train projection head** — `train_contrastive_steerer.py`
   - Architecture: `z_patch (2048) → Linear(2048, 512) → GELU → Linear(512, 256)`
   - Loss: InfoNCE with temperature τ=0.07
   - Training: repo-stratified split, 50 epochs, lr=1e-4, batch=512
   - Output: projection weights + per-fold AUROC on held-out repos

3. **Evaluate**
   - Score candidate patches by cosine to repo's merged-patch centroid in projection space
   - Compare AUROC to Exp 6.1 logistic baseline
   - Visualize t-SNE of projected embeddings colored by merged/closed

#### Files to create

| File | Purpose |
|------|---------|
| `build_contrastive_pairs.py` | Pair construction from prs_copy + python_js_ts_rust_closed_prs |
| `train_contrastive_steerer.py` | InfoNCE projection head training |
| `docs/phase6_2_contrastive_steerer.md` | Results |

---

### Experiment 6.3: Structural Surprise (Fully Self-Supervised)

**Objective**: Train a predictor on git history with zero labels. Use prediction error as
the steering signal.

#### Implementation steps

1. **Build commit-level training set** — `extract_commit_context_targets.py`
   - For each non-trivial commit in the 150-repo corpus (>5 lines changed, non-merge):
     - Extract pre-patch context: 2048-token window from surrounding files
       (imports, adjacent functions, module docstrings, test headers)
     - Extract post-patch state: modified files after commit
     - Run frozen teacher on both → `z_context` (2048-dim), `z_post_patch` (2048-dim)
   - Output: `data/commit_context_targets_500k.npz`
   - Target: ~500k commit pairs from 150 repos

2. **Train predictor** — `train_structural_predictor.py`
   - Architecture: Phase 1.2 transformer encoder (~45M params)
     - 6-layer transformer, 512 hidden, 8 heads over tokenized context
     - 2-layer cross-attention predictor → `z_predicted` (2048-dim)
   - Loss: SmoothL1(`z_predicted`, `z_post_patch`)
   - Training: repo-stratified split, DDP on 8 GPUs
   - Track: val loss, cosine similarity, structural surprise distribution

3. **Validate against merge outcomes** (labels for eval only)
   - Compute `surprise = ||z_predicted - z_actual||` for merged and closed PRs
   - Statistical test: Mann-Whitney U on surprise(merged) vs surprise(closed)
   - Plot surprise distribution by outcome

4. **FeatBench steering integration**
   - At agent inference time: run frozen teacher + predictor on candidate patch
   - Compute structural surprise
   - Map surprise to steering actions:
     - Low surprise (< p25): submit
     - Medium (p25–p75): light hints (test coverage, documentation)
     - High (> p75): strong hints (scope tighten, module boundary, split patch)
   - Run full FeatBench eval and compare to Phase 5.1

5. **Generalization test**
   - Hold out 20 repos entirely from training
   - Verify surprise still separates merged/closed on held-out repos
   - Verify FeatBench steering works for instances from held-out repos

#### Files to create

| File | Purpose |
|------|---------|
| `extract_commit_context_targets.py` | Build (context, post-patch) pairs from git history |
| `train_structural_predictor.py` | Train JEPA predictor on commit pairs |
| `compute_structural_surprise.py` | Score patches by prediction error |
| `data/commit_context_targets_500k.npz` | Cached training data |
| `docs/phase6_3_structural_surprise.md` | Results |

#### Dependencies

- Phase 1.2 transformer encoder architecture (reuse code from `train_student.py`)
- Phase 3.0 target computation trick (full sig+body context for teacher targets)
- 150-repo corpus with git history access
- FeatBench eval infrastructure from Phase 5.1

---

### Phase 6 timeline

| Experiment | Estimated effort | GPU-days | Gate |
|-----------|-----------------|----------|------|
| 6.1 Embedding steerer | 1 week | 1 | AUROC ≥ 0.71 to proceed |
| 6.2 Contrastive projection | 1 week | 2 | AUROC > 6.1 baseline |
| 6.3 Structural surprise | 2-3 weeks | 4-8 | surprise(merged) < surprise(closed) at p<0.01 |
| 6.3 FeatBench eval | 1 week | 0 (inference) | F2P ≥ Phase 5.1 |

Total: ~4-6 weeks, 7-11 GPU-days. 6.1 is the critical gate — if embeddings don't match
the feature baseline, the premise fails and 6.2/6.3 should be deprioritized.
