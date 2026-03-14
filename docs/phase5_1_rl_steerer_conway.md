# Experiment 5.1 — Conway-Aware RL Steerer

**Date**: 2026-03-10
**Status**: Training complete; agentic eval pending
**Model files**: `data/phase5_1_pr_steerer_model_v51_100k_conway.json`

---

## Overview

Experiment 5.1 trains a compact steering model to predict review friction, acceptance
likelihood, and longitudinal reward from PR-level features. The steerer is designed to
guide a large coding agent at decision time — not by retraining the agent, but by scoring
candidate actions and patch states against real-world review dynamics.

The central question: **do deep code-level signals grounded in Conway's Law (import
trust boundaries, error contract quality, ownership friction) predict review outcomes
significantly better than surface-level PR metadata alone?**

---

## What was built

### 1. Extended MDP dataset (`build_pr_mdp_dataset_v51.py`)

Each PR in `prs_copy` becomes a single state-action-reward transition:

```
s_t  = PR features at submission time
a_t  = "submit_or_update_pr" (single action per row)
s_t1 = review outcome: threads, comments, refactor requests, acceptance
r    = (1.0 if accepted else -1.0) - 0.2 * review_friction - 0.1 * refactor_requested
```

Eligibility filter: `changed_files ∈ [1, 120]`, `additions + deletions ∈ [5, 8000]`,
`patch IS NOT NULL`, `file_patches IS NOT NULL`.

### 2. Conway patch feature extractor (`extract_conway_patch_features.py`)

Tree-sitter AST parsing across 10 languages (Python, JavaScript, TypeScript, Go, Rust,
Java, Ruby, PHP, Kotlin, C++) combined with 28 regex pattern categories. Runs in
parallel with 64 threads. Key implementation notes:

- Patches truncated to 300 KB and added lines to 500 chars before regex scanning
  to prevent catastrophic backtracking on minified JS and large diffs (a single
  unconstrained `.*X.*Y` regex on a 100k-char line is O(n²) and holds the Python GIL)
- 99,923 / 100,000 PRs extracted successfully (77 recursion depth errors on pathological
  inputs, handled gracefully)

### 3. Steerer training (`train_pr_steerer_rl_v51.py`)

Four trained heads on the joined dataset:

| Head | Type | Target |
|------|------|--------|
| `acceptance` | Logistic regression | Binary: PR merged |
| `refactor` | Logistic regression | Binary: refactor language in review |
| `reward_model` | Ridge regression | Scalar reward r |
| `pairwise_reward_model` | Bradley-Terry (logistic on Δfeatures) | Pairwise preference |

Bradley-Terry training uses within-repo pairs where `|r_i − r_j| > 0.3`, with both
directions (winner→loser and loser→winner) to give balanced 50/50 labels.

---

## Feature set (28 signals in `s_t`)

### Group 1: PR scope (6 original features)

| Feature | Description |
|---------|-------------|
| `is_draft` | PR marked as draft at submission — indicates incomplete intent |
| `changed_files` | File count (log1p transformed) |
| `additions` | Lines added (log1p) |
| `deletions` | Lines deleted (log1p) |
| `requested_reviewers_count` | Reviewer count at open time (log1p) |
| `has_closing_issue` | Linked issue present at open time — signals scoped, deliberate work |

### Group 2: PR-level Conway proxies (4 v51 features)

| Feature | Description |
|---------|-------------|
| `cross_module_spread` | Distinct top-level directories touched — rough ownership friction |
| `has_tests` | Test file present in PR — coverage discipline proxy |
| `churn_asymmetry` | deletions / (additions + deletions + 1) — refactor vs pure addition |
| `followup_risk` | From Exp 4.3 probe cache per instance_id (0.0 for most; only 14 matched) |

### Group 3: Import trust-boundary signals (tree-sitter, 2 features)

| Feature | Description |
|---------|-------------|
| `imp_external` | Count of new third-party (non-stdlib, non-relative) imports added — each is a new upstream actor the org doesn't own |
| `imp_relative` | Count of intra-repo relative imports added — refactoring signal |

Import classification is per-language using curated stdlib sets. External imports are
anything that is not in the stdlib set and not a relative/local path. These are the
imports reviewers scrutinize for error handling, memory behaviour, and security.

### Group 4: Compound Conway scores (4 features)

| Feature | Formula | Meaning |
|---------|---------|---------|
| `trust_boundary_crossings` | `(imp_external>0) + (dep_file∧imp_external>0) + has_http_client + has_db_client + has_queue_client` | Count of distinct new external actors introduced |
| `error_contract_score` | `+try_catch +finally +raise_from +log_in_except +reraise − 2×bare_except − 3×except_pass` | Error handling quality: positive = careful, negative = silencing errors |
| `security_risk_score` | `+2×shell_true +2×sql_fstring +2×eval_exec +2×pickle_loads +3×hardcoded_cred +cred_in_log` | Security anti-pattern density |
| `operability_score` | `+2×metric_emit +log_warn_err +health_check +pool_config − 2×ext_client_no_obs − ext_client_no_log` | Observability quality: penalises external clients without monitoring |

> **Candidate extension — Design by Contract / self-checking mechanisms**: whether
> assertions, precondition checks, and invariant guards (`assert`, `Preconditions.checkArgument`,
> `Objects.requireNonNull`, Rust `debug_assert!`, property-based test annotations) predict
> PR merge outcomes is an open question. Initial hypothesis: explicit contracts signal defensive
> programming discipline similar to `has_try_catch`, but are confounded by language build modes
> (`assert` is disabled in Python `-O` and Java production JVMs; Rust `debug_assert!` is stripped
> in release builds). Soundly measuring this requires knowing the project's build configuration,
> which is not available in `prs_copy`. Tracked as part of Exp 5.1.1 language-level confound
> analysis; not included in the current feature set.

### Group 5: Conway binary signals (12 features)

**Error handling**

| Feature | What triggers review scrutiny |
|---------|-------------------------------|
| `has_try_catch` | Exception handling added — reviewers check completeness |
| `has_bare_except` | `except:` with no type — silences all exceptions including KeyboardInterrupt |
| `has_open_no_with` | `open(...)` without context manager — resource leak risk |
| `has_sql_fstring` | f-string or `%`-formatted SQL — SQL injection risk |

**External clients**

| Feature | What triggers review scrutiny |
|---------|-------------------------------|
| `has_db_client` | New database client — reviewers check connection pooling, error handling, memory |
| `has_http_client` | New HTTP client — reviewers check timeout, retry, circuit-breaker |
| `ext_client_no_obs` | External client (DB/HTTP/queue) with no metric_emit or health_check |

**Interface and schema**

| Feature | What triggers review scrutiny |
|---------|-------------------------------|
| `has_pub_func` | New public function/method export — API contract change, backwards compat |
| `has_schema_change` | ALTER TABLE / ADD COLUMN / ORM field — data contract, migration risk |
| `has_hardcoded_cred` | Literal `password=`, `api_key=` etc. — security red flag |
| `has_metric_emit` | Prometheus/StatsD/OpenTelemetry counter/gauge — infra coupling |
| `modifies_shared_util` | Edits to `utils/`, `common/`, `lib/`, `core/` paths — high blast radius |

---

## Results

### Dataset statistics

| Dataset | PRs | Repos | Accepted | Friction | Pairs |
|---------|-----|-------|----------|----------|-------|
| 30k baseline | 30,000 | 2,685 | 94.9% | 22.5% | 34,122 |
| 30k + Conway | 29,958 joined | 2,685 | 94.9% | 22.5% | 34,122 |
| 100k + Conway | 99,923 joined | 4,407 | 92.8% | 22.0% | 111,772 |

Join coverage on (repo, pull_number): **99.1%** for both scales.

### Model performance (5-fold cross-validation)

| Model | Acceptance AUROC | Refactor AUROC | Value R² | Value Spearman ρ |
|-------|-----------------|----------------|----------|-----------------|
| 30k, no Conway (6 scope features) | 0.769 | 0.644 | 0.122 | 0.004 |
| 30k + Conway (28 features) | **0.847** | 0.644 | 0.137 | 0.033 |
| 100k + Conway (28 features) | 0.808 | **0.660** | **0.174** | **0.141** |

**Conway features add +7.8pp acceptance AUROC** on 30k (0.769 → 0.847).

**Scaling from 30k → 100k** reduces acceptance AUROC (−3.9pp) but substantially
improves reward prediction (value ρ: 0.033 → 0.141, a 4.3× improvement). The
acceptance drop is expected: the 30k dataset is the most-recent, temporally
homogeneous slice; adding 70k older PRs introduces distribution shift in acceptance
dynamics across different repo eras and cultures. But the reward/value head benefits
from the additional diversity — reward prediction is more general than acceptance
prediction.

The 100k model is chosen for the RL phase because **reward prediction quality drives
RL update quality**, and value Spearman ρ = 0.141 vs 0.033 means 4.3× better
step-level guidance signal.

### Feature importance (from acceptance head coefficients, 100k model)

The trained model's strongest positive predictors (higher → more likely accepted):
- `has_tests` — test coverage discipline is the strongest single signal
- `has_closing_issue` — PR scoped to a tracked issue
- `requested_reviewers_count` — explicit reviewer assignment signals process maturity
- `churn_asymmetry` — refactoring PRs (high deletions ratio) accepted more readily

Strongest negative predictors (higher → more likely to face friction or rejection):
- `trust_boundary_crossings` — each additional external actor reduces acceptance odds
- `changed_files` — large scope increases review burden
- `has_bare_except` — error-silencing is a near-universal review blocker
- `security_risk_score` — security anti-patterns trigger mandatory change requests

### Conway signal analysis (100k dataset)

**Trust boundary crossings — dose-response**

The clearest causal signal. Each additional external actor (new import, DB client,
HTTP client) adds 5–8pp friction probability:

| TBC count | n PRs | Friction | Acceptance |
|-----------|-------|----------|------------|
| 0 | 56,268 | 16.8% | 91.8% |
| 1 | 36,647 | 27.8% | 94.6% |
| 2 | 6,402 | 32.1% | 92.0% |
| 3 | 558 | 37.1% | 88.0% |
| 4 | 47 | 44.7% | 83.0% |

Note: TBC=1 has slightly higher acceptance than TBC=0. These are well-scoped PRs that
introduce exactly one new integration (common for feature PRs that are professionally
crafted). The friction at TBC=1 rises because those integrations demand scrutiny. At
TBC≥2, both friction and rejection rise together.

**Top binary signals by friction lift (100k)**

| Signal | n PRs | Friction lift | Acceptance | Interpretation |
|--------|-------|--------------|------------|----------------|
| `has_pickle_loads` | 82 | **2.06×** | 81.7% | Deserialization of untrusted data → security/safety discussion always triggered |
| `has_shell_true` | 30 | **1.97×** | 73.3% | `subprocess(..., shell=True)` → injection risk, strong rejection signal |
| `has_version_suffix` | 501 | 1.72× | 95.8% | `v2`/`_v3` in function names → deprecation and migration discussions |
| `has_metric_emit` | 1,400 | 1.66× | 96.4% | Observability paradox: metric additions accompany complex infra PRs |
| `has_hardcoded_cred` | 760 | 1.63× | 91.3% | Literal credentials → security review mandatory |
| `has_schema_change` | 2,557 | 1.54× | 94.9% | DB schema changes → migration review, rollback planning |
| `has_thread_spawn` | 601 | 1.53× | 81.4% | New thread/goroutine/task → concurrency safety review |
| `has_db_client` | 1,651 | 1.51× | 92.2% | New DB connection → pooling, error handling, memory scrutiny |

**Interesting trend: observability paradox**
`has_metric_emit` has the 4th highest friction lift (1.66×) yet high acceptance (96.4%).
This is not contradictory: PRs that add Prometheus/OpenTelemetry counters are inherently
complex infrastructure changes (new client + config + instrumentation together), which
attract more review threads. But they are well-crafted PRs — the author thought about
operability — so they get merged. The steerer correctly learns `metric_emit` → higher
reward (good practice) but also → more friction (complex infra), placing these PRs in
the high-reward, high-effort quadrant.

**Interesting trend: `has_raise_from` acceptance = 80.2%**
PRs that add `raise X from Y` (Python exception chaining) have substantially lower
acceptance (80.2% vs 92.8% base). These typically appear in deep refactors of error
handling layers — exactly the kind of architectural change that reviewers push back on.
The chaining syntax itself signals "I'm restructuring how errors propagate" which is a
red flag for scope in otherwise small PRs.

**Strongest continuous predictors (Spearman ρ vs friction, 30k)**

| Feature | ρ vs friction |
|---------|--------------|
| `imp_total_new` | +0.191 |
| `imp_external` | +0.184 |
| `trust_boundary_crossings` | +0.168 |
| `has_pub_func` | +0.163 |
| `imp_relative` | +0.152 |
| `n_files` | +0.146 |
| `cross_module_spread` | +0.116 |

`imp_external` (raw count of new third-party imports) and `trust_boundary_crossings`
(compound score) are the two strongest Conway-derived predictors. Both are derived
from tree-sitter AST analysis — they would be invisible to a model using only PR
metadata labels.

---

## What the steerer does at inference time

At deployment, the steerer takes the current observable PR state and outputs:

```python
{
    "acceptance_prob": 0.91,     # P(merged | current state)
    "refactor_prob":  0.08,      # P(refactor requested | current state)
    "reward_estimate": 0.74,     # predicted reward r
    "pairwise_logit": 1.23,      # preference score vs comparison state
}
```

In the agentic eval (`run_phase5_1_agentic_eval.py`), these scores are used to:
1. Rank candidate patches before submission (pairwise logit)
2. Gate retry decisions (if acceptance_prob < threshold, retry rather than submit)
3. Guide ablation studies: compare `review-only`, `conway-only`, `review+conway` feature
   sets by zeroing non-active features (preserving coefficient alignment)

---

## Limitations and known confounds

**`has_pub_func` in Rust**: 65% of Rust PRs trigger this feature (because `pub fn` is
standard Rust visibility syntax). The model partially learns `Rust → high pub_func` as
a language confound. Consistent at inference time (agent-generated Rust code also uses
`pub fn`) but weakens the causal interpretation.

**`followup_risk` near-zero coverage**: Only 14 / 30,000 instance_ids matched the
Exp 4.3 probe cache. This feature is zero for virtually all training examples and
contributes nothing to the current steerer. A richer longitudinal signal would require
running the Exp 4.3 region localization probe at full scale.

**Temporal distribution shift**: The 30k model trains on the most recent 30k PRs;
the 100k model adds older PRs. Acceptance dynamics differ across repo eras (older
repos may have different review cultures). This is partly responsible for the −3.9pp
acceptance AUROC regression at 100k.

**ORM only, no step-level signal**: The steerer evaluates terminal PR state. It cannot
distinguish a good patch that was arrived at via a thoughtful exploration of the
architecture from one that stumbled into the same final state. Step-level process
reward is not yet implemented (see "Next steps" below).

---

## Experiment 5.1.2 — LLM-Judged Refactor Labels & Org-Specific Preferences

### Motivation

The regex-based refactor labels (`REFACTOR_RE`, `REFACTOR_SQL_REGEX`) miss semantically
diverse refactor requests ("use X instead", "this should be a separate function") and
produce false positives on comments that mention refactoring without requesting it.

We replaced regex labeling with LLM-judge labeling (`label_refactor_llm.py`) using
`qwen3.5_35b_a3b` across 5 replicas (2000 concurrency), classifying each review thread
for `refactor_requested` (bool) and `refactor_scope` (function / module / library_component).

### LLM Labeling Results (prs_copy)

| Metric | Value |
|--------|-------|
| PRs labeled | 41,376 |
| Refactor rate | 45.6% |
| Scope: function | 13,549 |
| Scope: module | 3,680 |
| Scope: library_component | 1,625 |
| Error threads | 17 / 145,224 (0.01%) |
| Throughput | ~62 PRs/s (~11 min total) |

### Retrained Steerer Metrics

**Binary refactor_requested (LLM labels)**

| Metric | Train | CV Mean |
|--------|-------|---------|
| AUROC | 0.670 | 0.663 |
| PR-AUC | 0.304 | 0.298 |

**Ordinal scope (none=0, function=1, module=2, library_component=3)**

| Metric | Train | CV Mean |
|--------|-------|---------|
| AUROC (weighted OVR) | 0.649 | 0.642 |
| PR-AUC (weighted) | 0.764 | 0.762 |

### Correlation with Conway Features

Most Conway features show weak univariate correlation with refactor labels, but the
multivariate model reveals conditional structure after controlling for patch size:

| Feature | r_pb (binary) | rho (scope) | Model coef (binary) |
|---------|--------------|-------------|---------------------|
| new_func_defs | **+0.174** | **+0.359** | +0.356 |
| imp_relative | **-0.072** | **-0.112** | — |
| additions | -0.017 | +0.051 | **+0.473** |
| changed_files | -0.020 | +0.013 | **-0.291** |
| cross_module_spread | -0.014 | +0.020 | **+0.100** |
| trust_boundary_crossings | -0.001 | +0.028 | +0.047 |

The scope model reveals directional structure invisible to the binary model:

| Feature | none | function | module | lib_component |
|---------|------|----------|--------|---------------|
| cross_module_spread | -0.098 | -0.009 | -0.029 | **+0.136** |
| additions | **-0.400** | +0.053 | **+0.208** | +0.140 |
| new_func_defs | -0.295 | +0.039 | +0.078 | **+0.177** |
| imp_total_new | -0.114 | -0.172 | **+0.138** | **+0.149** |

Key finding: `cross_module_spread` specifically predicts **library-component-level**
refactors (+0.136), not function-level ones. This aligns with the Conway thesis —
cross-team boundary friction generates broader restructuring demands, not local cleanup.

### Interpretation: Static Features as Prior, Org Preferences as Posterior

Moderate signal is recoverable from patch-level and static features (CV AUROC ~0.66),
but strong prediction requires learning reviewer preferences and org-specific processes.

This ceiling is not a data limitation. It reflects the fact that refactor requests are
driven by organizational structure and review culture that patch features alone cannot
capture. The connection to Conway's thesis is direct:

1. **The org chart is selected first.** Before any code is written, the organization's
   structure — teams, reporting lines, ownership boundaries — is established.

2. **Task subdivision narrows the design space.** Once work is delegated, the
   decomposition of the system mirrors the org structure. Each sub-delegation prunes
   the set of possible designs by making alternatives unpursuable. A reviewer asking
   for a function extraction, a module split, or a component boundary change is
   enforcing the org's structural priors on the code.

3. **Preferences are org-specific.** Whether a 200-line function should be split,
   whether a utility belongs in `common/` or stays local, whether an HTTP client
   needs a circuit breaker — these decisions vary by team. The static steerer learns
   the *average* org's preferences. Closing the gap to strong prediction requires
   RL/RLHF-style adaptation that learns each org's unique review culture.

4. **Legacy/deprecated code carries over additively.** As a consequence of software requirements
   and rapid iteration, deprecated code (or code clones) is likely to linger across interface boundaries.
   As the org and delegations evolve, code tends to compound additively, without refactor/ clean rewrites.

The static steerer provides a useful **prior** (e.g., "large PRs introducing new
functions across module boundaries are likely to draw refactor requests"). The
**posterior** — strong, actionable guidance — requires adaptation to the specific
org's processes, which is the role of the RL fine-tuning stage.

### How to Reproduce

```bash
source .venv/bin/activate

# 1. LLM labeling (any table: prs_copy, go_prs, go_prs_closed)
python label_refactor_llm.py --table prs_copy --concurrency 2000

# 2. Train binary refactor steerer
python train_pr_steerer.py \
  --llm-refactor-labels data/refactor_labels_llm_prs_copy.jsonl

# 3. Train scope steerer
python train_pr_steerer.py \
  --llm-refactor-labels data/refactor_labels_llm_prs_copy.jsonl \
  --refactor-scope-target \
  --model-out data/phase4_7_pr_steerer_model_scope.json \
  --metrics-out data/phase4_7_pr_steerer_metrics_scope.json

# 4. Language-specific (e.g., TypeScript)
python label_refactor_llm.py --table prs_copy --concurrency 2000  # if not already done
python train_pr_steerer.py \
  --llm-refactor-labels data/refactor_labels_llm_prs_copy.jsonl \
  --language typescript \
  --model-out data/pr_steerer_model_typescript.json \
  --metrics-out data/pr_steerer_metrics_typescript.json

# 5. Rebuild corpora with LLM labels
python build_go_pr_steerer_corpus.py --llm-labels data/refactor_labels_llm_go_prs.jsonl
python build_pr_mdp_dataset_v51.py --llm-labels data/refactor_labels_llm_prs_copy.jsonl
```

### Files

| File | Role |
|------|------|
| `label_refactor_llm.py` | LLM judge for review thread refactor classification |
| `data/refactor_labels_llm_prs_copy.jsonl` | LLM labels (41k PRs) |
| `data/phase4_7_pr_steerer_model.json` | Binary refactor steerer (LLM labels) |
| `data/phase4_7_pr_steerer_model_scope.json` | Ordinal scope steerer |

### Notes

- `label_refactor_llm.py` requires `chat_template_kwargs: {enable_thinking: False}` for
  qwen3.5_35b_a3b (thinking mode puts output in `reasoning_content`, leaving `content` null).
- The `--resume` flag skips PRs already in the output file, enabling interrupted runs to continue.
- LLM labels include `s_t1.refactor_requested` for direct compatibility with `_load_refactor_labels()`.

---

## Next steps

### Immediate
- Run agentic eval (`run_phase5_1_agentic_eval.py`) on the 7k baseline task set
- Compare: baseline prompt-only vs supervised steerer (Exp 4.7) vs 100k Conway steerer
- Compute `Avg@8`, `Pass@k`, `AcceptProxy@k` across ablation feature sets
- Run LLM labeling for TypeScript and other languages; compare per-language steerer metrics

### Process reward model (future)
The current steerer is an **outcome reward model (ORM)** — it scores the terminal state.
For step-level guidance, a **process reward model (PRM)** is needed.

Clean design using **potential-based reward shaping** (Ng et al., 1999):

```
r_dense(t) = γ · Φ(s_{t+1}) − Φ(s_t)
```

where `Φ(s_t) = steerer.reward_estimate(belief_state_at_step_t)`. The belief state
is updated incrementally as the agent observes files, imports, and module structure
during its trajectory.

Key anti-hacking properties of this design:
- **Theoretical guarantee**: potential-based shaping cannot change the optimal policy —
  it only speeds convergence to the same terminal optimum
- **State-grounded**: rewards require actual reductions in the Conway risk estimate, not
  just calling tools or writing arch-related text in thinking tokens
- **Cap**: total process reward bounded by `|R_terminal|` so bad outcomes remain bad

The ORM trained here becomes the `Φ` function for the PRM, making the 100k Conway
steerer the foundation for the next training stage.

---

## Files

| File | Description |
|------|-------------|
| `build_pr_mdp_dataset_v51.py` | MDP dataset builder with PR-level Conway proxies |
| `extract_conway_patch_features.py` | Tree-sitter + regex Conway signal extractor |
| `train_pr_steerer_rl_v51.py` | Steerer training (acceptance, refactor, value, pairwise heads) |
| `run_phase5_1_agentic_eval.py` | Agentic eval with ablation feature sets |
| `data/phase5_1_pr_mdp_dataset_v51_100k_conway.jsonl` | 100k training set (28 features) |
| `data/phase5_1_pr_steerer_model_v51_100k_conway.json` | Trained model (4 heads) |
| `data/phase5_1_pr_steerer_metrics_v51_100k_conway.json` | CV metrics |
| `data/conway_patch_features_100k.jsonl` | Raw per-PR Conway features (100k) |
| `data/conway_patch_features_summary_100k.json` | Friction/acceptance lift table |
| `label_refactor_llm.py` | LLM judge for review thread refactor classification |
| `data/refactor_labels_llm_prs_copy.jsonl` | LLM refactor labels (41k PRs) |
| `data/phase4_7_pr_steerer_model_scope.json` | Ordinal scope steerer model |
