# Experiment 7.1 — `h`-Full Student Steerer Plan

## Goal

Implement a **Trae-agent-compatible scaffold steerer** that keeps the successful
phase 6.2 v3 control loop, but replaces the hand-built patch review logic with
the trained phase 7.1 student checkpoint:

- checkpoint: `data/phase7_1/review_state_student_h_full_canonical.pt`
- metrics: `data/phase7_1/review_state_student_h_full_canonical_metrics.json`

This is the minimal "student replaces steerer" experiment:

1. Trae-agent solves the task normally.
2. The student reads the produced `patch.diff`.
3. The student predicts review-state signals.
4. A symbolic bridge converts those signals into concrete review feedback.
5. Trae-agent runs one targeted review pass with the same prompt injection
   structure as v3.

The key point is that **no learned text decoder is required**. The student only
predicts state. A deterministic bridge turns that state into the same kind of
promptable review issues already used by the v3 steerer.

## Operational Note

Before starting a FeatBench run, start the stale-container reaper in the
background:

```bash
bash /shared_workspace_mfs/arthur/coder/eval/FeatBench/docker_agent/scripts/kill_stale_containers.sh
```

This script removes FeatBench worker containers with `_w` in the name once they
have been running for over `1.5` hours. It helps clear long-stuck Conan-style
workers so the eval loop can continue making progress without manual cleanup.

---

## Reference Design: Phase 6.2 v3

The reference implementation is
`eval/FeatBench/docker_agent/agents/steered_trae_agent_v3.py`.

Its runtime flow is:

1. Run baseline Trae-agent with the clean problem statement.
2. Read `patch.diff` from the host swap directory.
3. Detect review issues from the generated patch.
4. If no issues are found, keep the first patch.
5. Otherwise build a review prompt containing:
   - the original problem
   - the previous patch
   - a short numbered issue list
6. Run Trae-agent a second time with that review prompt.
7. If the second pass fails or writes an empty patch, restore the original patch.

Important design choices to preserve:

- **Run 1 stays baseline-identical.**
- **The review pass is grounded in the actual produced patch.**
- **Prompt injection happens only in the second pass.**
- **The agent wrapper stays host-side; no container-side model changes are needed.**

So phase 7.1 should replace only the step that turns `patch.diff` into issue
messages. The review prompt builder and second-pass control flow should remain
v3-compatible.

---

## Available Student Signals

The trained checkpoint bundle already contains everything needed for runtime
inference:

- `state_dict`
- `tag_names`
- `cluster_label_values`
- `cluster_hints`
- model hyperparameters (`hash_vocab_size`, `max_tokens`, `embed_dim`,
  `hidden_dim`, `latent_dim`)

The student predicts four heads from the patch diff:

1. **Latent head**
   - predicts the phase 6.2 `h` representation
   - useful for logging and later nearest-neighbor analysis
   - not required for the minimal v7.1 prompt bridge

2. **Cluster head**
   - predicts a class index over the 21 canonical super-clusters
   - mapped back to original cluster ids through `cluster_label_values`

3. **Acceptance head**
   - predicts whether the teacher considered the patch accepted / clean
   - test AUROC from training run: `0.851`

4. **Tag head**
   - predicts 9 deterministic review tags
   - test macro F1 from training run: `0.124`

The current tag set is:

- `patch_too_large`
- `shared_without_tests`
- `api_without_tests`
- `eval_exec`
- `sql_fstring`
- `hardcoded_credential`
- `bare_except`
- `except_pass`
- `http_without_timeout`

These tags already align with promptable review feedback in
`experiment_7/review_state_bridge.py`.

---

## How Student Signals Map To Prompt Injection

### Core principle

Use a **symbolic bridge**, not a learned decoder:

`patch.diff -> student -> predicted state -> deterministic issue strings -> v3 review prompt`

That means the student does not have to write natural language. It only has to
predict structured signals that the bridge can map to text.

### Runtime mapping

| Student signal | Runtime use | Prompt mapping |
|---|---|---|
| `tag_probs[tag]` | Concrete issue detection | `render_review_messages()` turns high-probability tags into review bullets |
| `cluster_logits` | Historical pattern prior | predicted cluster id is mapped through `cluster_hints` into a risk message |
| `accept_prob` | Gate for whether to trigger review pass | low acceptance can force a review pass even if no single tag crosses threshold |
| `latent_hat` | Diagnostics only in v7.1 | log only; do not use in first implementation |

### Recommended minimal bridge

1. Read `patch.diff`.
2. Run student inference on the patch text.
3. Build issue list from:
   - high-confidence tag messages
   - high-risk cluster hint, if present
4. If the acceptance score is low but the issue list is empty:
   - use the top 1-2 predicted tags above a lower fallback threshold
   - optionally add one generic cluster-risk message
5. Feed those issues into the existing v3 review prompt builder.

This preserves the v3 UX:

- numbered issues
- previous patch included inline
- instruction to make targeted fixes, not rewrite from scratch

### Why this is the right minimal design

- It directly tests whether the **student can replace the steerer's review-state
  estimator**.
- It avoids the harder problem of generating free-text rationale from scratch.
- It keeps the successful v3 review-pass prompt structure unchanged, which makes
  the comparison cleaner.

---

## Proposed Runtime Policy

### Review trigger

Use the student to decide whether a second pass should run:

- trigger if any tag probability is `>= 0.50`
- or trigger if predicted cluster has a `HIGH` risk tier
- or trigger if `accept_prob <= 0.65`

This uses the stronger head for gating:

- the acceptance head is the most accurate student signal
- tag predictions are noisier, so they should not be the only trigger

### Review message construction

Initial policy:

- primary tag threshold: `0.50`
- fallback tag threshold when `accept_prob` is low: `0.20`
- max issues: `4`
- include cluster hint when:
  - cluster risk tier is `HIGH`, or
  - `bugfix_rate >= 0.60`

### Prompt injection shape

Reuse the same prompt shape as v3:

~~~text
{original problem}

---
Your previous attempt produced the patch below. A code review found these
issues. Please fix them while keeping the solution correct:

Issues:
  1. ...
  2. ...

Previous patch:
```diff
...
```

Fix the issues above. Do NOT rewrite the solution from scratch — make targeted
improvements to address each issue.
~~~

Only the issue generation changes. The prompt wrapper should stay the same.

---

## Implementation Plan

### 1. Add a runtime inference helper

Create a small host-side loader, for example:

- `experiment_7/review_state_runtime.py`

Responsibilities:

- load `review_state_student_h_full_canonical.pt`
- reconstruct the `EmbeddingBagStudent`
- hash-tokenize patch text using checkpoint settings
- run CPU inference
- return a structured prediction object:
  - `accept_prob`
  - `tag_probs`
  - `cluster_class`
  - `cluster_id`
  - `cluster_confidence`
  - `latent`

Important runtime rule:

- use the **checkpoint’s own** `max_tokens`, `hash_vocab_size`, and tag ordering
- do not hardcode these values separately in the steerer

### 2. Extend the bridge for runtime rendering

Reuse `experiment_7/review_state_bridge.py`, but add a small runtime-oriented
wrapper, for example:

- `render_student_review_messages(prediction, tag_threshold, fallback_threshold, max_issues)`

Behavior:

- call existing `render_review_messages()` for normal thresholded tags
- append cluster hint from bundle `cluster_hints`
- if `accept_prob` is low and issues are still empty, add top predicted tag
  messages above fallback threshold

This is the symbolic "decoder".

### 3. Implement the new scaffold agent

Create:

- `eval/FeatBench/docker_agent/agents/steered_trae_agent_phase7_1_h_full.py`

This class should be structurally almost identical to
`SteeredTraeAgentV3`, with one replacement:

- replace `_detect_issues(patch_text, model)` with
  `student_predict_then_render_issues(patch_text)`

The rest should stay the same:

- first baseline solve
- read `patch.diff`
- back up original patch before review pass
- second review pass only when issues exist
- revert to original patch if review pass fails or writes an empty patch

### 4. Wire the new agent into the manager

Update:

- `eval/FeatBench/docker_agent/agents/manager.py`

Add a new agent name, e.g.:

- `qwen35-steered-phase7_1-h-full`

and map it to the new class.

### 5. Add env-configurable runtime knobs

Recommended env vars:

- `REVIEW_STATE_STUDENT_PATH`
- `REVIEW_STATE_ACCEPT_THRESHOLD`
- `REVIEW_STATE_TAG_THRESHOLD`
- `REVIEW_STATE_TAG_FALLBACK_THRESHOLD`
- `REVIEW_STATE_MAX_ISSUES`

This allows threshold sweeps without changing code.

### 6. Add logging for diagnosis

Each run should log:

- predicted `accept_prob`
- top tag probabilities
- predicted cluster id and confidence
- rendered issue list
- whether review pass was triggered

This will be critical because the acceptance head is strong but the tag head is
still weak. We need to see whether prompts are being driven by sensible signals.

---

## Minimal Pseudocode

```python
success, output = super().run(problem_statement, instance_id, repo_name)
if not success:
    return success, output

patch_text = read_patch_diff(...)
if not patch_text.strip():
    return success, output

pred = review_state_student.predict(patch_text)
issues = render_student_review_messages(pred)

should_review = bool(issues)
if not should_review and pred.accept_prob <= ACCEPT_THRESHOLD:
    issues = render_student_review_messages(
        pred,
        tag_threshold=FALLBACK_THRESHOLD,
    )
    should_review = bool(issues)

if not should_review:
    return success, output

review_prompt = build_review_prompt(problem_statement, patch_text, issues[:MAX_ISSUES])
success2, output2 = super().run(review_prompt, instance_id, repo_name)

if success2 and non_empty_patch_written():
    return success2, output + "\n--- REVIEW PASS ---\n" + output2

restore_original_patch()
return success, output
```

---

## Evaluation Plan

### Offline sanity checks

Before full FeatBench reruns:

1. pick a few saved patch diffs from the v3 run
2. run student inference on them
3. inspect:
   - acceptance score
   - predicted tags
   - predicted cluster hint
   - final rendered issue list
4. compare those issue lists against:
   - deterministic `detect_review_issue_flags()`
   - actual v3 `_detect_issues()` output

Goal:

- verify that the new prompt bullets are concrete and non-generic
- verify that low-acceptance patches actually trigger meaningful review prompts

### End-to-end evaluation

Primary evaluation:

- baseline vs `qwen35-steered-phase7_1-h-full`
- same FeatBench paired set used for phase 6.2
- same in-scaffold judge panel used for the v3 final comparison

Success criteria for the minimal replacement claim:

- clearly above baseline under in-scaffold judging
- within a few points of the v3 steerer, or better
- no substantial increase in scaffold failures

If that holds, we can claim:

- the **student-predicted review state is sufficient to drive the same scaffold
  improvement loop**, without the original hand-built v3 review detector

---

## Known Limitations

1. **The student sees only the first `max_tokens=384` hashed tokens of the diff**
   in this v0 checkpoint.
   - This is good enough for the first replacement experiment.
   - It is not full-patch semantic coverage.

2. **Tag head quality is still modest**
   - test macro F1 is `0.124`
   - therefore acceptance score should be used for gating, not tags alone

3. **Latent prediction is not yet exploited directly**
   - the checkpoint predicts `h`
   - but v7.1 minimal replacement does not need nearest-neighbor or centroid
     decoding yet

4. **This is still post-submit steering**
   - it proves "student replaces review-state steerer"
   - it does not yet prove pre-generation task-only steering

---

## Recommended First Implementation Scope

Keep the first code change set narrow:

1. runtime student loader
2. bridge wrapper from predictions to issue strings
3. new `SteeredTraeAgentPhase71HFull`
4. agent-manager registration
5. smoke evaluation on a few saved patches

Do **not** add:

- a learned natural-language decoder
- task-only steering
- retrieval augmentation from nearest neighbors
- extra container-side dependencies

That keeps Experiment 7.1 faithful to the minimal claim:

> the trained `h`-full student can replace the v3 steerer’s patch-review state
> estimator and drive the same in-scaffold prompt injection loop.

---

## Appendix: Steering Trigger Discussion

The phase 7.1 steerer emits a **limited fixed tag set** of 9 review tags:

- `patch_too_large`
- `shared_without_tests`
- `api_without_tests`
- `eval_exec`
- `sql_fstring`
- `hardcoded_credential`
- `bare_except`
- `except_pass`
- `http_without_timeout`

However, runtime steering is **not tag-only**. The actual review trigger is a
combination of:

- thresholded tag messages
- cluster-risk hints from the predicted HDBSCAN super-cluster
- the acceptance head `accept_prob`

The host-side control flow in
`eval/FeatBench/docker_agent/agents/steered_trae_agent_phase7_1_h_full.py`
only injects a second-pass review prompt when the rendered `issues` list is
non-empty. If no issues are rendered, the first-pass patch is kept as-is.

Empirical trigger counts from the qwen35 phase 7.1 FeatBench run:

- main 156-instance run:
  - `128` trajectories reached review-state inference
  - `79` launched a review pass
  - `49` logged `Review: no issues found, keeping patch as-is`
- targeted rerun:
  - `14` trajectories reached review-state inference
  - `6` launched a review pass
  - `8` logged `Review: no issues found, keeping patch as-is`

Combined:

- `142` trajectories reached review-state inference
- `85` launched a review pass
- `57` kept the first-pass patch with no steering prompt

So the right interpretation is:

- yes, the symbolic output vocabulary is narrow
- no, it is not true that *most* trajectories had no steering input
- the run was close to evenly split, with review prompting on about `60%` of
  trajectories that reached the review-state stage

The more important limitation is not only the small tag set; it is that many
second-pass prompts were driven by coarse cluster-risk messages rather than
high-fidelity contract or architecture diagnostics. That likely contributed to
the weaker end-to-end quality relative to the phase 6.2 v3 steerer.
