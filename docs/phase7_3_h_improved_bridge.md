# Experiment 7.3 — `h`-Full Steerer With Improved Bridge

## Goal

Re-run the full in-scaffold FeatBench evaluation for the `h`-full JEPA steerer
after expanding the review ontology beyond the original 9-tag bridge.

This note records two closely related steps:

1. the initial bridge-only ablation, which added richer host-side review
   heuristics on top of the old 7.1 checkpoint
2. the final reported run, which retrained the student on the richer tag set
   and then reran FeatBench with that retrained checkpoint

The main question is whether better bridge specificity improves:

- FeatBench F2P / P2P
- in-scaffold judge preference
- agreement with frontier judge reasoning

## Retrain steerer

### Export to sqlite
```bash
cd /shared_workspace_mfs/arthur/coder
  mkdir -p /shared_workspace_mfs/arthur/coder/data/phase7_2/logs
  python3 -u /shared_workspace_mfs/arthur/coder/experiment_7/export_patch_store_sqlite.py \
    --projected /shared_workspace_mfs/arthur/coder/data/phase6_2_canonical_latest/projected_embeddings.npz \
    --pg-config /shared_workspace_mfs/arthur/coder/postgres_connection.yaml \
    --out /shared_workspace_mfs/arthur/coder/data/phase7_2/patch_store_canonical.sqlite \
    --summary-out /shared_workspace_mfs/arthur/coder/data/phase7_2/patch_store_canonical_summary.json \
    --fetch-chunk-size 1024 \
    2>&1 | tee /shared_workspace_mfs/arthur/coder/data/phase7_2/logs/export_patch_store_sqlite.log
```

### Shard preprocessing

```bash
cd /shared_workspace_mfs/arthur/coder
  PARTITION=debug \
  NODES=22 \
  CPUS_PER_TASK=192 \
  RUN_TAG=phase7_2_review_state_preprocessed_richer_tags_v1 \
  PROJECTED=/shared_workspace_mfs/arthur/coder/data/phase6_2_canonical_latest/projected_embeddings.npz \
  SUPER_CLUSTERS=/shared_workspace_mfs/arthur/coder/data/phase6_2_canonical_latest/super_cluster_assignments.npz \
  PATCH_SQLITE=/shared_workspace_mfs/arthur/coder/data/phase7_2/patch_store_canonical.sqlite \
  FETCH_CHUNK_SIZE=1024 \
  PREPROCESS_PROGRESS_EVERY=1024 \
  TAG_BATCH_SIZE=64 \
  bash /shared_workspace_mfs/arthur/coder/experiment_7/run_review_state_preprocess_srun.sh
```

### Train

```bash
cd /shared_workspace_mfs/arthur/coder
  CUDA_VISIBLE_DEVICES=0 python3 -u /shared_workspace_mfs/arthur/coder/experiment_7/train_review_state_student.py \
    --device cuda \
    --epochs 5 \
    --batch-size 64 \
    --preprocessed-dir /shared_workspace_mfs/arthur/coder/data/phase7_2_review_state_preprocessed_richer_tags_v1/shards \
    --model-out /shared_workspace_mfs/arthur/coder/data/phase7_2/review_state_student_h_full_richer_tags.pt \
    --metrics-out /shared_workspace_mfs/arthur/coder/data/phase7_2/review_state_student_h_full_richer_tags_metrics.json
```

## Bridge Changes

The bridge work started as a host-side expansion in:

- `experiment_7/review_state_bridge.py`
- `eval/FeatBench/docker_agent/agents/review_state_runtime.py`
- `eval/FeatBench/docker_agent/agents/steered_trae_agent_phase7_1_h_full.py`

The new runtime-only heuristics add concrete review messages for:

- public/shared contract changes without clearly targeted regression coverage
- schema / kwargs / serializer contract risk
- sentinel or identity misuse
- tests that rely on private attributes
- async / event-loop boundary changes
- type / annotation handling risk
- broader-than-necessary helper rewrites

These heuristics do **not** change the checkpoint schema. The student still
predicts the original 9 tags; the bridge now supplements them with diff-side
contract/test diagnostics.

That bridge-only variant was useful for error analysis, but it was not the
final reported configuration. The final phase 7.3 result below uses the
retrained richer-tag student from phase 7.2 instead of relying on runtime-only
heuristics layered on top of the old 9-tag checkpoint.

## Retrained Student

The final reported phase 7.3 run uses the retrained richer-tag student from
phase 7.2, not the earlier runtime-only heuristic bridge ablation.

Checkpoint:

- `/shared_workspace_mfs/arthur/coder/data/phase7_2/review_state_student_h_full_richer_tags.pt`

Test metrics:

- latent cosine: `0.3964`
- cluster accuracy: `0.3767`
- acceptance AUROC: `0.8502`
- acceptance PR AUC: `0.9005`
- tag macro F1: `0.1806`

The richer learned tag set contains `16` tags. This improved tag supervision
materially raised tag quality over the original 9-tag phase 7.1 student, while
keeping the acceptance head roughly flat.

## Final Reported Eval

The first `64`-worker richer-tag attempts were not stable enough to treat as
the final result. The reported phase 7.3 numbers come from the stable rerun
after the 20-instance overlap spot check:

- eval dir:
  `/shared_workspace_mfs/arthur/coder/eval/FeatBench/docker_agent/eval_runs/2026-04-04_phase7_3_h_improved_bridge_richer_tags_qwen35_full32_after_spotcheck`
- results:
  `/shared_workspace_mfs/arthur/coder/eval/FeatBench/docker_agent/eval_runs/2026-04-04_phase7_3_h_improved_bridge_richer_tags_qwen35_full32_after_spotcheck/evaluation_results_steered_phase7_3_h_improved_bridge_richer_tags_qwen35.json`
- log:
  `/shared_workspace_mfs/arthur/coder/eval/FeatBench/docker_agent/eval_runs/2026-04-04_phase7_3_h_improved_bridge_richer_tags_qwen35_full32_after_spotcheck/parallel_eval_qwen35_phase73_richer_tags.log`

Setup:

- agent: `qwen35-steered-phase7_1-h-full`
- coding model: `hosted_vllm/qwen3.5_35b_a3b`
- endpoint: `http://10.10.110.70:24000/v1`
- workers: `32`

Final coding outcomes on `156` FeatBench instances:

- `F2P`: `64 / 156`
- `P2P`: `17 / 156`
- `Both`: `7 / 156`
- `None`: `4 / 156`
- non-empty patches: `127 / 156`

## Final Comparison

Against the fully rerun references:

| Run | F2P | P2P | Both |
|---|---:|---:|---:|
| Baseline | `74 / 156` | `15 / 156` | `6 / 156` |
| HDBSCAN 6.2 steerer | `72 / 156` | `15 / 156` | `8 / 156` |
| JEPA-steerer v1 | `79 / 156` | `14 / 156` | `7 / 156` |
| Phase 7.3 richer-tags | `64 / 156` | `17 / 156` | `7 / 156` |

Interpretation:

- phase 7.3 improved `P2P` the most
- phase 7.3 did not recover `F2P` to baseline or JEPA v1
- `Both` tied JEPA v1, exceeded baseline, and stayed below HDBSCAN v3

## Scaffold Judge Panel

Final scaffold judging was recomputed after the Claude API recovered and the
old Claude `no_verdict` rows were retried in place.

HDBSCAN 6.2 scaffold panel:

- summary:
  `/shared_workspace_mfs/arthur/coder/eval/FeatBench/docker_agent/eval_runs/2026-03-25_phase6_2/judge_panel_v3_scaffold_clean/summary.json`
- panel: `605 / 1083 = 55.86%`
- Claude after cleanup: `65 / 120 = 54.17%`, `0` errors

Retrained JEPA scaffold panel:

- summary:
  `/shared_workspace_mfs/arthur/coder/eval/FeatBench/docker_agent/eval_runs/2026-04-04_phase7_3_h_improved_bridge_richer_tags_qwen35_judge_scaffold/summary.json`
- panel: `605 / 1016 = 59.55%`
- Claude after cleanup: `66 / 115 = 57.39%`, `0` errors

These panel totals are over valid judgments only. Some non-Claude judges still
have residual errors, so the denominator differs across panels.

## Conclusion

Phase 7.3 did not beat the best coding-task `F2P` reference, but it did
improve `P2P` and materially improved scaffold judge preference relative to the
fully rerun HDBSCAN 6.2 panel:

- coding tasks: stronger preservation / review quality signal, weaker direct
  target-fix rate than baseline and JEPA v1
- scaffold judging: `59.55%` for retrained richer-tags vs `55.86%` for rerun
  HDBSCAN v3

So the richer learned ontology looks directionally useful for the review-state
bridge, but the current coding model still converts that extra signal into
better `P2P` more reliably than better `F2P`.
