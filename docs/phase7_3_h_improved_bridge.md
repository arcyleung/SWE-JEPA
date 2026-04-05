# Experiment 7.3 — `h`-Full Steerer With Improved Bridge

## Goal

Re-run the full in-scaffold FeatBench evaluation for the phase 7.1 `h`-full
student steerer after improving the host-side review bridge.

This experiment keeps the same trained 7.1 checkpoint and coding model, but
expands the symbolic bridge so the steerer can emit more concrete review
guidance from the actual patch diff instead of relying as heavily on:

- the original 9 learned tags
- generic HDBSCAN cluster-risk messages

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

The updated bridge lives in:

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

## Eval Setup

Run date: `2026-04-02`

Coding agent:

- agent: `qwen35-steered-phase7_1-h-full`
- model: `hosted_vllm/qwen3.5_35b_a3b`
- endpoint: `http://10.10.110.65:24000/v1`

Concurrency:

- `64` workers

Run outputs:

- eval dir:
  `/shared_workspace_mfs/arthur/coder/eval/FeatBench/docker_agent/eval_runs/2026-04-02_phase7_3_h_improved_bridge_qwen35`
- results:
  `/shared_workspace_mfs/arthur/coder/eval/FeatBench/docker_agent/eval_runs/2026-04-02_phase7_3_h_improved_bridge_qwen35/evaluation_results_steered_phase7_3_h_improved_bridge_qwen35.json`
- log:
  `/shared_workspace_mfs/arthur/coder/eval/FeatBench/docker_agent/eval_runs/2026-04-02_phase7_3_h_improved_bridge_qwen35/parallel_eval_qwen35_phase73.log`
- swap root:
  `/tmp/20260402_qwen35_eval_phase73_full156_swap`

## Status

The full 156-instance in-scaffold rerun has been launched and is currently in
progress.

At the time this note was created:

- the controller process was alive
- the eval directory existed
- the run had reached full `64`-worker container occupancy
- the active wave was still in package install / setup inside the worker containers
- no completed result rows had been written yet

## Planned Comparison

When the run finishes, compare against the previous 7.1 qwen35 steered run and
the merged baseline:

- prior 7.1 steered:
  `eval/FeatBench/docker_agent/eval_runs/2026-03-31_phase7_1_h_full_qwen35/`
- merged baseline:
  `eval/FeatBench/docker_agent/eval_runs/2026-03-31_qwen35_baseline_rerun_infra/`

Primary metrics:

- `success_f2p`
- `success_p2p`
- `success` (both)

Secondary follow-up:

- rerun the in-scaffold judge panel on the improved-bridge outputs
- compare judge win rate against:
  - phase 7.1 scaffold: `52.26%`
  - phase 7.1 patch-only: `55.07%`

## Initial Expectation

The improved bridge should help most on cases where frontier judges previously
preferred the baseline because the old bridge missed concrete defect classes
such as:

- schema / kwargs contract drift
- wrong sentinel handling
- missing targeted regression tests
- async boundary risk
- unnecessary helper rewrites

So the expected benefit is not “stronger generic steering,” but “more specific
second-pass review prompts that better match what the stronger judges actually
criticized.”
