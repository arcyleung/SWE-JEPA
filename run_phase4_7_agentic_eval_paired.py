#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import UTC, datetime

import run_phase4_7_agentic_eval as base
import run_phase4_7_agentic_eval_steered as steered


ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_pair_keys(path: str | None) -> set[tuple]:
    if not path or not os.path.exists(path):
        return set()
    out: set[tuple] = set()
    with open(path) as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except Exception:
                continue
            repo = r.get("repo")
            pull = int(r.get("pull_number") or 0)
            if repo and pull > 0:
                out.add((repo, pull))
    return out


def _load_task_keys(path: str | None) -> list[tuple[str, int]] | None:
    if not path:
        return None
    keys = []
    seen = set()
    with open(path) as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except Exception:
                continue
            repo = r.get("repo")
            pull = int(r.get("pull_number") or 0)
            if not repo or pull <= 0:
                continue
            k = (repo, pull)
            if k in seen:
                continue
            seen.add(k)
            keys.append(k)
    return keys


def _worker(
    t: dict,
    model_cfg: dict,
    repo_dirs: dict[str, str],
    tokens: list[str],
    baseline_traj_dir: str,
    baseline_patch_dir: str,
    steered_traj_dir: str,
    steered_patch_dir: str,
    step_limit: int,
    timeout_sec: int,
    agent_python: str,
    api_base: str | None,
    api_key: str | None,
    litellm_model: str | None,
    steerer_obj: steered.Steerer,
    steer_max_attempts: int,
    steer_accept_threshold: float,
    steer_refactor_threshold: float,
    steer_retry_temperature: float,
) -> dict:
    t0 = time.time()
    # Baseline arm
    b = steered._run_one_task(
        deepcopy(t),
        model_cfg,
        repo_dirs,
        tokens,
        baseline_traj_dir,
        baseline_patch_dir,
        step_limit,
        timeout_sec,
        agent_python,
        api_base,
        api_key,
        litellm_model,
        None,
        1,
        steer_accept_threshold,
        steer_refactor_threshold,
        0.0,
    )
    # Steered arm
    s = steered._run_one_task(
        deepcopy(t),
        model_cfg,
        repo_dirs,
        tokens,
        steered_traj_dir,
        steered_patch_dir,
        step_limit,
        timeout_sec,
        agent_python,
        api_base,
        api_key,
        litellm_model,
        steerer_obj,
        steer_max_attempts,
        steer_accept_threshold,
        steer_refactor_threshold,
        steer_retry_temperature,
    )
    return {
        "repo": t["repo"],
        "instance_id": t["instance_id"],
        "pull_number": t["pull_number"],
        "elapsed_sec_pair": time.time() - t0,
        "baseline": b,
        "steered": s,
        "pair_has_patch": bool(b.get("has_patch")) and bool(s.get("has_patch")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="qwen3_coder_30b")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--source-multiplier", type=int, default=8)
    ap.add_argument("--pr-category", default="feature")
    ap.add_argument("--task-keys-jsonl", default=None, help="Optional fixed cohort JSONL with {repo,pull_number}")
    ap.add_argument("--skip-existing-pairs-jsonl", default=None, help="Skip already-run pair keys from this paired JSONL")
    ap.add_argument("--concurrency", type=int, default=200)
    ap.add_argument("--step-limit", type=int, default=80)
    ap.add_argument("--timeout-sec", type=int, default=1800)
    ap.add_argument("--agent-python", default="/usr/bin/python3")
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--litellm-model", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steerer-model", default=os.path.join(ROOT, "data", "phase4_7_pr_steerer_model.json"))
    ap.add_argument("--steer-max-attempts", type=int, default=3)
    ap.add_argument("--steer-accept-threshold", type=float, default=0.65)
    ap.add_argument("--steer-refactor-threshold", type=float, default=0.35)
    ap.add_argument("--steer-w-accept", type=float, default=1.0)
    ap.add_argument("--steer-w-refactor", type=float, default=1.0)
    ap.add_argument("--steer-scope-penalty", type=float, default=0.15)
    ap.add_argument("--steer-retry-temperature", type=float, default=0.25)
    ap.add_argument("--out-jsonl", default=os.path.join(ROOT, "data", "phase4_7_agentic_eval_pairs_4_7_1.jsonl"))
    ap.add_argument("--out-summary", default=os.path.join(ROOT, "data", "phase4_7_agentic_eval_pairs_4_7_1_summary.json"))
    ap.add_argument("--baseline-traj-dir", default=os.path.join(ROOT, "data", "phase4_7_trajectories_paired_baseline"))
    ap.add_argument("--baseline-patch-dir", default=os.path.join(ROOT, "data", "phase4_7_patches_paired_baseline"))
    ap.add_argument("--steered-traj-dir", default=os.path.join(ROOT, "data", "phase4_7_trajectories_paired_steered"))
    ap.add_argument("--steered-patch-dir", default=os.path.join(ROOT, "data", "phase4_7_patches_paired_steered"))
    args = ap.parse_args()

    t0 = time.time()
    model_cfg = base._load_model_cfg(args.model_name)
    repo_dirs = base._repo_dir_map()
    tokens = base._load_tokens()

    if args.task_keys_jsonl:
        keys = _load_task_keys(args.task_keys_jsonl) or []
        tasks = steered._fetch_tasks_by_keys(keys, args.seed)
    else:
        tasks = base._fetch_tasks(args.limit, args.seed, args.pr_category, args.source_multiplier)
    before_repo = len(tasks)
    tasks = [t for t in tasks if t["repo"] in repo_dirs]
    skipped_repo = before_repo - len(tasks)

    done_pairs = _load_pair_keys(args.skip_existing_pairs_jsonl)
    skipped_existing = 0
    if done_pairs:
        kept = []
        for t in tasks:
            k = (t["repo"], int(t.get("pull_number") or 0))
            if k in done_pairs:
                skipped_existing += 1
                continue
            kept.append(t)
        tasks = kept
    tasks = tasks[: args.limit]
    print(
        f"tasks selected: {len(tasks)} (requested={args.limit})"
        f" skipped_repo_missing={skipped_repo} skipped_existing_pairs={skipped_existing}",
        flush=True,
    )
    print(f"concurrency={args.concurrency}", flush=True)
    if args.pr_category:
        print(f"pr_category filter: {args.pr_category}", flush=True)

    steerer_obj = steered._load_steerer(args.steerer_model, args.steer_w_accept, args.steer_w_refactor, args.steer_scope_penalty)
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    os.makedirs(args.baseline_traj_dir, exist_ok=True)
    os.makedirs(args.baseline_patch_dir, exist_ok=True)
    os.makedirs(args.steered_traj_dir, exist_ok=True)
    os.makedirs(args.steered_patch_dir, exist_ok=True)

    rows = []
    with open(args.out_jsonl, "w") as out_handle:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = [
                ex.submit(
                    _worker,
                    t,
                    model_cfg,
                    repo_dirs,
                    tokens,
                    args.baseline_traj_dir,
                    args.baseline_patch_dir,
                    args.steered_traj_dir,
                    args.steered_patch_dir,
                    args.step_limit,
                    args.timeout_sec,
                    args.agent_python,
                    args.api_base,
                    args.api_key,
                    args.litellm_model,
                    steerer_obj,
                    args.steer_max_attempts,
                    args.steer_accept_threshold,
                    args.steer_refactor_threshold,
                    args.steer_retry_temperature,
                )
                for t in tasks
            ]
            done = 0
            for fut in as_completed(futs):
                done += 1
                r = fut.result()
                rows.append(r)
                out_handle.write(json.dumps(r) + "\n")
                out_handle.flush()
                if done % 10 == 0 or done == len(futs):
                    both_patch = sum(1 for x in rows if x.get("pair_has_patch"))
                    print(f"  [{done}/{len(futs)}] pair_has_patch={both_patch}", flush=True)

    summary = {
        "date_utc": datetime.now(UTC).isoformat(),
        "args": vars(args),
        "n_pairs": len(rows),
        "pair_has_patch_rate": 0.0,
        "elapsed_total_sec": time.time() - t0,
        "out_jsonl": args.out_jsonl,
    }
    if rows:
        summary["pair_has_patch_rate"] = sum(1 for r in rows if r.get("pair_has_patch")) / len(rows)
        summary["baseline_ok_rate"] = sum(1 for r in rows if r.get("baseline", {}).get("status") == "ok") / len(rows)
        summary["steered_ok_rate"] = sum(1 for r in rows if r.get("steered", {}).get("status") == "ok") / len(rows)
        summary["baseline_has_patch_rate"] = sum(1 for r in rows if r.get("baseline", {}).get("has_patch")) / len(rows)
        summary["steered_has_patch_rate"] = sum(1 for r in rows if r.get("steered", {}).get("has_patch")) / len(rows)

    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
