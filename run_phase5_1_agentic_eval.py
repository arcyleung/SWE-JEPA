#!/usr/bin/env python3
"""Experiment 5.1 — Agentic Evaluation with Conway-Aware RL Steerer.

Extends run_phase4_7_agentic_eval_steered.py with:
- SteererV51: supports v51 model blob (reward_model + pairwise_reward_model heads)
- Extended features: cross_module_spread, has_tests, churn_asymmetry, followup_risk
- Multi-sample metrics: Avg@N, Pass@k, AcceptProxy@k (ThinkLogit-style)
- Always runs all N attempts (no early stopping) to compute full Avg@N curves
- Ablation mode: --feature-set {review|conway|review+conway|full}

Usage:
  python run_phase5_1_agentic_eval.py \\
      --steerer-model data/phase5_1_pr_steerer_model_v51.json \\
      --steer-max-attempts 8 \\
      --task-keys-jsonl data/phase4_7_baseline_subset_7k_keys.jsonl \\
      --limit 100
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime

ROOT = os.path.dirname(os.path.abspath(__file__))

OUT_JSONL = os.path.join(ROOT, "data", "phase5_1_agentic_eval_results.jsonl")
OUT_SUMMARY = os.path.join(ROOT, "data", "phase5_1_agentic_eval_summary.json")
DEFAULT_STEERER_MODEL = os.path.join(ROOT, "data", "phase5_1_pr_steerer_model_v51.json")

# ── Load infrastructure from base eval harness ─────────────────────────────

def _load_base_module():
    spec = importlib.util.spec_from_file_location(
        "base_eval",
        os.path.join(ROOT, "run_phase4_7_agentic_eval_steered.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["base_eval"] = mod  # needed for @dataclass resolution
    spec.loader.exec_module(mod)    # if __name__ == "__main__" won't fire (name is "base_eval")
    return mod


_base = _load_base_module()

# Re-export infrastructure from base module
_load_pg_cfg = _base._load_pg_cfg
_load_tokens = _base._load_tokens
_repo_dir_map = _base._repo_dir_map
_sha_available = _base._sha_available
_fetch_sha = _base._fetch_sha
_mount_overlay = _base._mount_overlay
_umount_overlay = _base._umount_overlay
_create_worktree = _base._create_worktree
_remove_worktree = _base._remove_worktree
_load_model_cfg = _base._load_model_cfg
_normalize_api_base = _base._normalize_api_base
_fetch_tasks = _base._fetch_tasks
_fetch_tasks_by_keys = _base._fetch_tasks_by_keys
_load_task_keys = _base._load_task_keys
_run_attempt = _base._run_attempt
_task_prompt = _base._task_prompt
_steer_hint = _base._steer_hint

import subprocess  # noqa: E402 (needed for _run_one_task)


# ── Feature set definitions ────────────────────────────────────────────────

FEATURE_SETS = {
    "review": ["is_draft", "changed_files", "additions", "deletions",
               "requested_reviewers_count", "has_closing_issue"],
    "conway": ["cross_module_spread", "has_tests", "churn_asymmetry"],
    "review+conway": ["is_draft", "changed_files", "additions", "deletions",
                      "requested_reviewers_count", "has_closing_issue",
                      "cross_module_spread", "has_tests", "churn_asymmetry"],
    "full": ["is_draft", "changed_files", "additions", "deletions",
             "requested_reviewers_count", "has_closing_issue",
             "cross_module_spread", "has_tests", "churn_asymmetry", "followup_risk"],
}

_FEATURE_TRANSFORMS = {
    "is_draft": lambda t, cf: float(t.get("is_draft", 0)),
    "changed_files": lambda t, cf: math.log1p(float(max(cf, 0))),
    "additions": lambda t, cf: math.log1p(float(max(t.get("additions", 0), 0))),
    "deletions": lambda t, cf: math.log1p(float(max(t.get("deletions", 0), 0))),
    "requested_reviewers_count": lambda t, cf: math.log1p(float(max(t.get("requested_reviewers_count", 0), 0))),
    "has_closing_issue": lambda t, cf: float(t.get("has_closing_issue", 0)),
    "cross_module_spread": lambda t, cf: math.log1p(float(max(t.get("cross_module_spread", 0), 0))),
    "has_tests": lambda t, cf: float(t.get("has_tests", 0)),
    "churn_asymmetry": lambda t, cf: float(t.get("churn_asymmetry", 0.0)),
    "followup_risk": lambda t, cf: float(t.get("followup_risk", 0.0)),
}


def _build_features_v51(
    task: dict, changed_files_after: int, all_features: list[str], active_set: set[str]
) -> list[float]:
    """Build full-length feature vector; zero out features not in active_set (ablation)."""
    out = []
    for f in all_features:
        if f in active_set:
            out.append(_FEATURE_TRANSFORMS[f](task, changed_files_after))
        else:
            out.append(0.0)
    return out


# ── Extended Steerer (v51) ─────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class SteererV51:
    """Drop-in Steerer upgrade for v51 model blobs.

    Score priority:
      1. reward_model (value regression) — if present
      2. pairwise_reward_model (Bradley-Terry) — if present
      3. acceptance - refactor composite (backward compat)

    feature_set ablation: zeros out features not in the selected set, keeping
    the model's original coefficient positions intact (correct behaviour).
    """
    blob: dict
    w_accept: float = 1.0
    w_refactor: float = 1.0
    scope_penalty: float = 0.15
    feature_set: str = "full"
    _all_features: list[str] = field(default_factory=list, init=False, repr=False)
    _active_set: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self):
        self._all_features = self.blob.get("features", list(FEATURE_SETS["full"]))
        fs = FEATURE_SETS.get(self.feature_set, FEATURE_SETS["full"])
        self._active_set = set(fs)

    @property
    def _feature_names(self) -> list[str]:
        """Active feature names (for introspection)."""
        return [f for f in self._all_features if f in self._active_set]

    def _apply_linear(self, head_blob: dict, x: list[float]) -> float:
        mean = head_blob["scaler_mean"]
        scale = head_blob["scaler_scale"]
        coef = head_blob["coef"]
        inter = float(head_blob["intercept"])
        z = inter
        n = min(len(coef), len(x), len(mean), len(scale))
        for i in range(n):
            xi = (x[i] - float(mean[i])) / (float(scale[i]) + 1e-8)
            z += float(coef[i]) * xi
        return z

    def score(self, task: dict, changed_files_after: int) -> tuple[float, dict]:
        x = _build_features_v51(task, changed_files_after, self._all_features, self._active_set)
        diag: dict = {}

        # Always compute supervised heads for early-stopping compatibility
        p_acc = _sigmoid(self._apply_linear(self.blob["acceptance"], x))
        p_ref = _sigmoid(self._apply_linear(self.blob["refactor"], x))
        exp_files = max(1.0, float(task.get("changed_files", 1)))
        scope_drift = abs(float(changed_files_after) - exp_files) / exp_files
        supervised_score = self.w_accept * p_acc - self.w_refactor * p_ref - self.scope_penalty * scope_drift
        diag.update({"p_accept": p_acc, "p_refactor": p_ref, "scope_drift": scope_drift})

        # Use RL-trained reward model if available (primary scorer)
        if "reward_model" in self.blob:
            rm = self.blob["reward_model"]
            reward_score = self._apply_linear(rm, x)  # raw linear (not sigmoid for regression)
            diag["reward_model_score"] = reward_score
            final_score = reward_score
        elif "pairwise_reward_model" in self.blob:
            pm = self.blob["pairwise_reward_model"]
            pairwise_score = _sigmoid(self._apply_linear(pm, x))
            diag["pairwise_score"] = pairwise_score
            final_score = pairwise_score
        else:
            final_score = supervised_score

        diag["score"] = final_score
        return final_score, diag


def _load_steerer_v51(
    path: str | None,
    w_accept: float,
    w_refactor: float,
    scope_penalty: float,
    feature_set: str,
) -> SteererV51 | None:
    if not path:
        return None
    blob = json.load(open(path))
    if "acceptance" not in blob or "refactor" not in blob:
        raise ValueError(f"Invalid steerer model: {path} (missing acceptance/refactor heads)")
    return SteererV51(
        blob=blob,
        w_accept=w_accept,
        w_refactor=w_refactor,
        scope_penalty=scope_penalty,
        feature_set=feature_set,
    )


# ── Multi-sample metrics ────────────────────────────────────────────────────

def _compute_multi_sample_metrics(rows: list[dict], k_values: list[int]) -> dict:
    """Compute Avg@N, Pass@k, AcceptProxy@k across all eval results.

    - Pass@k: fraction of tasks where any of top-k attempts produced a patch (returncode=0)
    - AcceptProxy@k: fraction of tasks where any of top-k attempts has p_accept >= 0.5
    - Avg@N: mean p_accept over all N attempts (across tasks)
    """
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("steer_attempts")]
    if not ok_rows:
        return {}

    metrics: dict = {"n_tasks": len(ok_rows)}
    for k in k_values:
        pass_k = 0
        accept_proxy_k = 0
        for r in ok_rows:
            attempts = r["steer_attempts"][:k]
            if any(a.get("returncode", -1) == 0 for a in attempts):
                pass_k += 1
            if any(a.get("diag", {}).get("p_accept", 0.0) >= 0.5 for a in attempts):
                accept_proxy_k += 1
        n = len(ok_rows)
        metrics[f"pass_at_{k}"] = pass_k / n
        metrics[f"accept_proxy_at_{k}"] = accept_proxy_k / n

    # Avg@N: mean p_accept across all attempts per task
    all_p_acc = []
    for r in ok_rows:
        for a in r["steer_attempts"]:
            pa = a.get("diag", {}).get("p_accept", 0.0)
            all_p_acc.append(pa)
    metrics["avg_n_p_accept"] = float(sum(all_p_acc) / len(all_p_acc)) if all_p_acc else 0.0

    # Score distribution
    all_scores = []
    for r in ok_rows:
        for a in r["steer_attempts"]:
            s = a.get("diag", {}).get("score", 0.0)
            all_scores.append(s)
    if all_scores:
        metrics["score_mean"] = float(sum(all_scores) / len(all_scores))
        sorted_s = sorted(all_scores)
        metrics["score_p25"] = float(sorted_s[len(sorted_s) // 4])
        metrics["score_p75"] = float(sorted_s[3 * len(sorted_s) // 4])

    return metrics


# ── Run one task (v51 — no early stopping, collect all N attempts) ──────────

def _run_one_task_v51(
    t: dict,
    model_cfg: dict,
    repo_dirs: dict[str, str],
    tokens: list[str],
    out_traj_dir: str,
    mini_step_limit: int,
    timeout_sec: int,
    agent_python: str,
    api_base_override: str | None,
    api_key_override: str | None,
    litellm_model_override: str | None,
    steerer: SteererV51 | None,
    n_attempts: int,
) -> dict:
    """Run all n_attempts for one task, collecting full attempt history for Avg@N metrics."""
    repo = t["repo"]
    repo_dir = repo_dirs.get(repo)
    if not repo_dir:
        return {"repo": repo, "instance_id": t["instance_id"], "status": "missing_repo"}

    tag = f"p51_{t.get('_task_idx', 0)}_{abs(hash((repo, t['pull_number'], t['base_sha'], t['instance_id']))) % (10**12)}"
    merged = upper = work = None
    workspace_method = ""
    t0 = time.time()
    try:
        if __import__("shutil").which("fuse-overlayfs") and __import__("shutil").which("fusermount3"):
            merged, upper, work = _mount_overlay(repo_dir, tag)
            workspace_method = "overlayfs"
        else:
            merged = _create_worktree(repo_dir, tag)
            workspace_method = "git_worktree"

        if not _sha_available(merged, t["base_sha"]):
            ok = False
            for tok in tokens[:3]:
                if _fetch_sha(merged, t["base_sha"], repo, tok):
                    ok = True
                    break
            if not ok and not _fetch_sha(merged, t["base_sha"], repo, None):
                return {"repo": repo, "instance_id": t["instance_id"], "status": "missing_base_sha"}

        ck = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", merged, "checkout", "--force", t["base_sha"]],
            capture_output=True, text=True,
        )
        if ck.returncode != 0:
            return {"repo": repo, "instance_id": t["instance_id"], "status": "checkout_failed",
                    "reason": (ck.stderr or ck.stdout)[:300]}

        os.makedirs(out_traj_dir, exist_ok=True)
        attempts = []
        best = None
        best_diag = {"score": -1e9}

        for ai in range(max(1, n_attempts)):
            subprocess.run(
                ["git", "-c", "safe.directory=*", "-C", merged, "reset", "--hard", t["base_sha"]],
                capture_output=True,
            )
            hint = _steer_hint(ai, max(1, int(t.get("changed_files", 1))), best_diag if ai > 0 else None)
            task_text = _task_prompt(t, hint if steerer else "")
            traj = os.path.join(
                out_traj_dir,
                f"{t['instance_id']}__pr{t['pull_number']}__a{ai}.traj.json",
            )
            rr = _run_attempt(
                merged, task_text, traj, model_cfg,
                mini_step_limit, timeout_sec, agent_python,
                api_base_override, api_key_override, litellm_model_override,
            )
            diag = {"score": 0.0, "p_accept": 0.0, "p_refactor": 0.0, "scope_drift": 0.0}
            if steerer:
                score, diag = steerer.score(t, rr["changed_files_after"])
                diag["score"] = score
            row = {
                "attempt_idx": ai,
                "hint": hint,
                "returncode": rr["returncode"],
                "traj_path": rr["traj_path"],
                "changed_files_after": rr["changed_files_after"],
                "diag": diag,
            }
            attempts.append(row)
            if diag["score"] > best_diag["score"]:
                best = rr
                best_diag = diag

        return {
            "repo": repo,
            "instance_id": t["instance_id"],
            "pull_number": t["pull_number"],
            "status": "ok" if best and best["returncode"] == 0 else "agent_failed",
            "returncode": -1 if not best else best["returncode"],
            "elapsed_sec": time.time() - t0,
            "workspace_method": workspace_method,
            "traj_path": "" if not best else best["traj_path"],
            "stdout_tail": "" if not best else best.get("stdout_tail", ""),
            "stderr_tail": "" if not best else best.get("stderr_tail", ""),
            "changed_files_after": 0 if not best else best["changed_files_after"],
            "changed_files_list_head": [] if not best else best.get("changed_files_list_head", []),
            "steered": bool(steerer),
            "steer_attempts": attempts,
            "steer_best_diag": best_diag,
        }
    except subprocess.TimeoutExpired:
        return {"repo": repo, "instance_id": t["instance_id"], "status": "timeout",
                "elapsed_sec": time.time() - t0}
    except Exception as e:
        return {"repo": repo, "instance_id": t["instance_id"], "status": "error",
                "reason": str(e)[:500]}
    finally:
        if workspace_method == "overlayfs" and merged and upper and work:
            _umount_overlay(merged, upper, work)
        elif workspace_method == "git_worktree" and merged:
            _remove_worktree(repo_dir, merged)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="qwen3_coder_30b")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--step-limit", type=int, default=20)
    ap.add_argument("--timeout-sec", type=int, default=900)
    ap.add_argument("--agent-python", default="/usr/bin/python3")
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--litellm-model", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task-keys-jsonl", default=None)
    ap.add_argument("--out-jsonl", default=OUT_JSONL)
    ap.add_argument("--out-summary", default=OUT_SUMMARY)
    ap.add_argument("--traj-dir", default=os.path.join(ROOT, "data", "phase5_1_trajectories"))
    ap.add_argument("--steerer-model", default=DEFAULT_STEERER_MODEL)
    ap.add_argument("--n-attempts", type=int, default=8,
                    help="Number of attempts per task for Avg@N/Pass@k metrics")
    ap.add_argument("--steer-w-accept", type=float, default=1.0)
    ap.add_argument("--steer-w-refactor", type=float, default=1.0)
    ap.add_argument("--steer-scope-penalty", type=float, default=0.15)
    ap.add_argument("--feature-set", default="full",
                    choices=list(FEATURE_SETS.keys()),
                    help="Ablation feature subset (default: full)")
    ap.add_argument("--disable-steering", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    model_cfg = _load_model_cfg(args.model_name)
    repo_dirs = _repo_dir_map()
    tokens = _load_tokens()

    if args.task_keys_jsonl:
        keys = _load_task_keys(args.task_keys_jsonl)
        tasks = _fetch_tasks_by_keys(keys, args.seed)
        if args.limit > 0:
            tasks = tasks[: args.limit]
    else:
        tasks = _fetch_tasks(args.limit, args.seed)
    tasks = [t for t in tasks if t["repo"] in repo_dirs]
    print(f"tasks: {len(tasks)}  concurrency: {args.concurrency}  n_attempts: {args.n_attempts}")

    steerer = None
    if not args.disable_steering and os.path.exists(args.steerer_model):
        steerer = _load_steerer_v51(
            args.steerer_model,
            args.steer_w_accept,
            args.steer_w_refactor,
            args.steer_scope_penalty,
            args.feature_set,
        )
        version = steerer.blob.get("version", "unknown")
        has_rl = "reward_model" in steerer.blob or "pairwise_reward_model" in steerer.blob
        print(f"steerer: {version}  feature_set={args.feature_set}  rl_head={has_rl}")
    else:
        print("steerer: disabled")

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    os.makedirs(args.traj_dir, exist_ok=True)

    rows = []
    with open(args.out_jsonl, "w") as out_handle:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = [
                ex.submit(
                    _run_one_task_v51,
                    t, model_cfg, repo_dirs, tokens, args.traj_dir,
                    args.step_limit, args.timeout_sec, args.agent_python,
                    args.api_base, args.api_key, args.litellm_model,
                    steerer, args.n_attempts,
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
                    ok = sum(1 for x in rows if x.get("status") == "ok")
                    print(f"  [{done}/{len(futs)}] ok={ok}", flush=True)

    k_values = [1, 2, 3, 5, 8]
    multi_metrics = _compute_multi_sample_metrics(rows, k_values)

    status_counts: dict[str, int] = {}
    for r in rows:
        s = r.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    summary = {
        "date_utc": datetime.now(UTC).isoformat(),
        "exp": "5.1",
        "steerer_model": args.steerer_model if not args.disable_steering else None,
        "feature_set": args.feature_set,
        "n_attempts": args.n_attempts,
        "n_tasks": len(rows),
        "status_counts": status_counts,
        "ok_rate": status_counts.get("ok", 0) / len(rows) if rows else 0.0,
        "avg_elapsed_sec": sum(float(r.get("elapsed_sec", 0.0)) for r in rows) / len(rows) if rows else 0.0,
        "avg_changed_files_after": sum(float(r.get("changed_files_after", 0.0)) for r in rows) / len(rows) if rows else 0.0,
        "multi_sample_metrics": multi_metrics,
        "elapsed_total_sec": time.time() - t0,
        "out_jsonl": args.out_jsonl,
    }
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
