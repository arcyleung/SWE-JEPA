#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pg8000.native
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
PG_CONFIG_FILE = os.path.join(ROOT, "postgres_connection.yaml")
MODELS_YAML = os.path.join(ROOT, "models.yaml")
TOKENS_YAML = os.path.join(ROOT, "crawl_tokens.yaml")

REPOS_BASE = "/shared_workspace_mfs/repos"
OVERLAY_MERGED_BASE = "/shared_workspace_mfs/repos_tmp_overlayfs"
OVERLAY_SHM_BASE = "/dev/shm"
WORKTREE_BASE = "/shared_workspace_mfs/repos_tmp_worktrees"

MINI_SRC = os.path.join(ROOT, "agentic_scaffold", "mini-swe-agent", "src")
MINI_CFG = os.path.join(MINI_SRC, "minisweagent", "config", "mini.yaml")

OUT_JSONL = os.path.join(ROOT, "docs", "phase4_7_agentic_eval_results.jsonl")
OUT_SUMMARY = os.path.join(ROOT, "docs", "phase4_7_agentic_eval_summary.json")


def _load_pg_cfg() -> dict:
    cfg = yaml.safe_load(open(PG_CONFIG_FILE))
    return dict(
        host=cfg["ip"],
        port=cfg.get("port", 9999),
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
    )


def _load_tokens() -> list[str]:
    try:
        cfg = yaml.safe_load(open(TOKENS_YAML))
        return cfg.get("gh_tokens", []) or []
    except Exception:
        return []


def _repo_dir_map() -> dict[str, str]:
    m: dict[str, str] = {}
    for entry in os.listdir(REPOS_BASE):
        p = os.path.join(REPOS_BASE, entry)
        if not os.path.isdir(p):
            continue
        parts = entry.split("__")
        if len(parts) < 3:
            continue
        owner = parts[1]
        name = "__".join(parts[2:])
        m[f"{owner}/{name}"] = p
    return m


def _sha_available(repo_dir: str, sha: str) -> bool:
    r = subprocess.run(["git", "-c", "safe.directory=*", "-C", repo_dir, "cat-file", "-e", sha], capture_output=True)
    return r.returncode == 0


def _fetch_sha(repo_dir: str, sha: str, repo_slug: str, gh_token: str | None) -> bool:
    url = f"https://github.com/{repo_slug}.git"
    if gh_token:
        url = f"https://{gh_token}@github.com/{repo_slug}.git"
    r = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", repo_dir, "fetch", "--depth=1", url, sha],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return r.returncode == 0


def _mount_overlay(repo_path: str, tag: str) -> tuple[str, str, str]:
    upper = os.path.join(OVERLAY_SHM_BASE, f"ovl-upper-{tag}")
    work = os.path.join(OVERLAY_SHM_BASE, f"ovl-work-{tag}")
    merged = os.path.join(OVERLAY_MERGED_BASE, tag)
    for d in (upper, work, merged):
        os.makedirs(d, exist_ok=True)
    subprocess.run(
        ["fuse-overlayfs", "-o", f"lowerdir={repo_path},upperdir={upper},workdir={work}", merged],
        check=True,
        capture_output=True,
    )
    return merged, upper, work


def _umount_overlay(merged: str, upper: str, work: str):
    for cmd in (["fusermount3", "-u", merged], ["fusermount3", "-u", "-z", merged]):
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        if r.returncode == 0:
            break
    for d in (upper, work):
        shutil.rmtree(d, ignore_errors=True)


def _create_worktree(repo_path: str, tag: str) -> str:
    os.makedirs(WORKTREE_BASE, exist_ok=True)
    wt = os.path.join(WORKTREE_BASE, tag)
    os.makedirs(wt, exist_ok=True)
    subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", repo_path, "worktree", "add", "--detach", wt, "HEAD"],
        check=True,
        capture_output=True,
    )
    return wt


def _remove_worktree(repo_path: str, wt: str):
    subprocess.run(["git", "-c", "safe.directory=*", "-C", repo_path, "worktree", "remove", "--force", wt], capture_output=True)
    shutil.rmtree(wt, ignore_errors=True)


def _load_model_cfg(model_name: str) -> dict:
    cfg = yaml.safe_load(open(MODELS_YAML))
    for m in cfg.get("model_list", []):
        if m.get("model_name") == model_name:
            return m
    raise ValueError(f"model_name={model_name} not found in models.yaml")


def _normalize_api_base(base: str) -> str:
    b = (base or "").rstrip("/")
    if not b.endswith("/v1"):
        b = b + "/v1"
    return b


def _fetch_tasks(limit: int, seed: int) -> list[dict]:
    db = _load_pg_cfg()
    conn = pg8000.native.Connection(**db)
    rows = conn.run(
        """
        SELECT
            repo, instance_id, pull_number, base_sha, pr_title, pr_body,
            problem_statement, hints_text, changed_files, additions, deletions,
            requested_reviewers, closing_issue_id
        FROM prs_copy
        WHERE pr_merged = TRUE
          AND base_sha IS NOT NULL
          AND patch IS NOT NULL
          AND changed_files BETWEEN 1 AND 60
        ORDER BY created_at DESC
        LIMIT :lim
        """,
        lim=limit * 6,
    )
    conn.close()
    out = []
    for r in rows:
        out.append(
            {
                "repo": r[0],
                "instance_id": r[1],
                "pull_number": int(r[2] or 0),
                "base_sha": r[3],
                "pr_title": r[4] or "",
                "pr_body": (r[5] or "")[:2000],
                "problem_statement": (r[6] or "")[:2000],
                "hints_text": (r[7] or "")[:1200],
                "changed_files": int(r[8] or 0),
                "additions": int(r[9] or 0),
                "deletions": int(r[10] or 0),
                "requested_reviewers": r[11],
                "closing_issue_id": r[12],
            }
        )
    random.Random(seed).shuffle(out)
    return out[:limit]


def _task_prompt(t: dict) -> str:
    if t["problem_statement"]:
        core = t["problem_statement"]
    else:
        core = (
            f"PR title: {t['pr_title']}\n\n"
            f"PR body (truncated):\n{t['pr_body']}\n\n"
            "Please make focused code improvements related to this PR context."
        )
    hints = f"\nHints:\n{t['hints_text']}\n" if t["hints_text"] else ""
    return (
        f"{core}\n{hints}\n"
        "Goal: produce a high-quality, merge-ready patch with minimal unnecessary scope.\n"
        "Run relevant tests/checks if possible, then finish with COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT."
    )


def _run_one_task(
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
) -> dict:
    repo = t["repo"]
    repo_dir = repo_dirs.get(repo)
    if not repo_dir:
        return {"repo": repo, "instance_id": t["instance_id"], "status": "missing_repo"}

    tag = f"p47_{abs(hash((repo, t['instance_id']))) % (10**12)}"
    merged = upper = work = None
    workspace_method = ""
    t0 = time.time()
    try:
        # Prefer overlayfs when available; fallback to git worktree.
        if shutil.which("fuse-overlayfs") and shutil.which("fusermount3"):
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
                return {
                    "repo": repo,
                    "instance_id": t["instance_id"],
                    "status": "missing_base_sha",
                }
        ck = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", merged, "checkout", "--force", t["base_sha"]],
            capture_output=True,
            text=True,
        )
        if ck.returncode != 0:
            return {
                "repo": repo,
                "instance_id": t["instance_id"],
                "status": "checkout_failed",
                "reason": (ck.stderr or ck.stdout)[:300],
            }

        os.makedirs(out_traj_dir, exist_ok=True)
        traj = os.path.join(out_traj_dir, f"{t['instance_id']}.traj.json")
        base_url = _normalize_api_base(api_base_override or model_cfg["litellm_params"]["api_base"])
        api_key = api_key_override or model_cfg["litellm_params"]["api_key"]
        # mini-swe-agent uses litellm directly, so model name must include provider prefix
        # (e.g. hosted_vllm/<alias>) rather than a raw backend model id/path.
        model_name = litellm_model_override or model_cfg["litellm_params"]["model"]
        task = _task_prompt(t)

        env = os.environ.copy()
        env["PYTHONPATH"] = MINI_SRC + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["MSWEA_SILENT_STARTUP"] = "1"
        env["MSWEA_CONFIGURED"] = "true"
        env["MSWEA_MODEL_NAME"] = model_name
        cmd = [
            agent_python,
            "-m",
            "minisweagent.run.mini",
            "--model",
            model_name,
            "--model-class",
            "litellm",
            "--agent-class",
            "default",
            "--environment-class",
            "local",
            "--task",
            task,
            "-o",
            traj,
            "--yolo",
            "--exit-immediately",
            "-c",
            MINI_CFG,
            "-c",
            f"environment.cwd={merged}",
            "-c",
            f"agent.step_limit={mini_step_limit}",
            "-c",
            "model.cost_tracking=ignore_errors",
            "-c",
            f"model.model_kwargs.api_base={base_url}",
            "-c",
            f"model.model_kwargs.api_key={api_key}",
            "-c",
            "model.model_kwargs.temperature=0.0",
        ]
        rr = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_sec)

        out = {
            "repo": repo,
            "instance_id": t["instance_id"],
            "pull_number": t["pull_number"],
            "status": "ok" if rr.returncode == 0 else "agent_failed",
            "returncode": rr.returncode,
            "elapsed_sec": time.time() - t0,
            "workspace_method": workspace_method,
            "traj_path": traj if os.path.exists(traj) else "",
            "stdout_tail": (rr.stdout or "")[-2000:],
            "stderr_tail": (rr.stderr or "")[-1200:],
        }
        gd = subprocess.run(["git", "-c", "safe.directory=*", "-C", merged, "diff", "--name-only"], capture_output=True, text=True)
        files = [ln.strip() for ln in gd.stdout.splitlines() if ln.strip()] if gd.returncode == 0 else []
        out["changed_files_after"] = len(files)
        out["changed_files_list_head"] = files[:20]

        if out["traj_path"]:
            try:
                tj = json.load(open(out["traj_path"]))
                info = tj.get("info", {})
                out["exit_status"] = info.get("exit_status", "")
                out["submission"] = info.get("submission", "")
                ms = info.get("model_stats", {})
                out["model_calls"] = ms.get("api_calls", 0)
                out["model_cost"] = ms.get("instance_cost", 0.0)
            except Exception as e:
                out["traj_parse_error"] = str(e)[:300]
        return out
    except subprocess.TimeoutExpired:
        return {"repo": repo, "instance_id": t["instance_id"], "status": "timeout", "elapsed_sec": time.time() - t0}
    except Exception as e:
        return {"repo": repo, "instance_id": t["instance_id"], "status": "error", "reason": str(e)[:500]}
    finally:
        if workspace_method == "overlayfs" and merged and upper and work:
            _umount_overlay(merged, upper, work)
        elif workspace_method == "git_worktree" and merged:
            _remove_worktree(repo_dir, merged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="qwen3_coder_30b")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--step-limit", type=int, default=20)
    ap.add_argument("--timeout-sec", type=int, default=900)
    ap.add_argument("--agent-python", default="/usr/bin/python3")
    ap.add_argument("--api-base", default=None, help="Override model api_base from models.yaml")
    ap.add_argument("--api-key", default=None, help="Override model api_key from models.yaml")
    ap.add_argument("--litellm-model", default=None, help="Override model id/name used by litellm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-jsonl", default=OUT_JSONL)
    ap.add_argument("--out-summary", default=OUT_SUMMARY)
    ap.add_argument("--traj-dir", default=os.path.join(ROOT, "docs", "phase4_7_trajectories"))
    args = ap.parse_args()

    t0 = time.time()
    model_cfg = _load_model_cfg(args.model_name)
    repo_dirs = _repo_dir_map()
    tokens = _load_tokens()
    tasks = _fetch_tasks(args.limit, args.seed)
    tasks = [t for t in tasks if t["repo"] in repo_dirs]
    print(f"tasks selected: {len(tasks)} (requested={args.limit})", flush=True)
    print(f"concurrency={args.concurrency}", flush=True)

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    os.makedirs(args.traj_dir, exist_ok=True)
    rows = []
    out_handle = open(args.out_jsonl, "w")
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = [
            ex.submit(
                _run_one_task,
                t,
                model_cfg,
                repo_dirs,
                tokens,
                args.traj_dir,
                args.step_limit,
                args.timeout_sec,
                args.agent_python,
                args.api_base,
                args.api_key,
                args.litellm_model,
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
    out_handle.close()

    summary = {
        "date_utc": datetime.now(UTC).isoformat(),
        "args": vars(args),
        "n_tasks": len(rows),
        "status_counts": {},
        "ok_rate": 0.0,
        "avg_elapsed_sec": 0.0,
        "avg_changed_files_after": 0.0,
        "elapsed_total_sec": time.time() - t0,
        "out_jsonl": args.out_jsonl,
    }
    for r in rows:
        s = r.get("status", "unknown")
        summary["status_counts"][s] = summary["status_counts"].get(s, 0) + 1
    if rows:
        summary["ok_rate"] = summary["status_counts"].get("ok", 0) / len(rows)
        summary["avg_elapsed_sec"] = sum(float(r.get("elapsed_sec", 0.0)) for r in rows) / len(rows)
        summary["avg_changed_files_after"] = sum(float(r.get("changed_files_after", 0.0)) for r in rows) / len(rows)

    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
