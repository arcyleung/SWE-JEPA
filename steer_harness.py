#!/usr/bin/env python3
"""Standalone steering harness for mini-swe-agent (or any agent subprocess).

Runs an agent subprocess, extracts the patch via ``git diff``, scores it with
a trained steerer model, and retries with injected hints if the score is below
threshold. Agent-agnostic: works with mini-swe-agent, OpenHands, or any tool
that modifies a git worktree.

Example::

    python steer_harness.py \
        --steerer-model data/phase4_7_pr_steerer_model.json \
        --repo-dir /tmp/my_repo \
        --base-sha abc123 \
        --task "Fix the login timeout bug" \
        --max-attempts 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# Ensure the project root is importable so steerer.py can find
# extract_conway_patch_features if present.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import steerer module directly to avoid triggering minisweagent.__init__
# dependencies (platformdirs, rich, etc.) that may not be installed.
MINI_SRC = os.path.join(ROOT, "agentic_scaffold", "mini-swe-agent", "src")
_steerer_path = os.path.join(MINI_SRC, "minisweagent", "steerer.py")
import importlib.util
_spec = importlib.util.spec_from_file_location("steerer", _steerer_path)
_steerer_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _steerer_mod
_spec.loader.exec_module(_steerer_mod)
Steerer = _steerer_mod.Steerer
load_steerer = _steerer_mod.load_steerer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git(repo_dir: str, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", repo_dir, *args],
        capture_output=True, text=True, timeout=timeout,
    )


def _changed_files(repo_dir: str) -> list[str]:
    r = _git(repo_dir, "diff", "--name-only")
    if r.returncode != 0:
        return []
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def _get_patch(repo_dir: str) -> str:
    r = _git(repo_dir, "diff", "--binary")
    return r.stdout if r.returncode == 0 else ""


def _reset_hard(repo_dir: str, sha: str):
    _git(repo_dir, "reset", "--hard", sha)


# ---------------------------------------------------------------------------
# Agent runner — runs mini-swe-agent as a subprocess
# ---------------------------------------------------------------------------

def _build_task_prompt(task: str, hint: str = "") -> str:
    steer = f"\nSteering constraints:\n{hint}\n" if hint else ""
    return (
        f"{task}\n{steer}\n"
        "Goal: produce a high-quality, merge-ready patch with minimal unnecessary scope.\n"
        "Before making broad edits to shared, public, or long-lived code, inspect local ownership and change history"
        " with `git blame` or `git log -L` on the exact file/function you plan to modify.\n"
        "Run relevant tests/checks if possible, then finish with COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT."
    )


def _run_agent(
    repo_dir: str,
    task_text: str,
    traj_path: str,
    agent_cmd: list[str],
    timeout_sec: int,
    env_overrides: dict[str, str] | None = None,
) -> dict:
    """Run the agent subprocess and return result metadata."""
    env = os.environ.copy()
    if MINI_SRC not in env.get("PYTHONPATH", ""):
        env["PYTHONPATH"] = MINI_SRC + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if env_overrides:
        env.update(env_overrides)

    cmd = agent_cmd + [
        "--task", task_text,
        "-o", traj_path,
        "-c", f"environment.cwd={repo_dir}",
    ]

    t0 = time.time()
    try:
        rr = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "elapsed": time.time() - t0, "error": "timeout"}

    files = _changed_files(repo_dir)
    result = {
        "returncode": rr.returncode,
        "elapsed": time.time() - t0,
        "changed_files": files,
        "stdout_tail": (rr.stdout or "")[-1000:],
        "stderr_tail": (rr.stderr or "")[-500:],
    }
    if os.path.exists(traj_path):
        try:
            tj = json.load(open(traj_path))
            info = tj.get("info", {})
            result["exit_status"] = info.get("exit_status", "")
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Main steering loop
# ---------------------------------------------------------------------------

def steer_loop(
    steerer: Steerer,
    repo_dir: str,
    base_sha: str,
    task: str,
    agent_cmd: list[str],
    max_attempts: int = 3,
    timeout_sec: int = 1800,
    traj_dir: str = ".",
    expected_files: int = 1,
    retry_temperature: float = 0.3,
) -> dict:
    """Run agent with steering loop. Returns best attempt result."""
    best_score = float("-inf")
    best_patch = ""
    best_diag: dict | None = None
    best_result: dict = {}
    all_attempts = []

    for ai in range(max(1, max_attempts)):
        _reset_hard(repo_dir, base_sha)

        hint = ""
        if ai > 0 and best_diag is not None:
            hint = steerer.hint(best_diag, attempt_idx=ai, expected_files=expected_files)

        task_text = _build_task_prompt(task, hint)
        traj_path = os.path.join(traj_dir, f"attempt_{ai}.traj.json")

        # Inject temperature for retries via env var (agent-specific)
        env_overrides = {}
        if ai > 0:
            env_overrides["STEER_TEMPERATURE"] = f"{min(0.8, retry_temperature * ai):.3f}"

        result = _run_agent(repo_dir, task_text, traj_path, agent_cmd, timeout_sec, env_overrides)

        patch_text = _get_patch(repo_dir)
        files = _changed_files(repo_dir)
        score, diag = steerer.score(
            patch_text, files,
            expected_files=expected_files,
            repo_dir=repo_dir, base_sha=base_sha, workspace_dir=repo_dir,
        )

        attempt = {
            "attempt_idx": ai,
            "score": score,
            "diag": diag,
            "changed_files": len(files),
            "patch_chars": len(patch_text),
            "hint": hint,
            "returncode": result.get("returncode"),
            "elapsed": result.get("elapsed"),
        }
        all_attempts.append(attempt)
        print(f"[steer] attempt {ai}: score={score:.3f}  p_accept={diag['p_accept']:.3f}  "
              f"p_refactor={diag['p_refactor']:.3f}  files={len(files)}  chars={len(patch_text)}")

        if score > best_score:
            best_score = score
            best_patch = patch_text
            best_diag = diag
            best_result = result

        # Accept early if score is above threshold
        if score >= steerer.threshold:
            print(f"[steer] accepted at attempt {ai} (score {score:.3f} >= threshold {steerer.threshold:.3f})")
            break

    return {
        "best_score": best_score,
        "best_patch": best_patch,
        "best_diag": best_diag,
        "best_result": best_result,
        "attempts": all_attempts,
        "n_attempts": len(all_attempts),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Steering harness: run an agent subprocess with score-and-retry loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--steerer-model", required=True, help="Path to steerer model JSON")
    ap.add_argument("--repo-dir", required=True, help="Path to the git repo worktree")
    ap.add_argument("--base-sha", required=True, help="Git SHA to reset to before each attempt")
    ap.add_argument("--task", required=True, help="Task description for the agent")
    ap.add_argument("--max-attempts", type=int, default=3, help="Max steering retries (default: 3)")
    ap.add_argument("--timeout", type=int, default=1800, help="Timeout per attempt in seconds (default: 1800)")
    ap.add_argument("--threshold", type=float, default=0.3, help="Score threshold for early acceptance (default: 0.3)")
    ap.add_argument("--expected-files", type=int, default=1, help="Expected number of changed files")
    ap.add_argument("--retry-temperature", type=float, default=0.3, help="Temperature multiplier for retries")
    ap.add_argument("--traj-dir", default=".", help="Directory to save trajectory files")
    ap.add_argument("--output", default=None, help="Path to write final result JSON")
    ap.add_argument("--agent-cmd", nargs=argparse.REMAINDER, default=None,
                     help="Agent command (everything after --agent-cmd is passed through)")

    args = ap.parse_args()

    steerer = load_steerer(args.steerer_model, threshold=args.threshold)

    # Default agent command: mini-swe-agent
    agent_cmd = args.agent_cmd or [
        sys.executable, "-m", "minisweagent.run.mini",
        "--agent-class", "default",
        "--environment-class", "local",
        "--yolo",
        "--exit-immediately",
    ]

    os.makedirs(args.traj_dir, exist_ok=True)

    result = steer_loop(
        steerer=steerer,
        repo_dir=args.repo_dir,
        base_sha=args.base_sha,
        task=args.task,
        agent_cmd=agent_cmd,
        max_attempts=args.max_attempts,
        timeout_sec=args.timeout,
        traj_dir=args.traj_dir,
        expected_files=args.expected_files,
        retry_temperature=args.retry_temperature,
    )

    # Write best patch to stdout
    print(f"\n[steer] done — {result['n_attempts']} attempts, best score: {result['best_score']:.3f}")

    if args.output:
        # Exclude the raw patch text from JSON output (can be huge)
        out = {k: v for k, v in result.items() if k != "best_patch"}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[steer] results written to {args.output}")


if __name__ == "__main__":
    main()
