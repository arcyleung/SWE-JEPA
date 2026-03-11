#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime

import pg8000.native
import yaml

from extract_conway_patch_features import extract_features as extract_conway_patch_features

ROOT = os.path.dirname(os.path.abspath(__file__))
PG_CONFIG_FILE = os.path.join(ROOT, "postgres_connection.yaml")
MODELS_YAML = os.path.join(ROOT, "models.yaml")
TOKENS_YAML = os.path.join(ROOT, "crawl_tokens.yaml")

REPOS_BASE = "/shared_workspace_mfs/repos"
TMP_EXEC_BASE = "/work/repos_tmp_worktrees"
OVERLAY_MERGED_BASE = os.path.join(TMP_EXEC_BASE, "overlay_merged")
OVERLAY_SHM_BASE = "/dev/shm"
WORKTREE_BASE = os.path.join(TMP_EXEC_BASE, "worktrees")

MINI_SRC = os.path.join(ROOT, "agentic_scaffold", "mini-swe-agent", "src")
MINI_CFG = os.path.join(MINI_SRC, "minisweagent", "config", "mini.yaml")

OUT_JSONL = os.path.join(ROOT, "data", "phase4_7_agentic_eval_results_steered.jsonl")
OUT_SUMMARY = os.path.join(ROOT, "data", "phase4_7_agentic_eval_summary_steered.json")


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
    # Use unique worktree paths to avoid collisions across retries/reruns.
    unique = f"{tag}_{os.getpid()}_{int(time.time() * 1000)}_{random.randint(0, 999999)}"
    wt = os.path.join(WORKTREE_BASE, unique)
    # Prune stale worktree metadata before adding.
    subprocess.run(["git", "-c", "safe.directory=*", "-C", repo_path, "worktree", "prune"], capture_output=True)
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
        lim=max(limit * 4, limit),
    )
    conn.close()

    out = []
    seen: set[tuple] = set()
    for r in rows:
        repo = r[0]
        instance_id = r[1]
        pull_number = int(r[2] or 0)
        base_sha = r[3]
        if pull_number > 0:
            key = ("pr", repo, pull_number)
        else:
            key = ("sha", repo, base_sha, instance_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "repo": repo,
                "instance_id": instance_id,
                "pull_number": pull_number,
                "base_sha": base_sha,
                "pr_title": r[4] or "",
                "pr_body": (r[5] or "")[:2000],
                "problem_statement": (r[6] or "")[:2000],
                "hints_text": (r[7] or "")[:1200],
                "changed_files": int(r[8] or 0),
                "additions": int(r[9] or 0),
                "deletions": int(r[10] or 0),
                "requested_reviewers_count": len(r[11] or []),
                "has_closing_issue": 1.0 if r[12] is not None else 0.0,
            }
        )
    random.Random(seed).shuffle(out)
    out = out[:limit]
    for i, t in enumerate(out):
        t["_task_idx"] = i
    return out


def _load_task_keys(path: str) -> list[tuple[str, int]]:
    keys: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
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


def _fetch_tasks_by_keys(task_keys: list[tuple[str, int]], seed: int) -> list[dict]:
    db = _load_pg_cfg()
    conn = pg8000.native.Connection(**db)
    out = []
    seen: set[tuple] = set()
    chunk = 250
    for i in range(0, len(task_keys), chunk):
        part = task_keys[i : i + chunk]
        values = []
        params = {}
        for j, (repo, pull) in enumerate(part):
            params[f"r{j}"] = repo
            params[f"p{j}"] = pull
            values.append(f"((:r{j})::text, (:p{j})::integer)")
        q = f"""
        SELECT
            pc.repo, pc.instance_id, pc.pull_number, pc.base_sha, pc.pr_title, pc.pr_body,
            pc.problem_statement, pc.hints_text, pc.changed_files, pc.additions, pc.deletions,
            pc.requested_reviewers, pc.closing_issue_id
        FROM prs_copy pc
        JOIN (VALUES {', '.join(values)}) AS k(repo, pull_number)
          ON pc.repo = k.repo AND pc.pull_number = k.pull_number
        WHERE pc.pr_merged = TRUE
          AND pc.base_sha IS NOT NULL
          AND pc.patch IS NOT NULL
          AND pc.changed_files BETWEEN 1 AND 60
        """
        rows = conn.run(q, **params)
        for r in rows:
            repo = r[0]
            instance_id = r[1]
            pull_number = int(r[2] or 0)
            base_sha = r[3]
            key = ("pr", repo, pull_number)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "repo": repo,
                    "instance_id": instance_id,
                    "pull_number": pull_number,
                    "base_sha": base_sha,
                    "pr_title": r[4] or "",
                    "pr_body": (r[5] or "")[:2000],
                    "problem_statement": (r[6] or "")[:2000],
                    "hints_text": (r[7] or "")[:1200],
                    "changed_files": int(r[8] or 0),
                    "additions": int(r[9] or 0),
                    "deletions": int(r[10] or 0),
                    "requested_reviewers_count": len(r[11] or []),
                    "has_closing_issue": 1.0 if r[12] is not None else 0.0,
                }
            )
    conn.close()
    random.Random(seed).shuffle(out)
    for i, t in enumerate(out):
        t["_task_idx"] = i
    return out


LEGACY_FEATURES = [
    "is_draft",
    "changed_files",
    "additions",
    "deletions",
    "requested_reviewers_count",
    "has_closing_issue",
]
LEGACY_LOG1P_FEATURES = frozenset(["changed_files", "additions", "deletions", "requested_reviewers_count"])


def _count_patch_lines(patch_text: str) -> tuple[int, int]:
    additions = 0
    deletions = 0
    for line in patch_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1
    return additions, deletions


def _build_patch_feature_map(
    patch_text: str,
    changed_files_list: list[str],
    repo_dir: str | None = None,
    base_sha: str | None = None,
    workspace_dir: str | None = None,
) -> dict[str, float]:
    additions, deletions = _count_patch_lines(patch_text)
    feats = extract_conway_patch_features(
        patch_text,
        changed_files_list,
        False,
        0,
        0,
        "",
        repo_dir=repo_dir,
        base_sha=base_sha,
        workspace_dir=workspace_dir,
    )
    return {
        "changed_files": float(len(changed_files_list)),
        "additions": float(additions),
        "deletions": float(deletions),
        **{k: v for k, v in feats.items() if isinstance(v, (int, float))},
    }


def _legacy_feature_map(task: dict, changed_files_after: int) -> dict[str, float]:
    return {
        "is_draft": 0.0,
        "changed_files": float(max(changed_files_after, 0)),
        "additions": float(max(task.get("additions", 0), 0)),
        "deletions": float(max(task.get("deletions", 0), 0)),
        "requested_reviewers_count": float(max(task.get("requested_reviewers_count", 0), 0)),
        "has_closing_issue": float(task.get("has_closing_issue", 0.0)),
    }


def _vectorize_feature_map(feature_map: dict[str, float], feature_names: list[str], log1p_features: set[str]) -> list[float]:
    out = []
    for name in feature_names:
        value = float(feature_map.get(name, 0.0) or 0.0)
        if name in log1p_features:
            value = math.log1p(max(value, 0.0))
        out.append(value)
    return out


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class Steerer:
    blob: dict
    w_accept: float = 1.0
    w_refactor: float = 1.0
    scope_penalty: float = 0.15
    _feature_names: list[str] = None  # type: ignore[assignment]
    _log1p_features: set[str] = None  # type: ignore[assignment]
    _feature_source: str = ""

    def __post_init__(self):
        self._feature_names = list(self.blob.get("features", LEGACY_FEATURES))
        self._log1p_features = set(self.blob.get("log1p_features", sorted(LEGACY_LOG1P_FEATURES)))
        self._feature_source = self.blob.get("feature_source", "legacy_metadata")

    def _pred_head(self, head: str, x: list[float]) -> float:
        h = self.blob[head]
        mean = h["scaler_mean"]
        scale = h["scaler_scale"]
        coef = h["coef"]
        inter = float(h["intercept"])
        z = inter
        for i in range(len(coef)):
            xi = (x[i] - float(mean[i])) / (float(scale[i]) + 1e-8)
            z += float(coef[i]) * xi
        return _sigmoid(z)

    def score(
        self,
        task: dict,
        changed_files_after: int,
        patch_text: str = "",
        changed_files_list: list[str] | None = None,
        repo_dir: str | None = None,
        base_sha: str | None = None,
        workspace_dir: str | None = None,
    ) -> tuple[float, dict]:
        if self._feature_source == "conway_patch_features":
            feature_map = _build_patch_feature_map(
                patch_text,
                changed_files_list or [],
                repo_dir=repo_dir,
                base_sha=base_sha,
                workspace_dir=workspace_dir,
            )
        else:
            feature_map = _legacy_feature_map(task, changed_files_after)
        x = _vectorize_feature_map(feature_map, self._feature_names, self._log1p_features)
        p_acc = self._pred_head("acceptance", x)
        p_ref = self._pred_head("refactor", x)
        exp_files = max(1.0, float(task.get("changed_files", 1)))
        scope_drift = abs(float(changed_files_after) - exp_files) / exp_files
        s = self.w_accept * p_acc - self.w_refactor * p_ref - self.scope_penalty * scope_drift
        return s, {
            "p_accept": p_acc,
            "p_refactor": p_ref,
            "scope_drift": scope_drift,
            "patch_chars": len(patch_text),
            "feature_source": self._feature_source,
            "blame_unique_authors": float(feature_map.get("blame_unique_authors", 0.0)),
            "blame_top_author_share": float(feature_map.get("blame_top_author_share", 0.0)),
            "ownership_diffusion": float(feature_map.get("ownership_diffusion", 0.0)),
            "api_change_without_tests": float(feature_map.get("api_change_without_tests", 0.0)),
            "shared_change_isolated": float(feature_map.get("shared_change_isolated", 0.0)),
            "boundary_crossing_without_obs": float(feature_map.get("boundary_crossing_without_obs", 0.0)),
            "score": s,
        }


def _task_prompt(t: dict, steer_hint: str = "") -> str:
    if t["problem_statement"]:
        core = t["problem_statement"]
    else:
        core = (
            f"PR title: {t['pr_title']}\n\n"
            f"PR body (truncated):\n{t['pr_body']}\n\n"
            "Please make focused code improvements related to this PR context."
        )
    hints = f"\nHints:\n{t['hints_text']}\n" if t["hints_text"] else ""
    steer = f"\nSteering constraints:\n{steer_hint}\n" if steer_hint else ""
    return (
        f"{core}\n{hints}{steer}\n"
        "Goal: produce a high-quality, merge-ready patch with minimal unnecessary scope.\n"
        "Before making broad edits to shared, public, or long-lived code, inspect local ownership and change history"
        " with `git blame` or `git log -L` on the exact file/function you plan to modify.\n"
        "Run relevant tests/checks if possible, then finish with COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT."
    )


def _changed_files(repo_dir: str) -> list[str]:
    gd = subprocess.run(["git", "-c", "safe.directory=*", "-C", repo_dir, "diff", "--name-only"], capture_output=True, text=True)
    if gd.returncode != 0:
        return []
    return [ln.strip() for ln in gd.stdout.splitlines() if ln.strip()]


def _run_attempt(
    merged: str,
    task_text: str,
    traj: str,
    model_cfg: dict,
    mini_step_limit: int,
    timeout_sec: int,
    agent_python: str,
    api_base_override: str | None,
    api_key_override: str | None,
    litellm_model_override: str | None,
    temperature: float = 0.0,
) -> dict:
    base_url = _normalize_api_base(api_base_override or model_cfg["litellm_params"]["api_base"])
    api_key = api_key_override or model_cfg["litellm_params"]["api_key"]
    model_name = litellm_model_override or model_cfg["litellm_params"]["model"]
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
        task_text,
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
        f"model.model_kwargs.temperature={float(temperature):.3f}",
    ]
    rr = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_sec)
    files = _changed_files(merged)
    out = {
        "returncode": rr.returncode,
        "stdout_tail": (rr.stdout or "")[-2000:],
        "stderr_tail": (rr.stderr or "")[-1200:],
        "changed_files_after": len(files),
        "changed_files_list": files,
        "changed_files_list_head": files[:20],
        "touched_tests": any(("/test" in f.lower()) or ("tests/" in f.lower()) or ("_test." in f.lower()) for f in files),
        "traj_path": traj if os.path.exists(traj) else "",
        "temperature": float(temperature),
    }
    if out["traj_path"]:
        try:
            tj = json.load(open(out["traj_path"]))
            info = tj.get("info", {})
            out["exit_status"] = info.get("exit_status", "")
            ms = info.get("model_stats", {})
            out["model_calls"] = ms.get("api_calls", 0)
            out["model_cost"] = ms.get("instance_cost", 0.0)
        except Exception:
            pass
    return out


def _steer_hint(attempt_idx: int, base_expected_files: int, prev_diag: dict | None) -> str:
    payload = {
        "attempt_idx": int(attempt_idx),
        "p_accept": None,
        "p_refactor": None,
        "scope_drift": None,
        "score": None,
        "blame_unique_authors": None,
        "blame_top_author_share": None,
        "ownership_diffusion": None,
        "api_change_without_tests": None,
        "shared_change_isolated": None,
        "boundary_crossing_without_obs": None,
    }
    if prev_diag is not None:
        payload["p_accept"] = float(prev_diag.get("p_accept", 0.0))
        payload["p_refactor"] = float(prev_diag.get("p_refactor", 0.0))
        payload["scope_drift"] = float(prev_diag.get("scope_drift", 0.0))
        payload["score"] = float(prev_diag.get("score", 0.0))
        payload["blame_unique_authors"] = float(prev_diag.get("blame_unique_authors", 0.0))
        payload["blame_top_author_share"] = float(prev_diag.get("blame_top_author_share", 0.0))
        payload["ownership_diffusion"] = float(prev_diag.get("ownership_diffusion", 0.0))
        payload["api_change_without_tests"] = float(prev_diag.get("api_change_without_tests", 0.0))
        payload["shared_change_isolated"] = float(prev_diag.get("shared_change_isolated", 0.0))
        payload["boundary_crossing_without_obs"] = float(prev_diag.get("boundary_crossing_without_obs", 0.0))
    return json.dumps(payload, ensure_ascii=True)


def _run_one_task(
    t: dict,
    model_cfg: dict,
    repo_dirs: dict[str, str],
    tokens: list[str],
    out_traj_dir: str,
    out_patch_dir: str | None,
    mini_step_limit: int,
    timeout_sec: int,
    agent_python: str,
    api_base_override: str | None,
    api_key_override: str | None,
    litellm_model_override: str | None,
    steerer: Steerer | None,
    steer_max_attempts: int,
    steer_accept_threshold: float,
    steer_refactor_threshold: float,
    steer_retry_temperature: float,
) -> dict:
    repo = t["repo"]
    repo_dir = repo_dirs.get(repo)
    if not repo_dir:
        return {"repo": repo, "instance_id": t["instance_id"], "status": "missing_repo"}

    tag = f"p47s_{t.get('_task_idx', 0)}_{abs(hash((repo, t['pull_number'], t['base_sha'], t['instance_id']))) % (10**12)}"
    merged = upper = work = None
    workspace_method = ""
    t0 = time.time()
    try:
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
                return {"repo": repo, "instance_id": t["instance_id"], "pull_number": t["pull_number"], "status": "missing_base_sha"}

        ck = subprocess.run(["git", "-c", "safe.directory=*", "-C", merged, "checkout", "--force", t["base_sha"]], capture_output=True, text=True)
        if ck.returncode != 0:
            return {
                "repo": repo,
                "instance_id": t["instance_id"],
                "pull_number": t["pull_number"],
                "status": "checkout_failed",
                "reason": (ck.stderr or ck.stdout)[:300],
            }

        os.makedirs(out_traj_dir, exist_ok=True)
        if out_patch_dir:
            os.makedirs(out_patch_dir, exist_ok=True)
        attempts = []
        best = None
        best_diag = None
        best_patch_text = ""

        for ai in range(max(1, steer_max_attempts)):
            subprocess.run(["git", "-c", "safe.directory=*", "-C", merged, "reset", "--hard", t["base_sha"]], capture_output=True)
            hint = _steer_hint(ai, max(1, int(t.get("changed_files", 1))), best_diag if ai > 0 else None)
            task_text = _task_prompt(t, hint if steerer else "")
            traj = os.path.join(out_traj_dir, f"{t['instance_id']}__pr{t['pull_number']}__a{ai}.traj.json")
            attempt_temperature = 0.0 if (not steerer or ai == 0) else min(0.8, float(steer_retry_temperature) * float(ai))
            rr = _run_attempt(
                merged,
                task_text,
                traj,
                model_cfg,
                mini_step_limit,
                timeout_sec,
                agent_python,
                api_base_override,
                api_key_override,
                litellm_model_override,
                attempt_temperature,
            )
            gp = subprocess.run(["git", "-c", "safe.directory=*", "-C", merged, "diff", "--binary"], capture_output=True, text=True)
            patch_text = gp.stdout if gp.returncode == 0 else ""
            diag = {"score": 0.0, "p_accept": 0.0, "p_refactor": 0.0, "scope_drift": 0.0}
            if steerer:
                score, diag = steerer.score(
                    t,
                    rr["changed_files_after"],
                    patch_text=patch_text,
                    changed_files_list=rr.get("changed_files_list", []),
                    repo_dir=repo_dir,
                    base_sha=t.get("base_sha"),
                    workspace_dir=merged,
                )
                diag["score"] = score
            row = {
                "attempt_idx": ai,
                "hint": hint,
                "temperature": rr.get("temperature", attempt_temperature),
                "returncode": rr["returncode"],
                "traj_path": rr["traj_path"],
                "changed_files_after": rr["changed_files_after"],
                "diag": diag,
            }
            attempts.append(row)
            if (best is None) or (diag["score"] > best_diag["score"]):
                best = rr
                best_diag = diag
                best_patch_text = patch_text

            if steerer:
                if diag["p_accept"] >= steer_accept_threshold and diag["p_refactor"] <= steer_refactor_threshold:
                    break
            else:
                break

        out = {
            "repo": repo,
            "instance_id": t["instance_id"],
            "pull_number": t["pull_number"],
            "status": "ok" if best and best["returncode"] == 0 else "agent_failed",
            "returncode": -1 if not best else best["returncode"],
            "elapsed_sec": time.time() - t0,
            "workspace_method": workspace_method,
            "traj_path": "" if not best else best["traj_path"],
            "stdout_tail": "" if not best else best["stdout_tail"],
            "stderr_tail": "" if not best else best["stderr_tail"],
            "changed_files_after": 0 if not best else best["changed_files_after"],
            "changed_files_list_head": [] if not best else best["changed_files_list_head"],
            "steered": bool(steerer),
            "steer_attempts": attempts,
            "steer_best_diag": best_diag,
        }
        if out_patch_dir:
            patch_path = os.path.join(out_patch_dir, f"{t['instance_id']}__pr{t['pull_number']}.patch")
            with open(patch_path, "w") as pf:
                pf.write(best_patch_text)
            out["patch_path"] = patch_path
            out["patch_chars"] = len(best_patch_text)
            out["has_patch"] = len(best_patch_text.strip()) > 0
        return out
    except subprocess.TimeoutExpired:
        return {"repo": repo, "instance_id": t["instance_id"], "pull_number": t["pull_number"], "status": "timeout", "elapsed_sec": time.time() - t0}
    except Exception as e:
        return {"repo": repo, "instance_id": t["instance_id"], "pull_number": t["pull_number"], "status": "error", "reason": str(e)[:500]}
    finally:
        if workspace_method == "overlayfs" and merged and upper and work:
            _umount_overlay(merged, upper, work)
        elif workspace_method == "git_worktree" and merged:
            _remove_worktree(repo_dir, merged)


def _load_steerer(path: str | None, w_accept: float, w_refactor: float, scope_penalty: float) -> Steerer | None:
    if not path:
        return None
    blob = json.load(open(path))
    if "acceptance" not in blob or "refactor" not in blob:
        raise ValueError(f"Invalid steerer model file: {path}")
    return Steerer(blob=blob, w_accept=w_accept, w_refactor=w_refactor, scope_penalty=scope_penalty)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="qwen3_coder_30b")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--step-limit", type=int, default=20)
    ap.add_argument("--timeout-sec", type=int, default=900)
    ap.add_argument("--agent-python", default="/usr/bin/python3")
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--litellm-model", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task-keys-jsonl", default=None, help="Optional JSONL of {repo,pull_number} to run exact cohort")
    ap.add_argument("--out-jsonl", default=OUT_JSONL)
    ap.add_argument("--out-summary", default=OUT_SUMMARY)
    ap.add_argument("--traj-dir", default=os.path.join(ROOT, "data", "phase4_7_trajectories_steered"))
    ap.add_argument("--patch-dir", default=os.path.join(ROOT, "data", "phase4_7_patches_steered"))
    ap.add_argument("--steerer-model", default=os.path.join(ROOT, "data", "phase4_7_pr_steerer_model.json"))
    ap.add_argument("--steer-max-attempts", type=int, default=3)
    ap.add_argument("--steer-accept-threshold", type=float, default=0.65)
    ap.add_argument("--steer-refactor-threshold", type=float, default=0.35)
    ap.add_argument("--steer-w-accept", type=float, default=1.0)
    ap.add_argument("--steer-w-refactor", type=float, default=1.0)
    ap.add_argument("--steer-scope-penalty", type=float, default=0.15)
    ap.add_argument("--steer-retry-temperature", type=float, default=0.25)
    ap.add_argument("--disable-steering", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    model_cfg = _load_model_cfg(args.model_name)
    repo_dirs = _repo_dir_map()
    tokens = _load_tokens()
    if args.task_keys_jsonl:
        keys = _load_task_keys(args.task_keys_jsonl)
        print(f"loaded task keys: {len(keys)} from {args.task_keys_jsonl}", flush=True)
        tasks = _fetch_tasks_by_keys(keys, args.seed)
        if args.limit > 0:
            tasks = tasks[: args.limit]
    else:
        tasks = _fetch_tasks(args.limit, args.seed)
    tasks = [t for t in tasks if t["repo"] in repo_dirs]
    print(f"tasks selected: {len(tasks)} (requested={args.limit})", flush=True)
    print(f"concurrency={args.concurrency}", flush=True)

    steerer = None
    if not args.disable_steering:
        steerer = _load_steerer(args.steerer_model, args.steer_w_accept, args.steer_w_refactor, args.steer_scope_penalty)
        print(f"steerer: enabled ({args.steerer_model})", flush=True)
    else:
        print("steerer: disabled", flush=True)

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    os.makedirs(args.traj_dir, exist_ok=True)
    os.makedirs(args.patch_dir, exist_ok=True)

    rows = []
    with open(args.out_jsonl, "w") as out_handle:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = [
                ex.submit(
                    _run_one_task,
                    t,
                    model_cfg,
                    repo_dirs,
                    tokens,
                    args.traj_dir,
                    args.patch_dir,
                    args.step_limit,
                    args.timeout_sec,
                    args.agent_python,
                    args.api_base,
                    args.api_key,
                    args.litellm_model,
                    steerer,
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
                    ok = sum(1 for x in rows if x.get("status") == "ok")
                    print(f"  [{done}/{len(futs)}] ok={ok}", flush=True)

    summary = {
        "date_utc": datetime.now(UTC).isoformat(),
        "args": vars(args),
        "n_tasks": len(rows),
        "status_counts": {},
        "ok_rate": 0.0,
        "avg_elapsed_sec": 0.0,
        "avg_changed_files_after": 0.0,
        "avg_steer_attempts": 0.0,
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
        summary["avg_steer_attempts"] = sum(len(r.get("steer_attempts", [])) for r in rows) / len(rows)

    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
