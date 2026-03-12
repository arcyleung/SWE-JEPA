#!/usr/bin/env python3
"""Extract cumulative PR commit snapshots and test Conway trend hypotheses.

For each merged multi-commit PR in prs_copy:
- read the ordered commit list from `commits`
- compute cumulative patch snapshots: `git diff base_sha..<commit_i>`
- extract Conway patch features for each snapshot
- align review events to commit intervals

Outputs:
- JSONL with one row per cumulative snapshot
- JSON summary with first->final and post-review delta statistics
- Markdown report with the same key findings
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pg8000.native
import yaml

from build_pr_mdp_dataset_v51 import REFACTOR_RE
from extract_conway_patch_features import extract_features, _repo_dir_map

ROOT = os.path.dirname(os.path.abspath(__file__))
PG_CONFIG_FILE = os.path.join(ROOT, "postgres_connection.yaml")
TOKENS_YAML = os.path.join(ROOT, "crawl_tokens.yaml")
OUT_JSONL = os.path.join(ROOT, "data", "phase4_7_2_pr_refinement_history.jsonl")
OUT_SUMMARY = os.path.join(ROOT, "data", "phase4_7_2_pr_refinement_history_summary.json")
OUT_REPORT = os.path.join(ROOT, "docs", "phase4_7_2_pr_refinement_history.md")
GIT_TIMEOUT_SEC = int(os.environ.get("PR_REFINEMENT_GIT_TIMEOUT_SEC", "600"))

LOWER_BETTER_METRICS = [
    "conway_risk_proxy",
    "conway_risk_flags",
    "cross_module_spread",
    "boundary_density",
    "shared_change_isolated",
    "api_change_without_tests",
    "schema_change_without_migration",
    "boundary_crossing_without_obs",
    "public_api_without_docs",
    "dependency_change_without_tests",
    "external_io_without_safety",
    "ownership_diffusion",
    "blame_unique_authors",
    "blame_multi_author_hunks",
    "trust_boundary_crossings",
    "security_risk_score",
]

HIGHER_BETTER_METRICS = [
    "error_contract_score",
    "operability_score",
    "test_file_ratio",
    "docs_file_ratio",
]

_WRITE_LOCK = threading.Lock()


def _load_db() -> pg8000.native.Connection:
    cfg = yaml.safe_load(open(PG_CONFIG_FILE))
    return pg8000.native.Connection(
        host=cfg["ip"],
        port=cfg.get("port", 9999),
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
    )


def _j(v: Any) -> Any:
    if isinstance(v, (list, dict)):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return []
    return []


def _load_tokens() -> list[str]:
    try:
        cfg = yaml.safe_load(open(TOKENS_YAML))
        toks = cfg.get("gh_tokens", []) or []
        return [str(t).strip() for t in toks if str(t).strip()]
    except Exception:
        return []


def _parse_ts(v: Any) -> dt.datetime | None:
    if not v:
        return None
    if isinstance(v, dt.datetime):
        return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)
    s = str(v).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        out = dt.datetime.fromisoformat(s)
        return out if out.tzinfo else out.replace(tzinfo=dt.timezone.utc)
    except Exception:
        return None


def _iso(v: dt.datetime | None) -> str | None:
    return v.isoformat() if v else None


def _run_git(repo_dir: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", repo_dir, *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(
            ["git", "-c", "safe.directory=*", "-C", repo_dir, *args],
            124,
            stdout=e.stdout or "",
            stderr=(e.stderr or "") + f"\nTIMEOUT after {GIT_TIMEOUT_SEC}s",
        )


def _git_fetch(repo_dir: str, args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", repo_dir, "fetch", *args],
            capture_output=True,
            text=True,
            timeout=min(timeout, GIT_TIMEOUT_SEC),
        )
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(
            ["git", "-c", "safe.directory=*", "-C", repo_dir, "fetch", *args],
            124,
            stdout=e.stdout or "",
            stderr=(e.stderr or "") + f"\nTIMEOUT after {min(timeout, GIT_TIMEOUT_SEC)}s",
        )


def _sha_available(repo_dir: str, sha: str) -> bool:
    if not sha:
        return False
    rr = _run_git(repo_dir, ["cat-file", "-e", f"{sha}^{{commit}}"])
    return rr.returncode == 0


def _github_remote(repo_slug: str, gh_token: str | None) -> str:
    if gh_token:
        return f"https://{gh_token}@github.com/{repo_slug}.git"
    return f"https://github.com/{repo_slug}.git"


def _try_fetch_pr_history(
    repo_dir: str,
    repo_slug: str,
    pull_number: int,
    head_branch: str | None,
    commit_shas: list[str],
    gh_tokens: list[str],
) -> str:
    wanted = [sha for sha in commit_shas if sha]
    if wanted and all(_sha_available(repo_dir, sha) for sha in wanted):
        return "local"

    auths: list[str | None] = [None]
    for tok in gh_tokens:
        if tok not in auths:
            auths.append(tok)

    if pull_number > 0:
        for tok in auths:
            rr = _git_fetch(
                repo_dir,
                [
                    "--no-tags",
                    "--depth=4096",
                    _github_remote(repo_slug, tok),
                    f"+refs/pull/{pull_number}/head:refs/remotes/origin/pr/{pull_number}",
                ],
                timeout=300,
            )
            if rr.returncode == 0 and all(_sha_available(repo_dir, sha) for sha in wanted):
                return "pull_ref"

    if head_branch:
        safe_head = re.sub(r"[^A-Za-z0-9._/-]+", "_", str(head_branch).strip())
        if safe_head:
            for tok in auths:
                rr = _git_fetch(
                    repo_dir,
                    [
                        "--no-tags",
                        "--depth=4096",
                        _github_remote(repo_slug, tok),
                        f"+refs/heads/{head_branch}:refs/remotes/origin/recovered/{safe_head}",
                    ],
                    timeout=300,
                )
                if rr.returncode == 0 and all(_sha_available(repo_dir, sha) for sha in wanted):
                    return "head_branch"

    missing = [sha for sha in wanted if not _sha_available(repo_dir, sha)]
    for sha in missing:
        ok = False
        for tok in auths:
            rr = _git_fetch(
                repo_dir,
                ["--depth=1", _github_remote(repo_slug, tok), sha],
                timeout=180,
            )
            if rr.returncode == 0 and _sha_available(repo_dir, sha):
                ok = True
                break
        if not ok:
            return "unrecovered"
    return "direct_sha" if missing else "local"


def _commit_list(commits_raw: Any) -> list[dict[str, Any]]:
    commits = _j(commits_raw)
    if not isinstance(commits, list):
        return []
    out: list[dict[str, Any]] = []
    for idx, c in enumerate(commits):
        if not isinstance(c, dict):
            continue
        node = c.get("commit") if isinstance(c.get("commit"), dict) else c
        author = node.get("author") if isinstance(node.get("author"), dict) else {}
        author_user = author.get("user") if isinstance(author.get("user"), dict) else {}
        sha = str(node.get("oid") or c.get("hash") or "").strip()
        if not sha:
            continue
        committed = (
            _parse_ts(node.get("committedDate"))
            or _parse_ts(node.get("committed_date"))
            or _parse_ts(c.get("committed_date"))
            or _parse_ts(c.get("committedDate"))
            or _parse_ts(author.get("date"))
        )
        authored = (
            _parse_ts(node.get("authoredDate"))
            or _parse_ts(node.get("authored_date"))
            or _parse_ts(c.get("authored_date"))
            or _parse_ts(c.get("authoredDate"))
            or _parse_ts(author.get("date"))
            or committed
        )
        out.append(
            {
                "commit_idx_hint": idx,
                "hash": sha,
                "authored_date": authored,
                "committed_date": committed,
                "message_headline": str(
                    node.get("messageHeadline")
                    or c.get("message_headline")
                    or node.get("message")
                    or c.get("message")
                    or ""
                ).splitlines()[0][:300],
                "author_name": author.get("name") or c.get("author_name"),
                "author_github": author_user.get("login") or c.get("author_github"),
            }
        )
    out.sort(key=lambda c: ((c["committed_date"] or dt.datetime.min.replace(tzinfo=dt.timezone.utc)), c["commit_idx_hint"]))
    for idx, c in enumerate(out):
        c["commit_idx"] = idx + 1
    return out


def _sample_commits(commits: list[dict[str, Any]], max_snapshots: int) -> list[dict[str, Any]]:
    if max_snapshots <= 0 or len(commits) <= max_snapshots:
        return commits
    if max_snapshots == 1:
        return [commits[-1]]
    positions: list[int] = []
    for i in range(max_snapshots):
        pos = round(i * (len(commits) - 1) / (max_snapshots - 1))
        if not positions or pos != positions[-1]:
            positions.append(pos)
    if positions[-1] != len(commits) - 1:
        positions[-1] = len(commits) - 1
    return [commits[pos] for pos in positions]


def _review_events(review_threads_raw: Any, submitted_reviews_raw: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for th in _j(review_threads_raw):
        if not isinstance(th, dict):
            continue
        thread_id = th.get("thread_id") or th.get("id")
        file_path = th.get("file_path") or th.get("path")
        comments = th.get("comments", [])
        if isinstance(comments, dict):
            comments = comments.get("nodes", [])
        for c in _j(comments):
            if not isinstance(c, dict):
                continue
            body = str(c.get("body") or "")
            author = c.get("author") if isinstance(c.get("author"), dict) else {}
            commit = c.get("commit") if isinstance(c.get("commit"), dict) else {}
            created_at = _parse_ts(c.get("created_at")) or _parse_ts(c.get("createdAt"))
            if not created_at:
                continue
            events.append(
                {
                    "kind": "review_comment",
                    "created_at": created_at,
                    "commit_hash": str(c.get("commit_hash") or commit.get("oid") or "").strip(),
                    "body": body,
                    "file_path": file_path or c.get("file_path") or c.get("path"),
                    "author": c.get("author") if not author else author.get("login") or author.get("name"),
                    "thread_id": thread_id,
                    "is_refactor": bool(REFACTOR_RE.search(body)),
                }
            )
    for r in _j(submitted_reviews_raw):
        if not isinstance(r, dict):
            continue
        author = r.get("author") if isinstance(r.get("author"), dict) else {}
        submitted_at = _parse_ts(r.get("submitted_at")) or _parse_ts(r.get("submittedAt"))
        if not submitted_at:
            continue
        state = str(r.get("state") or "").upper()
        events.append(
            {
                "kind": "submitted_review",
                "created_at": submitted_at,
                "state": state,
                "reviewer": r.get("reviewer") or author.get("login") or author.get("name"),
                "body": str(r.get("body") or ""),
                "is_refactor": False,
            }
        )
    events.sort(key=lambda e: e["created_at"])
    return events


def _events_before(events: list[dict[str, Any]], cutoff: dt.datetime | None) -> list[dict[str, Any]]:
    if cutoff is None:
        return []
    return [e for e in events if e["created_at"] <= cutoff]


def _events_between(events: list[dict[str, Any]], start: dt.datetime | None, end: dt.datetime | None) -> list[dict[str, Any]]:
    if end is None:
        return []
    if start is None:
        return [e for e in events if e["created_at"] <= end]
    return [e for e in events if start < e["created_at"] <= end]


def _count_states(events: list[dict[str, Any]]) -> tuple[int, int]:
    approvals = 0
    changes_requested = 0
    for e in events:
        if e.get("kind") != "submitted_review":
            continue
        state = str(e.get("state") or "")
        if state == "APPROVED":
            approvals += 1
        elif state == "CHANGES_REQUESTED":
            changes_requested += 1
    return approvals, changes_requested


def _linked_comment_count(events: list[dict[str, Any]], commit_sha: str) -> int:
    return sum(1 for e in events if e.get("kind") == "review_comment" and e.get("commit_hash") == commit_sha)


def _file_patch_stub(paths: list[str]) -> list[dict[str, str]]:
    return [{"file": p} for p in paths if p]


def _conway_risk_proxy(feats: dict[str, Any]) -> tuple[float, int]:
    flag_keys = [
        "api_change_without_tests",
        "schema_change_without_migration",
        "boundary_crossing_without_obs",
        "public_api_without_docs",
        "dependency_change_without_tests",
        "shared_change_isolated",
        "external_io_without_safety",
        "has_shell_true",
        "has_eval_exec",
        "has_pickle_loads",
        "has_sql_fstring",
        "has_hardcoded_cred",
        "ext_client_no_obs",
        "ext_client_no_log",
    ]
    flag_count = sum(int(float(feats.get(k, 0.0)) > 0.0) for k in flag_keys)
    score = (
        1.0 * flag_count
        + 0.35 * float(feats.get("cross_module_spread", 0.0))
        + 0.75 * float(feats.get("boundary_density", 0.0))
        + 1.0 * float(feats.get("ownership_diffusion", 0.0))
        + 0.25 * float(feats.get("blame_multi_author_hunks", 0.0))
        + 0.20 * float(feats.get("trust_boundary_crossings", 0.0))
        + 0.40 * float(feats.get("security_risk_score", 0.0))
        - 0.20 * max(0.0, float(feats.get("error_contract_score", 0.0)))
        - 0.20 * max(0.0, float(feats.get("operability_score", 0.0)))
    )
    return float(score), int(flag_count)


def _extract_snapshot_rows(
    row: tuple[Any, ...],
    repo_dirs: dict[str, str],
    gh_tokens: list[str],
    max_snapshots: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    (
        repo,
        instance_id,
        pull_number,
        base_sha,
        head_sha,
        head_branch,
        total_commits,
        commits_raw,
        review_threads_raw,
        submitted_reviews_raw,
        created_at,
        merged_at,
    ) = row
    repo_dir = repo_dirs.get(repo)
    if not repo_dir or not os.path.isdir(repo_dir):
        return [], {"repo": repo, "instance_id": instance_id, "status": "missing_repo_dir"}
    if not base_sha or not _sha_available(repo_dir, str(base_sha)):
        return [], {"repo": repo, "instance_id": instance_id, "status": "missing_base_sha"}

    commits = _commit_list(commits_raw)
    if len(commits) < 2:
        return [], {"repo": repo, "instance_id": instance_id, "status": "insufficient_commits"}
    sampled_commits = _sample_commits(commits, max_snapshots)
    recovery_mode = _try_fetch_pr_history(
        repo_dir,
        str(repo),
        int(pull_number or 0),
        str(head_branch or ""),
        [c["hash"] for c in commits],
        gh_tokens,
    )
    events = _review_events(review_threads_raw, submitted_reviews_raw)
    created_dt = _parse_ts(created_at)
    merged_dt = _parse_ts(merged_at)

    rows_out: list[dict[str, Any]] = []
    prev_commit_dt: dt.datetime | None = None
    for sample_idx, commit in enumerate(sampled_commits, start=1):
        commit_sha = commit["hash"]
        commit_dt = commit["committed_date"]
        if not _sha_available(repo_dir, commit_sha):
            return rows_out, {
                "repo": repo,
                "instance_id": instance_id,
                "status": "missing_commit_sha",
                "commit_sha": commit_sha,
                "recovery_mode": recovery_mode,
                "emitted_rows": len(rows_out),
            }
        diff_rr = _run_git(repo_dir, ["diff", "--find-renames", "--find-copies=50%", "--binary", str(base_sha), commit_sha])
        if diff_rr.returncode != 0:
            return rows_out, {
                "repo": repo,
                "instance_id": instance_id,
                "status": "diff_failed",
                "commit_sha": commit_sha,
                "recovery_mode": recovery_mode,
                "stderr": diff_rr.stderr[:400],
                "emitted_rows": len(rows_out),
            }
        names_rr = _run_git(repo_dir, ["diff", "--name-only", "--find-renames", str(base_sha), commit_sha])
        if names_rr.returncode != 0:
            return rows_out, {
                "repo": repo,
                "instance_id": instance_id,
                "status": "name_only_failed",
                "commit_sha": commit_sha,
                "recovery_mode": recovery_mode,
                "stderr": names_rr.stderr[:400],
                "emitted_rows": len(rows_out),
            }
        fnames = [line.strip() for line in names_rr.stdout.splitlines() if line.strip()]
        patch = diff_rr.stdout
        events_before = _events_before(events, commit_dt)
        between = _events_between(events, prev_commit_dt, commit_dt)
        review_comments_before = sum(1 for e in events_before if e["kind"] == "review_comment")
        refactor_comments_before = sum(1 for e in events_before if e["kind"] == "review_comment" and e["is_refactor"])
        submitted_reviews_before = sum(1 for e in events_before if e["kind"] == "submitted_review")
        approvals_before, changes_requested_before = _count_states(events_before)
        review_events_between = sum(1 for e in between if e["kind"] == "review_comment")
        refactor_events_between = sum(1 for e in between if e["kind"] == "review_comment" and e["is_refactor"])
        submitted_between = sum(1 for e in between if e["kind"] == "submitted_review")
        features = extract_features(
            patch,
            _file_patch_stub(fnames),
            True,
            review_comments_before,
            review_comments_before,
            "",
            repo_dir=repo_dir,
            base_sha=str(base_sha),
            workspace_dir=None,
        )
        risk_proxy, risk_flags = _conway_risk_proxy(features)
        rows_out.append(
            {
                "repo": repo,
                "instance_id": instance_id,
                "pull_number": int(pull_number or 0),
                "base_sha": str(base_sha),
                "head_sha": str(head_sha or ""),
                "head_branch": str(head_branch or ""),
                "total_commits": int(total_commits or len(commits)),
                "sampled_commit_count": int(len(sampled_commits)),
                "sampled_commit_rank": int(sample_idx),
                "history_recovery_mode": recovery_mode,
                "commit_sha": commit_sha,
                "commit_idx": int(commit["commit_idx"]),
                "commit_message_headline": commit["message_headline"],
                "commit_author_name": commit.get("author_name"),
                "commit_author_github": commit.get("author_github"),
                "authored_date": _iso(commit.get("authored_date")),
                "committed_date": _iso(commit_dt),
                "pr_created_at": _iso(created_dt),
                "pr_merged_at": _iso(merged_dt),
                "snapshot_kind": "cumulative_from_base",
                "snapshot_changed_files": len(fnames),
                "review_comments_before": review_comments_before,
                "refactor_comments_before": refactor_comments_before,
                "submitted_reviews_before": submitted_reviews_before,
                "approvals_before": approvals_before,
                "changes_requested_before": changes_requested_before,
                "review_events_between_prev_commit": review_events_between,
                "refactor_events_between_prev_commit": refactor_events_between,
                "submitted_reviews_between_prev_commit": submitted_between,
                "linked_review_comments": _linked_comment_count(events, commit_sha),
                "is_post_review_revision": int((review_events_between + submitted_between) > 0 and commit["commit_idx"] > 1),
                "conway_risk_proxy": risk_proxy,
                "conway_risk_flags": risk_flags,
                **features,
            }
        )
        prev_commit_dt = commit_dt
    return rows_out, None


def _metric_summary(deltas: list[float], better: str) -> dict[str, float]:
    if not deltas:
        return {"n": 0, "mean_delta": 0.0, "median_delta": 0.0, "improved_fraction": 0.0}
    improved = [d < 0 for d in deltas] if better == "lower" else [d > 0 for d in deltas]
    s = sorted(deltas)
    mid = len(s) // 2
    median = s[mid] if len(s) % 2 == 1 else 0.5 * (s[mid - 1] + s[mid])
    return {
        "n": len(deltas),
        "mean_delta": float(sum(deltas) / len(deltas)),
        "median_delta": float(median),
        "improved_fraction": float(sum(improved) / len(improved)),
    }


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["instance_id"]].append(row)
    for vals in groups.values():
        vals.sort(key=lambda r: r["commit_idx"])

    review_event_total = sum(int(r.get("review_events_between_prev_commit", 0)) for r in rows)
    review_submissions_total = sum(int(r.get("submitted_reviews_between_prev_commit", 0)) for r in rows)
    first_final_deltas: dict[str, list[float]] = defaultdict(list)
    response_deltas: dict[str, list[float]] = defaultdict(list)
    risk_first: list[float] = []
    risk_final: list[float] = []
    risk_response_pre: list[float] = []
    risk_response_post: list[float] = []
    prs_with_review_response = 0
    response_transitions = 0

    for vals in groups.values():
        if len(vals) < 2:
            continue
        first = vals[0]
        final = vals[-1]
        risk_first.append(float(first.get("conway_risk_proxy", 0.0)))
        risk_final.append(float(final.get("conway_risk_proxy", 0.0)))
        for metric in LOWER_BETTER_METRICS + HIGHER_BETTER_METRICS:
            if metric in first and metric in final:
                first_final_deltas[metric].append(float(final[metric]) - float(first[metric]))
        saw_response = False
        for prev, cur in zip(vals, vals[1:]):
            if int(cur.get("review_events_between_prev_commit", 0)) + int(cur.get("submitted_reviews_between_prev_commit", 0)) <= 0:
                continue
            saw_response = True
            response_transitions += 1
            risk_response_pre.append(float(prev.get("conway_risk_proxy", 0.0)))
            risk_response_post.append(float(cur.get("conway_risk_proxy", 0.0)))
            for metric in LOWER_BETTER_METRICS + HIGHER_BETTER_METRICS:
                if metric in prev and metric in cur:
                    response_deltas[metric].append(float(cur[metric]) - float(prev[metric]))
        if saw_response:
            prs_with_review_response += 1

    first_final_summary = {}
    response_summary = {}
    for metric in LOWER_BETTER_METRICS:
        first_final_summary[metric] = _metric_summary(first_final_deltas[metric], better="lower")
        response_summary[metric] = _metric_summary(response_deltas[metric], better="lower")
    for metric in HIGHER_BETTER_METRICS:
        first_final_summary[metric] = _metric_summary(first_final_deltas[metric], better="higher")
        response_summary[metric] = _metric_summary(response_deltas[metric], better="higher")

    def _median(vals: list[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        mid = len(s) // 2
        return float(s[mid] if len(s) % 2 == 1 else 0.5 * (s[mid - 1] + s[mid]))

    return {
        "n_rows": len(rows),
        "n_prs": len(groups),
        "n_prs_multi_commit": sum(1 for vals in groups.values() if len(vals) >= 2),
        "review_comment_events": review_event_total,
        "submitted_review_events": review_submissions_total,
        "n_prs_with_review_response": prs_with_review_response,
        "n_review_response_transitions": response_transitions,
        "median_risk_first": _median(risk_first),
        "median_risk_final": _median(risk_final),
        "median_risk_response_pre": _median(risk_response_pre),
        "median_risk_response_post": _median(risk_response_post),
        "first_to_final": first_final_summary,
        "post_review_transition": response_summary,
    }


def _write_report(
    summary: dict[str, Any],
    out_path: str,
    limit: int | None,
    data_path: str,
    summary_path: str,
) -> None:
    def fmt(v: float) -> str:
        return f"{v:.3f}"

    key_metrics = [
        "conway_risk_proxy",
        "conway_risk_flags",
        "api_change_without_tests",
        "public_api_without_docs",
        "shared_change_isolated",
        "ownership_diffusion",
        "boundary_density",
        "operability_score",
    ]

    lines = [
        "# Experiment 4.7.2 — PR Refinement History and Conway Drift",
        "",
        "## Setup",
        "",
        "- Dataset: merged PRs from `prs_copy` with `total_commits >= 2` and stored commit history.",
        "- Snapshot type: cumulative patch `base_sha..commit_i` for each commit in the PR.",
        "- Review alignment: review-thread comments and submitted reviews are attached by timestamp to commit intervals.",
        f"- PR limit for this run: `{limit if limit is not None else 'all'}`.",
        "",
        "## Main result",
        "",
        f"- PRs analyzed: `{summary['n_prs']}`",
        f"- Commit snapshots analyzed: `{summary['n_rows']}`",
        f"- Review-response transitions: `{summary['n_review_response_transitions']}`",
        f"- Median heuristic Conway risk proxy, first -> final: `{fmt(summary['median_risk_first'])}` -> `{fmt(summary['median_risk_final'])}`",
        f"- Median heuristic Conway risk proxy, pre-review -> post-review response: `{fmt(summary['median_risk_response_pre'])}` -> `{fmt(summary['median_risk_response_post'])}`",
        "",
        "The risk proxy is a heuristic aggregate over Conway-style patch signals. It is used here as a compact summary; the more important evidence is the direction of the individual raw metrics below.",
        "",
        "## Metric deltas",
        "",
        "| Metric | First->final median delta | First->final improved | Post-review median delta | Post-review improved |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in key_metrics:
        ff = summary["first_to_final"][metric]
        pr = summary["post_review_transition"][metric]
        lines.append(
            f"| `{metric}` | {fmt(ff['median_delta'])} | {100.0 * ff['improved_fraction']:.1f}% | "
            f"{fmt(pr['median_delta'])} | {100.0 * pr['improved_fraction']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Negative deltas are better for risk metrics; positive deltas are better for quality metrics such as `operability_score`.",
            "- If post-review transitions improve more consistently than first->final drift, that is stronger evidence that review rounds are actively shaping the Conway state rather than the effect being only due to PR completion.",
            "- This analysis still only sees the surviving PR commit history in `prs_copy`. Force-pushed-away commits are not recovered here.",
            "",
            "## Artifacts",
            "",
            f"- Snapshot rows: `data/{os.path.basename(data_path)}`",
            f"- Summary JSON: `data/{os.path.basename(summary_path)}`",
        ]
    )
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _default_progress_log_path(out_path: str) -> str:
    stem, _ = os.path.splitext(out_path)
    return stem + ".progress.log"


def _default_partial_summary_path(summary_path: str) -> str:
    stem, ext = os.path.splitext(summary_path)
    return stem + ".partial" + ext


def _append_progress_log(path: str, payload: dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_partial_summary(
    path: str,
    rows: list[dict[str, Any]],
    counters: dict[str, int],
    errors: list[dict[str, Any]],
    processed: int,
    total: int,
) -> None:
    summary = _build_summary(rows)
    summary["errors"] = errors[:200]
    summary["ok_prs"] = counters["ok_prs"]
    summary["err_prs"] = counters["err_prs"]
    summary["processed_prs"] = processed
    summary["total_prs"] = total
    summary["progress_fraction"] = float(processed / total) if total > 0 else 0.0
    summary["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--source-limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default=OUT_JSONL)
    ap.add_argument("--summary-out", default=OUT_SUMMARY)
    ap.add_argument("--report-out", default=OUT_REPORT)
    ap.add_argument("--progress-log", default=None)
    ap.add_argument("--partial-summary-out", default=None)
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--max-snapshots", type=int, default=5)
    ap.add_argument("--min-commits", type=int, default=2)
    ap.add_argument("--order", choices=("newest", "oldest", "random"), default="newest")
    ap.add_argument("--min-files", type=int, default=1)
    ap.add_argument("--max-files", type=int, default=40)
    ap.add_argument("--min-lines", type=int, default=10)
    ap.add_argument("--max-lines", type=int, default=4000)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--instance-id", default=None)
    ap.add_argument("--instance-ids-file", default=None)
    ap.add_argument("--require-review-events", action="store_true")
    args = ap.parse_args()

    conn = _load_db()
    order_sql = {
        "newest": "created_at DESC NULLS LAST",
        "oldest": "created_at ASC NULLS LAST",
        "random": "random()",
    }[args.order]
    source_limit = args.source_limit or args.limit
    repo_clause = "AND repo = :repo_filter" if args.repo else ""
    iid_clause = "AND instance_id = :iid_filter" if args.instance_id else ""
    review_clause = "AND (COALESCE(total_review_threads, 0) > 0 OR COALESCE(submitted_reviews::text, '') <> '')" if args.require_review_events else ""
    instance_ids_filter: set[str] = set()
    if args.instance_ids_file:
        with open(args.instance_ids_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    instance_ids_filter.add(line)
    iid_list_clause = ""
    if instance_ids_filter:
        literals = ", ".join("'" + iid.replace("'", "''") + "'" for iid in sorted(instance_ids_filter))
        iid_list_clause = f"AND instance_id IN ({literals})"
    rows = conn.run(
        f"""
        WITH latest AS (
            SELECT DISTINCT ON (instance_id)
                repo, instance_id, pull_number,
                base_sha, head_sha, head_branch,
                total_commits, commits,
                review_threads, submitted_reviews,
                created_at, merged_at, crawl_time,
                changed_files, additions, deletions
            FROM prs_copy
            WHERE pr_merged = TRUE
              AND base_sha IS NOT NULL
              AND commits IS NOT NULL
              {review_clause}
              {repo_clause}
              {iid_clause}
              {iid_list_clause}
            ORDER BY instance_id, crawl_time DESC NULLS LAST
        )
        SELECT
            repo, instance_id, pull_number,
            base_sha, head_sha, head_branch,
            total_commits, commits,
            review_threads, submitted_reviews,
            created_at, merged_at
        FROM latest
        WHERE jsonb_array_length(commits::jsonb) >= :min_commits
          AND changed_files BETWEEN :min_files AND :max_files
          AND (COALESCE(additions, 0) + COALESCE(deletions, 0)) BETWEEN :min_lines AND :max_lines
        ORDER BY {order_sql}
        LIMIT :source_lim
        """,
        min_commits=args.min_commits,
        source_lim=source_limit,
        min_files=args.min_files,
        max_files=args.max_files,
        min_lines=args.min_lines,
        max_lines=args.max_lines,
        repo_filter=args.repo,
        iid_filter=args.instance_id,
    )
    conn.close()
    print(f"Fetched {len(rows)} PR rows")

    repo_dirs = _repo_dir_map()
    gh_tokens = _load_tokens()
    args.progress_log = args.progress_log or _default_progress_log_path(args.out)
    args.partial_summary_out = args.partial_summary_out or _default_partial_summary_path(args.summary_out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.progress_log), exist_ok=True)
    os.makedirs(os.path.dirname(args.partial_summary_out), exist_ok=True)

    counters = {"ok_prs": 0, "err_prs": 0, "rows": 0}
    errors: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    total_futs = len(rows)
    progress_every = max(1, int(args.progress_every))

    with open(args.out, "w"):
        pass
    with open(args.progress_log, "w") as f:
        f.write(
            json.dumps(
                {
                    "event": "start",
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "total_prs": total_futs,
                    "workers": int(args.workers),
                    "out": args.out,
                    "summary_out": args.summary_out,
                    "partial_summary_out": args.partial_summary_out,
                },
                sort_keys=True,
            )
            + "\n"
        )

    with open(args.out, "a") as out_f, ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(_extract_snapshot_rows, row, repo_dirs, gh_tokens, args.max_snapshots) for row in rows]
        for idx, fut in enumerate(as_completed(futs), start=1):
            snapshot_rows, err = fut.result()
            with _WRITE_LOCK:
                if snapshot_rows:
                    all_rows.extend(snapshot_rows)
                    counters["ok_prs"] += 1
                    counters["rows"] += len(snapshot_rows)
                    for row in snapshot_rows:
                        out_f.write(json.dumps(row) + "\n")
                    out_f.flush()
                if err:
                    counters["err_prs"] += 1
                    errors.append(err)
                if idx % 25 == 0:
                    print(
                        f"  processed={idx} ok_prs={counters['ok_prs']} err_prs={counters['err_prs']} rows={counters['rows']}",
                        flush=True,
                    )
                if idx % progress_every == 0 or idx == total_futs:
                    payload = {
                        "event": "progress",
                        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "processed_prs": idx,
                        "total_prs": total_futs,
                        "ok_prs": counters["ok_prs"],
                        "err_prs": counters["err_prs"],
                        "rows": counters["rows"],
                        "progress_fraction": float(idx / total_futs) if total_futs > 0 else 0.0,
                        "last_error_status": (errors[-1].get("status") if errors else None),
                    }
                    _append_progress_log(args.progress_log, payload)
                    _write_partial_summary(args.partial_summary_out, all_rows, counters, errors, idx, total_futs)

    all_rows.sort(key=lambda r: (r["repo"], r["instance_id"], r["commit_idx"]))
    with open(args.out, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    summary = _build_summary(all_rows)
    summary["errors"] = errors[:200]
    summary["ok_prs"] = counters["ok_prs"]
    summary["err_prs"] = counters["err_prs"]
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    _write_report(summary, args.report_out, args.limit, args.out, args.summary_out)
    _append_progress_log(
        args.progress_log,
        {
            "event": "done",
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "processed_prs": total_futs,
            "total_prs": total_futs,
            "ok_prs": counters["ok_prs"],
            "err_prs": counters["err_prs"],
            "rows": counters["rows"],
            "out": args.out,
            "summary_out": args.summary_out,
            "report_out": args.report_out,
        },
    )

    print(f"Done: {counters['rows']} commit snapshots from {counters['ok_prs']} PRs -> {args.out}")
    print(f"Summary -> {args.summary_out}")
    print(f"Report -> {args.report_out}")


if __name__ == "__main__":
    main()
