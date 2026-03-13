#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

import pg8000.native
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = "/shared_workspace_mfs/akki/scratch_mfs/arthur-task/enriched-all-unmerged"
TOKENS_YAML = os.path.join(ROOT, "crawl_tokens.yaml")
PG_CONFIG_FILE = os.path.join(ROOT, "postgres_connection.yaml")
DEFAULT_JSON = os.path.join(ROOT, "data", "phase4_7_3_closed_pr_ingest_summary.json")
DEFAULT_MD = os.path.join(ROOT, "docs", "phase4_7_3_closed_pr_ingestion.md")

TEST_FILE_RE = re.compile(
    r"(test[_/]|[_/]test|\.test\.|_spec\.|spec[_/]|[_/]spec|__tests?__)",
    re.IGNORECASE,
)
ISSUE_REF_RE = re.compile(r"(?<![A-Za-z0-9_/-])#(\d+)\b")
INT32_MAX = 2_147_483_647

PRS_COPY_COLUMNS = [
    "id",
    "crawl_time",
    "instance_id",
    "repo",
    "pull_number",
    "issue_numbers",
    "base_commit",
    "patch",
    "file_patches",
    "test_patch",
    "test_file_patches",
    "problem_statement",
    "hints_text",
    "pass_to_pass",
    "fail_to_pass",
    "repo_id",
    "stars",
    "forks",
    "primary_language",
    "pr_title",
    "pr_body",
    "pr_url",
    "pr_state",
    "pr_merged",
    "pr_is_draft",
    "pr_author",
    "pr_author_name",
    "pr_labels",
    "base_branch",
    "head_branch",
    "base_sha",
    "head_sha",
    "created_at",
    "updated_at",
    "total_commits",
    "commits",
    "total_comments",
    "comments",
    "total_review_threads",
    "review_threads",
    "requested_reviewers",
    "submitted_reviews",
    "additions",
    "deletions",
    "changed_files",
    "pr_category",
    "pr_category_confidence",
    "pr_category_reasoning",
    "linked_issues",
    "closing_issue_id",
    "merged_at",
    "merged_by",
]

FETCHABLE_COLUMNS = {"patch", "file_patches", "test_patch", "test_file_patches"}
UNAVAILABLE_COLUMNS = {
    "problem_statement",
    "hints_text",
    "pass_to_pass",
    "fail_to_pass",
    "merged_by",
}


def _load_db() -> pg8000.native.Connection:
    cfg = yaml.safe_load(open(PG_CONFIG_FILE))
    return pg8000.native.Connection(
        host=cfg["ip"],
        port=cfg.get("port", 9999),
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
    )


def _load_tokens(path: str) -> list[str]:
    try:
        cfg = yaml.safe_load(open(path))
    except FileNotFoundError:
        return []
    vals = cfg.get("gh_tokens", []) or []
    return [str(v).strip() for v in vals if str(v).strip()]


def _iter_source_rows(source_dir: str):
    base = Path(source_dir)
    for path in sorted(base.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield path.name, json.loads(line)


def _j(v: Any, default: Any = None) -> Any:
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return default
    return default


def _repo_slug(rec: dict[str, Any]) -> str:
    base_repo = (((rec.get("base") or {}).get("repo") or {}).get("full_name") or "").strip()
    if base_repo:
        return base_repo
    owner = str(rec.get("repo_owner") or "").strip()
    name = str(rec.get("repo_name") or "").strip()
    return f"{owner}/{name}".strip("/")


def _repo_id(rec: dict[str, Any]) -> int | None:
    rid = ((rec.get("base") or {}).get("repo") or {}).get("id")
    try:
        return int(rid) if rid is not None else None
    except Exception:
        return None


def _instance_id(rec: dict[str, Any]) -> str:
    rid = _repo_id(rec)
    owner = str(rec.get("repo_owner") or "").strip()
    name = str(rec.get("repo_name") or "").strip()
    if rid is not None and owner and name:
        return f"{rid}__{owner}__{name}"
    slug = _repo_slug(rec).replace("/", "__")
    return slug


def _pr_graphql(rec: dict[str, Any]) -> dict[str, Any]:
    return ((rec.get("graphql_enrichment") or {}).get("pull_request") or {})


def _is_closed_unmerged(rec: dict[str, Any]) -> bool:
    pr = _pr_graphql(rec)
    merged = bool(pr.get("merged") or rec.get("merged_at"))
    state = str(pr.get("state") or rec.get("state") or "").strip().lower()
    closed_at = pr.get("closedAt") or rec.get("closed_at")
    return (not merged) and (state in {"closed", ""}) and bool(closed_at)


def _to_jsonb(v: Any) -> str | None:
    if v is None:
        return None
    return json.dumps(v, ensure_ascii=False)


def _safe_int32(v: Any) -> int | None:
    try:
        iv = int(v)
    except Exception:
        return None
    if -INT32_MAX - 1 <= iv <= INT32_MAX:
        return iv
    return None


def _page_url(url: str, page: int, per_page: int = 100) -> str:
    parsed = urlparse(url)
    q = dict(parse_qsl(parsed.query, keep_blank_values=True))
    q["per_page"] = str(per_page)
    q["page"] = str(page)
    return urlunparse(parsed._replace(query=urlencode(q)))


class GitHubClient:
    def __init__(self, tokens: list[str], timeout_sec: float = 30.0):
        self.tokens = list(tokens)
        random.shuffle(self.tokens)
        self.timeout_sec = timeout_sec
        self._idx = 0
        self._lock = threading.Lock()

    def _next_token(self) -> str | None:
        if not self.tokens:
            return None
        with self._lock:
            tok = self.tokens[self._idx % len(self.tokens)]
            self._idx += 1
        return tok

    def _open(self, url: str, accept: str) -> bytes:
        attempts = max(1, len(self.tokens) or 1)
        last_err: Exception | None = None
        for _ in range(attempts):
            token = self._next_token()
            headers = {
                "Accept": accept,
                "User-Agent": "phase4_7_3-closed-pr-loader",
            }
            if token:
                headers["Authorization"] = f"Bearer {token}"
            req = Request(url, headers=headers)
            try:
                with urlopen(req, timeout=self.timeout_sec) as resp:
                    return resp.read()
            except HTTPError as e:
                last_err = e
                if e.code in (403, 429):
                    continue
                if e.code == 404:
                    return b""
            except URLError as e:
                last_err = e
                continue
        if last_err is not None:
            raise last_err
        return b""

    def get_json(self, url: str) -> Any:
        raw = self._open(url, "application/vnd.github+json")
        if not raw:
            return None
        return json.loads(raw.decode("utf-8", errors="replace"))

    def get_text(self, url: str, accept: str) -> str | None:
        raw = self._open(url, accept)
        if not raw:
            return None
        return raw.decode("utf-8", errors="replace")

    def get_paginated_json(self, url: str, max_pages: int = 20) -> list[Any]:
        out: list[Any] = []
        for page in range(1, max_pages + 1):
            data = self.get_json(_page_url(url, page))
            if not data:
                break
            if not isinstance(data, list):
                break
            out.extend(data)
            if len(data) < 100:
                break
        return out


def _parse_issue_numbers(text: str | None, linked_issues: list[dict[str, Any]]) -> list[int]:
    nums = {int(m.group(1)) for m in ISSUE_REF_RE.finditer(text or "")}
    for item in linked_issues:
        try:
            num = item.get("number")
            if num is not None:
                nums.add(int(num))
        except Exception:
            pass
    return sorted(nums)


def _linked_issues(rec: dict[str, Any]) -> list[dict[str, Any]]:
    pr = _pr_graphql(rec)
    nodes = ((pr.get("closingIssuesReferences") or {}).get("nodes") or [])
    return nodes if isinstance(nodes, list) else []


def _closing_issue_id(linked_issues: list[dict[str, Any]]) -> str | None:
    if not linked_issues:
        return None
    first = linked_issues[0]
    if first.get("id") is not None:
        return str(first["id"])
    if first.get("number") is not None:
        return str(first["number"])
    return None


def _pr_category(rec: dict[str, Any]) -> tuple[str | None, float | None, str | None]:
    if not rec.get("classification_success"):
        return None, None, None
    raw = str(rec.get("pr_type") or "").strip().lower()
    mapping = {
        "feat": "feature",
        "feature": "feature",
        "fix": "bugfix",
        "bugfix": "bugfix",
        "bug": "bugfix",
        "docs": "docs",
        "doc": "docs",
        "chore": "maintenance",
        "refactor": "maintenance",
        "maintenance": "maintenance",
        "style": "maintenance",
        "test": "maintenance",
        "perf": "maintenance",
        "ci": "maintenance",
    }
    cat = mapping.get(raw, "feature" if bool(rec.get("is_feature")) else "maintenance")
    conf = rec.get("confidence")
    try:
        conf_v = float(conf) if conf is not None else None
    except Exception:
        conf_v = None
    reasoning = f"source=top_level_classifier pr_type={raw or 'unknown'} is_feature={bool(rec.get('is_feature'))}"
    return cat, conf_v, reasoning


def _normalize_requested_reviewers(rec: dict[str, Any]) -> list[Any]:
    out: list[Any] = []
    users = rec.get("requested_reviewers") or []
    teams = rec.get("requested_teams") or []
    if isinstance(users, list):
        out.extend(users)
    if isinstance(teams, list):
        out.extend([{"team": team} if isinstance(team, dict) else team for team in teams])
    return out


def _comments_payload(rec: dict[str, Any]) -> tuple[int, list[Any], bool]:
    pr = _pr_graphql(rec)
    comments = pr.get("comments") or {}
    total = comments.get("totalCount")
    nodes = comments.get("nodes") or []
    page_info = comments.get("pageInfo") or {}
    partial = bool(page_info.get("hasNextPage")) or (
        total is not None and isinstance(nodes, list) and len(nodes) < int(total or 0)
    )
    return int(total or len(nodes)), list(nodes), partial


def _review_threads_payload(rec: dict[str, Any]) -> tuple[int, list[Any], bool]:
    pr = _pr_graphql(rec)
    threads = pr.get("reviewThreads") or {}
    total = threads.get("totalCount")
    nodes = threads.get("nodes") or []
    page_info = threads.get("pageInfo") or {}
    partial = bool(page_info.get("hasNextPage")) or (
        total is not None and isinstance(nodes, list) and len(nodes) < int(total or 0)
    )
    return int(total or len(nodes)), list(nodes), partial


def _reviews_payload(rec: dict[str, Any]) -> list[Any]:
    pr = _pr_graphql(rec)
    reviews = pr.get("reviews") or {}
    return list(reviews.get("nodes") or [])


def _commits_payload(rec: dict[str, Any]) -> tuple[int, list[Any], bool]:
    pr = _pr_graphql(rec)
    commits = pr.get("commits") or {}
    total = commits.get("totalCount")
    nodes = commits.get("nodes") or []
    page_info = commits.get("pageInfo") or {}
    partial = bool(page_info.get("hasNextPage")) or (
        total is not None and isinstance(nodes, list) and len(nodes) < int(total or 0)
    )
    return int(total or len(nodes)), list(nodes), partial


def _fetch_commits(client: GitHubClient, rec: dict[str, Any]) -> list[Any]:
    url = rec.get("commits_url")
    if not url:
        return []
    items = client.get_paginated_json(url)
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        node = {
            "commit": {
                "oid": item.get("sha"),
                "message": ((item.get("commit") or {}).get("message") or ""),
                "author": {
                    "name": ((item.get("commit") or {}).get("author") or {}).get("name"),
                    "email": ((item.get("commit") or {}).get("author") or {}).get("email"),
                    "date": ((item.get("commit") or {}).get("author") or {}).get("date"),
                    "user": {"login": ((item.get("author") or {}).get("login"))} if item.get("author") else None,
                },
                "committer": {
                    "name": ((item.get("commit") or {}).get("committer") or {}).get("name"),
                    "email": ((item.get("commit") or {}).get("committer") or {}).get("email"),
                    "date": ((item.get("commit") or {}).get("committer") or {}).get("date"),
                    "user": {"login": ((item.get("committer") or {}).get("login"))} if item.get("committer") else None,
                },
            }
        }
        out.append(node)
    return out


def _fetch_comments(client: GitHubClient, rec: dict[str, Any]) -> list[Any]:
    url = rec.get("comments_url")
    if not url:
        return []
    items = client.get_paginated_json(url)
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "id": item.get("id"),
                "body": item.get("body"),
                "createdAt": item.get("created_at"),
                "updatedAt": item.get("updated_at"),
                "author": {"login": ((item.get("user") or {}).get("login"))} if item.get("user") else None,
            }
        )
    return out


def _fetch_files(client: GitHubClient, rec: dict[str, Any]) -> list[Any]:
    slug = _repo_slug(rec)
    number = rec.get("number")
    if not slug or not number:
        return []
    url = f"https://api.github.com/repos/{slug}/pulls/{number}/files"
    items = client.get_paginated_json(url, max_pages=30)
    return [item for item in items if isinstance(item, dict)]


def _test_file_patches(file_patches: list[Any]) -> list[Any]:
    out = []
    for item in file_patches:
        if isinstance(item, dict):
            filename = str(item.get("filename") or item.get("path") or "")
            if filename and TEST_FILE_RE.search(filename):
                out.append(item)
        elif isinstance(item, str) and TEST_FILE_RE.search(item):
            out.append(item)
    return out


def _test_patch_text(test_files: list[Any]) -> str | None:
    parts = []
    for item in test_files:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename") or item.get("path") or "")
        patch = item.get("patch")
        if filename and patch:
            parts.append(f"diff --git a/{filename} b/{filename}\n{patch}")
    return "\n".join(parts) if parts else None


def _normalize_one(
    rec: dict[str, Any],
    now_iso: str,
    client: GitHubClient | None,
    fetch_missing: bool,
) -> tuple[dict[str, Any], Counter[str]]:
    stats: Counter[str] = Counter()
    repo_slug = _repo_slug(rec)
    repo_id = _repo_id(rec)
    pr = _pr_graphql(rec)
    linked_issues = _linked_issues(rec)
    issue_numbers = _parse_issue_numbers(str(rec.get("body") or ""), linked_issues)
    total_commits, commits, commits_partial = _commits_payload(rec)
    total_comments, comments, comments_partial = _comments_payload(rec)
    total_review_threads, review_threads, review_threads_partial = _review_threads_payload(rec)
    submitted_reviews = _reviews_payload(rec)
    patch_text = None
    file_patches: list[Any] | None = None

    if fetch_missing and client is not None:
        try:
            patch_text = client.get_text(str(rec.get("patch_url") or ""), "application/vnd.github.v3.patch")
        except Exception:
            stats["patch_fetch_failed"] += 1
        try:
            file_patches = _fetch_files(client, rec)
        except Exception:
            stats["files_fetch_failed"] += 1
        if commits_partial or not commits:
            try:
                commits = _fetch_commits(client, rec)
                total_commits = len(commits) or total_commits
                stats["commits_fetched"] += 1
            except Exception:
                stats["commits_fetch_failed"] += 1
        if comments_partial:
            try:
                comments = _fetch_comments(client, rec)
                total_comments = len(comments) or total_comments
                stats["comments_fetched"] += 1
            except Exception:
                stats["comments_fetch_failed"] += 1

    test_files = _test_file_patches(file_patches or [])
    pr_category, pr_confidence, pr_reasoning = _pr_category(rec)
    created_at = pr.get("createdAt") or rec.get("created_at")
    updated_at = pr.get("updatedAt") or rec.get("updated_at")
    row = {
        "id": _safe_int32(rec.get("id")),
        "crawl_time": now_iso,
        "instance_id": _instance_id(rec),
        "repo": repo_slug or None,
        "pull_number": rec.get("number"),
        "issue_numbers": issue_numbers,
        "base_commit": ((rec.get("base") or {}).get("sha") or pr.get("baseRefOid")),
        "patch": patch_text,
        "file_patches": file_patches,
        "test_patch": _test_patch_text(test_files),
        "test_file_patches": test_files or None,
        "problem_statement": None,
        "hints_text": None,
        "pass_to_pass": None,
        "fail_to_pass": None,
        "repo_id": repo_id,
        "stars": ((rec.get("base") or {}).get("repo") or {}).get("stargazers_count"),
        "forks": ((rec.get("base") or {}).get("repo") or {}).get("forks_count"),
        "primary_language": ((rec.get("base") or {}).get("repo") or {}).get("language"),
        "pr_title": pr.get("title") or rec.get("title"),
        "pr_body": pr.get("body") or rec.get("body"),
        "pr_url": rec.get("html_url") or pr.get("url") or rec.get("url"),
        "pr_state": (pr.get("state") or rec.get("state")),
        "pr_merged": bool(pr.get("merged") or rec.get("merged_at")),
        "pr_is_draft": bool(pr.get("isDraft") if pr.get("isDraft") is not None else rec.get("draft")),
        "pr_author": ((rec.get("user") or {}).get("login")),
        "pr_author_name": ((pr.get("author") or {}).get("login")) if isinstance(pr.get("author"), dict) else None,
        "pr_labels": rec.get("labels") or pr.get("labels"),
        "base_branch": ((rec.get("base") or {}).get("ref") or pr.get("baseRefName")),
        "head_branch": ((rec.get("head") or {}).get("ref") or pr.get("headRefName")),
        "base_sha": ((rec.get("base") or {}).get("sha") or pr.get("baseRefOid")),
        "head_sha": ((rec.get("head") or {}).get("sha") or pr.get("headRefOid")),
        "created_at": created_at,
        "updated_at": updated_at,
        "total_commits": total_commits,
        "commits": commits,
        "total_comments": total_comments,
        "comments": comments,
        "total_review_threads": total_review_threads,
        "review_threads": review_threads,
        "requested_reviewers": _normalize_requested_reviewers(rec),
        "submitted_reviews": submitted_reviews,
        "additions": pr.get("additions"),
        "deletions": pr.get("deletions"),
        "changed_files": pr.get("changedFiles"),
        "pr_category": pr_category,
        "pr_category_confidence": pr_confidence,
        "pr_category_reasoning": pr_reasoning,
        "linked_issues": linked_issues,
        "closing_issue_id": _closing_issue_id(linked_issues),
        "merged_at": pr.get("mergedAt") or rec.get("merged_at"),
        "merged_by": None,
    }

    if commits_partial:
        stats["commits_partial_source"] += 1
    if comments_partial:
        stats["comments_partial_source"] += 1
    if review_threads_partial:
        stats["review_threads_partial_source"] += 1
    if file_patches:
        stats["with_file_patches"] += 1
    if patch_text:
        stats["with_patch"] += 1
    if rec.get("id") is not None and row["id"] is None:
        stats["id_overflow"] += 1
    return row, stats


def _ensure_table(conn: pg8000.native.Connection, table_name: str) -> None:
    conn.run(f"CREATE TABLE IF NOT EXISTS {table_name} (LIKE prs_copy INCLUDING DEFAULTS)")
    conn.run(f"CREATE UNIQUE INDEX IF NOT EXISTS {table_name}_repo_pull_idx ON {table_name} (repo, pull_number)")
    conn.run(f"CREATE INDEX IF NOT EXISTS {table_name}_instance_id_idx ON {table_name} (instance_id)")


def _upsert_row(conn: pg8000.native.Connection, table_name: str, row: dict[str, Any]) -> None:
    params = {k: (_to_jsonb(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
    col_sql = ", ".join(PRS_COPY_COLUMNS)
    val_sql = ", ".join(f":{c}" for c in PRS_COPY_COLUMNS)
    upd_cols = [c for c in PRS_COPY_COLUMNS if c not in ("repo", "pull_number")]
    upd_sql = ", ".join(f"{c}=EXCLUDED.{c}" for c in upd_cols)
    conn.run(
        f"""
        INSERT INTO {table_name} ({col_sql})
        VALUES ({val_sql})
        ON CONFLICT (repo, pull_number) DO UPDATE
        SET {upd_sql}
        """,
        **params,
    )


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    by_col = {}
    for col in PRS_COPY_COLUMNS:
        non_null = sum(1 for row in rows if row.get(col) not in (None, [], {}))
        by_col[col] = {
            "non_null": non_null,
            "coverage": (non_null / total) if total else 0.0,
            "class": (
                "fetchable"
                if col in FETCHABLE_COLUMNS
                else ("unavailable" if col in UNAVAILABLE_COLUMNS else "direct_or_derived")
            ),
        }
    return by_col


def _render_report(summary: dict[str, Any], out_path: str) -> None:
    def pct(v: float) -> str:
        return f"{100.0 * v:.1f}%"

    lines = [
        "# Experiment 4.7.3 — Closed PR Ingestion into prs_copy_closed",
        "",
        "This report measures how much of the `prs_copy` schema can be recovered from the",
        "closed/unmerged JSONL corpus in `/shared_workspace_mfs/akki/scratch_mfs/arthur-task/enriched-all-unmerged`.",
        "",
        f"- Source PR rows: `{summary['source_rows']}`",
        f"- Repo files: `{summary['source_files']}`",
        f"- Skipped merged/non-closed tail: `{summary['skipped_non_closed_or_merged_rows']}`",
        f"- GraphQL enrichment success: `{summary['graphql_ok']}` / `{summary['source_rows']}`",
        f"- Overflowed GitHub PR ids coerced to NULL: `{summary['id_overflow_rows']}`",
        f"- Commit payload partial rows: `{summary['commit_partial_rows']}`",
        f"- Comment payload partial rows: `{summary['comments_partial_rows']}`",
        f"- Review-thread partial rows: `{summary['review_threads_partial_rows']}`",
        "",
        "## Recoverability classes",
        "",
        "- `direct_or_derived`: present locally or derivable from the JSONL",
        "- `fetchable`: missing locally but recoverable from GitHub REST/patch endpoints",
        "- `unavailable`: not present in this corpus and not targeted for recovery in 4.7.3",
        "",
        "## Column coverage",
        "",
        "| Column | Class | Non-null rows | Coverage |",
        "|---|---|---:|---:|",
    ]
    for col in PRS_COPY_COLUMNS:
        item = summary["coverage"][col]
        lines.append(f"| `{col}` | `{item['class']}` | {item['non_null']} | {pct(item['coverage'])} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `patch` and `file_patches` are the main fetch-required artifacts for downstream patch-based experiments.",
            "- `comments` / `commits` are mostly present locally; the crawler only needs to repair a small partial tail.",
            "- `review_threads` are already present in almost all rows that have them; the remaining partial rows are small enough to defer.",
        ]
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default=SOURCE_DIR)
    ap.add_argument("--table", default="prs_copy_closed")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fetch-missing", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--include-merged-tail", action="store_true")
    ap.add_argument("--summary-out", default=DEFAULT_JSON)
    ap.add_argument("--report-out", default=DEFAULT_MD)
    args = ap.parse_args()

    raw_rows = []
    skipped_merged_tail = 0
    source_files = 0
    seen_files = set()
    for file_name, rec in _iter_source_rows(args.source_dir):
        if not args.include_merged_tail and not _is_closed_unmerged(rec):
            skipped_merged_tail += 1
            continue
        raw_rows.append(rec)
        seen_files.add(file_name)
        if args.limit and len(raw_rows) >= args.limit:
            break
    source_files = len(seen_files)

    tokens = _load_tokens(TOKENS_YAML)
    client = GitHubClient(tokens) if args.fetch_missing else None
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    out_rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    lock = threading.Lock()

    def _work(rec: dict[str, Any]) -> tuple[dict[str, Any], Counter[str]]:
        return _normalize_one(rec, now_iso, client, args.fetch_missing)

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = [ex.submit(_work, rec) for rec in raw_rows]
        for fut in as_completed(futures):
            row, row_stats = fut.result()
            with lock:
                out_rows.append(row)
                stats.update(row_stats)

    out_rows.sort(key=lambda r: (str(r.get("repo") or ""), int(r.get("pull_number") or 0)))
    coverage = _coverage(out_rows)
    summary = {
        "source_dir": args.source_dir,
        "source_files": source_files,
        "source_rows": len(raw_rows),
        "skipped_non_closed_or_merged_rows": skipped_merged_tail,
        "graphql_ok": sum(1 for rec in raw_rows if rec.get("graphql_enrichment_success") is True),
        "commit_partial_rows": int(stats.get("commits_partial_source", 0)),
        "comments_partial_rows": int(stats.get("comments_partial_source", 0)),
        "review_threads_partial_rows": int(stats.get("review_threads_partial_source", 0)),
        "with_patch": int(stats.get("with_patch", 0)),
        "with_file_patches": int(stats.get("with_file_patches", 0)),
        "id_overflow_rows": int(stats.get("id_overflow", 0)),
        "fetch_missing": bool(args.fetch_missing),
        "table": args.table,
        "coverage": coverage,
    }

    if not args.analyze_only:
        conn = _load_db()
        try:
            _ensure_table(conn, args.table)
            for idx, row in enumerate(out_rows, 1):
                _upsert_row(conn, args.table, row)
                if idx % 500 == 0:
                    print(f"upserted {idx}/{len(out_rows)} rows", flush=True)
        finally:
            conn.close()

    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    _render_report(summary, args.report_out)
    print(json.dumps(
        {
            "rows": len(out_rows),
            "table": args.table,
            "fetch_missing": bool(args.fetch_missing),
            "analyze_only": bool(args.analyze_only),
            "summary_out": args.summary_out,
            "report_out": args.report_out,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
