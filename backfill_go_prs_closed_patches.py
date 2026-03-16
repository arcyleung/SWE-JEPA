#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pg8000.native
import yaml

from ingest_prs_copy_closed import GitHubClient, _load_tokens

ROOT = os.path.dirname(os.path.abspath(__file__))
PG_CONFIG_FILE = os.path.join(ROOT, "postgres_connection.yaml")
TOKENS_YAML = os.path.join(ROOT, "crawl_tokens.yaml")
SOURCE_DIR = "/shared_workspace_mfs/akki/scratch_mfs/arthur-task/enriched-all-unmerged"
DEFAULT_JSON = os.path.join(ROOT, "data", "phase4_7_3_go_prs_closed_patch_backfill_summary.json")
DEFAULT_MD = os.path.join(ROOT, "docs", "phase4_7_3_go_prs_closed_patch_backfill.md")

TEST_INDICATORS = [
    "/test",
    "test_",
    "_test",
    "tests/",
    "spec/",
    "_spec",
    ".spec.",
    "__test__",
]

ADDED_COLUMNS = {
    "file_patches": "jsonb",
    "test_file_patches": "jsonb",
    "non_test_patch": "text",
    "non_test_patch_files": "text[]",
    "test_patch_files": "text[]",
    "is_splittable": "boolean",
    "has_fix_patch": "boolean",
    "has_test_patch": "boolean",
    "repo_language": "text",
    "extracted_language": "text",
    "step13_error": "text",
    "graphql_viewer_login": "text",
    "graphql_rate_limit_remaining": "integer",
    "graphql_rate_limit_reset_at": "timestamp with time zone",
    "graphql_rate_limit_cost": "integer",
    "graphql_errors_text": "text",
    "graphql_errors_count": "integer",
    "raw_json_text": "text",
    "inserted_at": "timestamp with time zone",
    "updated_at_db": "timestamp with time zone",
}


def _table_slug(table: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(table).strip().lower()).strip("_") or "closed_prs"


def _default_outputs(table: str) -> tuple[str, str]:
    if table == "go_prs_closed":
        return DEFAULT_JSON, DEFAULT_MD
    slug = _table_slug(table)
    return (
        os.path.join(ROOT, "data", f"{slug}_patch_backfill_summary.json"),
        os.path.join(ROOT, "docs", f"{slug}_patch_backfill.md"),
    )


def _load_db() -> pg8000.native.Connection:
    cfg = yaml.safe_load(open(PG_CONFIG_FILE))
    return pg8000.native.Connection(
        host=cfg["ip"],
        port=cfg.get("port", 9999),
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
    )


def _to_jsonb(v: Any) -> str | None:
    if v is None:
        return None
    return json.dumps(_sanitize_json_value(v), ensure_ascii=False)


def _clean_text(v: str | None) -> str | None:
    if v is None:
        return None
    return v.replace("\x00", "")


def _sanitize_json_value(v: Any) -> Any:
    if isinstance(v, str):
        return _clean_text(v)
    if isinstance(v, list):
        return [_sanitize_json_value(item) for item in v]
    if isinstance(v, dict):
        return {str(k): _sanitize_json_value(val) for k, val in v.items()}
    return v


def _parse_sections(patch: str) -> list[tuple[str | None, list[str]]]:
    if not patch:
        return []
    sections: list[tuple[str | None, list[str]]] = []
    cur: list[str] = []
    cur_path: str | None = None

    def flush() -> None:
        nonlocal cur, cur_path
        if cur:
            sections.append((cur_path, cur))
        cur = []
        cur_path = None

    for raw in patch.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("diff --git "):
            flush()
            cur = [line]
            m = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
            if m:
                cur_path = m.group(2)
            continue
        if not cur:
            continue
        cur.append(line)
        if line.startswith("+++ "):
            path = line[4:].strip().split()[0]
            if path != "/dev/null":
                cur_path = path[2:] if path.startswith("b/") else path
        elif line.startswith("--- ") and cur_path is None:
            path = line[4:].strip().split()[0]
            if path != "/dev/null":
                cur_path = path[2:] if path.startswith("a/") else path
    flush()
    return sections


def _is_test_file(path: str | None) -> bool:
    if not path:
        return False
    lower = path.lower()
    return any(tok in lower for tok in TEST_INDICATORS)


def _split_patch(patch: str) -> dict[str, Any]:
    test_sections: list[str] = []
    non_test_sections: list[str] = []
    test_files: list[str] = []
    non_test_files: list[str] = []

    seen_test: set[str] = set()
    seen_non_test: set[str] = set()

    for path, lines in _parse_sections(patch):
        block = "\n".join(lines).strip()
        if not block:
            continue
        if _is_test_file(path):
            test_sections.append(block)
            if path and path not in seen_test:
                seen_test.add(path)
                test_files.append(path)
        else:
            non_test_sections.append(block)
            if path and path not in seen_non_test:
                seen_non_test.add(path)
                non_test_files.append(path)

    has_fix_patch = bool(non_test_sections)
    has_test_patch = bool(test_sections)
    return {
        "non_test_patch": "\n".join(non_test_sections) if non_test_sections else None,
        "test_patch": "\n".join(test_sections) if test_sections else None,
        "non_test_patch_files": non_test_files or None,
        "test_patch_files": test_files or None,
        "has_fix_patch": has_fix_patch,
        "has_test_patch": has_test_patch,
        "is_splittable": bool(has_fix_patch and has_test_patch),
    }


def _fetch_files(client: GitHubClient, repo: str, pull_number: int) -> list[dict[str, Any]]:
    url = f"https://api.github.com/repos/{repo}/pulls/{pull_number}/files"
    items = client.get_paginated_json(url, max_pages=30)
    return [item for item in items if isinstance(item, dict)]


def _fetch_patch(client: GitHubClient, repo: str, pull_number: int) -> str | None:
    url = f"https://api.github.com/repos/{repo}/pulls/{pull_number}"
    return client.get_text(url, "application/vnd.github.v3.patch")


def _test_file_patches(file_patches: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    out = []
    for item in file_patches:
        filename = str(item.get("filename") or "")
        if filename and _is_test_file(filename):
            out.append(item)
    return out or None


def _raw_source_map(source_dir: str) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    base = Path(source_dir)
    for path in sorted(base.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                repo = f"{rec.get('repo_owner')}/{rec.get('repo_name')}"
                try:
                    pr = int(rec.get("number"))
                except Exception:
                    continue
                out[(repo, pr)] = rec
    return out


def _ensure_columns(conn: pg8000.native.Connection, table: str) -> None:
    existing = {
        row[0]
        for row in conn.run(
            "select column_name from information_schema.columns where table_name = :t",
            t=table,
        )
    }
    for col, col_type in ADDED_COLUMNS.items():
        if col not in existing:
            conn.run(f"alter table {table} add column {col} {col_type}")


def _get_rows(
    conn: pg8000.native.Connection,
    table: str,
    limit: int,
    language: str | None = None,
) -> list[tuple[Any, ...]]:
    lim_sql = f"limit {int(limit)}" if limit > 0 else ""
    params: dict[str, Any] = {}
    language_sql = ""
    if language:
        language_sql = "and lower(primary_language) = :language_lower"
        params["language_lower"] = str(language).lower()
    return conn.run(
        f"""
        select repo, pull_number, primary_language, patch, file_patches, pr_url
        from {table}
        where patch is null
           or file_patches is null
           or non_test_patch is null
           or test_file_patches is null
           or has_fix_patch is null
          {language_sql}
        order by created_at desc nulls last
        {lim_sql}
        """,
        **params,
    )


def _update_row(
    conn: pg8000.native.Connection,
    table: str,
    repo: str,
    pull_number: int,
    primary_language: str | None,
    patch: str | None,
    file_patches: list[dict[str, Any]] | None,
    split: dict[str, Any],
    source_meta: dict[str, Any] | None,
    step13_error: str | None,
) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    test_file_patches = _test_file_patches(file_patches or []) if file_patches is not None else None
    graphql = ((source_meta or {}).get("graphql_enrichment") or {})
    rate_limit = graphql.get("rate_limit") or {}
    errors = graphql.get("errors") or []
    viewer = graphql.get("viewer") or {}
    resolved_language = str(primary_language).strip() if primary_language else None
    patch = _clean_text(patch)
    test_patch = _clean_text(split.get("test_patch"))
    non_test_patch = _clean_text(split.get("non_test_patch"))
    step13_error = _clean_text(step13_error)
    conn.run(
        f"""
        update {table}
        set patch = :patch,
            file_patches = cast(:file_patches as jsonb),
            test_patch = :test_patch,
            test_file_patches = cast(:test_file_patches as jsonb),
            non_test_patch = :non_test_patch,
            non_test_patch_files = :non_test_patch_files,
            test_patch_files = :test_patch_files,
            has_fix_patch = :has_fix_patch,
            has_test_patch = :has_test_patch,
            is_splittable = :is_splittable,
            repo_language = coalesce(repo_language, primary_language, :resolved_language),
            extracted_language = coalesce(extracted_language, :resolved_language),
            step13_error = :step13_error,
            graphql_viewer_login = :graphql_viewer_login,
            graphql_rate_limit_remaining = :graphql_rate_limit_remaining,
            graphql_rate_limit_reset_at = :graphql_rate_limit_reset_at,
            graphql_rate_limit_cost = :graphql_rate_limit_cost,
            graphql_errors_text = :graphql_errors_text,
            graphql_errors_count = :graphql_errors_count,
            raw_json_text = coalesce(raw_json_text, :raw_json_text),
            inserted_at = coalesce(inserted_at, :inserted_at),
            updated_at_db = :updated_at_db
        where repo = :repo and pull_number = :pull_number
        """,
        repo=repo,
        pull_number=int(pull_number),
        patch=patch,
        file_patches=_to_jsonb(file_patches),
        test_patch=test_patch,
        test_file_patches=_to_jsonb(test_file_patches),
        non_test_patch=non_test_patch,
        non_test_patch_files=split.get("non_test_patch_files"),
        test_patch_files=split.get("test_patch_files"),
        has_fix_patch=split.get("has_fix_patch"),
        has_test_patch=split.get("has_test_patch"),
        is_splittable=split.get("is_splittable"),
        resolved_language=resolved_language,
        step13_error=step13_error,
        graphql_viewer_login=viewer.get("login"),
        graphql_rate_limit_remaining=rate_limit.get("remaining"),
        graphql_rate_limit_reset_at=rate_limit.get("resetAt"),
        graphql_rate_limit_cost=rate_limit.get("cost"),
        graphql_errors_text=json.dumps(errors, ensure_ascii=False) if errors else None,
        graphql_errors_count=len(errors),
        raw_json_text=json.dumps(_sanitize_json_value(source_meta), ensure_ascii=False) if source_meta is not None else None,
        inserted_at=now.isoformat(),
        updated_at_db=now.isoformat(),
    )


def _worker(client: GitHubClient, repo: str, pull_number: int) -> tuple[str | None, list[dict[str, Any]] | None, str | None]:
    try:
        patch = _fetch_patch(client, repo, pull_number)
        files = _fetch_files(client, repo, pull_number)
        return patch, files, None
    except Exception as e:
        return None, None, str(e)


def _render_report(summary: dict[str, Any], out_path: str) -> None:
    lines = [
        f"# Closed PR Patch Recovery: {summary['table']}",
        "",
        f"- Rows targeted: `{summary['target_rows']}`",
        f"- Rows updated: `{summary['updated_rows']}`",
        f"- Patch fetch success: `{summary['patch_success']}`",
        f"- File-list fetch success: `{summary['files_success']}`",
        "",
        "## Split quality",
        "",
        f"- `has_fix_patch`: `{summary['has_fix_patch']}`",
        f"- `has_test_patch`: `{summary['has_test_patch']}`",
        f"- `is_splittable`: `{summary['is_splittable']}`",
        "",
        "## Notes",
        "",
        "- Patch splitting follows the legacy `explore_agent_simple.py` test-file heuristic.",
        "- The current implementation stores both `prs_copy`-style JSON patch fields (`file_patches`, `test_file_patches`) and split patch fields (`non_test_patch`, `*_patch_files`, split flags).",
        "- `repo_language` / `extracted_language` now default from each row's `primary_language`, so mixed-language closed tables can be backfilled with one run.",
    ]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="go_prs_closed")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--source-dir", default=SOURCE_DIR)
    ap.add_argument("--language", default=None, help="Optional primary_language filter for mixed-language tables.")
    ap.add_argument("--summary-out", default=None)
    ap.add_argument("--report-out", default=None)
    args = ap.parse_args()

    if args.summary_out is None or args.report_out is None:
        default_summary_out, default_report_out = _default_outputs(args.table)
        args.summary_out = args.summary_out or default_summary_out
        args.report_out = args.report_out or default_report_out

    tokens = _load_tokens(TOKENS_YAML)
    client = GitHubClient(tokens)
    source_map = _raw_source_map(args.source_dir)
    conn = _load_db()
    stats: Counter[str] = Counter()
    try:
        _ensure_columns(conn, args.table)
        rows = _get_rows(conn, args.table, args.limit, args.language)
        stats["target_rows"] = len(rows)
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            futs = {
                ex.submit(_worker, client, str(repo), int(pull_number)): (
                    str(repo),
                    int(pull_number),
                    str(_primary_language) if _primary_language else None,
                )
                for repo, pull_number, _primary_language, _patch, _file_patches, _pr_url in rows
            }
            done = 0
            for fut in as_completed(futs):
                repo, pull_number, primary_language = futs[fut]
                patch, files, err = fut.result()
                source_meta = source_map.get((repo, pull_number))
                if patch:
                    stats["patch_success"] += 1
                if files is not None:
                    stats["files_success"] += 1
                if err:
                    stats["fetch_errors"] += 1
                split = _split_patch(patch or "")
                if split.get("has_fix_patch"):
                    stats["has_fix_patch"] += 1
                if split.get("has_test_patch"):
                    stats["has_test_patch"] += 1
                if split.get("is_splittable"):
                    stats["is_splittable"] += 1
                _update_row(
                    conn,
                    args.table,
                    repo,
                    pull_number,
                    primary_language,
                    patch,
                    files,
                    split,
                    source_meta,
                    err,
                )
                done += 1
                stats["updated_rows"] = done
                if done % 100 == 0 or done == len(futs):
                    print(f"updated {done}/{len(futs)} rows", flush=True)
    finally:
        conn.close()

    summary = {
        "table": args.table,
        "language": args.language,
        "target_rows": int(stats.get("target_rows", 0)),
        "updated_rows": int(stats.get("updated_rows", 0)),
        "patch_success": int(stats.get("patch_success", 0)),
        "files_success": int(stats.get("files_success", 0)),
        "fetch_errors": int(stats.get("fetch_errors", 0)),
        "has_fix_patch": int(stats.get("has_fix_patch", 0)),
        "has_test_patch": int(stats.get("has_test_patch", 0)),
        "is_splittable": int(stats.get("is_splittable", 0)),
    }
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    _render_report(summary, args.report_out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
