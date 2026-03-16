#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import pg8000.native
import yaml

from extract_conway_patch_features import (
    extract_features,
    _filenames,
    _parse_unified_diff,
    _repo_dir_map,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
PG_CONFIG_FILE = os.path.join(ROOT, "postgres_connection.yaml")
DEFAULT_OUT = os.path.join(ROOT, "data", "conway_patch_features_go_merged_closed.jsonl")
DEFAULT_LABELS = os.path.join(ROOT, "data", "phase4_7_3_go_pr_labels.jsonl")
DEFAULT_SUMMARY = os.path.join(ROOT, "data", "phase4_7_3_go_pr_corpus_summary.json")

REFACTOR_SQL_REGEX = (
    r"(refactor|restructure|cleanup|architecture|design|rewrite|reorganize|rework|"
    r"simplif|extract|move (to|into)|split (into|out)|split this|separate|decouple|"
    r"consolidat|encapsulat|naming|too large|too big)"
)

_REPO_DIRS: dict[str, str] = {}
_LLM_LABEL_MAP: dict[tuple[str, int], dict] = {}


def _slug(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _default_outputs(
    merged_table: str,
    closed_table: str,
    language: str | None,
) -> tuple[str, str, str]:
    lang_slug = _slug(language)
    if merged_table == "go_prs" and closed_table == "go_prs_closed" and lang_slug in ("", "go"):
        return DEFAULT_OUT, DEFAULT_LABELS, DEFAULT_SUMMARY
    base = lang_slug or f"{_slug(merged_table)}_{_slug(closed_table)}"
    return (
        os.path.join(ROOT, "data", f"conway_patch_features_{base}_merged_closed.jsonl"),
        os.path.join(ROOT, "data", f"{base}_pr_labels.jsonl"),
        os.path.join(ROOT, "data", f"{base}_pr_corpus_summary.json"),
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


def _j(v: Any) -> Any:
    if isinstance(v, (list, dict)):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return []
    return []


def _go_file_list(file_patches: Any, non_test_patch_files: Any, test_patch_files: Any, patch: str) -> Any:
    """Recover a filename list from either prs_copy-style JSON or split patch fields."""
    names = _filenames(file_patches)
    if names:
        return names
    out: list[str] = []
    for raw in (non_test_patch_files, test_patch_files):
        vals = _j(raw)
        if isinstance(vals, list):
            for item in vals:
                if isinstance(item, str) and item:
                    out.append(item)
    if out:
        dedup = []
        seen = set()
        for name in out:
            if name not in seen:
                dedup.append(name)
                seen.add(name)
        return dedup
    parsed = []
    seen = set()
    for fd in _parse_unified_diff(patch or ""):
        path = fd.new_path or fd.old_path
        if path and path not in seen:
            parsed.append(path)
            seen.add(path)
    return parsed


def _synthetic_instance_id(repo: str, pull_number: int, source_table: str) -> str:
    safe_repo = repo.replace("/", "__")
    return f"{source_table}__{safe_repo}__{pull_number}"


def _fetch_rows(
    conn: pg8000.native.Connection,
    table: str,
    limit: int,
    shard_modulus: int,
    shard_remainder: int,
    language: str | None,
) -> list[tuple[Any, ...]]:
    limit_sql = f"limit {int(limit)}" if limit > 0 else ""
    shard_sql = ""
    params: dict[str, Any] = {"ref_regex": REFACTOR_SQL_REGEX}
    language_sql = ""
    if language:
        language_sql = "and lower(primary_language) = :language_lower"
        params["language_lower"] = str(language).lower()
    if shard_modulus > 0:
        shard_sql = (
            "and mod(abs(hashtext(repo || '#' || pull_number::text)), :shard_modulus) = :shard_remainder"
        )
        params["shard_modulus"] = int(shard_modulus)
        params["shard_remainder"] = int(shard_remainder)
    if table == "prs_copy":
        return conn.run(
            f"""
            select
                repo,
                instance_id,
                pull_number,
                base_sha,
                pr_merged,
                pr_is_draft,
                changed_files,
                additions,
                deletions,
                total_review_threads,
                total_comments,
                ((coalesce(review_threads::text, '') || ' ' || coalesce(comments::text, '')) ~* :ref_regex) as refactor_requested,
                patch,
                file_patches,
                null::text as non_test_patch_files,
                null::text as test_patch_files
            from prs_copy
            where patch is not null
              and file_patches is not null
              {language_sql}
              {shard_sql}
            order by created_at desc nulls last
            {limit_sql}
            """,
            **params,
        )
    if table == "go_prs":
        return conn.run(
            f"""
            select
                repo,
                null::text as instance_id,
                pull_number,
                base_sha,
                pr_merged,
                pr_is_draft,
                changed_files,
                additions,
                deletions,
                total_review_threads,
                total_comments,
                ((coalesce(review_threads::text, '') || ' ' || coalesce(comments::text, '')) ~* :ref_regex) as refactor_requested,
                patch,
                null::text as file_patches,
                non_test_patch_files,
                test_patch_files
            from go_prs
            where patch is not null
              and non_test_patch is not null
              {language_sql}
              {shard_sql}
            order by created_at desc nulls last
            {limit_sql}
            """,
            **params,
        )
    if table == "go_prs_closed":
        return conn.run(
            f"""
            select
                repo,
                null::text as instance_id,
                pull_number,
                base_sha,
                pr_merged,
                pr_is_draft,
                changed_files,
                additions,
                deletions,
                total_review_threads,
                total_comments,
                ((coalesce(review_threads::text, '') || ' ' || coalesce(comments::text, '')) ~* :ref_regex) as refactor_requested,
                patch,
                file_patches,
                non_test_patch_files,
                test_patch_files
            from go_prs_closed
            where patch is not null
              and non_test_patch is not null
              {language_sql}
              {shard_sql}
            order by created_at desc nulls last
            {limit_sql}
            """,
            **params,
        )
    if table == "python_js_ts_rust_closed_prs":
        return conn.run(
            f"""
            select
                repo,
                null::text as instance_id,
                pull_number,
                base_sha,
                pr_merged,
                pr_is_draft,
                changed_files,
                additions,
                deletions,
                total_review_threads,
                total_comments,
                ((coalesce(review_threads::text, '') || ' ' || coalesce(comments::text, '')) ~* :ref_regex) as refactor_requested,
                patch,
                null::text as file_patches,
                non_test_patch_files,
                test_patch_files
            from python_js_ts_rust_closed_prs
            where patch is not null
              and non_test_patch is not null
              {language_sql}
              {shard_sql}
            order by created_at desc nulls last
            {limit_sql}
            """,
            **params,
        )
    raise ValueError(f"unsupported table: {table}")


def _worker(
    row: tuple[Any, ...],
    source_table: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, Counter]:
    (
        repo,
        instance_id,
        pull_number,
        base_sha,
        pr_merged,
        pr_is_draft,
        changed_files,
        additions,
        deletions,
        total_review_threads,
        total_comments,
        refactor_requested,
        patch,
        file_patches,
        non_test_patch_files,
        test_patch_files,
    ) = row

    stats = Counter()
    patch_text = patch if isinstance(patch, str) else ""
    if not repo or not pull_number or not patch_text:
        stats["skipped_missing_core"] += 1
        return None, None, stats

    fp_raw = _go_file_list(file_patches, non_test_patch_files, test_patch_files, patch_text)
    # Override with LLM labels if available
    llm_key = (str(repo), int(pull_number))
    if _LLM_LABEL_MAP:
        llm_row = _LLM_LABEL_MAP.get(llm_key)
        refactor_requested = int(bool(llm_row["refactor_requested"])) if llm_row else 0
    else:
        refactor_requested = int(bool(refactor_requested))

    try:
        feats = extract_features(
            patch_text,
            fp_raw,
            bool(pr_merged),
            int(total_review_threads or 0),
            int(total_comments or 0),
            "",
            repo_dir=_REPO_DIRS.get(str(repo)),
            base_sha=base_sha or "",
        )
    except Exception as exc:
        stats["extract_error"] += 1
        stats[f"extract_error__{type(exc).__name__}"] += 1
        return None, None, stats

    iid = str(instance_id or "") or _synthetic_instance_id(str(repo), int(pull_number), source_table)
    feature_row = {
        "repo": str(repo),
        "instance_id": iid,
        "pull_number": int(pull_number or 0),
        "source_table": source_table,
        "is_draft": int(bool(pr_is_draft)),
        "changed_files": int(changed_files or 0),
        "additions": int(additions or 0),
        "deletions": int(deletions or 0),
        **feats,
    }
    label_row = {
        "repo": str(repo),
        "instance_id": iid,
        "pull_number": int(pull_number or 0),
        "source_table": source_table,
        "s_t1": {
            "accepted": int(bool(pr_merged)),
            "refactor_requested": refactor_requested,
            "review_friction": int(
                (int(total_review_threads or 0) >= 3)
                or (int(total_comments or 0) >= 8)
                or refactor_requested
            ),
        },
    }
    stats["rows_ok"] += 1
    stats[f"rows_ok__{source_table}"] += 1
    stats[f"accepted__{int(bool(pr_merged))}"] += 1
    stats[f"refactor_requested__{refactor_requested}"] += 1
    return feature_row, label_row, stats


def _init_worker(repo_dirs: dict[str, str], llm_label_map: dict[tuple[str, int], dict] | None = None) -> None:
    global _REPO_DIRS, _LLM_LABEL_MAP
    _REPO_DIRS = repo_dirs
    if llm_label_map is not None:
        _LLM_LABEL_MAP = llm_label_map


def _load_llm_labels(paths: list[str]) -> dict[tuple[str, int], dict]:
    """Load pre-computed LLM refactor labels from one or more label_refactor_llm.py outputs."""
    labels: dict[tuple[str, int], dict] = {}
    for path in paths:
        with open(path) as f:
            for ln in f:
                if not ln.strip():
                    continue
                row = json.loads(ln)
                key = (str(row["repo"]), int(row["pull_number"]))
                labels[key] = row
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a merged+closed PR steerer corpus from Go-specific or language-filtered PR tables.",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--labels-out", default=None)
    ap.add_argument("--summary-out", default=None)
    ap.add_argument("--merged-table", default="go_prs", help="Merged PR source table (e.g. go_prs, prs_copy).")
    ap.add_argument(
        "--closed-table",
        default="go_prs_closed",
        help="Closed PR source table (e.g. go_prs_closed, python_js_ts_rust_closed_prs).",
    )
    ap.add_argument("--language", default=None, help="Optional primary_language filter (e.g. 'python', 'go').")
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--submit-batch-size", type=int, default=4096)
    ap.add_argument("--limit-merged", type=int, default=0)
    ap.add_argument("--limit-closed", type=int, default=0)
    ap.add_argument("--shard-modulus", type=int, default=0)
    ap.add_argument("--shard-remainder", type=int, default=0)
    ap.add_argument(
        "--llm-labels",
        nargs="+",
        default=None,
        help="One or more LLM-judged refactor label JSONL files. When provided, replaces SQL regex labeling.",
    )
    args = ap.parse_args()

    if args.out is None or args.labels_out is None or args.summary_out is None:
        default_out, default_labels, default_summary = _default_outputs(
            args.merged_table,
            args.closed_table,
            args.language,
        )
        args.out = args.out or default_out
        args.labels_out = args.labels_out or default_labels
        args.summary_out = args.summary_out or default_summary

    conn = _load_db()
    merged_rows = _fetch_rows(
        conn,
        args.merged_table,
        args.limit_merged,
        args.shard_modulus,
        args.shard_remainder,
        args.language,
    )
    print(f"fetched {args.merged_table} rows: {len(merged_rows)}", flush=True)
    closed_rows = _fetch_rows(
        conn,
        args.closed_table,
        args.limit_closed,
        args.shard_modulus,
        args.shard_remainder,
        args.language,
    )
    print(f"fetched {args.closed_table} rows: {len(closed_rows)}", flush=True)
    conn.close()

    rows: list[tuple[str, tuple[Any, ...]]] = [
        (args.merged_table, r) for r in merged_rows
    ] + [
        (args.closed_table, r) for r in closed_rows
    ]
    repo_dirs = _repo_dir_map()
    print(f"repo_dir map size: {len(repo_dirs)}", flush=True)

    llm_label_map: dict[tuple[str, int], dict] | None = None
    if args.llm_labels:
        llm_label_map = _load_llm_labels(args.llm_labels)
        print(f"Loaded {len(llm_label_map)} LLM refactor labels from {len(args.llm_labels)} file(s)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.labels_out), exist_ok=True)

    counters = Counter()
    with open(args.out, "w") as f_feat, open(args.labels_out, "w") as f_lbl:
        with ProcessPoolExecutor(
            max_workers=max(1, args.workers),
            initializer=_init_worker,
            initargs=(repo_dirs, llm_label_map),
        ) as pool:
            processed = 0
            batch_size = max(1, int(args.submit_batch_size))
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                futures = [
                    pool.submit(_worker, row, source_table)
                    for source_table, row in batch
                ]
                for fut in as_completed(futures):
                    try:
                        feature_row, label_row, stats = fut.result()
                    except Exception as exc:
                        counters["worker_exception"] += 1
                        counters[f"worker_exception__{type(exc).__name__}"] += 1
                        processed += 1
                        continue
                    counters.update(stats)
                    if feature_row is not None and label_row is not None:
                        f_feat.write(json.dumps(feature_row) + "\n")
                        f_lbl.write(json.dumps(label_row) + "\n")
                    processed += 1
                    if processed % 2000 == 0:
                        print(
                            f"  processed={processed} ok={counters.get('rows_ok', 0)} "
                            f"err={counters.get('extract_error', 0)}",
                            flush=True,
                        )

    summary = {
        "language": args.language,
        "merged_table": args.merged_table,
        "closed_table": args.closed_table,
        "rows_requested": len(rows),
        "rows_emitted": int(counters.get("rows_ok", 0)),
        "rows_merged_source": len(merged_rows),
        "rows_closed_source": len(closed_rows),
        "accepted_rate": float(counters.get("accepted__1", 0) / max(1, counters.get("rows_ok", 0))),
        "refactor_requested_rate": float(counters.get("refactor_requested__1", 0) / max(1, counters.get("rows_ok", 0))),
        "errors": {
            k: int(v)
            for k, v in sorted(counters.items())
            if k.startswith("extract_error") or k.startswith("skipped_")
        },
        "by_source": {
            table: int(counters.get(f"rows_ok__{table}", 0))
            for table in dict.fromkeys([args.merged_table, args.closed_table])
        },
        "out": args.out,
        "labels_out": args.labels_out,
    }
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
