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
) -> list[tuple[Any, ...]]:
    limit_sql = f"limit {int(limit)}" if limit > 0 else ""
    shard_sql = ""
    params: dict[str, Any] = {"ref_regex": REFACTOR_SQL_REGEX}
    if shard_modulus > 0:
        shard_sql = (
            "and mod(abs(hashtext(repo || '#' || pull_number::text)), :shard_modulus) = :shard_remainder"
        )
        params["shard_modulus"] = int(shard_modulus)
        params["shard_remainder"] = int(shard_remainder)
    if table == "go_prs":
        return conn.run(
            f"""
            select
                repo,
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

    iid = _synthetic_instance_id(str(repo), int(pull_number), source_table)
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


def _init_worker(repo_dirs: dict[str, str]) -> None:
    global _REPO_DIRS
    _REPO_DIRS = repo_dirs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--labels-out", default=DEFAULT_LABELS)
    ap.add_argument("--summary-out", default=DEFAULT_SUMMARY)
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--submit-batch-size", type=int, default=4096)
    ap.add_argument("--limit-merged", type=int, default=0)
    ap.add_argument("--limit-closed", type=int, default=0)
    ap.add_argument("--shard-modulus", type=int, default=0)
    ap.add_argument("--shard-remainder", type=int, default=0)
    args = ap.parse_args()

    conn = _load_db()
    merged_rows = _fetch_rows(conn, "go_prs", args.limit_merged, args.shard_modulus, args.shard_remainder)
    print(f"fetched go_prs rows: {len(merged_rows)}", flush=True)
    closed_rows = _fetch_rows(conn, "go_prs_closed", args.limit_closed, args.shard_modulus, args.shard_remainder)
    print(f"fetched go_prs_closed rows: {len(closed_rows)}", flush=True)
    conn.close()

    rows: list[tuple[str, tuple[Any, ...]]] = [("go_prs", r) for r in merged_rows] + [("go_prs_closed", r) for r in closed_rows]
    repo_dirs = _repo_dir_map()
    print(f"repo_dir map size: {len(repo_dirs)}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.labels_out), exist_ok=True)

    counters = Counter()
    with open(args.out, "w") as f_feat, open(args.labels_out, "w") as f_lbl:
        with ProcessPoolExecutor(
            max_workers=max(1, args.workers),
            initializer=_init_worker,
            initargs=(repo_dirs,),
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
            "go_prs": int(counters.get("rows_ok__go_prs", 0)),
            "go_prs_closed": int(counters.get("rows_ok__go_prs_closed", 0)),
        },
        "out": args.out,
        "labels_out": args.labels_out,
    }
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
