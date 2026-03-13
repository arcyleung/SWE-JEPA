#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any

from extract_pr_refinement_history import (
    CHECKPOINT_TABLE,
    _commit_list,
    _is_merge_headline,
    _load_db,
)


def _select_rows(conn: Any, run_tag: str, only_missing: bool) -> list[tuple[str, str, str, Any]]:
    missing_clause = ""
    if only_missing:
        missing_clause = "AND payload_json::text NOT LIKE '%is_merge_commit_sampled%'"
    return conn.run(
        f"""
        WITH latest_pr AS (
            SELECT DISTINCT ON (instance_id)
                instance_id,
                commits
            FROM prs_copy
            WHERE commits IS NOT NULL
            ORDER BY instance_id, crawl_time DESC NULLS LAST
        )
        SELECT
            cp.instance_id,
            cp.repo,
            cp.payload_json::text,
            lp.commits
        FROM {CHECKPOINT_TABLE} cp
        LEFT JOIN latest_pr lp
          ON lp.instance_id = cp.instance_id
        WHERE cp.run_tag = :run_tag
          AND cp.status = 'ok'
          {missing_clause}
        ORDER BY cp.updated_at ASC
        """,
        run_tag=run_tag,
    )


def _commit_lookup(commits_raw: Any) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    commits = _commit_list(commits_raw)
    return (
        {str(commit.get("hash") or ""): commit for commit in commits if commit.get("hash")},
        {
            "merge_commit_count": sum(1 for commit in commits if commit.get("is_merge_commit")),
            "non_merge_commit_count": sum(1 for commit in commits if not commit.get("is_merge_commit")),
        },
    )


def _update_checkpoint(
    conn: Any,
    run_tag: str,
    instance_id: str,
    payload_json: str,
) -> None:
    conn.run(
        f"""
        UPDATE {CHECKPOINT_TABLE}
        SET payload_json = CAST(:payload_json AS jsonb),
            updated_at = now()
        WHERE run_tag = :run_tag
          AND instance_id = :instance_id
        """,
        run_tag=run_tag,
        instance_id=instance_id,
        payload_json=payload_json,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--batch-size", type=int, default=250)
    ap.add_argument("--only-missing", action="store_true")
    args = ap.parse_args()

    conn = _load_db()
    try:
        rows = _select_rows(conn, args.run_tag, args.only_missing)
        total = len(rows)
        print(f"Backfilling merge flags for run_tag={args.run_tag} rows={total}", flush=True)
        updated = 0
        scanned = 0
        for instance_id, repo, payload_text, commits_raw in rows:
            scanned += 1
            try:
                payload = json.loads(payload_text or "{}")
            except Exception:
                payload = {}
            snapshot_rows = payload.get("snapshot_rows")
            if not isinstance(snapshot_rows, list) or not snapshot_rows:
                continue
            commit_by_sha, stats = _commit_lookup(commits_raw)
            changed = False
            for row in snapshot_rows:
                if not isinstance(row, dict):
                    continue
                commit_sha = str(row.get("commit_sha") or "")
                commit_meta = commit_by_sha.get(commit_sha)
                if commit_meta is not None:
                    merge_flag = int(bool(commit_meta.get("is_merge_commit")))
                    parent_count = commit_meta.get("merge_commit_parent_count")
                    detection_mode = str(commit_meta.get("merge_commit_detection_mode") or "")
                else:
                    merge_flag = int(_is_merge_headline(str(row.get("commit_message_headline") or "")))
                    parent_count = None
                    detection_mode = "headline_heuristic_row_fallback"
                desired = {
                    "is_merge_commit_sampled": merge_flag,
                    "merge_commit_parent_count": parent_count,
                    "merge_commit_detection_mode": detection_mode,
                    "merge_commits_total": int(stats["merge_commit_count"]),
                    "non_merge_commits_total": int(stats["non_merge_commit_count"]),
                    "merge_commits_skipped_from_sampling": int(
                        stats["merge_commit_count"] if stats["non_merge_commit_count"] > 0 else 0
                    ),
                }
                for key, value in desired.items():
                    if row.get(key) != value:
                        row[key] = value
                        changed = True
            if changed:
                payload["snapshot_rows"] = snapshot_rows
                _update_checkpoint(conn, args.run_tag, str(instance_id), json.dumps(payload, sort_keys=True))
                updated += 1
            if scanned % max(1, args.batch_size) == 0 or scanned == total:
                print(f"  scanned={scanned}/{total} updated={updated}", flush=True)
        print(f"Backfill complete: updated={updated} scanned={scanned}", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
