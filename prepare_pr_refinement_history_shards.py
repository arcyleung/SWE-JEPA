#!/usr/bin/env python3
"""Prepare balanced shard files for Slurm PR refinement-history extraction."""
from __future__ import annotations

import argparse
import json
import os

import pg8000.native
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
PG_CONFIG_FILE = os.path.join(ROOT, "postgres_connection.yaml")


def _load_db() -> pg8000.native.Connection:
    cfg = yaml.safe_load(open(PG_CONFIG_FILE))
    return pg8000.native.Connection(
        host=cfg["ip"],
        port=cfg.get("port", 9999),
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-shards", type=int, default=12)
    ap.add_argument("--heavy-out-dir", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--min-commits", type=int, default=2)
    ap.add_argument("--min-files", type=int, default=1)
    ap.add_argument("--max-files", type=int, default=40)
    ap.add_argument("--min-lines", type=int, default=10)
    ap.add_argument("--max-lines", type=int, default=10000)
    ap.add_argument("--order", choices=("newest", "oldest", "random"), default="newest")
    ap.add_argument("--require-review-events", action="store_true")
    ap.add_argument("--heavy-cost-threshold", type=float, default=1200.0)
    ap.add_argument("--heavy-commits-threshold", type=int, default=80)
    ap.add_argument("--heavy-files-threshold", type=int, default=30)
    ap.add_argument("--heavy-lines-threshold", type=int, default=6000)
    args = ap.parse_args()

    order_sql = {
        "newest": "created_at DESC NULLS LAST",
        "oldest": "created_at ASC NULLS LAST",
        "random": "random()",
    }[args.order]
    review_clause = "AND (COALESCE(total_review_threads, 0) > 0 OR COALESCE(submitted_reviews::text, '') <> '')" if args.require_review_events else ""
    limit_sql = f"LIMIT {int(args.limit)}" if args.limit and args.limit > 0 else ""

    conn = _load_db()
    rows = conn.run(
        f"""
        WITH latest AS (
            SELECT DISTINCT ON (instance_id)
                repo, instance_id, total_commits, COALESCE(changed_files, 0) AS changed_files, additions, deletions, created_at
            FROM prs_copy
            WHERE pr_merged = TRUE
              AND base_sha IS NOT NULL
              AND commits IS NOT NULL
              {review_clause}
            ORDER BY instance_id, crawl_time DESC NULLS LAST
        )
        SELECT
            repo,
            instance_id,
            total_commits,
            changed_files,
            COALESCE(additions, 0) + COALESCE(deletions, 0) AS total_lines,
            created_at
        FROM latest
        WHERE total_commits >= :min_commits
          AND changed_files BETWEEN :min_files AND :max_files
          AND (COALESCE(additions, 0) + COALESCE(deletions, 0)) BETWEEN :min_lines AND :max_lines
        ORDER BY {order_sql}
        {limit_sql}
        """,
        min_commits=args.min_commits,
        min_files=args.min_files,
        max_files=args.max_files,
        min_lines=args.min_lines,
        max_lines=args.max_lines,
    )
    conn.close()

    os.makedirs(args.out_dir, exist_ok=True)
    heavy_out_dir = args.heavy_out_dir or os.path.join(args.out_dir, "heavy_queue")
    os.makedirs(heavy_out_dir, exist_ok=True)
    shards = [[] for _ in range(args.n_shards)]
    shard_cost = [0.0] * args.n_shards
    weighted = []
    heavy_records = []

    def is_heavy(rec: dict[str, object]) -> bool:
        cost = float(rec["cost"])
        total_commits = int(rec["total_commits"])
        changed_files = int(rec["changed_files"])
        total_lines = int(rec["total_lines"])
        return (
            (args.heavy_cost_threshold > 0 and cost >= args.heavy_cost_threshold)
            or (args.heavy_commits_threshold > 0 and total_commits >= args.heavy_commits_threshold)
            or (args.heavy_files_threshold > 0 and changed_files >= args.heavy_files_threshold)
            or (args.heavy_lines_threshold > 0 and total_lines >= args.heavy_lines_threshold)
        )

    for repo, iid, total_commits, changed_files, total_lines, created_at in rows:
        cost = float(total_commits or 0) * max(1.0, float(total_lines or 0) / 200.0)
        rec = {
            "repo": repo,
            "instance_id": iid,
            "total_commits": int(total_commits or 0),
            "changed_files": int(changed_files or 0),
            "total_lines": int(total_lines or 0),
            "created_at": str(created_at or ""),
            "cost": cost,
        }
        if is_heavy(rec):
            heavy_records.append(rec)
        else:
            weighted.append(rec)
    weighted.sort(key=lambda r: r["cost"], reverse=True)
    for rec in weighted:
        idx = min(range(args.n_shards), key=lambda i: shard_cost[i])
        shards[idx].append(rec)
        shard_cost[idx] += rec["cost"]

    heavy_records.sort(key=lambda r: r["cost"], reverse=True)
    heavy_txt_path = os.path.join(heavy_out_dir, "heavy_queue.txt")
    heavy_json_path = os.path.join(heavy_out_dir, "heavy_queue.json")
    with open(heavy_txt_path, "w") as f:
        for rec in heavy_records:
            f.write(f"{rec['instance_id']}\n")
    with open(heavy_json_path, "w") as f:
        json.dump(heavy_records, f, indent=2)

    manifest = {
        "n_rows": len(weighted) + len(heavy_records),
        "n_regular_rows": len(weighted),
        "n_heavy_rows": len(heavy_records),
        "n_shards": args.n_shards,
        "heavy_queue": {
            "count": len(heavy_records),
            "cost": float(sum(rec["cost"] for rec in heavy_records)),
            "txt": heavy_txt_path,
            "json": heavy_json_path,
            "thresholds": {
                "cost": args.heavy_cost_threshold,
                "commits": args.heavy_commits_threshold,
                "files": args.heavy_files_threshold,
                "lines": args.heavy_lines_threshold,
            },
        },
        "shards": [],
    }
    for idx, records in enumerate(shards):
        txt_path = os.path.join(args.out_dir, f"shard_{idx:03d}.txt")
        json_path = os.path.join(args.out_dir, f"shard_{idx:03d}.json")
        with open(txt_path, "w") as f:
            for rec in records:
                f.write(f"{rec['instance_id']}\n")
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2)
        manifest["shards"].append(
            {
                "shard": idx,
                "count": len(records),
                "cost": shard_cost[idx],
                "txt": txt_path,
                "json": json_path,
            }
        )

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"Prepared {len(weighted)} regular PRs across {args.n_shards} shards; "
        f"deferred {len(heavy_records)} heavy PRs",
    )
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
