#!/usr/bin/env python3
"""Prepare balanced shard files for Slurm patch-embedding extraction (Experiment 6.1)."""
from __future__ import annotations

import argparse
import json
import os

import pg8000.native
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
PG_CONFIG_FILE = os.path.join(PROJECT_ROOT, "postgres_connection.yaml")


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
    ap.add_argument("--n-shards", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    conn = _load_db()
    limit_sql = f"LIMIT {int(args.limit)}" if args.limit > 0 else ""
    rows = conn.run(
        f"""
        SELECT * FROM (
            SELECT instance_id, repo, COALESCE(additions, 0) + COALESCE(deletions, 0) AS total_lines
            FROM prs_copy
            WHERE patch IS NOT NULL
            UNION ALL
            SELECT
                REPLACE(repo, '/', '__') || '__' || pull_number AS instance_id,
                repo,
                COALESCE(additions, 0) + COALESCE(deletions, 0) AS total_lines
            FROM python_js_ts_rust_closed_prs
            WHERE patch IS NOT NULL
        ) AS combined
        {limit_sql}
        """
    )
    conn.close()
    print(f"Fetched {len(rows)} rows from DB", flush=True)

    # Greedy cost-balanced sharding by total_lines
    records = []
    for instance_id, repo, total_lines in rows:
        cost = max(float(total_lines or 0), 1.0)
        records.append({"instance_id": instance_id, "repo": repo, "total_lines": int(total_lines or 0), "cost": cost})

    records.sort(key=lambda r: r["cost"], reverse=True)
    shards: list[list[dict]] = [[] for _ in range(args.n_shards)]
    shard_cost = [0.0] * args.n_shards

    for rec in records:
        idx = min(range(args.n_shards), key=lambda i: shard_cost[i])
        shards[idx].append(rec)
        shard_cost[idx] += rec["cost"]

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {
        "n_rows": len(records),
        "n_shards": args.n_shards,
        "shards": [],
    }

    for idx, shard_records in enumerate(shards):
        txt_path = os.path.join(args.out_dir, f"shard_{idx:03d}.txt")
        json_path = os.path.join(args.out_dir, f"shard_{idx:03d}.json")
        with open(txt_path, "w") as f:
            for rec in shard_records:
                f.write(f"{rec['instance_id']}\n")
        with open(json_path, "w") as f:
            json.dump(shard_records, f, indent=2)
        manifest["shards"].append({
            "shard": idx,
            "count": len(shard_records),
            "cost": shard_cost[idx],
            "txt": txt_path,
            "json": json_path,
        })

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Prepared {len(records)} PRs across {args.n_shards} shards")
    for s in manifest["shards"]:
        print(f"  shard {s['shard']:03d}: {s['count']:,} rows, cost={s['cost']:.0f}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
