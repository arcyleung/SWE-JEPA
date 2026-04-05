#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import pg8000.native
import yaml

from patch_store import fetch_remote_patch_chunk, init_patch_sqlite, open_patch_sqlite

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

DEFAULT_PROJECTED = os.path.join(PROJECT_ROOT, "data", "phase6_2_canonical_latest", "projected_embeddings.npz")
DEFAULT_PG_CONFIG = os.path.join(PROJECT_ROOT, "postgres_connection.yaml")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "data", "phase7_2", "patch_store_canonical.sqlite")
DEFAULT_SUMMARY_OUT = os.path.join(PROJECT_ROOT, "data", "phase7_2", "patch_store_canonical_summary.json")


def _load_db(cfg_path: str) -> pg8000.native.Connection:
    cfg = yaml.safe_load(open(cfg_path))
    return pg8000.native.Connection(
        host=cfg["ip"],
        port=cfg.get("port", 9999),
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
    )


def _load_metadata(projected_path: str) -> list[dict[str, str]]:
    projected = np.load(projected_path, allow_pickle=True)
    instance_ids = projected["instance_ids"]
    repos = projected["repos"]
    rows = [{"instance_id": str(iid), "repo": str(repo)} for iid, repo in zip(instance_ids, repos)]
    dup_counter = Counter(row["instance_id"] for row in rows)
    if any(v > 1 for v in dup_counter.values()):
        raise RuntimeError("projected embeddings contain duplicate instance_ids")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projected", default=DEFAULT_PROJECTED)
    ap.add_argument("--pg-config", default=DEFAULT_PG_CONFIG)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    ap.add_argument("--fetch-chunk-size", type=int, default=1024)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rows = _load_metadata(args.projected)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)

    if args.overwrite and os.path.exists(args.out):
        os.remove(args.out)

    sqlite_conn = open_patch_sqlite(args.out)
    init_patch_sqlite(sqlite_conn)
    db_conn = _load_db(args.pg_config)

    existing_ids = {
        str(instance_id)
        for (instance_id,) in sqlite_conn.execute("SELECT instance_id FROM patches")
    }
    if existing_ids:
        print(
            f"Resuming export with {len(existing_ids):,} existing rows already present.",
            flush=True,
        )

    n_total = len(rows)
    rows = [row for row in rows if row["instance_id"] not in existing_ids]
    n_remaining = len(rows)
    n_written = len(existing_ids)
    n_missing = 0
    started = time.time()
    try:
        for start in range(0, n_remaining, args.fetch_chunk_size):
            chunk_rows = rows[start:start + args.fetch_chunk_size]
            t0 = time.time()
            fetched = fetch_remote_patch_chunk(db_conn, chunk_rows)
            sqlite_conn.executemany(
                "INSERT OR REPLACE INTO patches(instance_id, repo, patch) VALUES (?, ?, ?)",
                [
                    (row["instance_id"], row["repo"], fetched[row["instance_id"]])
                    for row in chunk_rows
                    if row["instance_id"] in fetched
                ],
            )
            sqlite_conn.commit()
            n_written += len(fetched)
            n_missing += len(chunk_rows) - len(fetched)
            print(
                f"chunk {start + 1:,}-{start + len(chunk_rows):,}/{n_remaining:,} remaining "
                f"(total_written={n_written:,}/{n_total:,}): "
                f"fetched={len(fetched):,} missing={len(chunk_rows) - len(fetched):,} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    finally:
        db_conn.close()
        sqlite_conn.close()

    summary = {
        "projected": args.projected,
        "pg_config": args.pg_config,
        "out": args.out,
        "n_total_rows": n_total,
        "n_written": n_written,
        "n_missing": n_missing,
        "fetch_chunk_size": args.fetch_chunk_size,
        "elapsed_sec": round(time.time() - started, 2),
    }
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
