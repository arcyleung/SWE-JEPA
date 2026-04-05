#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from typing import Any

import pg8000.native


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def parse_closed_pr_key(instance_id: str, repo: str) -> int | None:
    prefix = repo.replace("/", "__") + "__"
    if not instance_id.startswith(prefix):
        return None
    suffix = instance_id[len(prefix):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def fetch_remote_patch_chunk(
    conn: pg8000.native.Connection,
    metadata_rows: list[dict[str, Any]],
) -> dict[str, str]:
    if not metadata_rows:
        return {}

    out: dict[str, str] = {}
    instance_ids = [str(row["instance_id"]) for row in metadata_rows]
    quoted = ", ".join(sql_quote(iid) for iid in instance_ids)
    prs_rows = conn.run(
        f"""
        SELECT instance_id, patch
        FROM prs_copy
        WHERE patch IS NOT NULL
          AND instance_id IN ({quoted})
        """
    )
    for instance_id, patch in prs_rows:
        out[str(instance_id)] = str(patch)

    missing_rows = [row for row in metadata_rows if str(row["instance_id"]) not in out]
    if not missing_rows:
        return out

    values_rows = []
    for row in missing_rows:
        instance_id = str(row["instance_id"])
        repo = str(row["repo"])
        pull_number = parse_closed_pr_key(instance_id, repo)
        if pull_number is None:
            continue
        values_rows.append(
            f"({sql_quote(instance_id)}, {sql_quote(repo)}, {pull_number})"
        )
    if not values_rows:
        return out

    closed_rows = conn.run(
        f"""
        WITH wanted(instance_id_key, repo, pull_number) AS (
            VALUES {", ".join(values_rows)}
        )
        SELECT wanted.instance_id_key, c.patch
        FROM wanted
        JOIN python_js_ts_rust_closed_prs c
          ON c.repo = wanted.repo
         AND c.pull_number = wanted.pull_number
        WHERE c.patch IS NOT NULL
        """
    )
    for instance_id, patch in closed_rows:
        out[str(instance_id)] = str(patch)
    return out


def open_patch_sqlite(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_patch_sqlite(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS patches (
            instance_id TEXT PRIMARY KEY,
            repo TEXT NOT NULL,
            patch TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_patches_repo ON patches(repo)")
    conn.commit()


def fetch_sqlite_patch_chunk(
    conn: sqlite3.Connection,
    instance_ids: Iterable[str],
) -> dict[str, str]:
    ids = [str(x) for x in instance_ids]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT instance_id, patch FROM patches WHERE instance_id IN ({placeholders})",
        ids,
    ).fetchall()
    return {str(instance_id): str(patch) for instance_id, patch in rows}

