#!/usr/bin/env python3
"""
Language-agnostic PR title classifier for any Postgres table.

Reads PR rows from a Postgres table, classifies titles via LLM, and writes
pr_category / pr_category_confidence / pr_category_reasoning back to the DB.

Idempotent: skips rows where pr_category_confidence IS NOT NULL unless --force.

Usage:
    source .venv/bin/activate
    # Label go_prs with local qwen3.5 model
    python pr_label.py \
        --table go_prs \
        --pk-cols repo pull_number \
        --title-col pr_title \
        --model qwen3.5_35b_a3b \
        --concurrency 32 \
        --batch-size 50

    # Label a subset of repos
    python pr_label.py \
        --table go_prs \
        --where "repo IN ('owner/foo','owner/bar')" \
        --model qwen3.5_35b_a3b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import psycopg2
import psycopg2.extras
import yaml
from openai import OpenAI

ROOT = os.path.dirname(os.path.abspath(__file__))

VALID_CATEGORIES = {"feature", "bugfix", "maintenance", "docs", "test", "other"}

BATCH_PROMPT_TEMPLATE = """\
Classify these {n} pull request titles into categories.

Categories:
- feature: New functionality, enhancements, or additions
- bugfix: Bug fixes, error corrections, issue/crash resolutions
- maintenance: Refactoring, dependency bumps, code cleanup, minor improvements
- docs: Documentation updates, comment additions
- test: Test additions, test modifications, test fixes
- other: Miscellaneous changes that don't fit above categories

PR Titles (JSON array, 0-indexed):
{titles_json}

Return a JSON array with exactly {n} objects (one per title, in order):
[
  {{"index": 0, "category": "<category>", "confidence": <1-5>, "reasoning": "<one sentence>"}},
  ...
]
IMPORTANT: output ONLY the JSON array, no prose."""


def _pg_connect(pg_yaml: str):
    cfg = yaml.safe_load(open(pg_yaml))
    return psycopg2.connect(
        host=cfg["ip"], port=cfg["port"],
        user=cfg["user"], password=cfg["password"],
        database=cfg["database"],
    )


def _load_model_cfg(models_yaml: str, model_name: str) -> dict:
    cfg = yaml.safe_load(open(models_yaml))
    for m in cfg.get("model_list", []):
        if m["model_name"] == model_name:
            return m
    raise ValueError(f"Model '{model_name}' not found in {models_yaml}")


def _make_client(model_cfg: dict) -> tuple[OpenAI, str]:
    """Return (OpenAI client, clean model id) from model config."""
    lp = model_cfg["litellm_params"]
    api_base = lp.get("api_base", "")
    api_key  = lp.get("api_key", "sk-placeholder")
    model_id = model_cfg.get("model_id") or lp.get("model", model_cfg["model_name"])
    # Strip litellm prefix like "hosted_vllm/"
    if "/" in model_id and not model_id.startswith("/"):
        model_id = model_id.split("/", 1)[1]
    # Strip trailing slash (vLLM serves without it)
    model_id = model_id.rstrip("/")
    client = OpenAI(base_url=api_base, api_key=api_key)
    return client, model_id


def _extract_json(text: str) -> str:
    """Strip markdown fences if present."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


def _classify_batch(
    client: OpenAI,
    model_id: str,
    titles: list[str],
    temperature: float = 0.2,
    retries: int = 3,
) -> list[dict]:
    """
    Classify a batch of PR titles via LLM.
    Returns list of {category, confidence, reasoning} dicts, same length as titles.
    Falls back to "other"/1 on failure.
    """
    prompt = BATCH_PROMPT_TEMPLATE.format(
        n=len(titles),
        titles_json=json.dumps(titles, ensure_ascii=False, indent=2),
    )
    fallback = [{"category": "other", "confidence": 1, "reasoning": "classification failed"} for _ in titles]

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=4096,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = resp.choices[0].message.content
            if raw is None:
                # Some thinking-mode models put output in reasoning_content when content=None
                raw = getattr(resp.choices[0].message, "reasoning_content", None) or ""
            content = _extract_json(raw)
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                raise ValueError("expected JSON array")
            parsed.sort(key=lambda x: x.get("index", 0))
            results = []
            for item in parsed:
                cat = item.get("category", "other").lower()
                if cat not in VALID_CATEGORIES:
                    cat = "other"
                results.append({
                    "category":   cat,
                    "confidence": max(1, min(5, int(item.get("confidence", 1)))),
                    "reasoning":  str(item.get("reasoning", ""))[:500],
                })
            # Pad if model returned fewer items
            while len(results) < len(titles):
                results.append({"category": "other", "confidence": 1, "reasoning": "missing"})
            return results[:len(titles)]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [classify_batch] Failed after {retries} attempts: {e}", flush=True)
    return fallback


def _ensure_columns(conn, table: str) -> None:
    """Add classification columns if they don't exist."""
    cur = conn.cursor()
    for col, col_type in [
        ("pr_category_confidence", "smallint"),
        ("pr_category_reasoning",  "text"),
    ]:
        cur.execute(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_name=%s AND column_name=%s
            """,
            (table, col),
        )
        if not cur.fetchone():
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            print(f"  Added column {table}.{col}", flush=True)
    conn.commit()
    cur.close()


def _fetch_rows(
    conn,
    table: str,
    pk_cols: list[str],
    title_col: str,
    where: Optional[str],
    force: bool,
    limit: Optional[int],
) -> list[dict]:
    """Fetch rows needing classification."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    select_cols = ", ".join(pk_cols + [title_col])
    conditions = []
    if not force:
        conditions.append("pr_category_confidence IS NULL")
    if where:
        conditions.append(f"({where})")
    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"SELECT {select_cols} FROM {table} {where_clause} {limit_clause}"
    cur.execute(query)
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    return rows


def _update_rows(
    conn,
    table: str,
    pk_cols: list[str],
    updates: list[dict],
) -> None:
    """Batch-update classification columns in DB."""
    if not updates:
        return
    cur = conn.cursor()
    pk_where = " AND ".join(f"{c}=%s" for c in pk_cols)
    sql = (
        f"UPDATE {table} SET "
        f"pr_category=%s, pr_category_confidence=%s, pr_category_reasoning=%s "
        f"WHERE {pk_where}"
    )
    args = [
        (u["category"], u["confidence"], u["reasoning"]) + tuple(u[c] for c in pk_cols)
        for u in updates
    ]
    cur.executemany(sql, args)
    conn.commit()
    cur.close()


def _process_chunk(
    rows: list[dict],
    pk_cols: list[str],
    title_col: str,
    client: OpenAI,
    model_id: str,
    batch_size: int,
    temperature: float,
) -> list[dict]:
    """Classify a chunk of rows and return list of update dicts."""
    updates = []
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i + batch_size]
        titles = [str(r.get(title_col) or "") for r in batch_rows]
        results = _classify_batch(client, model_id, titles, temperature)
        for row, res in zip(batch_rows, results):
            upd = {c: row[c] for c in pk_cols}
            upd.update(res)
            updates.append(upd)
    return updates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table",       default="go_prs", help="Postgres table name")
    ap.add_argument("--pk-cols",     nargs="+", default=["repo", "pull_number"],
                    help="Primary key column(s)")
    ap.add_argument("--title-col",   default="pr_title", help="Column with PR title text")
    ap.add_argument("--where",       default=None, help="Optional SQL WHERE clause to restrict rows")
    ap.add_argument("--model",       default="qwen3.5_35b_a3b", help="Model name from models.yaml")
    ap.add_argument("--models-yaml", default=os.path.join(ROOT, "models.yaml"))
    ap.add_argument("--pg-yaml",     default=os.path.join(ROOT, "postgres_connection.yaml"))
    ap.add_argument("--concurrency", type=int, default=16, help="Concurrent API threads")
    ap.add_argument("--batch-size",  type=int, default=50, help="Titles per API call")
    ap.add_argument("--chunk-size",  type=int, default=500,
                    help="Rows per thread chunk (committed together)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--limit",       type=int, default=None, help="Max rows to process (for testing)")
    ap.add_argument("--force",       action="store_true",
                    help="Re-classify even if pr_category_confidence is already set")
    args = ap.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    model_cfg = _load_model_cfg(args.models_yaml, args.model)
    client, model_id = _make_client(model_cfg)
    print(f"Model: {args.model}  (id={model_id})", flush=True)

    conn = _pg_connect(args.pg_yaml)
    _ensure_columns(conn, args.table)

    # ── Fetch rows ───────────────────────────────────────────────────────────
    rows = _fetch_rows(
        conn, args.table, args.pk_cols, args.title_col,
        args.where, args.force, args.limit,
    )
    print(f"Rows to classify: {len(rows)}", flush=True)
    if not rows:
        conn.close()
        return

    # ── Classify in parallel chunks ──────────────────────────────────────────
    chunks = [rows[i:i + args.chunk_size] for i in range(0, len(rows), args.chunk_size)]
    total_done = 0
    category_counts: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                _process_chunk,
                chunk, args.pk_cols, args.title_col,
                client, model_id, args.batch_size, args.temperature,
            ): len(chunk)
            for chunk in chunks
        }
        for fut in as_completed(futures):
            n_rows = futures[fut]
            try:
                updates = fut.result()
            except Exception as exc:
                print(f"  Chunk failed: {exc}", flush=True)
                continue
            _update_rows(conn, args.table, args.pk_cols, updates)
            total_done += n_rows
            for u in updates:
                category_counts[u["category"]] = category_counts.get(u["category"], 0) + 1
            print(f"  {total_done}/{len(rows)} rows committed  "
                  f"dist={dict(sorted(category_counts.items()))}", flush=True)

    conn.close()
    print(f"\nDone. {total_done} rows classified.", flush=True)
    print("Category distribution:")
    for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt} ({cnt/total_done:.1%})")


if __name__ == "__main__":
    main()
