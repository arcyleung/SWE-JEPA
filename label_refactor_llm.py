#!/usr/bin/env python3
"""LLM-judged refactor labeling for PR review comments.

Reads review comments from Postgres (prs_copy / go_prs / go_prs_closed /
python_js_ts_rust_closed_prs),
sends each PR's review threads to qwen3.5_35b_a3b via local replicas
(round-robin), and writes labeled JSONL.

Usage:
    python label_refactor_llm.py --table prs_copy --limit 20 --concurrency 4
    python label_refactor_llm.py --table go_prs --concurrency 2000
    python label_refactor_llm.py --table python_js_ts_rust_closed_prs --language python --concurrency 2000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any

import aiohttp
import pg8000.native
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
PG_CONFIG_FILE = os.path.join(ROOT, "postgres_connection.yaml")
MODELS_YAML = os.path.join(ROOT, "models.yaml")


def _slug(value: str | None) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip())
    return out.strip("_")

SYSTEM_PROMPT = """\
You are a code review analyst. You classify whether a review thread \
requests a structural code change.\
"""

USER_PROMPT_TEMPLATE = """\
Read the following review thread from a pull request and determine:
1. Does this thread request a structural code change (refactoring, reorganization, extraction, moving code, renaming for clarity, splitting, etc.)?
2. If yes, what is the scope of the requested change?

A refactor request includes any suggestion to restructure, reorganize, extract, split, rename, move, simplify, or redesign code — even if phrased as "use X instead", "this should be Y", "consider making this a Z", or "fix X to be Y" when it implies structural change. It does NOT include bug fixes, feature requests, style nits, or typo corrections.

Review thread:
{thread_text}

Respond in exactly this JSON format:
{{"refactor_requested": true, "scope": "function", "reasoning": "one sentence"}}

- refactor_requested: true or false
- scope: "function", "module", "library_component", or "none"
- reasoning: one sentence explanation\
"""

# ── Helpers ─────────────────────────────────────────────────────────────────


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


def _extract_json_from_response(raw: str) -> str:
    """Robustly extract JSON object from model response."""
    text = raw.strip()

    if "```json" in text:
        try:
            s = text[text.index("```json") + 7:]
            return s[:s.index("```")].strip()
        except ValueError:
            pass

    if "```" in text:
        try:
            s = text[text.index("```") + 3:]
            return s[:s.index("```")].strip()
        except ValueError:
            pass

    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start: i + 1].strip()

    return text


def _load_replicas() -> list[dict]:
    """Load qwen3.5_35b_a3b replicas from models.yaml."""
    cfg = yaml.safe_load(open(MODELS_YAML))
    for m in cfg["model_list"]:
        if m["model_name"] == "qwen3.5_35b_a3b":
            replicas_urls = m.get("api_base_replicas", [])
            if not replicas_urls:
                replicas_urls = [m["litellm_params"]["api_base"]]
            api_key = m["litellm_params"]["api_key"]
            model = m["litellm_params"]["model"]
            result = []
            for url in replicas_urls:
                # Normalize: ensure ends with /v1
                if url.endswith("/chat/completions"):
                    url = url[: -len("/chat/completions")]
                if not url.endswith("/v1"):
                    url = url.rstrip("/") + "/v1"
                result.append({"api_base": url, "api_key": api_key, "model": model})
            return result
    raise ValueError("qwen3.5_35b_a3b not found in models.yaml")


# ── Thread extraction ───────────────────────────────────────────────────────


def _extract_threads(review_threads: Any, comments: Any) -> list[str]:
    """Extract review threads as concatenated text blocks."""
    threads: list[str] = []

    for th in _j(review_threads):
        if not isinstance(th, dict):
            continue
        parts: list[str] = []
        for c in _j(th.get("comments", [])):
            if isinstance(c, dict):
                body = str(c.get("body", "")).strip()
                author = str(c.get("author", ""))
                if body:
                    parts.append(f"[{author}]: {body}" if author else body)
        if parts:
            threads.append("\n".join(parts))

    # Standalone comments (not in review threads) — group as one thread
    standalone: list[str] = []
    for c in _j(comments):
        if isinstance(c, dict):
            body = str(c.get("body", "")).strip()
            author = str(c.get("author", ""))
            if body:
                standalone.append(f"[{author}]: {body}" if author else body)
    if standalone:
        threads.append("\n".join(standalone))

    return threads


# ── DB queries ──────────────────────────────────────────────────────────────


def _fetch_prs(
    conn: pg8000.native.Connection,
    table: str,
    limit: int,
    language: str | None = None,
) -> list[dict]:
    limit_sql = f"LIMIT {int(limit)}" if limit > 0 else ""
    params: dict[str, Any] = {}
    language_sql = ""
    if language:
        language_sql = "AND lower(primary_language) = :language_lower"
        params["language_lower"] = str(language).lower()

    if table == "prs_copy":
        rows = conn.run(
            f"""
            SELECT repo, pull_number, review_threads, comments
            FROM prs_copy
            WHERE review_threads IS NOT NULL
              AND review_threads != '[]'
              {language_sql}
            ORDER BY created_at DESC NULLS LAST
            {limit_sql}
            """,
            **params,
        )
    elif table in ("go_prs", "go_prs_closed", "python_js_ts_rust_closed_prs"):
        rows = conn.run(
            f"""
            SELECT repo, pull_number, review_threads, comments
            FROM {table}
            WHERE review_threads IS NOT NULL
              AND review_threads != '[]'
              {language_sql}
            ORDER BY created_at DESC NULLS LAST
            {limit_sql}
            """,
            **params,
        )
    else:
        raise ValueError(f"unsupported table: {table}")

    results = []
    for r in rows:
        repo, pull_number, review_threads, comments = r
        threads = _extract_threads(review_threads, comments)
        if threads:
            results.append({
                "repo": str(repo),
                "pull_number": int(pull_number),
                "threads": threads,
            })
    return results


# ── Async LLM calls ────────────────────────────────────────────────────────


_rr_index = 0


async def _call_llm(
    session,  # aiohttp.ClientSession
    thread_text: str,
    replicas: list[dict],
    semaphore: asyncio.Semaphore,
    retries: int = 3,
) -> dict:
    """Call LLM for a single thread. Returns parsed verdict dict."""
    global _rr_index
    _rr_index += 1
    replica = replicas[_rr_index % len(replicas)]

    # Truncate very long threads
    if len(thread_text) > 12000:
        thread_text = thread_text[:12000] + "\n... [truncated]"

    prompt = USER_PROMPT_TEMPLATE.format(thread_text=thread_text)
    payload = {
        "model": replica["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 256,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {
        "Authorization": f"Bearer {replica['api_key']}",
        "Content-Type": "application/json",
    }
    url = replica["api_base"].rstrip("/") + "/chat/completions"

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            async with semaphore:
                async with session.post(url, json=payload, headers=headers, timeout=60) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {body[:300]}")
                    data = await resp.json()
            content = data["choices"][0]["message"]["content"] or ""
            if not content.strip():
                last_exc = ValueError("empty_response")
                await asyncio.sleep(2 ** attempt)
                continue
            text = _extract_json_from_response(content)
            parsed = json.loads(text)
            return {
                "refactor_requested": bool(parsed.get("refactor_requested", False)),
                "scope": str(parsed.get("scope", "none")),
                "reasoning": str(parsed.get("reasoning", "")),
            }
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)

    return {
        "refactor_requested": False,
        "scope": "none",
        "reasoning": f"error: {last_exc}",
        "error": True,
    }


async def _label_pr(
    session,
    pr: dict,
    replicas: list[dict],
    semaphore: asyncio.Semaphore,
) -> dict:
    """Label all threads in a PR, return aggregated result."""
    tasks = [
        _call_llm(session, thread_text, replicas, semaphore)
        for thread_text in pr["threads"]
    ]
    verdicts = await asyncio.gather(*tasks)

    refactor_threads = [v for v in verdicts if v.get("refactor_requested")]
    error_count = sum(1 for v in verdicts if v.get("error"))

    # Aggregate scope: take the broadest scope found
    scope_order = {"none": 0, "function": 1, "module": 2, "library_component": 3}
    best_scope = "none"
    for v in refactor_threads:
        s = v.get("scope", "none")
        if scope_order.get(s, 0) > scope_order.get(best_scope, 0):
            best_scope = s

    return {
        "repo": pr["repo"],
        "pull_number": pr["pull_number"],
        "refactor_requested": len(refactor_threads) > 0,
        "refactor_scope": best_scope,
        "refactor_thread_count": len(refactor_threads),
        "total_threads": len(verdicts),
        "error_count": error_count,
        "thread_verdicts": verdicts,
        # Compatibility with _load_refactor_labels() in train_pr_steerer.py
        "s_t1": {
            "refactor_requested": int(len(refactor_threads) > 0),
        },
    }


async def _run(
    prs: list[dict],
    replicas: list[dict],
    concurrency: int,
    out_path: str,
    resume_keys: set[tuple[str, int]],
) -> list[dict]:
    import aiohttp

    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    done = 0
    total = len(prs)
    t0 = time.time()

    connector = aiohttp.TCPConnector(limit=2000)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Process in batches to avoid creating millions of coroutines
        batch_size = min(concurrency * 2, total)
        for batch_start in range(0, total, batch_size):
            batch = prs[batch_start:batch_start + batch_size]
            tasks = []
            for pr in batch:
                key = (pr["repo"], pr["pull_number"])
                if key in resume_keys:
                    done += 1
                    continue
                tasks.append(_label_pr(session, pr, replicas, semaphore))

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            with open(out_path, "a") as fout:
                for r in batch_results:
                    if isinstance(r, Exception):
                        done += 1
                        continue
                    results.append(r)
                    fout.write(json.dumps(r) + "\n")
                    done += 1

            elapsed = time.time() - t0
            refactor_count = sum(1 for r in results if r.get("refactor_requested"))
            rate = done / elapsed if elapsed > 0 else 0
            print(
                f"  [{done}/{total}]  refactor={refactor_count}  "
                f"rate={rate:.0f} PRs/s  elapsed={elapsed:.0f}s",
                flush=True,
            )

    return results


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="LLM-judged refactor labeling")
    ap.add_argument(
        "--table",
        required=True,
        choices=["prs_copy", "go_prs", "go_prs_closed", "python_js_ts_rust_closed_prs"],
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSONL path (default adds table name, plus _{language} when --language is set).",
    )
    ap.add_argument(
        "--language",
        default=None,
        help="Optional primary_language filter for mixed-language tables (e.g. 'python').",
    )
    ap.add_argument("--concurrency", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="Limit PRs to process (0 = all)")
    ap.add_argument("--resume", action="store_true", help="Skip PRs already in output file")
    ap.add_argument("--replicas", nargs="*", default=None, help="Override replica URLs")
    args = ap.parse_args()

    if args.out:
        out_path = args.out
    else:
        suffix = f"_{_slug(args.language)}" if args.language else ""
        out_path = os.path.join(ROOT, "data", f"refactor_labels_llm_{args.table}{suffix}.jsonl")
    summary_path = out_path.replace(".jsonl", "_summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load replicas
    if args.replicas:
        # Manual override
        base_replicas = _load_replicas()
        api_key = base_replicas[0]["api_key"]
        model = base_replicas[0]["model"]
        replicas = [{"api_base": u, "api_key": api_key, "model": model} for u in args.replicas]
    else:
        replicas = _load_replicas()
    print(f"Replicas: {len(replicas)}  concurrency: {args.concurrency}")

    # Resume support
    resume_keys: set[tuple[str, int]] = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for ln in f:
                if not ln.strip():
                    continue
                try:
                    row = json.loads(ln)
                    resume_keys.add((row["repo"], row["pull_number"]))
                except Exception:
                    pass
        print(f"Resume: {len(resume_keys)} PRs already labeled")
    elif not args.resume:
        # Truncate output file for fresh run
        open(out_path, "w").close()

    # Fetch from DB
    print(f"Fetching PRs from {args.table}...", flush=True)
    conn = _load_db()
    prs = _fetch_prs(conn, args.table, args.limit, args.language)
    conn.close()
    print(f"  {len(prs)} PRs with review threads", flush=True)

    # Run async labeling
    results = asyncio.run(_run(prs, replicas, args.concurrency, out_path, resume_keys))

    # Summary
    n = len(results)
    refactor_count = sum(1 for r in results if r.get("refactor_requested"))
    error_threads = sum(r.get("error_count", 0) for r in results)
    total_threads = sum(r.get("total_threads", 0) for r in results)

    scope_counts = {"none": 0, "function": 0, "module": 0, "library_component": 0}
    for r in results:
        s = r.get("refactor_scope", "none")
        scope_counts[s] = scope_counts.get(s, 0) + 1

    summary = {
        "table": args.table,
        "prs_labeled": n,
        "prs_with_refactor": refactor_count,
        "refactor_rate": round(refactor_count / n, 4) if n else 0.0,
        "total_threads": total_threads,
        "error_threads": error_threads,
        "scope_distribution": scope_counts,
        "out": out_path,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
