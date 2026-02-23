"""
Experiment 0.2 documentation run.

Pipeline:
  1. KNN retrieval — re-runs the 8 query functions from Exp 0.2 against the
     Qwen2.5-Coder-3B FAISS index (k=5 neighbours each → 40 pairs total)
  2. Source fetch — retrieves function source from git for every unique
     (query, neighbour) ID via the existing llm_similarity_judge helper
  3. LLM judge — scores each pair at 3 granularity levels (signature,
     body, full) using the local vLLM endpoint with up to 50 concurrent
     requests
  4. JSONL output — one line per (query, neighbour) pair written to
     docs/phase0_2_knn_results_raw.jsonl

Usage:
    python run_exp02_eval.py
    python run_exp02_eval.py --k 5 --concurrency 50
"""

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import re

import numpy as np
import yaml
import pg8000.native
import faiss
from openai import AsyncOpenAI

# ── Paths & config ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PG_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'postgres_connection.yaml')
INDEX_DIR      = os.path.join(os.path.dirname(__file__), 'faiss_indices')
OUTPUT_FILE    = os.path.join(os.path.dirname(__file__), 'docs',
                              'phase0_2_knn_results_raw.jsonl')

_pg_cfg = yaml.safe_load(open(PG_CONFIG_FILE))
DB = dict(
    host=_pg_cfg['ip'],
    port=_pg_cfg.get('port') or 9999,
    user=_pg_cfg['user'],
    password=_pg_cfg['password'],
    database=_pg_cfg['database'],
)

LLM_ENDPOINT = 'http://10.10.110.65:24000/v1'
LLM_API_KEY  = 'apikey1234'
LLM_MODEL    = 'qwen3_coder_30b'

# The 8 query function IDs from Experiment 0.2 (Qwen2.5-Coder-3B rows)
EXP02_QUERY_IDS = [10691, 19774, 12162, 23057, 140, 17448, 25332, 4716]

# ── KNN retrieval ───────────────────────────────────────────────────────────

def run_knn(query_ids: list[int], k: int, model_tag: str = '_Qwen2.5-Coder-3B_L18'
            ) -> list[dict]:
    """
    For each query ID, fetch its embedding and retrieve k nearest neighbours.
    Returns a list of pair dicts (without source / judge scores yet).
    """
    index_path = os.path.join(INDEX_DIR, f'function_embeddings{model_tag}.faiss')
    ids_path   = os.path.join(INDEX_DIR, f'function_embeddings{model_tag}_ids.npy')
    index = faiss.read_index(index_path)
    ids   = np.load(ids_path)
    print(f"Loaded FAISS index: {index.ntotal} vectors (dim={index.d})", flush=True)

    conn = pg8000.native.Connection(**DB)

    # Bulk-fetch embeddings and metadata for all query IDs
    id_list = ', '.join(str(i) for i in query_ids)
    meta_rows = conn.run(
        f"SELECT id, instance_id, feature_file, feature_function, embedding "
        f"FROM function_embeddings WHERE id IN ({id_list})"
    )
    by_id = {r[0]: r for r in meta_rows}
    conn.close()

    pairs = []
    for qid in query_ids:
        if qid not in by_id:
            print(f"  WARNING: query id={qid} not found in DB, skipping", flush=True)
            continue
        _, q_iid, q_file, q_func, q_emb = by_id[qid]
        q_vec = np.array(q_emb, dtype=np.float32)
        q_vec /= (np.linalg.norm(q_vec) + 1e-9)

        sims, positions = index.search(q_vec.reshape(1, -1), k + 1)
        nbr_pg_ids = ids[positions[0]]
        nbr_sims   = sims[0]

        # Drop self-match
        mask       = nbr_pg_ids != qid
        nbr_pg_ids = nbr_pg_ids[mask][:k]
        nbr_sims   = nbr_sims[mask][:k]

        # Fetch neighbour metadata
        nbr_id_list = ', '.join(str(int(i)) for i in nbr_pg_ids)
        conn2 = pg8000.native.Connection(**DB)
        nbr_rows = conn2.run(
            f"SELECT id, instance_id, feature_file, feature_function "
            f"FROM function_embeddings WHERE id IN ({nbr_id_list})"
        )
        conn2.close()
        nbr_meta = {r[0]: r for r in nbr_rows}

        print(f"  query id={qid} ({q_func}): {len(nbr_pg_ids)} neighbours", flush=True)

        for rank, (nid, sim) in enumerate(zip(nbr_pg_ids, nbr_sims), 1):
            nid = int(nid)
            nm  = nbr_meta.get(nid, (nid, '', '', ''))
            pairs.append({
                'query_id':           qid,
                'query_function':     q_func,
                'query_file':         q_file,
                'query_instance_id':  q_iid,
                'neighbour_id':       nid,
                'neighbour_function': nm[3],
                'neighbour_file':     nm[2],
                'neighbour_instance_id': nm[1],
                'rank':               rank,
                'cosine_similarity':  float(sim),
                'model_tag':          model_tag.lstrip('_'),
                # filled in later:
                'query_source':       None,
                'neighbour_source':   None,
                'judge':              None,
            })
    return pairs


# ── Source fetching ─────────────────────────────────────────────────────────

def _fetch_source_via_overlay(
    pg_id: int,
    conn: pg8000.native.Connection,
) -> str:
    """
    Fetch the source of a single named function using the same overlayfs + git
    fetch approach as phase0_1_similarity_matrix.py.  Falls back to '' on failure.
    """
    from phase0_1_similarity_matrix import (
        REPOS_BASE, OVERLAY_MERGED_BASE,
        _mount_overlay, _umount_overlay, _sha_available, _fetch_sha,
    )
    from llm_similarity_judge import _extract_named_function

    # 1. Look up metadata from function_embeddings
    rows = conn.run(
        "SELECT instance_id, feature_file, feature_function "
        "FROM function_embeddings WHERE id = :id", id=pg_id)
    if not rows:
        return ''
    instance_id, feature_file, feature_function = rows[0]

    # 2. Look up PR info
    pr_rows = conn.run(
        "SELECT repo, repo_id, base_commit FROM prs WHERE instance_id = :iid",
        iid=instance_id)
    if not pr_rows:
        return ''
    repo_slug, repo_id, base_commit = pr_rows[0]
    if not base_commit:
        return ''

    # 3. Find on-disk repo directory
    repo_dir = None
    try:
        for entry in os.listdir(REPOS_BASE):
            if entry.startswith(str(repo_id) + '_'):
                repo_dir = os.path.join(REPOS_BASE, entry)
                break
    except OSError:
        return ''
    if not repo_dir:
        return ''

    # 4. Mount overlayfs, fetch commit if needed, git show
    tag = f'exp02src-{repo_id}-{pg_id}'
    os.makedirs(OVERLAY_MERGED_BASE, exist_ok=True)
    try:
        merged, upper, work = _mount_overlay(repo_dir, tag)
    except Exception:
        return ''

    src = ''
    try:
        import subprocess
        if not _sha_available(merged, base_commit):
            _fetch_sha(merged, base_commit, repo_slug, gh_token=None)
        if _sha_available(merged, base_commit):
            r = subprocess.run(
                ['git', '-C', merged, 'show', f'{base_commit}:{feature_file}'],
                capture_output=True, timeout=30)
            if r.returncode == 0:
                full = r.stdout.decode(errors='replace')
                extracted = _extract_named_function(full, feature_function)
                src = extracted or ''
    finally:
        _umount_overlay(merged, upper, work)

    return src


def fetch_sources(pairs: list[dict]) -> dict[int, str]:
    """
    Fetch source code for every unique function ID appearing in pairs.
    Uses overlayfs + git fetch (same as the embedding pipeline) so we can
    reach commits that aren't in the shallow clone.
    Returns {pg_id: source_str}.
    """
    unique_ids = set()
    for p in pairs:
        unique_ids.add(p['query_id'])
        unique_ids.add(p['neighbour_id'])

    conn  = pg8000.native.Connection(**DB)
    cache: dict[int, str] = {}
    total = len(unique_ids)
    for i, fid in enumerate(sorted(unique_ids), 1):
        print(f"  [{i}/{total}] fetching source for id={fid} …", flush=True)
        src = _fetch_source_via_overlay(fid, conn)
        if not src:
            print(f"    WARNING: could not retrieve source for id={fid}", flush=True)
        cache[fid] = src
    conn.close()
    return cache


# ── Async LLM judge ─────────────────────────────────────────────────────────

SCORE_RUBRIC = """\
Score scale:
  3 – functionally equivalent / same algorithm
  2 – moderately similar (same purpose or similar approach, notable differences)
  1 – weakly related (same broad domain or category)
  0 – unrelated"""

_SYSTEM = "You are a senior software engineer evaluating Python function similarity."


def _build_prompt_a(sig_a: str, sig_b: str) -> str:
    return (
        "Evaluate the functional similarity of these two Python functions based ONLY on "
        "their names, parameters, return types, and docstrings — do NOT infer anything "
        "about their implementations.\n\n"
        f"Function A (signature only):\n```python\n{sig_a}\n```\n\n"
        f"Function B (signature only):\n```python\n{sig_b}\n```\n\n"
        f"{SCORE_RUBRIC}\n\n"
        'Respond with a single JSON object:\n{"score": <0-3>, "reasoning": "<one concise sentence>"}'
    )


def _build_prompt_b(body_a: str, body_b: str) -> str:
    return (
        "Evaluate the algorithmic similarity of these two Python functions. The function "
        "names have been replaced with generic labels so you cannot use naming as a signal. "
        "Judge based ONLY on what the code actually does.\n\n"
        f"Function A (name redacted):\n```python\n{body_a}\n```\n\n"
        f"Function B (name redacted):\n```python\n{body_b}\n```\n\n"
        f"{SCORE_RUBRIC}\n\n"
        'Respond with a single JSON object:\n{"score": <0-3>, "reasoning": "<one concise sentence>"}'
    )


def _build_prompt_c(full_a: str, full_b: str) -> str:
    return (
        "Evaluate the overall functional similarity of these two Python functions, "
        "considering both their intent (names, docstrings) and their implementation.\n\n"
        f"Function A:\n```python\n{full_a}\n```\n\n"
        f"Function B:\n```python\n{full_b}\n```\n\n"
        f"{SCORE_RUBRIC}\n\n"
        'Respond with a single JSON object:\n{"score": <0-3>, "reasoning": "<one concise sentence>"}'
    )


async def _call_llm_async(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    prompt: str,
) -> dict:
    """Single async LLM call, bounded by the semaphore."""
    async with sem:
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
    raw = response.choices[0].message.content.strip()
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if not m:
        raise ValueError(f"LLM response did not contain JSON: {raw!r}")
    data = json.loads(m.group(0))
    return {'score': int(data['score']), 'reasoning': data.get('reasoning', '')}


async def judge_pair_async(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    source_a: str,
    source_b: str,
) -> dict:
    """
    Judge one pair at all 3 levels concurrently (a, b, c fire simultaneously).
    Returns a dict with score_a/b/c, reasoning_a/b/c, disconnect fields.
    """
    from llm_similarity_judge import extract_signature, redact_for_body_scoring

    sig_a  = extract_signature(source_a)
    sig_b  = extract_signature(source_b)
    body_a = redact_for_body_scoring(source_a, 'func_A')
    body_b = redact_for_body_scoring(source_b, 'func_B')

    res_a, res_b, res_c = await asyncio.gather(
        _call_llm_async(client, sem, _build_prompt_a(sig_a, sig_b)),
        _call_llm_async(client, sem, _build_prompt_b(body_a, body_b)),
        _call_llm_async(client, sem, _build_prompt_c(source_a, source_b)),
    )

    score_a, score_b = res_a['score'], res_b['score']
    gap  = abs(score_a - score_b)
    disc = gap >= 2
    if disc:
        dtype = 'name_misleads' if score_b > score_a else 'name_overclaims'
        note  = (
            f"Algorithms are similar (body={score_b}) but names suggest different "
            f"purposes (sig={score_a})." if dtype == 'name_misleads' else
            f"Names suggest similar purpose (sig={score_a}) but implementations "
            f"differ (body={score_b})."
        )
    else:
        dtype, note = '', ''

    return {
        'score_a': score_a, 'reasoning_a': res_a['reasoning'],
        'score_b': score_b, 'reasoning_b': res_b['reasoning'],
        'score_c': res_c['score'], 'reasoning_c': res_c['reasoning'],
        'disconnect': disc,
        'disconnect_type': dtype,
        'disconnect_note': note,
    }


async def run_judges(pairs: list[dict], source_cache: dict[int, str],
                     concurrency: int) -> list[dict]:
    """Fire all pair judgements with bounded concurrency."""
    client = AsyncOpenAI(base_url=LLM_ENDPOINT, api_key=LLM_API_KEY)
    sem    = asyncio.Semaphore(concurrency)

    async def _judge_one(i: int, pair: dict) -> dict:
        src_q = source_cache.get(pair['query_id'], '')
        src_n = source_cache.get(pair['neighbour_id'], '')
        if not src_q or not src_n or 'Could not retrieve' in src_q or 'Could not retrieve' in src_n:
            print(f"  [{i+1}/{len(pairs)}] SKIP (no source): "
                  f"{pair['query_function']} → {pair['neighbour_function']}", flush=True)
            pair['judge'] = None
            return pair
        try:
            judge = await judge_pair_async(client, sem, src_q, src_n)
            pair['judge'] = judge
            disc_flag = ' ⚠ DISCONNECT' if judge['disconnect'] else ''
            print(
                f"  [{i+1}/{len(pairs)}] "
                f"{pair['query_function']!r} → {pair['neighbour_function']!r}  "
                f"(sim={pair['cosine_similarity']:.4f}) "
                f"a={judge['score_a']} b={judge['score_b']} c={judge['score_c']}"
                f"{disc_flag}",
                flush=True,
            )
        except Exception as e:
            print(f"  [{i+1}/{len(pairs)}] ERROR: {e}", flush=True)
            pair['judge'] = {'error': str(e)}
        return pair

    tasks   = [_judge_one(i, p) for i, p in enumerate(pairs)]
    results = await asyncio.gather(*tasks)
    return list(results)


# ── JSONL serialisation ─────────────────────────────────────────────────────

def write_jsonl(pairs: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for p in pairs:
            # Don't embed full source in JSONL (too bulky); just store lengths
            out = {k: v for k, v in p.items()
                   if k not in ('query_source', 'neighbour_source')}
            out['query_source_len']     = len(p.get('query_source') or '')
            out['neighbour_source_len'] = len(p.get('neighbour_source') or '')
            f.write(json.dumps(out) + '\n')
    print(f"\nWrote {len(pairs)} lines → {path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k',           type=int, default=5,
                    help='Neighbours per query (default 5)')
    ap.add_argument('--concurrency', type=int, default=50,
                    help='Max concurrent LLM requests (default 50)')
    ap.add_argument('--query-ids',   type=int, nargs='+',
                    default=EXP02_QUERY_IDS,
                    help='Postgres IDs to query (default: Exp 0.2 set)')
    ap.add_argument('--skip-judge',  action='store_true',
                    help='Only run KNN, skip LLM judge (for quick testing)')
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print(f"Experiment 0.2 documentation run")
    print(f"  Query IDs  : {args.query_ids}")
    print(f"  k          : {args.k}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Output     : {OUTPUT_FILE}")
    print(f"{'='*60}\n")

    # ── Step 1: KNN retrieval ──────────────────────────────────────────────
    print("Step 1: KNN retrieval …")
    pairs = run_knn(args.query_ids, args.k)
    print(f"  → {len(pairs)} (query, neighbour) pairs\n")

    # ── Step 2: Source fetch ───────────────────────────────────────────────
    print("Step 2: Fetching function source code …")
    source_cache = fetch_sources(pairs)
    # Attach to pairs for context (not written to JSONL)
    for p in pairs:
        p['query_source']     = source_cache.get(p['query_id'], '')
        p['neighbour_source'] = source_cache.get(p['neighbour_id'], '')
    print(f"  → {len(source_cache)} unique functions fetched\n")

    # ── Step 3: LLM judge ─────────────────────────────────────────────────
    if not args.skip_judge:
        print(f"Step 3: LLM judge ({args.concurrency} concurrent) …")
        pairs = asyncio.run(run_judges(pairs, source_cache, args.concurrency))
        print()
    else:
        print("Step 3: Skipped (--skip-judge)\n")

    # ── Step 4: Write JSONL ────────────────────────────────────────────────
    print("Step 4: Writing JSONL …")
    write_jsonl(pairs, OUTPUT_FILE)

    # ── Summary ────────────────────────────────────────────────────────────
    judged = [p for p in pairs if p.get('judge') and 'error' not in p['judge']]
    if judged:
        print(f"\nSummary of {len(judged)} judged pairs:")
        print(f"  {'Query':<30} {'Neighbour':<30} {'sim':>6}  a  b  c  disc")
        print(f"  {'-'*90}")
        for p in judged:
            j = p['judge']
            disc = '⚠' if j['disconnect'] else ' '
            print(
                f"  {p['query_function']:<30} {p['neighbour_function']:<30} "
                f"{p['cosine_similarity']:>6.4f}  {j['score_a']}  {j['score_b']}  "
                f"{j['score_c']}  {disc}"
            )


if __name__ == '__main__':
    main()
