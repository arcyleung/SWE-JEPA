"""
Full-scale Experiment 0.2 evaluation.

Queries ALL 113,600 rows from function_embeddings (4 models × 28,400 functions)
against their respective FAISS indices with k=10, producing up to 1.136M
(query, neighbour) pairs.

Source code is fetched in bulk — one overlayfs mount per PR (~366 mounts total)
rather than per function — then cached for the judge step.

LLM judging is applied to:
  - All cross-repo pairs  (query and neighbour from different GitHub repos)
  - Same-repo pairs with cosine_similarity in [0.80, 0.999)
    (non-trivial matches that aren't exact PR-version copies)

Output:
  docs/phase0_2_knn_results_full.jsonl  — all pairs, judge field populated where run

Usage:
    python run_full_knn_eval.py
    python run_full_knn_eval.py --concurrency 200 --judge-min-sim 0.80
    python run_full_knn_eval.py --skip-judge   # KNN + source only, no LLM
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import yaml
import pg8000.native
import faiss
from openai import AsyncOpenAI

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Config ──────────────────────────────────────────────────────────────────
PG_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'postgres_connection.yaml')
INDEX_DIR      = os.path.join(os.path.dirname(__file__), 'faiss_indices')
OUTPUT_FILE    = os.path.join(os.path.dirname(__file__), 'docs',
                              'phase0_2_knn_results_full.jsonl')
TOKENS_FILE    = os.path.join(os.path.dirname(__file__), 'crawl_tokens.yaml')

LLM_ENDPOINT = 'http://10.10.110.65:24000/v1'
LLM_API_KEY  = 'apikey1234'
LLM_MODEL    = 'qwen3_coder_30b'

_pg_cfg = yaml.safe_load(open(PG_CONFIG_FILE))
DB = dict(
    host=_pg_cfg['ip'],
    port=_pg_cfg.get('port') or 9999,
    user=_pg_cfg['user'],
    password=_pg_cfg['password'],
    database=_pg_cfg['database'],
)

MODEL_TAGS = {
    'Qwen2.5-Coder-3B':          '_Qwen2.5-Coder-3B_L18',
    'Qwen2.5-Coder-3B-Instruct': '_Qwen2.5-Coder-3B-Instruct_L18',
    'Qwen3-8B':                   '_Qwen3-8B_L18',
    'Qwen3-8B-base':              '_Qwen3-8B-base_L18',
}

REPOS_BASE          = '/shared_workspace_mfs/repos'
OVERLAY_MERGED_BASE = '/shared_workspace_mfs/repos_tmp_overlayfs'


def _repo_slug(instance_id: str) -> str:
    """'owner__repo__123' → 'owner/repo'"""
    parts = instance_id.split('__')
    return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else instance_id


# ── Step 1: KNN retrieval (all models) ──────────────────────────────────────

def run_knn_all(k: int) -> list[dict]:
    """
    For each model, load its FAISS index and query all 28,400 functions.
    Returns a flat list of pair dicts (no source or judge yet).
    """
    conn = pg8000.native.Connection(**DB)

    all_pairs: list[dict] = []

    for model_name, tag in MODEL_TAGS.items():
        index_path = os.path.join(INDEX_DIR, f'function_embeddings{tag}.faiss')
        ids_path   = os.path.join(INDEX_DIR, f'function_embeddings{tag}_ids.npy')
        index = faiss.read_index(index_path)
        ids   = np.load(ids_path)
        print(f"\n  [{model_name}] index: {index.ntotal} vectors  dim={index.d}",
              flush=True)

        # Fetch all rows for this model
        rows = conn.run("""
            SELECT id, instance_id, feature_file, feature_function, embedding
            FROM function_embeddings
            WHERE model_name = :m
            ORDER BY id
        """, m=model_name)
        print(f"  [{model_name}] {len(rows)} query functions", flush=True)

        # Build query matrix
        emb_matrix = np.array([r[4] for r in rows], dtype=np.float32)
        norms      = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9
        emb_matrix /= norms

        # Batch FAISS search (all at once)
        sims_mat, pos_mat = index.search(emb_matrix, k + 1)

        # Convert all FAISS positions → postgres IDs, then fetch metadata in bulk
        # (pos_mat contains FAISS index positions; ids[] maps them to postgres IDs)
        valid_mask       = (pos_mat >= 0) & (pos_mat < len(ids))
        all_nbr_pg_ids   = set(int(ids[p]) for p in pos_mat[valid_mask])
        nbr_id_list      = ', '.join(str(i) for i in all_nbr_pg_ids)
        nbr_meta_rows    = conn.run(
            f"SELECT id, instance_id, feature_file, feature_function "
            f"FROM function_embeddings WHERE id IN ({nbr_id_list})"
        ) if nbr_id_list else []
        nbr_meta = {r[0]: r for r in nbr_meta_rows}

        model_pairs = 0
        for row, sims, positions in zip(rows, sims_mat, pos_mat):
            qid, q_iid, q_file, q_func, _ = row
            q_repo = _repo_slug(q_iid)

            # Drop self and re-slice to k
            mask       = ids[positions] != qid
            nbr_pg_ids = ids[positions][mask][:k]
            nbr_sims   = sims[mask][:k]

            for rank, (nid, sim) in enumerate(zip(nbr_pg_ids, nbr_sims), 1):
                nid = int(nid)
                nm  = nbr_meta.get(nid)
                if nm is None:
                    continue
                n_iid, n_file, n_func = nm[1], nm[2], nm[3]
                n_repo = _repo_slug(n_iid)
                cross  = q_repo != n_repo

                all_pairs.append({
                    'query_id':              qid,
                    'query_model':           model_name,
                    'query_function':        q_func,
                    'query_file':            q_file,
                    'query_instance_id':     q_iid,
                    'query_repo':            q_repo,
                    'neighbour_id':          nid,
                    'neighbour_function':    n_func,
                    'neighbour_file':        n_file,
                    'neighbour_instance_id': n_iid,
                    'neighbour_repo':        n_repo,
                    'rank':                  rank,
                    'cosine_similarity':     float(sim),
                    'cross_repo':            cross,
                    'judge':                 None,
                })
            model_pairs += 1

        print(f"  [{model_name}] {model_pairs} queries → "
              f"{len([p for p in all_pairs if p['query_model']==model_name])} pairs",
              flush=True)

    conn.close()
    return all_pairs


# ── Step 2: Bulk source fetching ─────────────────────────────────────────────
#
# Group all needed function positions by instance_id (≡ PR).
# For each PR: one overlayfs mount, fetch all needed files, extract functions.
# Returns a map: (instance_id, feature_file, feature_function) → source_str
#
# Runs in subprocess workers (spawn) so overlayfs mounts are isolated.

def _fetch_pr_sources_worker(
    pr_row: tuple,          # (repo_slug, repo_id, base_commit, instance_id)
    needed: list[tuple],    # [(feature_file, feature_function), ...]
    gh_token: str | None,
) -> dict[tuple, str]:
    """Worker: mount overlayfs for one PR, extract all needed function sources."""
    import shutil
    from phase0_1_similarity_matrix import (
        _mount_overlay, _umount_overlay, _sha_available, _fetch_sha,
    )
    from llm_similarity_judge import _extract_named_function

    repo_slug, repo_id, base_commit, instance_id = pr_row
    if not base_commit:
        return {}

    # Find repo directory
    repo_dir = None
    for entry in os.listdir(REPOS_BASE):
        if entry.startswith(str(repo_id) + '_'):
            repo_dir = os.path.join(REPOS_BASE, entry)
            break
    if not repo_dir:
        return {}

    tag = f'fulleval-{repo_id}-{instance_id[-6:]}'
    os.makedirs(OVERLAY_MERGED_BASE, exist_ok=True)
    try:
        merged, upper, work = _mount_overlay(repo_dir, tag)
    except Exception:
        return {}

    results: dict[tuple, str] = {}
    try:
        if not _sha_available(merged, base_commit):
            _fetch_sha(merged, base_commit, repo_slug, gh_token)
        if not _sha_available(merged, base_commit):
            return results

        # Group needed funcs by file to minimise git-show calls
        by_file: dict[str, list[str]] = {}
        for (feature_file, feature_function) in needed:
            by_file.setdefault(feature_file, []).append(feature_function)

        for feature_file, func_names in by_file.items():
            r = subprocess.run(
                ['git', '-C', merged, 'show', f'{base_commit}:{feature_file}'],
                capture_output=True, timeout=30)
            if r.returncode != 0:
                continue
            full_source = r.stdout.decode(errors='replace')
            for fn in func_names:
                src = _extract_named_function(full_source, fn)
                if src:
                    results[(feature_file, fn)] = src

    finally:
        _umount_overlay(merged, upper, work)

    return results


def fetch_all_sources_bulk(
    needed_positions: set[tuple],   # {(instance_id, feature_file, feature_function)}
) -> dict[tuple, str]:
    """
    Fetch source for all unique (instance_id, feature_file, feature_function) tuples.
    Groups by instance_id → one overlayfs mount per PR.
    Returns {(instance_id, feature_file, feature_function): source_str}.
    """
    # Load tokens
    try:
        tokens = yaml.safe_load(open(TOKENS_FILE)).get('gh_tokens', [])
    except Exception:
        tokens = []
    if not tokens:
        tokens = [None]

    # Group needed positions by instance_id
    by_iid: dict[str, list[tuple]] = {}
    for (iid, ff, fn) in needed_positions:
        by_iid.setdefault(iid, []).append((ff, fn))

    # Fetch PR metadata for all instance_ids
    conn = pg8000.native.Connection(**DB)
    iid_list = list(by_iid.keys())
    pr_rows_raw = conn.run("""
        SELECT DISTINCT ON (instance_id)
               instance_id, repo, repo_id, base_commit
        FROM prs
        WHERE instance_id = ANY(:iids)
          AND base_commit IS NOT NULL AND base_commit != ''
    """, iids=iid_list)
    conn.close()

    pr_map = {r[0]: (r[1], r[2], r[3], r[0]) for r in pr_rows_raw}
    print(f"  {len(pr_map)} PRs to mount for source fetch "
          f"({len(needed_positions)} function positions)", flush=True)

    source_cache: dict[tuple, str] = {}
    max_workers = min(len(tokens), 14)
    completed = 0
    total = len(pr_map)

    with ProcessPoolExecutor(max_workers=max_workers,
                             mp_context=mp.get_context('spawn')) as executor:
        futures = {
            executor.submit(
                _fetch_pr_sources_worker,
                pr_map[iid],
                by_iid[iid],
                tokens[i % len(tokens)],
            ): iid
            for i, iid in enumerate(pr_map)
        }
        for future in as_completed(futures):
            completed += 1
            iid = futures[future]
            try:
                partial = future.result()
                for (ff, fn), src in partial.items():
                    source_cache[(iid, ff, fn)] = src
            except Exception as e:
                print(f"  WARNING pr {iid}: {e}", flush=True)
            if completed % 50 == 0 or completed == total:
                print(f"  [{completed}/{total}] PRs done, "
                      f"{len(source_cache)} sources cached", flush=True)

    return source_cache


# ── Step 3: Async LLM judge ──────────────────────────────────────────────────

SCORE_RUBRIC = """\
Score scale:
  3 – functionally equivalent / same algorithm
  2 – moderately similar (same purpose or similar approach, notable differences)
  1 – weakly related (same broad domain or category)
  0 – unrelated"""

_SYSTEM = "You are a senior software engineer evaluating Python function similarity."


async def _llm_call(client: AsyncOpenAI, sem: asyncio.Semaphore, prompt: str) -> dict:
    async with sem:
        resp = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user",   "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
    raw = resp.choices[0].message.content.strip()
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in: {raw!r}")
    d = json.loads(m.group(0))
    return {'score': int(d['score']), 'reasoning': d.get('reasoning', '')}


def _prompt_a(sa, sb):
    return (
        "Evaluate the functional similarity based ONLY on names, parameters, "
        "return types, and docstrings — do NOT infer implementations.\n\n"
        f"Function A (signature):\n```python\n{sa}\n```\n\n"
        f"Function B (signature):\n```python\n{sb}\n```\n\n"
        f"{SCORE_RUBRIC}\n\n"
        'Respond with a single JSON object: {"score": <0-3>, "reasoning": "<one sentence>"}'
    )


def _prompt_b(ba, bb):
    return (
        "Evaluate the algorithmic similarity. Names are redacted — judge on "
        "what the code actually does.\n\n"
        f"Function A (redacted):\n```python\n{ba}\n```\n\n"
        f"Function B (redacted):\n```python\n{bb}\n```\n\n"
        f"{SCORE_RUBRIC}\n\n"
        'Respond with a single JSON object: {"score": <0-3>, "reasoning": "<one sentence>"}'
    )


def _prompt_c(fa, fb):
    return (
        "Evaluate overall functional similarity (intent and implementation).\n\n"
        f"Function A:\n```python\n{fa}\n```\n\n"
        f"Function B:\n```python\n{fb}\n```\n\n"
        f"{SCORE_RUBRIC}\n\n"
        'Respond with a single JSON object: {"score": <0-3>, "reasoning": "<one sentence>"}'
    )


async def judge_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    src_a: str,
    src_b: str,
) -> dict:
    from llm_similarity_judge import extract_signature, redact_for_body_scoring
    sig_a  = extract_signature(src_a)
    sig_b  = extract_signature(src_b)
    body_a = redact_for_body_scoring(src_a, 'func_A')
    body_b = redact_for_body_scoring(src_b, 'func_B')

    ra, rb, rc = await asyncio.gather(
        _llm_call(client, sem, _prompt_a(sig_a, sig_b)),
        _llm_call(client, sem, _prompt_b(body_a, body_b)),
        _llm_call(client, sem, _prompt_c(src_a, src_b)),
    )
    sa, sb = ra['score'], rb['score']
    disc = abs(sa - sb) >= 2
    if disc:
        dtype = 'name_misleads' if sb > sa else 'name_overclaims'
        note  = (f"body={sb} vs sig={sa}" if dtype == 'name_misleads'
                 else f"sig={sa} vs body={sb}")
    else:
        dtype = note = ''
    return {
        'score_a': sa, 'reasoning_a': ra['reasoning'],
        'score_b': sb, 'reasoning_b': rb['reasoning'],
        'score_c': rc['score'], 'reasoning_c': rc['reasoning'],
        'disconnect': disc, 'disconnect_type': dtype, 'disconnect_note': note,
    }


async def run_judges_async(
    pairs_to_judge: list[dict],
    source_cache: dict[tuple, str],
    concurrency: int,
    progress_every: int = 500,
) -> None:
    """Judge all pairs in-place (mutates pair['judge'])."""
    client = AsyncOpenAI(base_url=LLM_ENDPOINT, api_key=LLM_API_KEY)
    sem    = asyncio.Semaphore(concurrency)
    done   = 0
    lock   = asyncio.Lock()

    async def _one(pair: dict):
        nonlocal done
        src_a = source_cache.get((pair['query_instance_id'],
                                  pair['query_file'], pair['query_function']), '')
        src_b = source_cache.get((pair['neighbour_instance_id'],
                                  pair['neighbour_file'], pair['neighbour_function']), '')
        if not src_a or not src_b:
            pair['judge'] = {'skipped': 'no_source'}
        else:
            try:
                pair['judge'] = await judge_one(client, sem, src_a, src_b)
            except Exception as e:
                pair['judge'] = {'error': str(e)}
        async with lock:
            done += 1
            if done % progress_every == 0 or done == len(pairs_to_judge):
                print(f"  [{done}/{len(pairs_to_judge)}] judged", flush=True)

    await asyncio.gather(*[_one(p) for p in pairs_to_judge])


# ── JSONL helpers ────────────────────────────────────────────────────────────

def write_jsonl(pairs: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for p in pairs:
            f.write(json.dumps(p) + '\n')
    print(f"Wrote {len(pairs):,} lines → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k',             type=int, default=10)
    ap.add_argument('--concurrency',   type=int, default=200)
    ap.add_argument('--judge-min-sim', type=float, default=0.80,
                    help='Min cosine sim for same-repo pairs to be judged (default 0.80)')
    ap.add_argument('--skip-judge',    action='store_true')
    ap.add_argument('--skip-knn',      action='store_true',
                    help='Load existing JSONL and only run judge (resume mode)')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # ── Step 1: KNN ─────────────────────────────────────────────────────────
    if args.skip_knn and os.path.exists(OUTPUT_FILE):
        print("Step 1: Loading existing JSONL …")
        with open(OUTPUT_FILE) as f:
            pairs = [json.loads(l) for l in f]
        print(f"  Loaded {len(pairs):,} pairs")
    else:
        print(f"\n{'='*60}")
        print(f"Step 1: KNN retrieval (all models, k={args.k}) …")
        print(f"{'='*60}")
        pairs = run_knn_all(args.k)
        n_cross = sum(1 for p in pairs if p['cross_repo'])
        print(f"\n  Total pairs : {len(pairs):,}")
        print(f"  Cross-repo  : {n_cross:,}  ({100*n_cross/len(pairs):.1f}%)")

    if args.skip_judge:
        write_jsonl(pairs, OUTPUT_FILE)
        return

    # ── Step 2: Source fetch ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 2: Bulk source fetch (one overlayfs mount per PR) …")
    print(f"{'='*60}")

    # Determine which pairs need judging to scope the source fetch
    to_judge = [
        p for p in pairs
        if (p['cross_repo'] and p['cosine_similarity'] >= args.judge_min_sim)
        or (not p['cross_repo'] and args.judge_min_sim <= p['cosine_similarity'] < 0.999)
    ]
    print(f"  Pairs selected for judging: {len(to_judge):,} "
          f"(cross-repo≥{args.judge_min_sim} + same-repo [{args.judge_min_sim}, 0.999))")

    # Collect all needed (instance_id, file, func) positions
    needed: set[tuple] = set()
    for p in to_judge:
        needed.add((p['query_instance_id'],     p['query_file'],    p['query_function']))
        needed.add((p['neighbour_instance_id'], p['neighbour_file'],p['neighbour_function']))

    source_cache = fetch_all_sources_bulk(needed)
    hit = sum(1 for k in needed if k in source_cache)
    print(f"  Source cache: {hit:,}/{len(needed):,} positions resolved")

    # ── Step 3: LLM judge ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 3: LLM judge ({args.concurrency} concurrent) …")
    print(f"{'='*60}")
    asyncio.run(run_judges_async(to_judge, source_cache, args.concurrency))

    # Merge judge results back into main pairs list
    judge_map = {id(p): p['judge'] for p in to_judge}
    for p in pairs:
        if id(p) in judge_map:
            p['judge'] = judge_map[id(p)]

    # ── Step 4: Write JSONL ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 4: Writing JSONL …")
    write_jsonl(pairs, OUTPUT_FILE)

    # ── Summary ───────────────────────────────────────────────────────────────
    judged = [p for p in pairs
              if p.get('judge') and 'error' not in p['judge']
              and 'skipped' not in p['judge']]
    disconnects = [p for p in judged if p['judge'].get('disconnect')]
    cross_judged = [p for p in judged if p['cross_repo']]

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Total pairs      : {len(pairs):,}")
    print(f"  Cross-repo pairs : {sum(1 for p in pairs if p['cross_repo']):,}")
    print(f"  Judged pairs     : {len(judged):,}")
    print(f"    Cross-repo     : {len(cross_judged):,}")
    print(f"    Disconnects    : {len(disconnects):,}")

    if cross_judged:
        # Score distributions
        for level, key in [('a (signature)', 'score_a'),
                            ('b (body)',      'score_b'),
                            ('c (overall)',   'score_c')]:
            scores = [p['judge'][key] for p in cross_judged]
            avg = sum(scores) / len(scores)
            print(f"  Cross-repo score {level}: avg={avg:.2f}  "
                  f"[{scores.count(0)}×0  {scores.count(1)}×1  "
                  f"{scores.count(2)}×2  {scores.count(3)}×3]")

    if disconnects:
        print(f"\n  Top disconnects (name_misleads — same algo, different name):")
        ml = [p for p in disconnects if p['judge']['disconnect_type']=='name_misleads']
        for p in sorted(ml, key=lambda x: x['cosine_similarity'], reverse=True)[:10]:
            j = p['judge']
            print(f"    sim={p['cosine_similarity']:.4f}  "
                  f"{p['query_function']!r} ({p['query_repo']}) "
                  f"↔ {p['neighbour_function']!r} ({p['neighbour_repo']})  "
                  f"a={j['score_a']} b={j['score_b']}")


if __name__ == '__main__':
    main()
