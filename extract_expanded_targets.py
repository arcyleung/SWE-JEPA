"""
Expand function_student_targets with a stratified sample from followups_function.

Source:  followups_function table (144 repos, ~62 k functions).
Target:  ~50 k new rows in function_embeddings + function_student_targets
         (sig_embedding, body_embedding, sig_text, body_text all written in one pass).

Pipeline:
  Phase 1 – Stratified sampling from followups_function by feature_repo
  Phase 2 – Fetch sources via overlayfs (CPU parallel, per PR)
  Phase 3 – 8-way GPU batch inference (one Qwen2.5-Coder-3B per GPU)
  Phase 4 – Store results in postgres

Usage:
    python extract_expanded_targets.py
    python extract_expanded_targets.py --target 50000 --batch-size 64
"""

import argparse
import ast
import os
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import random

import numpy as np
import yaml
import pg8000.native
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase0_1_similarity_matrix import (
    DB,
    OVERLAY_MERGED_BASE,
    _mount_overlay,
    _umount_overlay,
    _sha_available,
    _fetch_sha,
)
from llm_similarity_judge import _extract_named_function

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEACHER_PATH       = '/home/original_models/Qwen2.5-Coder-3B'
TEACHER_MODEL_NAME = 'Qwen2.5-Coder-3B'
TEACHER_LAYER      = 18
MAX_TOKENS         = 512
MIN_BODY_CHARS     = 5
REPOS_BASE         = '/shared_workspace_mfs/repos'
TOKENS_FILE        = os.path.join(os.path.dirname(__file__), 'crawl_tokens.yaml')


# ── Source splitting ──────────────────────────────────────────────────────────

def split_source(src: str) -> tuple[str, str] | None:
    src = textwrap.dedent(src)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not funcs:
        return None
    func = funcs[0]
    if not func.body:
        return None
    lines = src.splitlines()
    first_stmt = func.body[0]
    if (isinstance(first_stmt, ast.Expr)
            and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)):
        split_line = first_stmt.end_lineno
    else:
        split_line = first_stmt.lineno - 1
    sig_text  = '\n'.join(lines[:split_line])
    body_text = '\n'.join(lines[split_line:])
    if len(body_text.strip()) < MIN_BODY_CHARS:
        return None
    return sig_text, body_text


# ── Phase 2: per-PR overlayfs fetch (subprocess worker) ──────────────────────

def _fetch_pr_worker(pr_row, needed, gh_token):
    """
    pr_row: (repo_slug, repo_id, base_commit, instance_id)
    needed: [(feature_file, feature_function, row_key), ...]
    Returns: {row_key: (sig_text, body_text)}
    """
    repo_slug, repo_id, base_commit, instance_id = pr_row
    if not base_commit:
        return {}

    repo_dir = None
    for entry in os.listdir(REPOS_BASE):
        if entry.startswith(str(repo_id) + '_'):
            repo_dir = os.path.join(REPOS_BASE, entry)
            break
    if not repo_dir:
        return {}

    tag = f'expanded-{repo_id}-{instance_id[-6:]}'
    os.makedirs(OVERLAY_MERGED_BASE, exist_ok=True)
    try:
        merged, upper, work = _mount_overlay(repo_dir, tag)
    except Exception:
        return {}

    results = {}
    try:
        if not _sha_available(merged, base_commit):
            _fetch_sha(merged, base_commit, repo_slug, gh_token)
        if not _sha_available(merged, base_commit):
            return results

        by_file = {}
        for (ff, fn, rk) in needed:
            by_file.setdefault(ff, []).append((fn, rk))

        for feature_file, fn_rk_pairs in by_file.items():
            r = subprocess.run(
                ['git', '-C', merged, 'show', f'{base_commit}:{feature_file}'],
                capture_output=True, timeout=30)
            if r.returncode != 0:
                continue
            full_source = r.stdout.decode(errors='replace')
            for (fn, rk) in fn_rk_pairs:
                src = _extract_named_function(full_source, fn)
                if src:
                    split = split_source(src)
                    if split is not None:
                        results[rk] = split
    finally:
        _umount_overlay(merged, upper, work)

    return results


def fetch_all_sources(sampled_rows: list, tokens: list) -> dict:
    """
    sampled_rows: [(row_key, feature_instance_id, feature_file, feature_function), ...]
    Returns: {row_key: (sig_text, body_text)}
    """
    if not tokens:
        tokens = [None]

    conn = pg8000.native.Connection(**DB)
    iid_list = list({r[1] for r in sampled_rows})
    pr_rows_raw = conn.run("""
        SELECT DISTINCT ON (instance_id) instance_id, repo, repo_id, base_commit
        FROM prs
        WHERE instance_id = ANY(:iids)
          AND base_commit IS NOT NULL AND base_commit != ''
    """, iids=iid_list)
    conn.close()
    pr_map = {r[0]: (r[1], r[2], r[3], r[0]) for r in pr_rows_raw}

    by_iid = {}
    for (rk, iid, ff, fn) in sampled_rows:
        if iid in pr_map:
            by_iid.setdefault(iid, []).append((ff, fn, rk))

    print(f"  Fetching from {len(by_iid):,} PRs for {len(sampled_rows):,} functions …",
          flush=True)

    source_cache = {}
    max_workers  = min(len(tokens), 14)
    done = 0
    total = len(by_iid)

    with ProcessPoolExecutor(max_workers=max_workers,
                             mp_context=mp.get_context('spawn')) as executor:
        futures = {
            executor.submit(_fetch_pr_worker, pr_map[iid], by_iid[iid],
                            tokens[i % len(tokens)]): iid
            for i, iid in enumerate(by_iid)
        }
        for future in as_completed(futures):
            done += 1
            try:
                source_cache.update(future.result())
            except Exception as e:
                print(f"  WARNING {futures[future]}: {e}", flush=True)
            if done % 100 == 0 or done == total:
                print(f"  [{done}/{total}] PRs done, {len(source_cache):,} splits",
                      flush=True)

    return source_cache


# ── Phase 3: GPU batch inference (per-process worker) ────────────────────────

def _embed_worker(rank: int, work_chunk: list, batch_size: int) -> list:
    """
    rank:       CUDA device index
    work_chunk: [(row_key, sig_text, body_text), ...]
    Returns:    [(row_key, sig_emb_list, body_emb_list, sig_tok, body_tok), ...]
    """
    device = f'cuda:{rank}'
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH,
        output_hidden_states=True,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    ).eval()

    row_keys   = [item[0] for item in work_chunk]
    sig_texts  = [item[1] for item in work_chunk]
    body_texts = [item[2] for item in work_chunk]

    def embed_batch(texts):
        all_embs, all_toks = [], []
        for start in range(0, len(texts), batch_size):
            batch = texts[start: start + batch_size]
            enc = tokenizer(
                batch, return_tensors='pt', truncation=True,
                max_length=MAX_TOKENS, padding=True,
            ).to(device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            hs   = out.hidden_states[TEACHER_LAYER]          # (B, T, D)
            mask = enc['attention_mask'].unsqueeze(-1).float()
            embs = (hs * mask).sum(1) / mask.sum(1)          # (B, D)
            all_embs.append(embs.float().cpu().numpy())
            all_toks.extend(int(enc['attention_mask'][i].sum())
                            for i in range(len(batch)))
        return np.vstack(all_embs), all_toks

    print(f"  [GPU {rank}] embedding {len(sig_texts):,} sig+body pairs …", flush=True)
    sig_embs,  sig_toks  = embed_batch(sig_texts)
    body_embs, body_toks = embed_batch(body_texts)

    return [
        (row_keys[i],
         sig_embs[i].tolist(), body_embs[i].tolist(),
         sig_toks[i], body_toks[i])
        for i in range(len(row_keys))
    ]


def run_gpu_inference(splits: list, n_gpus: int, batch_size: int) -> list:
    """
    splits:  [(row_key, sig_text, body_text), ...]
    Returns: [(row_key, sig_emb_list, body_emb_list, sig_tok, body_tok), ...]
    """
    # Distribute work evenly across GPUs
    chunks = [[] for _ in range(n_gpus)]
    for i, item in enumerate(splits):
        chunks[i % n_gpus].append(item)

    args = [(rank, chunk, batch_size) for rank, chunk in enumerate(chunks) if chunk]
    results = []

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=n_gpus, mp_context=ctx) as executor:
        futures = {executor.submit(_embed_worker, *a): a[0] for a in args}
        for future in as_completed(futures):
            rank = futures[future]
            try:
                results.extend(future.result())
            except Exception as e:
                print(f"  ERROR GPU {rank}: {e}", flush=True)

    return results


# ── Phase 1: stratified sampling ─────────────────────────────────────────────

def stratified_sample(conn, target: int, seed: int = 42) -> list:
    """
    Returns a list of (row_key, feature_instance_id, feature_file, feature_function)
    drawn from followups_function, stratified by feature_repo.

    row_key = followups_function.id
    """
    # Get all rows not yet in function_student_targets
    # (Check: if feature_instance_id+file+function already in function_embeddings AND
    #          function_student_targets, skip it)
    all_rows = conn.run("""
        SELECT ff.id,
               ff.feature_instance_id,
               ff.feature_repo,
               ff.feature_file,
               ff.feature_function
        FROM followups_function ff
        WHERE NOT EXISTS (
            SELECT 1
            FROM function_embeddings fe
            JOIN function_student_targets fst ON fst.function_id = fe.id
            WHERE fe.instance_id    = ff.feature_instance_id
              AND fe.feature_file   = ff.feature_file
              AND fe.feature_function = ff.feature_function
              AND fe.model_name     = 'Qwen2.5-Coder-3B'
        )
        ORDER BY ff.id
    """)
    print(f"  Available (not yet extracted): {len(all_rows):,}", flush=True)

    # Group by repo
    by_repo: dict[str, list] = {}
    for (rk, iid, repo, ff, fn) in all_rows:
        by_repo.setdefault(repo, []).append((rk, iid, ff, fn))

    # Binary search for per-repo cap to hit ~target total
    counts = [len(v) for v in by_repo.values()]
    total_available = sum(counts)
    if total_available <= target:
        cap = max(counts)   # take everything
    else:
        lo, hi = 1, max(counts)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if sum(min(c, mid) for c in counts) <= target:
                lo = mid
            else:
                hi = mid - 1
        cap = lo

    total_after_cap = sum(min(c, cap) for c in counts)
    print(f"  Per-repo cap: {cap}  →  {total_after_cap:,} functions "
          f"from {len(by_repo)} repos", flush=True)

    rng = random.Random(seed)
    sampled = []
    for repo, rows in sorted(by_repo.items()):
        rng.shuffle(rows)
        for (rk, iid, ff, fn) in rows[:cap]:
            sampled.append((rk, iid, ff, fn))

    rng.shuffle(sampled)
    return sampled


# ── Phase 4: DB store ─────────────────────────────────────────────────────────

def store_results(results: list, splits_map: dict, conn) -> tuple[int, int]:
    """
    results:   [(row_key, sig_emb_list, body_emb_list, sig_tok, body_tok), ...]
    splits_map: {row_key: (sig_text, body_text)}

    Inserts into function_embeddings (with NULL full-embedding — only needed for FK)
    then into function_student_targets with sig_text + body_text.

    Returns (inserted_fe, inserted_fst).
    """
    # Build lookup from row_key → (feature_instance_id, feature_file, feature_function)
    # We stored this in the sampled rows — rebuild from a query
    rk_set = [r[0] for r in results]

    conn2 = pg8000.native.Connection(**DB)
    meta_rows = conn2.run("""
        SELECT id, feature_instance_id, feature_file, feature_function
        FROM followups_function
        WHERE id = ANY(:ids)
    """, ids=rk_set)
    conn2.close()
    meta = {r[0]: (r[1], r[2], r[3]) for r in meta_rows}

    inserted_fe = inserted_fst = 0

    for (rk, sig_emb_list, body_emb_list, sig_tok, body_tok) in results:
        if rk not in meta or rk not in splits_map:
            continue
        iid, ff, fn = meta[rk]
        sig_text, body_text = splits_map[rk]

        # 1. Insert into function_embeddings; get id (upsert trick: update noop)
        #    embedding = body_emb (non-null placeholder; fe.embedding not used in training)
        fe_rows = conn.run("""
            INSERT INTO function_embeddings
                (instance_id, feature_file, feature_function,
                 model_name, layer_index, embedding)
            VALUES (:iid, :ff, :fn, :mn, :li, :emb)
            ON CONFLICT (instance_id, feature_file, feature_function,
                         model_name, layer_index)
            DO UPDATE SET instance_id = EXCLUDED.instance_id
            RETURNING id
        """, iid=iid, ff=ff, fn=fn, mn=TEACHER_MODEL_NAME, li=TEACHER_LAYER,
             emb=body_emb_list)
        if not fe_rows:
            continue
        function_id = fe_rows[0][0]
        inserted_fe += 1

        # 2. Insert into function_student_targets (sig_text + body_text included)
        conn.run("""
            INSERT INTO function_student_targets
                (function_id, sig_embedding, body_embedding,
                 sig_tokens, body_tokens, sig_text, body_text)
            VALUES (:fid, :sig, :body, :st, :bt, :stext, :btext)
            ON CONFLICT (function_id) DO NOTHING
        """,
            fid=function_id,
            sig=sig_emb_list,
            body=body_emb_list,
            st=sig_tok,
            bt=body_tok,
            stext=sig_text,
            btext=body_text,
        )
        inserted_fst += 1

        if inserted_fst % 2000 == 0:
            print(f"  {inserted_fst:,} rows written …", flush=True)

    return inserted_fe, inserted_fst


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target',     type=int, default=50_000,
                    help='Target total functions (default: 50000)')
    ap.add_argument('--gpus',       type=int, default=8,
                    help='Number of GPUs to use (default: 8)')
    ap.add_argument('--batch-size', type=int, default=64,
                    help='Inference batch size per GPU (default: 64)')
    ap.add_argument('--seed',       type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    print(f"SWE-JEPA Expanded Extraction")
    print(f"{'='*60}")
    print(f"  Target: {args.target:,} functions  |  GPUs: {args.gpus}  "
          f"|  Batch size: {args.batch_size}", flush=True)

    # Load GH tokens
    try:
        tokens = yaml.safe_load(open(TOKENS_FILE)).get('gh_tokens', [])
    except Exception:
        tokens = []

    conn = pg8000.native.Connection(**DB)

    # Ensure sig_text/body_text columns exist
    for col in ('sig_text TEXT', 'body_text TEXT'):
        try:
            conn.run(f'ALTER TABLE function_student_targets '
                     f'ADD COLUMN IF NOT EXISTS {col}')
        except Exception:
            pass

    # ── Phase 1: stratified sample ───────────────────────────────────────────
    print(f"\nPhase 1: Stratified sampling from followups_function …", flush=True)
    sampled = stratified_sample(conn, target=args.target, seed=args.seed)
    print(f"  Sampled: {len(sampled):,} functions", flush=True)
    if not sampled:
        print("Nothing to do.")
        conn.close()
        return

    # ── Phase 2: fetch sources ───────────────────────────────────────────────
    print(f"\nPhase 2: Fetching sources via overlayfs …", flush=True)
    source_cache = fetch_all_sources(sampled, tokens)
    print(f"  → {len(source_cache):,} valid (sig, body) splits", flush=True)

    if not source_cache:
        print("No sources found.")
        conn.close()
        return

    # Build work list for GPU
    rk_to_meta = {rk: (iid, ff, fn) for (rk, iid, ff, fn) in sampled}
    splits_list = [
        (rk, sig_text, body_text)
        for rk, (sig_text, body_text) in source_cache.items()
    ]

    # ── Phase 3: GPU inference ───────────────────────────────────────────────
    print(f"\nPhase 3: Teacher inference on {args.gpus} GPUs …", flush=True)
    embed_results = run_gpu_inference(splits_list, args.gpus, args.batch_size)
    print(f"  → {len(embed_results):,} embeddings computed", flush=True)

    # ── Phase 4: store ───────────────────────────────────────────────────────
    print(f"\nPhase 4: Storing in postgres …", flush=True)
    inserted_fe, inserted_fst = store_results(embed_results, source_cache, conn)
    conn.close()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} min:")
    print(f"  function_embeddings rows inserted/found: {inserted_fe:,}")
    print(f"  function_student_targets rows inserted:  {inserted_fst:,}", flush=True)

    # Report total corpus size
    conn2 = pg8000.native.Connection(**DB)
    total = conn2.run("""
        SELECT COUNT(*), COUNT(DISTINCT p.repo)
        FROM function_student_targets fst
        JOIN function_embeddings fe ON fe.id = fst.function_id
          AND fe.model_name = 'Qwen2.5-Coder-3B'
        JOIN prs p ON p.instance_id = fe.instance_id
        WHERE fst.sig_text IS NOT NULL
    """)[0]
    conn2.close()
    print(f"\n  Total corpus: {total[0]:,} functions from {total[1]} repos "
          f"(with sig_text populated)", flush=True)


if __name__ == '__main__':
    main()
