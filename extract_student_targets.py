"""
Experiment 1.1 data preparation: extract split (signature, body) embeddings
for every function in function_embeddings (model='Qwen2.5-Coder-3B').

Pipeline:
  Phase 1 – Fetch source code for all functions via overlayfs (per-PR mounts,
             ProcessPoolExecutor with spawn, same pattern as extract_static_props.py)
  Phase 2 – Split each source into (sig_text, body_text) using AST
  Phase 3 – GPU batch inference: run frozen teacher on all texts → mean-pool
             layer-18 hidden states → (sig_embedding, body_embedding)
  Phase 4 – Store in function_student_targets postgres table

The teacher is Qwen2.5-Coder-3B at layer 18 (best structural teacher from Exp 0.1/0.2).

Usage:
    python extract_student_targets.py
    python extract_student_targets.py --batch-size 64 --device cuda:1
"""

import argparse
import ast
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import torch
import yaml
import pg8000.native
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase0_1_similarity_matrix import (
    DB,
    OVERLAY_MERGED_BASE,
    _build_repo_id_map,
    _load_tokens,
    _mount_overlay,
    _umount_overlay,
    _sha_available,
    _fetch_sha,
)
from llm_similarity_judge import _extract_named_function

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PG_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'postgres_connection.yaml')
TOKENS_FILE    = os.path.join(os.path.dirname(__file__), 'crawl_tokens.yaml')
REPOS_BASE     = '/shared_workspace_mfs/repos'

TEACHER_MODEL  = 'Qwen2.5-Coder-3B'
TEACHER_PATH   = '/home/original_models/Qwen2.5-Coder-3B'
TEACHER_LAYER  = 18
MAX_TOKENS     = 512
MIN_BODY_CHARS = 5    # skip only truly empty bodies (pass, …)

DDL = """
CREATE TABLE IF NOT EXISTS function_student_targets (
    id               SERIAL  PRIMARY KEY,
    function_id      INTEGER NOT NULL UNIQUE,
    sig_embedding    REAL[]  NOT NULL,
    body_embedding   REAL[]  NOT NULL,
    sig_tokens       INTEGER,
    body_tokens      INTEGER,
    FOREIGN KEY (function_id) REFERENCES function_embeddings(id)
);
CREATE INDEX IF NOT EXISTS fst_func_idx ON function_student_targets (function_id);
"""


# ── Source split helpers ─────────────────────────────────────────────────────

def split_source(src: str) -> tuple[str, str] | None:
    """
    Split a function's source code into (signature_text, body_text).

    signature_text: def line(s) + docstring if present (the 'context')
    body_text:      remaining statements (the 'masked target')

    Returns None if the split is not meaningful (parse error, trivial body).
    """
    src = textwrap.dedent(src)  # class methods are extracted with leading indent
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not funcs:
        return None
    func = funcs[0]  # outermost function
    if not func.body:
        return None

    lines = src.splitlines()

    # Determine split point: after docstring if present, else after def line
    first_stmt = func.body[0]
    if (isinstance(first_stmt, ast.Expr)
            and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)):
        # Has docstring → include it in signature
        split_line = first_stmt.end_lineno  # 1-indexed, inclusive
    else:
        # No docstring → split after the def line(s)
        split_line = first_stmt.lineno - 1  # 1-indexed, exclusive

    sig_text  = '\n'.join(lines[:split_line])
    body_text = '\n'.join(lines[split_line:])

    if len(body_text.strip()) < MIN_BODY_CHARS:
        return None  # trivial body (pass, raise, return None, etc.)

    return sig_text, body_text


# ── Phase 1: per-PR overlayfs source fetch (subprocess worker) ───────────────

def _fetch_pr_sources_worker(
    pr_row: tuple,       # (repo_slug, repo_id, base_commit, instance_id)
    needed: list[tuple], # [(feature_file, feature_function, function_id), ...]
    gh_token: str | None,
) -> dict[int, str]:
    """Subprocess worker: mount one overlayfs, extract all needed function sources."""
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

    tag = f'sttarget-{repo_id}-{instance_id[-6:]}'
    os.makedirs(OVERLAY_MERGED_BASE, exist_ok=True)
    try:
        merged, upper, work = _mount_overlay(repo_dir, tag)
    except Exception:
        return {}

    results: dict[int, str] = {}
    try:
        if not _sha_available(merged, base_commit):
            _fetch_sha(merged, base_commit, repo_slug, gh_token)
        if not _sha_available(merged, base_commit):
            return results

        # Group by file to minimise git-show calls
        by_file: dict[str, list[tuple[str, int]]] = {}
        for (ff, fn, fid) in needed:
            by_file.setdefault(ff, []).append((fn, fid))

        for feature_file, fn_fid_pairs in by_file.items():
            r = subprocess.run(
                ['git', '-C', merged, 'show', f'{base_commit}:{feature_file}'],
                capture_output=True, timeout=30)
            if r.returncode != 0:
                continue
            full_source = r.stdout.decode(errors='replace')
            for (fn, fid) in fn_fid_pairs:
                src = _extract_named_function(full_source, fn)
                if src:
                    results[fid] = src
    finally:
        _umount_overlay(merged, upper, work)

    return results


def fetch_all_sources(function_rows: list) -> dict[int, str]:
    """
    Fetch source code for all functions.
    Returns {function_id: source_str}.
    """
    try:
        tokens = yaml.safe_load(open(TOKENS_FILE)).get('gh_tokens', [])
    except Exception:
        tokens = []
    if not tokens:
        tokens = [None]

    # Load PR metadata
    conn = pg8000.native.Connection(**DB)
    iid_list = list({r[1] for r in function_rows})  # unique instance_ids
    pr_rows_raw = conn.run("""
        SELECT DISTINCT ON (instance_id)
               instance_id, repo, repo_id, base_commit
        FROM prs
        WHERE instance_id = ANY(:iids)
          AND base_commit IS NOT NULL AND base_commit != ''
    """, iids=iid_list)
    conn.close()
    pr_map = {r[0]: (r[1], r[2], r[3], r[0]) for r in pr_rows_raw}

    # Group needed positions by instance_id
    by_iid: dict[str, list[tuple]] = {}
    for (fid, iid, ff, fn) in function_rows:
        if iid in pr_map:
            by_iid.setdefault(iid, []).append((ff, fn, fid))

    print(f"  Fetching from {len(by_iid)} PRs for {len(function_rows)} functions …",
          flush=True)

    source_cache: dict[int, str] = {}
    max_workers = min(len(tokens), 14)
    done = 0
    total = len(by_iid)

    with ProcessPoolExecutor(max_workers=max_workers,
                             mp_context=mp.get_context('spawn')) as executor:
        futures = {
            executor.submit(
                _fetch_pr_sources_worker,
                pr_map[iid],
                by_iid[iid],
                tokens[i % len(tokens)],
            ): iid
            for i, iid in enumerate(by_iid)
        }
        for future in as_completed(futures):
            done += 1
            try:
                source_cache.update(future.result())
            except Exception as e:
                print(f"  WARNING: {futures[future]}: {e}", flush=True)
            if done % 50 == 0 or done == total:
                print(f"  [{done}/{total}] PRs done, {len(source_cache)} sources",
                      flush=True)

    return source_cache


# ── Phase 3: GPU batch inference ─────────────────────────────────────────────

def embed_texts_batched(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    layer: int = TEACHER_LAYER,
    batch_size: int = 32,
) -> tuple[np.ndarray, list[int]]:
    """
    Mean-pool hidden states at `layer` for each text.
    Returns (embeddings, token_counts) where embeddings shape is (N, D).
    """
    all_embs: list[np.ndarray] = []
    token_counts: list[int] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors='pt',
            truncation=True,
            max_length=MAX_TOKENS,
            padding=True,
        ).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs   = out.hidden_states[layer]               # (B, T, D)
        mask = enc['attention_mask'].unsqueeze(-1).float()
        embs = (hs * mask).sum(1) / mask.sum(1)       # (B, D)
        all_embs.append(embs.float().cpu().numpy())
        token_counts.extend(int(enc['attention_mask'][i].sum()) for i in range(len(batch)))

    return np.vstack(all_embs), token_counts


# ── Phase 4: postgres upsert ─────────────────────────────────────────────────

def upsert_targets(rows: list[tuple], conn) -> int:
    """Upsert (function_id, sig_emb, body_emb, sig_tok, body_tok) rows."""
    inserted = 0
    for (fid, sig_emb, body_emb, sig_tok, body_tok) in rows:
        conn.run("""
            INSERT INTO function_student_targets
                (function_id, sig_embedding, body_embedding, sig_tokens, body_tokens)
            VALUES (:fid, :sig, :body, :st, :bt)
            ON CONFLICT (function_id) DO NOTHING
        """,
            fid=fid,
            sig=sig_emb.tolist(),
            body=body_emb.tolist(),
            st=sig_tok,
            bt=body_tok,
        )
        inserted += 1
    return inserted


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--device',     type=str, default='cuda:0')
    ap.add_argument('--limit',      type=int, default=0,
                    help='Process at most this many functions (0=all)')
    args = ap.parse_args()

    conn = pg8000.native.Connection(**DB)

    # Create table
    for stmt in DDL.strip().split(';'):
        stmt = stmt.strip()
        if stmt:
            conn.run(stmt)

    # Load functions not yet processed
    rows = conn.run("""
        SELECT fe.id, fe.instance_id, fe.feature_file, fe.feature_function
        FROM function_embeddings fe
        LEFT JOIN function_student_targets fst ON fst.function_id = fe.id
        WHERE fe.model_name = :m
          AND fst.function_id IS NULL
        ORDER BY fe.id
    """, m=TEACHER_MODEL)

    if args.limit:
        rows = rows[:args.limit]

    print(f"Functions to process: {len(rows):,}", flush=True)
    if not rows:
        print("Nothing to do — all functions already have targets.")
        conn.close()
        return

    # ── Phase 1: source fetch ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 1: Fetching function sources …")
    print(f"{'='*60}")
    source_cache = fetch_all_sources(rows)
    print(f"  → {len(source_cache):,} sources fetched", flush=True)

    # ── Phase 2: split sources ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 2: Splitting sources into signature / body …")
    print(f"{'='*60}")
    splits: list[tuple[int, str, str]] = []   # (function_id, sig_text, body_text)
    skipped_nosrc = skipped_split = 0

    for (fid, iid, ff, fn) in rows:
        src = source_cache.get(fid)
        if not src:
            skipped_nosrc += 1
            continue
        result = split_source(src)
        if result is None:
            skipped_split += 1
            continue
        sig_text, body_text = result
        splits.append((fid, sig_text, body_text))

    print(f"  Splittable: {len(splits):,}  (no_source: {skipped_nosrc:,}  "
          f"unsplittable: {skipped_split:,})", flush=True)

    if not splits:
        print("Nothing to embed.")
        conn.close()
        return

    # ── Phase 3: GPU batch inference ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 3: Teacher inference (layer {TEACHER_LAYER}) on {args.device} …")
    print(f"{'='*60}")

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"  Loading {TEACHER_PATH} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH,
        output_hidden_states=True,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    ).eval()

    fids        = [s[0] for s in splits]
    sig_texts   = [s[1] for s in splits]
    body_texts  = [s[2] for s in splits]

    print(f"  Embedding {len(sig_texts):,} signatures …", flush=True)
    sig_embs, sig_toks = embed_texts_batched(
        model, tokenizer, sig_texts, device, batch_size=args.batch_size)

    print(f"  Embedding {len(body_texts):,} bodies …", flush=True)
    body_embs, body_toks = embed_texts_batched(
        model, tokenizer, body_texts, device, batch_size=args.batch_size)

    # ── Phase 4: store ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 4: Storing targets in postgres …")
    print(f"{'='*60}")

    batch_rows = [
        (fids[i], sig_embs[i], body_embs[i], sig_toks[i], body_toks[i])
        for i in range(len(fids))
    ]
    inserted = upsert_targets(batch_rows, conn)
    conn.close()

    print(f"  → {inserted:,} rows written to function_student_targets", flush=True)

    # Quick sanity check
    conn2 = pg8000.native.Connection(**DB)
    total_stored = conn2.run("SELECT COUNT(*) FROM function_student_targets")[0][0]
    conn2.close()
    print(f"  Total rows in table: {total_stored:,}")

    # Cosine similarity stats between sig and body embeddings
    sig_n  = sig_embs  / (np.linalg.norm(sig_embs,  axis=1, keepdims=True) + 1e-9)
    body_n = body_embs / (np.linalg.norm(body_embs, axis=1, keepdims=True) + 1e-9)
    sims   = (sig_n * body_n).sum(axis=1)
    print(f"\nSig-body cosine similarity: "
          f"mean={sims.mean():.3f}  median={np.median(sims):.3f}  "
          f"p10={np.percentile(sims,10):.3f}  p90={np.percentile(sims,90):.3f}")
    print("(Higher = teacher thinks sig and body are similar; "
          "lower = more informative masking target)")


if __name__ == '__main__':
    main()
