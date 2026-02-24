"""
Build an expanded function_student_targets dataset by walking the HEAD state
of each target repo directly.

This avoids the stale-file-path problem in followups_function (many files have
been renamed/deleted since the shallow clones were made) and the dotted-name
matching issue with _extract_named_function.

Source:  followups_function → 144 distinct repos → walk HEAD Python files
Target:  ~50 k rows in function_embeddings + function_student_targets
         (sig_embedding, body_embedding, sig_text, body_text all in one pass)

Pipeline:
  Phase 1 – Enumerate target repos + get a representative instance_id per repo
  Phase 2 – Walk Python files at HEAD for each repo, extract + split functions
             (CPU, per-repo subprocess)
  Phase 3 – GPU batch inference on a single GPU (batch_size=32, safe with
             other processes using ~40-50 GB/card)
  Phase 4 – Store results in postgres

Usage:
    python extract_from_heads.py
    python extract_from_heads.py --target 50000 --device cuda:0 --batch-size 32
"""

import argparse
import ast
import os
import subprocess
import sys
import textwrap
import time
import random

import numpy as np
import pg8000.native
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase0_1_similarity_matrix import DB

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEACHER_PATH       = '/home/original_models/Qwen2.5-Coder-3B'
TEACHER_MODEL_NAME = 'Qwen2.5-Coder-3B'
TEACHER_LAYER      = 18
MAX_TOKENS         = 512
MIN_BODY_CHARS     = 5
REPOS_BASE         = '/shared_workspace_mfs/repos'


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
    if len(sig_text.strip()) < 5:   # single-line functions → empty sig
        return None
    return sig_text, body_text


# ── Phase 2: walk one repo's HEAD Python files ────────────────────────────────

def extract_repo_functions(repo_dir: str, instance_id: str, max_per_repo: int,
                           rng: random.Random) -> list[tuple]:
    """
    Walk all .py files at HEAD of repo_dir, extract functions with non-trivial
    bodies.  Reservoir-sample up to max_per_repo results.

    Returns list of (instance_id, feature_file, feature_function, sig_text, body_text).
    """
    # Get HEAD SHA to make sure we have a valid state
    r = subprocess.run(['git', '-C', repo_dir, 'rev-parse', 'HEAD'],
                       capture_output=True, timeout=10)
    if r.returncode != 0:
        return []
    head_sha = r.stdout.decode().strip()

    # List all Python files tracked at HEAD
    r2 = subprocess.run(
        ['git', '-C', repo_dir, 'ls-tree', '-r', '--name-only', 'HEAD'],
        capture_output=True, timeout=30)
    if r2.returncode != 0:
        return []
    all_files = [f for f in r2.stdout.decode().splitlines() if f.endswith('.py')]

    # Shuffle files so we don't always take functions from the same files
    rng.shuffle(all_files)

    results = []
    seen_keys = set()   # (feature_file, feature_function) — ensure uniqueness

    for py_file in all_files:
        if len(results) >= max_per_repo * 3:   # over-sample then reservoir-sample
            break

        r3 = subprocess.run(
            ['git', '-C', repo_dir, 'show', f'HEAD:{py_file}'],
            capture_output=True, timeout=15)
        if r3.returncode != 0:
            continue

        source = r3.stdout.decode(errors='replace')
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        lines = source.splitlines()

        # Extract top-level functions and class methods
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                src = '\n'.join(lines[node.lineno - 1: node.end_lineno])
                split = split_source(src)
                if split:
                    key = (py_file, func_name)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        results.append((instance_id, py_file, func_name,
                                        split[0], split[1]))

            elif isinstance(node, ast.ClassDef):
                for item in ast.iter_child_nodes(node):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        func_name = f"{node.name}.{item.name}"
                        src = '\n'.join(lines[item.lineno - 1: item.end_lineno])
                        split = split_source(src)
                        if split:
                            key = (py_file, func_name)
                            if key not in seen_keys:
                                seen_keys.add(key)
                                results.append((instance_id, py_file, func_name,
                                                split[0], split[1]))

    # Reservoir sample to max_per_repo
    rng.shuffle(results)
    return results[:max_per_repo]


# ── Phase 3: GPU batch inference ─────────────────────────────────────────────

def embed_texts_batched(model, tokenizer, texts, device, batch_size=32):
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target',     type=int,   default=50_000)
    ap.add_argument('--device',     type=str,   default='cuda:0')
    ap.add_argument('--batch-size', type=int,   default=32)
    ap.add_argument('--seed',       type=int,   default=42)
    args = ap.parse_args()

    t0 = time.time()
    print(f"SWE-JEPA Head-State Extraction")
    print(f"{'='*60}")
    print(f"  Target: {args.target:,}  |  Device: {args.device}  "
          f"|  Batch: {args.batch_size}", flush=True)

    conn = pg8000.native.Connection(**DB)

    # Ensure sig_text/body_text columns exist
    for col in ('sig_text TEXT', 'body_text TEXT'):
        try:
            conn.run(f'ALTER TABLE function_student_targets '
                     f'ADD COLUMN IF NOT EXISTS {col}')
        except Exception:
            pass

    # ── Phase 1: enumerate target repos ──────────────────────────────────────
    print(f"\nPhase 1: Enumerating target repos …", flush=True)

    # Get distinct repos from followups_function with a valid instance_id
    # Use a CTE to get MIN(id) per repo first — much faster than correlated subqueries
    repo_rows = conn.run("""
        WITH repo_prs AS (
            SELECT repo, repo_id, instance_id,
                   ROW_NUMBER() OVER (PARTITION BY repo ORDER BY id) AS rn
            FROM prs
            WHERE base_commit IS NOT NULL AND base_commit != ''
        ),
        first_pr AS (
            SELECT repo, repo_id, instance_id FROM repo_prs WHERE rn = 1
        ),
        target_repos AS (
            SELECT DISTINCT feature_repo FROM followups_function
        )
        SELECT t.feature_repo, fp.instance_id, fp.repo_id
        FROM target_repos t
        JOIN first_pr fp ON fp.repo = t.feature_repo
        ORDER BY t.feature_repo
    """)

    # Filter to repos with a valid instance_id and local directory
    repo_dirs = {}
    for entry in os.listdir(REPOS_BASE):
        parts = entry.split('_', 1)
        if parts[0].isdigit():
            repo_dirs[int(parts[0])] = os.path.join(REPOS_BASE, entry)

    valid_repos = []
    for feature_repo, instance_id, repo_id in repo_rows:
        if not instance_id or not repo_id:
            continue
        repo_dir = repo_dirs.get(repo_id)
        if repo_dir:
            valid_repos.append((feature_repo, instance_id, repo_id, repo_dir))

    print(f"  Valid repos with local dirs: {len(valid_repos)}", flush=True)

    # Count total existing function_student_targets rows (fast scalar query)
    total_existing = conn.run("""
        SELECT COUNT(*) FROM function_student_targets
    """)[0][0]
    done_per_repo = {}   # not tracking per-repo; cap per repo = target_new // n_repos
    print(f"  Already in function_student_targets: {total_existing:,}", flush=True)

    # Compute per-repo cap via binary search
    n_repos = len(valid_repos)
    target_new = max(0, args.target - total_existing)
    per_repo_cap = max(50, target_new // max(1, n_repos) + 100)  # initial estimate
    print(f"  Target new: {target_new:,}  |  Initial per-repo cap: {per_repo_cap}",
          flush=True)

    # ── Phase 2: walk repos ───────────────────────────────────────────────────
    print(f"\nPhase 2: Walking {len(valid_repos)} repo HEADs …", flush=True)
    rng = random.Random(args.seed)

    all_functions = []   # (instance_id, feature_file, feature_function, sig, body)
    for i, (feature_repo, instance_id, repo_id, repo_dir) in enumerate(valid_repos):
        funcs = extract_repo_functions(repo_dir, instance_id, per_repo_cap, rng)
        all_functions.extend(funcs)

        if (i + 1) % 20 == 0 or (i + 1) == len(valid_repos):
            print(f"  [{i+1}/{len(valid_repos)}] repos scanned, "
                  f"{len(all_functions):,} functions so far", flush=True)

    print(f"  → {len(all_functions):,} new functions extracted", flush=True)

    if not all_functions:
        print("No functions found. Exiting.")
        conn.close()
        return

    # Deduplicate against existing DB entries
    # (instance_id, feature_file, feature_function) must not exist in function_embeddings
    print(f"  Deduplicating against existing function_embeddings …", flush=True)
    existing_keys = set()
    existing_rows = conn.run("""
        SELECT instance_id, feature_file, feature_function
        FROM function_embeddings
        WHERE model_name = 'Qwen2.5-Coder-3B'
    """)
    for row in existing_rows:
        existing_keys.add((row[0], row[1], row[2]))

    all_functions = [
        f for f in all_functions
        if (f[0], f[1], f[2]) not in existing_keys
    ]
    print(f"  After dedup: {len(all_functions):,} functions", flush=True)

    # Global shuffle and trim to target
    rng.shuffle(all_functions)
    all_functions = all_functions[:target_new]
    print(f"  Trimmed to target: {len(all_functions):,}", flush=True)

    if not all_functions:
        print("Nothing new to extract.")
        conn.close()
        return

    # ── Phase 3: GPU inference ────────────────────────────────────────────────
    print(f"\nPhase 3: Teacher inference on {args.device} …", flush=True)
    device = args.device if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH,
        output_hidden_states=True,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    ).eval()

    sig_texts  = [f[3] for f in all_functions]
    body_texts = [f[4] for f in all_functions]
    N = len(sig_texts)

    print(f"  Embedding {N:,} signatures …", flush=True)
    sig_embs, sig_toks = embed_texts_batched(
        model, tokenizer, sig_texts, device, args.batch_size)

    print(f"  Embedding {N:,} bodies …", flush=True)
    body_embs, body_toks = embed_texts_batched(
        model, tokenizer, body_texts, device, args.batch_size)

    print(f"  → {N:,} embeddings computed", flush=True)

    # ── Phase 4: store ────────────────────────────────────────────────────────
    print(f"\nPhase 4: Storing in postgres …", flush=True)

    inserted_fe = inserted_fst = skipped = 0

    for i, (instance_id, ff, fn, sig_text, body_text) in enumerate(all_functions):
        sig_emb  = sig_embs[i].tolist()
        body_emb = body_embs[i].tolist()
        sig_tok  = sig_toks[i]
        body_tok = body_toks[i]

        # Insert into function_embeddings (body_emb used as non-null embedding)
        try:
            fe_rows = conn.run("""
                INSERT INTO function_embeddings
                    (instance_id, feature_file, feature_function,
                     model_name, layer_index, embedding)
                VALUES (:iid, :ff, :fn, :mn, :li, :emb)
                ON CONFLICT (instance_id, feature_file, feature_function,
                             model_name, layer_index)
                DO UPDATE SET instance_id = EXCLUDED.instance_id
                RETURNING id
            """, iid=instance_id, ff=ff, fn=fn,
                 mn=TEACHER_MODEL_NAME, li=TEACHER_LAYER, emb=body_emb)
        except Exception as e:
            skipped += 1
            continue

        if not fe_rows:
            skipped += 1
            continue
        function_id = fe_rows[0][0]
        inserted_fe += 1

        # Insert into function_student_targets
        try:
            conn.run("""
                INSERT INTO function_student_targets
                    (function_id, sig_embedding, body_embedding,
                     sig_tokens, body_tokens, sig_text, body_text)
                VALUES (:fid, :sig, :body, :st, :bt, :stext, :btext)
                ON CONFLICT (function_id) DO NOTHING
            """,
                fid=function_id,
                sig=sig_emb,
                body=body_emb,
                st=sig_tok,
                bt=body_tok,
                stext=sig_text,
                btext=body_text,
            )
            inserted_fst += 1
        except Exception as e:
            skipped += 1

        if inserted_fst % 2000 == 0 and inserted_fst > 0:
            print(f"  {inserted_fst:,} rows written …", flush=True)

    conn.close()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} min:")
    print(f"  function_embeddings rows inserted: {inserted_fe:,}")
    print(f"  function_student_targets inserted: {inserted_fst:,}")
    print(f"  skipped (errors/conflicts):        {skipped:,}", flush=True)

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
    print(f"\n  Total corpus: {total[0]:,} functions from {total[1]} repos",
          flush=True)


if __name__ == '__main__':
    main()
