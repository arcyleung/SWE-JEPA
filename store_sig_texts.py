"""
Add sig_text / body_text columns to function_student_targets and populate them
by re-fetching source via overlayfs (same pipeline as extract_student_targets.py).

No GPU required. Runs in ~30-40 minutes (366 PR overlayfs mounts).

Usage:
    python store_sig_texts.py
"""

import os
import subprocess
import sys
import textwrap
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import pg8000.native
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase0_1_similarity_matrix import (
    DB, OVERLAY_MERGED_BASE,
    _mount_overlay, _umount_overlay, _sha_available, _fetch_sha,
)
from llm_similarity_judge import _extract_named_function

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOKENS_FILE = os.path.join(os.path.dirname(__file__), 'crawl_tokens.yaml')
REPOS_BASE  = '/shared_workspace_mfs/repos'
MIN_BODY_CHARS = 5


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


def _fetch_pr_worker(pr_row, needed, gh_token):
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
    tag = f'sigtxt-{repo_id}-{instance_id[-6:]}'
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


def main():
    conn = pg8000.native.Connection(**DB)

    # Add columns if not present
    for col in ('sig_text TEXT', 'body_text TEXT'):
        try:
            conn.run(f'ALTER TABLE function_student_targets ADD COLUMN IF NOT EXISTS {col}')
        except Exception:
            pass

    # Load functions that still need texts
    rows = conn.run("""
        SELECT fst.function_id, fe.instance_id, fe.feature_file, fe.feature_function
        FROM function_student_targets fst
        JOIN function_embeddings fe ON fe.id = fst.function_id
          AND fe.model_name = 'Qwen2.5-Coder-3B'
        WHERE fst.sig_text IS NULL
        ORDER BY fst.function_id
    """)
    print(f"Functions needing texts: {len(rows):,}", flush=True)
    if not rows:
        print("All texts already stored.")
        conn.close()
        return

    # Fetch PR metadata
    iid_list = list({r[1] for r in rows})
    pr_rows_raw = conn.run("""
        SELECT DISTINCT ON (instance_id) instance_id, repo, repo_id, base_commit
        FROM prs
        WHERE instance_id = ANY(:iids)
          AND base_commit IS NOT NULL AND base_commit != ''
    """, iids=iid_list)
    pr_map = {r[0]: (r[1], r[2], r[3], r[0]) for r in pr_rows_raw}

    by_iid = {}
    for (fid, iid, ff, fn) in rows:
        if iid in pr_map:
            by_iid.setdefault(iid, []).append((ff, fn, fid))

    try:
        tokens = yaml.safe_load(open(TOKENS_FILE)).get('gh_tokens', [])
    except Exception:
        tokens = []
    if not tokens:
        tokens = [None]

    print(f"Fetching from {len(by_iid)} PRs …", flush=True)
    source_cache = {}
    max_workers = min(len(tokens), 14)
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
            if done % 50 == 0 or done == total:
                print(f"  [{done}/{total}] PRs done, {len(source_cache)} sources",
                      flush=True)

    # Split and store
    stored = skipped = 0
    for (fid, iid, ff, fn) in rows:
        src = source_cache.get(fid)
        if not src:
            skipped += 1
            continue
        result = split_source(src)
        if result is None:
            skipped += 1
            continue
        sig_text, body_text = result
        conn.run("""
            UPDATE function_student_targets
            SET sig_text = :sig, body_text = :body
            WHERE function_id = :fid
        """, sig=sig_text, body=body_text, fid=fid)
        stored += 1
        if stored % 1000 == 0:
            print(f"  {stored:,} rows updated …", flush=True)

    conn.close()
    print(f"\nDone: {stored:,} texts stored, {skipped:,} skipped", flush=True)


if __name__ == '__main__':
    main()
