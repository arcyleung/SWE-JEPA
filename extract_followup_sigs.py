"""
Experiment 4.1: Extract function signatures for defect-proneness prediction.

Source: followups_function table (15,296 distinct function anchors, 144 repos).
Output: followup_sigs.jsonl — one JSON per anchor:
  feature_instance_id, feature_repo, feature_file, feature_function,
  sig_text, has_bugfix, n_bugfix_prs, has_feature, max_bugfix_overlap

Pipeline:
  Phase 1 – Aggregate labels per anchor (SQL GROUP BY)
  Phase 2 – Fetch sources via overlayfs (CPU parallel, per feature PR)
  Phase 3 – Extract sig_text, write JSONL

Usage:
    python extract_followup_sigs.py
    python extract_followup_sigs.py --out followup_sigs.jsonl --workers 14
"""

import argparse
import ast
import json
import os
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import yaml
import pg8000.native

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase0_1_similarity_matrix import (
    OVERLAY_MERGED_BASE,
    _mount_overlay,
    _umount_overlay,
    _sha_available,
    _fetch_sha,
)
from llm_similarity_judge import _extract_named_function

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PG_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'postgres_connection.yaml')
_pg_cfg = yaml.safe_load(open(PG_CONFIG_FILE))
DB = dict(host=_pg_cfg['ip'], port=_pg_cfg.get('port', 9999),
          user=_pg_cfg['user'], password=_pg_cfg['password'],
          database=_pg_cfg['database'])

REPOS_BASE   = '/shared_workspace_mfs/repos'
TOKENS_FILE  = os.path.join(os.path.dirname(__file__), 'crawl_tokens.yaml')
DEFAULT_OUT  = os.path.join(os.path.dirname(__file__), 'followup_sigs.jsonl')


# ── Signature extraction ───────────────────────────────────────────────────────

def extract_sig(src: str) -> str | None:
    """Return the signature portion of a function: def line + docstring (if any).
    Falls back to the first non-empty line if AST parse fails."""
    src = textwrap.dedent(src)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # Fallback: just take lines up to but not including the first `pass`/body stmt
        lines = src.splitlines()
        sig_lines = []
        for ln in lines:
            sig_lines.append(ln)
            stripped = ln.strip()
            if stripped and not stripped.startswith('def ') and not stripped.startswith('async def ') and not stripped.startswith('#') and not stripped.startswith('@') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                break
        return '\n'.join(sig_lines).strip() or None

    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not funcs:
        return None
    func = funcs[0]
    lines = src.splitlines()

    # Find split point: after docstring (if present), else before first real stmt
    first_stmt = func.body[0] if func.body else None
    if first_stmt is None:
        return None
    if (isinstance(first_stmt, ast.Expr)
            and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)):
        split_line = first_stmt.end_lineno  # include docstring in sig
    else:
        split_line = first_stmt.lineno - 1  # stop before first real stmt

    sig = '\n'.join(lines[:split_line]).strip()
    return sig if len(sig) >= 5 else None


# ── Phase 2: per-PR overlayfs fetch ───────────────────────────────────────────

def _fetch_pr_worker(pr_row, needed, gh_token):
    """
    pr_row:  (repo_slug, repo_id, base_commit, instance_id)
    needed:  [(feature_file, feature_function, anchor_key), ...]
    Returns: {anchor_key: sig_text}
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

    tag = f'followup-{repo_id}-{instance_id[-6:]}'
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

        by_file: dict[str, list] = {}
        for (ff, fn, ak) in needed:
            by_file.setdefault(ff, []).append((fn, ak))

        for feature_file, fn_ak_pairs in by_file.items():
            r = subprocess.run(
                ['git', '-C', merged, 'show', f'{base_commit}:{feature_file}'],
                capture_output=True, timeout=30)
            if r.returncode != 0:
                continue
            full_source = r.stdout.decode(errors='replace')
            for (fn, ak) in fn_ak_pairs:
                # feature_function may be qualified: 'ClassName.method_name'.
                # _extract_named_function matches node.name which is just the
                # method name, so strip the class prefix if present.
                fn_bare = fn.split('.')[-1] if '.' in fn else fn
                raw = _extract_named_function(full_source, fn_bare)
                if raw:
                    sig = extract_sig(raw)
                    if sig:
                        results[ak] = sig
    finally:
        _umount_overlay(merged, upper, work)

    return results


def fetch_all_sigs(anchors: list, tokens: list) -> dict:
    """
    anchors: [(feature_instance_id, feature_file, feature_function), ...]
    Returns: {(feature_instance_id, feature_file, feature_function): sig_text}
    """
    if not tokens:
        tokens = [None]

    # Build PR map: instance_id → (repo_slug, repo_id, head_sha)
    # head_sha is the PR branch tip — the state AFTER the feature PR's changes,
    # where the feature functions actually exist.
    conn = pg8000.native.Connection(**DB)
    iid_list = list({a[0] for a in anchors})
    pr_rows = conn.run("""
        SELECT DISTINCT ON (instance_id)
               instance_id, repo, repo_id, head_sha
        FROM prs
        WHERE instance_id = ANY(:iids)
          AND head_sha IS NOT NULL AND head_sha != ''
    """, iids=iid_list)
    conn.close()
    pr_map = {r[0]: (r[1], r[2], r[3], r[0]) for r in pr_rows}

    # Group anchors by instance_id
    by_iid: dict[str, list] = {}
    for (iid, ff, fn) in anchors:
        if iid in pr_map:
            anchor_key = (iid, ff, fn)
            by_iid.setdefault(iid, []).append((ff, fn, anchor_key))

    print(f"  Fetching from {len(by_iid):,} PRs for {len(anchors):,} anchors …",
          flush=True)

    results = {}
    max_workers = min(len(tokens), 14)
    done = 0
    total = len(by_iid)

    with ProcessPoolExecutor(max_workers=max_workers,
                             mp_context=mp.get_context('spawn')) as executor:
        futures = {
            executor.submit(_fetch_pr_worker, pr_map[iid], needed,
                            tokens[i % len(tokens)]): iid
            for i, (iid, needed) in enumerate(by_iid.items())
        }
        for future in as_completed(futures):
            done += 1
            try:
                results.update(future.result())
            except Exception as e:
                print(f"  WARNING {futures[future]}: {e}", flush=True)
            if done % 20 == 0 or done == total:
                print(f"  [{done}/{total}] PRs done, {len(results):,} sigs found",
                      flush=True)

    return results


# ── Phase 1: load anchors + labels ────────────────────────────────────────────

def load_anchors() -> list[dict]:
    """
    Query followups_function, aggregate labels per distinct anchor.
    Returns list of dicts with keys:
      feature_instance_id, feature_repo, feature_file, feature_function,
      has_bugfix, n_bugfix_prs, has_feature, max_bugfix_overlap
    Excludes 'maintenance' category (dependency bumps, not defect signal).
    """
    conn = pg8000.native.Connection(**DB)
    rows = conn.run("""
        SELECT
            feature_instance_id,
            feature_repo,
            feature_file,
            feature_function,
            MAX(CASE WHEN followup_category = 'bugfix' THEN 1 ELSE 0 END)
                AS has_bugfix,
            COUNT(DISTINCT CASE WHEN followup_category = 'bugfix'
                                THEN followup_pr_number END)
                AS n_bugfix_prs,
            MAX(CASE WHEN followup_category = 'feature' THEN 1 ELSE 0 END)
                AS has_feature,
            MAX(CASE WHEN followup_category = 'bugfix'
                     THEN hunk_overlap_fraction END)
                AS max_bugfix_overlap,
            MAX(feature_function_start) AS feature_function_start,
            MAX(feature_function_end)   AS feature_function_end
        FROM followups_function
        GROUP BY feature_instance_id, feature_repo, feature_file, feature_function
        ORDER BY feature_instance_id, feature_file, feature_function
    """)
    conn.close()

    anchors = []
    for r in rows:
        anchors.append({
            'feature_instance_id':    r[0],
            'feature_repo':           r[1],
            'feature_file':           r[2],
            'feature_function':       r[3],
            'has_bugfix':             int(r[4]),
            'n_bugfix_prs':           int(r[5]),
            'has_feature':            int(r[6]),
            'max_bugfix_overlap':     float(r[7]) if r[7] is not None else 0.0,
            'feature_function_start': int(r[8]) if r[8] is not None else None,
            'feature_function_end':   int(r[9]) if r[9] is not None else None,
        })
    return anchors


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',     default=DEFAULT_OUT,
                    help='Output JSONL path (default: followup_sigs.jsonl)')
    ap.add_argument('--workers', type=int, default=14,
                    help='Max parallel overlayfs workers (default: 14)')
    args = ap.parse_args()

    t0 = time.time()
    print("SWE-JEPA Exp 4.1: Extract followup function signatures")
    print('=' * 60)

    # Load GH tokens
    try:
        tokens = yaml.safe_load(open(TOKENS_FILE)).get('gh_tokens', [])
        print(f"  Loaded {len(tokens)} GitHub tokens")
    except Exception:
        tokens = []
        print("  No GitHub tokens found — fetching unauthenticated")

    # Phase 1: aggregate labels
    print("\nPhase 1: Loading anchors and labels from followups_function …",
          flush=True)
    anchors = load_anchors()
    print(f"  {len(anchors):,} distinct anchors from "
          f"{len({a['feature_instance_id'] for a in anchors})} feature PRs")
    n_bugfix = sum(a['has_bugfix'] for a in anchors)
    print(f"  has_bugfix: {n_bugfix:,} ({100*n_bugfix/len(anchors):.1f}%) positive")

    # Phase 2: fetch sources via overlayfs
    print("\nPhase 2: Fetching sources via overlayfs …", flush=True)
    anchor_tuples = [(a['feature_instance_id'], a['feature_file'],
                      a['feature_function']) for a in anchors]
    sig_map = fetch_all_sigs(anchor_tuples, tokens)
    print(f"\n  Found sig_text for {len(sig_map):,} / {len(anchors):,} anchors "
          f"({100*len(sig_map)/len(anchors):.1f}%)", flush=True)

    # Phase 3: write JSONL
    print(f"\nPhase 3: Writing {args.out} …", flush=True)
    written = 0
    with open(args.out, 'w') as f:
        for a in anchors:
            key = (a['feature_instance_id'], a['feature_file'], a['feature_function'])
            sig = sig_map.get(key)
            if sig is None:
                continue
            record = {**a, 'sig_text': sig}
            f.write(json.dumps(record) + '\n')
            written += 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min:")
    print(f"  Written: {written:,} records → {args.out}")
    missing = len(anchors) - written
    print(f"  Missing (source not found): {missing:,} "
          f"({100*missing/len(anchors):.1f}%)")

    # Label stats for written records
    has_bugfix = sum(1 for a in anchors
                     if (a['feature_instance_id'], a['feature_file'],
                         a['feature_function']) in sig_map and a['has_bugfix'])
    print(f"\n  Label stats (written records):")
    print(f"    has_bugfix=1 : {has_bugfix:,} ({100*has_bugfix/written:.1f}%)")
    print(f"    has_bugfix=0 : {written-has_bugfix:,} "
          f"({100*(written-has_bugfix)/written:.1f}%)")


if __name__ == '__main__':
    main()
