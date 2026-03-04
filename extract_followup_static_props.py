"""
Extract static AST properties for followup anchors (Exp 4.4 support).

Why this exists:
- `function_static_props` built in Exp 0.3 mostly covered a different PR set.
- Exp 4.4 needs cyclomatic complexity aligned to followup anchors
  `(feature_instance_id, feature_file, feature_function)`.

This script reads anchors from `followups_function`, fetches source at the
feature PR `head_sha`, finds each target function (supports class-qualified
names like `ClassName.method`), computes static properties, and upserts into
`function_static_props`.

Usage:
  python extract_followup_static_props.py --create-table
  python extract_followup_static_props.py --limit-prs 20
  python extract_followup_static_props.py --workers 14 --batch 300
"""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
import os
import subprocess
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pg8000.native
import yaml

from extract_static_props import analyse_function, create_table, store_props
from phase0_1_similarity_matrix import (
    _load_tokens,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Upstream code often contains invalid escape sequences in string literals.
# We only need AST structure; suppress these noisy parse warnings.
warnings.filterwarnings("ignore", category=SyntaxWarning)

PG_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'postgres_connection.yaml')
_pg_cfg = yaml.safe_load(open(PG_CONFIG_FILE))
DB = dict(host=_pg_cfg['ip'], port=_pg_cfg.get('port', 9999),
          user=_pg_cfg['user'], password=_pg_cfg['password'],
          database=_pg_cfg['database'])

REPOS_BASE = '/shared_workspace_mfs/repos'
DEFAULT_SIGS = os.path.join(os.path.dirname(__file__), 'followup_sigs.jsonl')


def _git_base_cmd() -> list[str]:
    # Avoid "detected dubious ownership" across shared repo mounts.
    return ['git', '-c', 'safe.directory=*']


def _sha_available(repo_dir: str, sha: str) -> bool:
    r = subprocess.run(_git_base_cmd() + ['-C', repo_dir, 'cat-file', '-e', sha],
                       capture_output=True)
    return r.returncode == 0


def _fetch_sha(repo_dir: str, sha: str, repo_slug: str, gh_token: str | None) -> None:
    url = (f'https://{gh_token}@github.com/{repo_slug}.git' if gh_token
           else f'https://github.com/{repo_slug}.git')
    subprocess.run(
        _git_base_cmd() + ['-C', repo_dir, 'fetch', '--depth=1', url, sha],
        capture_output=True, timeout=120
    )


def _find_repo_dir(repo_id: int) -> str | None:
    for entry in os.listdir(REPOS_BASE):
        if entry.startswith(f"{repo_id}_"):
            return os.path.join(REPOS_BASE, entry)
    return None


def _iter_function_nodes_with_qname(tree: ast.AST):
    def walk(stmts, prefix: list[str]):
        for node in stmts:
            if isinstance(node, ast.ClassDef):
                yield from walk(node.body, prefix + [node.name])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qname = '.'.join(prefix + [node.name]) if prefix else node.name
                yield qname, node

    if isinstance(tree, ast.Module):
        yield from walk(tree.body, [])


def _build_node_maps(source: str) -> tuple[dict[str, ast.AST], dict[str, list[str]]]:
    """Return (qname->node, bare_name->list[qname])."""
    tree = ast.parse(source)
    qmap: dict[str, ast.AST] = {}
    bare: dict[str, list[str]] = {}
    for qname, node in _iter_function_nodes_with_qname(tree):
        qmap[qname] = node
        b = qname.split('.')[-1]
        bare.setdefault(b, []).append(qname)
    return qmap, bare


def _pick_function_node(feature_function: str,
                        qmap: dict[str, ast.AST],
                        bare_map: dict[str, list[str]]) -> ast.AST | None:
    # Exact first
    if feature_function in qmap:
        return qmap[feature_function]

    # If followup name is class-qualified with different nesting, try suffix match.
    if '.' in feature_function:
        suffix = feature_function.split('.')[-1]
        candidates = bare_map.get(suffix, [])
        if len(candidates) == 1:
            return qmap[candidates[0]]
        # Prefer exact tail match if possible
        for c in candidates:
            if c.endswith(feature_function):
                return qmap[c]

    # Bare fallback
    candidates = bare_map.get(feature_function, [])
    if len(candidates) == 1:
        return qmap[candidates[0]]
    return None


def _fetch_pr_worker(pr_row, needed, gh_token):
    """
    pr_row: (instance_id, repo_slug, repo_id, head_sha)
    needed: list[(feature_file, feature_function)]
    returns: list[record-for-store_props]
    """
    instance_id, repo_slug, repo_id, head_sha = pr_row
    if not head_sha:
        return []

    repo_dir = _find_repo_dir(repo_id)
    if not repo_dir:
        return []

    out = []
    if not _sha_available(repo_dir, head_sha):
        _fetch_sha(repo_dir, head_sha, repo_slug, gh_token)
    if not _sha_available(repo_dir, head_sha):
        return out

    by_file: dict[str, list[str]] = {}
    for fpath, fname in needed:
        by_file.setdefault(fpath, []).append(fname)

    for feature_file, fnames in by_file.items():
        r = subprocess.run(
            _git_base_cmd() + ['-C', repo_dir, 'show', f'{head_sha}:{feature_file}'],
            capture_output=True, timeout=30,
        )
        if r.returncode != 0:
            continue

        source = r.stdout.decode(errors='replace')
        try:
            qmap, bare_map = _build_node_maps(source)
        except SyntaxError:
            continue

        lines = source.splitlines()
        for feature_function in fnames:
            node = _pick_function_node(feature_function, qmap, bare_map)
            if node is None:
                continue
            props = analyse_function(node, lines)
            out.append({
                'instance_id': instance_id,
                'feature_file': feature_file,
                'feature_function': feature_function,
                **props,
            })

    return out


def _load_followup_anchors_db(conn) -> list[dict]:
    rows = conn.run("""
        SELECT DISTINCT
               ff.feature_instance_id,
               ff.feature_repo,
               p.repo_id,
               p.head_sha,
               ff.feature_file,
               ff.feature_function
        FROM followups_function ff
        JOIN prs p
          ON p.instance_id = ff.feature_instance_id
        WHERE p.head_sha IS NOT NULL
          AND p.head_sha != ''
        ORDER BY ff.feature_instance_id, ff.feature_file, ff.feature_function
    """)
    anchors = []
    for r in rows:
        anchors.append({
            'feature_instance_id': r[0],
            'feature_repo': r[1],
            'repo_id': r[2],
            'head_sha': r[3],
            'feature_file': r[4],
            'feature_function': r[5],
        })
    return anchors


def _load_followup_anchors_from_sigs(conn, sigs_file: str) -> list[dict]:
    """
    Load anchors from followup_sigs.jsonl (the same extractable subset used by
    Exp 4.1/4.3/4.4), then join to prs for repo_id/head_sha.
    """
    keep = []
    with open(sigs_file) as f:
        for line in f:
            r = json.loads(line)
            keep.append((r['feature_instance_id'], r['feature_file'], r['feature_function']))

    keep_set = set(keep)
    iids = sorted({k[0] for k in keep_set})
    rows = conn.run("""
        SELECT instance_id, repo, repo_id, head_sha
        FROM prs
        WHERE instance_id = ANY(:iids)
          AND head_sha IS NOT NULL
          AND head_sha != ''
    """, iids=iids)
    pr_map = {r[0]: (r[1], r[2], r[3]) for r in rows}

    anchors = []
    for iid, feature_file, feature_function in sorted(keep_set):
        pr = pr_map.get(iid)
        if not pr:
            continue
        repo, repo_id, head_sha = pr
        anchors.append({
            'feature_instance_id': iid,
            'feature_repo': repo,
            'repo_id': repo_id,
            'head_sha': head_sha,
            'feature_file': feature_file,
            'feature_function': feature_function,
        })
    return anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--create-table', action='store_true')
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--batch', type=int, default=300)
    ap.add_argument('--sigs-file', type=str, default=DEFAULT_SIGS,
                    help='JSONL anchors from extract_followup_sigs.py (default: followup_sigs.jsonl)')
    ap.add_argument('--all-followup-anchors', action='store_true',
                    help='Use all DB followup anchors (includes non-function symbols; usually lower yield)')
    ap.add_argument('--limit-prs', type=int, default=None,
                    help='Process only first N PRs (debug)')
    args = ap.parse_args()

    t0 = time.time()
    conn = pg8000.native.Connection(**DB)

    if args.create_table:
        print('Ensuring function_static_props exists …', flush=True)
        create_table(conn)

    if args.all_followup_anchors:
        print('Loading all followup anchors from DB …', flush=True)
        anchors = _load_followup_anchors_db(conn)
    else:
        print(f'Loading followup anchors from {args.sigs_file} …', flush=True)
        anchors = _load_followup_anchors_from_sigs(conn, args.sigs_file)
    print(f"  Anchors loaded: {len(anchors):,}", flush=True)

    done_rows = conn.run("""
        SELECT instance_id, feature_file, feature_function
        FROM function_static_props
    """)
    done = {tuple(r) for r in done_rows}

    remaining = [a for a in anchors if (
        a['feature_instance_id'], a['feature_file'], a['feature_function']) not in done]
    print(f"  Already in function_static_props: {len(done):,}", flush=True)
    print(f"  Remaining followup anchors: {len(remaining):,}", flush=True)

    by_pr: dict[str, dict] = {}
    for a in remaining:
        iid = a['feature_instance_id']
        entry = by_pr.setdefault(iid, {
            'pr_row': (iid, a['feature_repo'], a['repo_id'], a['head_sha']),
            'needed': [],
        })
        entry['needed'].append((a['feature_file'], a['feature_function']))

    pr_items = list(by_pr.values())
    if args.limit_prs is not None:
        pr_items = pr_items[:args.limit_prs]
    print(f"  PRs to process: {len(pr_items):,}", flush=True)

    if not pr_items:
        print('Nothing to do.')
        conn.close()
        return

    tokens = _load_tokens()
    if not tokens:
        tokens = [None]

    max_workers = max(1, min(args.workers, len(pr_items), len(tokens)))
    print(f"Processing with {max_workers} workers …", flush=True)

    buf = []
    written = 0
    done_prs = 0

    with ProcessPoolExecutor(max_workers=max_workers,
                             mp_context=mp.get_context('spawn')) as ex:
        futures = {}
        for i, item in enumerate(pr_items):
            token = tokens[i % len(tokens)]
            fut = ex.submit(_fetch_pr_worker, item['pr_row'], item['needed'], token)
            futures[fut] = item['pr_row'][0]

        for fut in as_completed(futures):
            done_prs += 1
            iid = futures[fut]
            try:
                recs = fut.result()
            except Exception as e:
                print(f"  WARNING {iid}: {e}", flush=True)
                recs = []

            buf.extend(recs)
            if len(buf) >= args.batch or done_prs == len(pr_items):
                if buf:
                    store_props(buf, conn)
                    written += len(buf)
                    buf = []

            if done_prs % 20 == 0 or done_prs == len(pr_items):
                print(f"  [{done_prs}/{len(pr_items)}] PRs done, {written:,} rows written",
                      flush=True)

    # Coverage check against the exact loaded anchor set.
    anchor_set = {
        (a['feature_instance_id'], a['feature_file'], a['feature_function'])
        for a in anchors
    }
    iid_list = sorted({a['feature_instance_id'] for a in anchors})
    sp_rows = conn.run("""
        SELECT instance_id, feature_file, feature_function
        FROM function_static_props
        WHERE instance_id = ANY(:iids)
    """, iids=iid_list)
    sp_set = {tuple(r) for r in sp_rows}
    matched = len(anchor_set & sp_set)
    total = len(anchor_set)
    conn.close()

    elapsed = (time.time() - t0) / 60.0
    pct = (100.0 * matched / total) if total else 0.0
    print(f"\nDone in {elapsed:.1f} min")
    print(f"  New/updated rows written: {written:,}")
    print(f"  Followup static-prop coverage: {matched:,}/{total:,} ({pct:.2f}%)")


if __name__ == '__main__':
    main()
