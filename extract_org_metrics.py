"""
Collect organizational proxy metrics for followup anchors.

Metrics implemented:
- ownership_friction: low ownership concentration / high author entropy on file history
- interface_stress: high cross-module co-change coupling for the file

Input:
- followup_sigs.jsonl (or compatible JSONL with feature_instance_id/repo/file/function)

Output:
- followup_org_metrics.jsonl
- optional postgres upsert into followup_org_metrics

Usage:
  python extract_org_metrics.py
  python extract_org_metrics.py --limit 1000 --workers 8
  python extract_org_metrics.py --store-db
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SIGS = os.path.join(ROOT, 'followup_sigs.jsonl')
DEFAULT_OUT = os.path.join(ROOT, 'followup_org_metrics.jsonl')
PG_CONFIG_FILE = os.path.join(ROOT, 'postgres_connection.yaml')
REPOS_BASE = '/shared_workspace_mfs/repos'

DDL = """
CREATE TABLE IF NOT EXISTS followup_org_metrics (
    id SERIAL PRIMARY KEY,
    feature_instance_id TEXT NOT NULL,
    feature_repo TEXT NOT NULL,
    feature_file TEXT NOT NULL,
    feature_function TEXT NOT NULL,
    commits_touching_file INTEGER,
    distinct_authors INTEGER,
    top_author_fraction DOUBLE PRECISION,
    author_entropy DOUBLE PRECISION,
    ownership_friction DOUBLE PRECISION,
    cochange_weighted_degree DOUBLE PRECISION,
    cochange_unique_neighbors INTEGER,
    cochange_cross_module_ratio DOUBLE PRECISION,
    interface_stress DOUBLE PRECISION,
    metric_source TEXT,
    coverage_flag TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (feature_instance_id, feature_file, feature_function)
);
CREATE INDEX IF NOT EXISTS fom_repo_idx ON followup_org_metrics(feature_repo);
CREATE INDEX IF NOT EXISTS fom_iid_idx ON followup_org_metrics(feature_instance_id);
ALTER TABLE followup_org_metrics ADD COLUMN IF NOT EXISTS metric_source TEXT;
"""

UPSERT = """
INSERT INTO followup_org_metrics (
    feature_instance_id, feature_repo, feature_file, feature_function,
    commits_touching_file, distinct_authors, top_author_fraction,
    author_entropy, ownership_friction, cochange_weighted_degree,
    cochange_unique_neighbors, cochange_cross_module_ratio,
    interface_stress, metric_source, coverage_flag
) VALUES (
    :feature_instance_id, :feature_repo, :feature_file, :feature_function,
    :commits_touching_file, :distinct_authors, :top_author_fraction,
    :author_entropy, :ownership_friction, :cochange_weighted_degree,
    :cochange_unique_neighbors, :cochange_cross_module_ratio,
    :interface_stress, :metric_source, :coverage_flag
)
ON CONFLICT (feature_instance_id, feature_file, feature_function) DO UPDATE SET
    feature_repo                 = EXCLUDED.feature_repo,
    commits_touching_file        = EXCLUDED.commits_touching_file,
    distinct_authors             = EXCLUDED.distinct_authors,
    top_author_fraction          = EXCLUDED.top_author_fraction,
    author_entropy               = EXCLUDED.author_entropy,
    ownership_friction           = EXCLUDED.ownership_friction,
    cochange_weighted_degree     = EXCLUDED.cochange_weighted_degree,
    cochange_unique_neighbors    = EXCLUDED.cochange_unique_neighbors,
    cochange_cross_module_ratio  = EXCLUDED.cochange_cross_module_ratio,
    interface_stress             = EXCLUDED.interface_stress,
    metric_source                = EXCLUDED.metric_source,
    coverage_flag                = EXCLUDED.coverage_flag,
    created_at                   = NOW()
"""


def _git_base() -> list[str]:
    return ['git', '-c', 'safe.directory=*']


def _repo_dir_map() -> dict[str, str]:
    """Map repo slug owner/name -> local clone path."""
    m: dict[str, str] = {}
    for entry in os.listdir(REPOS_BASE):
        p = os.path.join(REPOS_BASE, entry)
        if not os.path.isdir(p):
            continue
        parts = entry.split('__')
        if len(parts) < 3:
            continue
        owner = parts[1]
        name = '__'.join(parts[2:])
        m[f'{owner}/{name}'] = p
    return m


def _top_module(path: str) -> str:
    path = path.strip('/')
    if not path:
        return '__root__'
    return path.split('/', 1)[0]


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p + 1e-12)
    return h


def _ownership_friction(author_counts: Counter) -> tuple[int, float, float, float]:
    """Return n_authors, top_frac, entropy, friction [0,1]."""
    total = sum(author_counts.values())
    if total <= 0:
        return 0, 0.0, 0.0, 0.0
    n_auth = len(author_counts)
    top_frac = max(author_counts.values()) / total
    h = _entropy(author_counts)
    h_norm = h / math.log(max(2, n_auth)) if n_auth > 1 else 0.0
    # Higher with dispersion + larger author set.
    breadth = min(1.0, math.log1p(n_auth) / math.log(10.0))
    friction = float(max(0.0, min(1.0, (1.0 - top_frac) * 0.6 + h_norm * 0.4)) * breadth)
    return n_auth, float(top_frac), float(h), friction


def _parse_repo_history(repo_dir: str, needed_files: set[str],
                        max_files_per_commit: int = 80) -> dict[str, dict]:
    """
    Parse git log once and compute file-level author/co-change stats.
    Only metrics for needed_files are materialized.
    """
    import subprocess

    cmd = _git_base() + [
        '-C', repo_dir,
        'log', '--format=@@@%H\t%ae', '--name-only', '--diff-filter=AMRT', '--', '*.py'
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {}

    file_author_counts: dict[str, Counter] = defaultdict(Counter)
    file_commit_counts: Counter = Counter()
    # For each needed file, neighbors/weights from same commits.
    neigh_weights: dict[str, Counter] = defaultdict(Counter)

    cur_author = None
    cur_files: list[str] = []

    def flush_commit(author: str | None, files: list[str]) -> None:
        if not author or not files:
            return
        uniq = list(dict.fromkeys(f for f in files if f.endswith('.py')))
        if not uniq:
            return
        if len(uniq) > max_files_per_commit:
            # Very broad refactors are noisy for co-change signal.
            return

        needed_in_commit = [f for f in uniq if f in needed_files]
        if not needed_in_commit:
            return

        modules = {_f: _top_module(_f) for _f in uniq}
        for f in needed_in_commit:
            file_author_counts[f][author] += 1
            file_commit_counts[f] += 1

        # Update co-change only for needed files vs all files in same commit.
        for f in needed_in_commit:
            for g in uniq:
                if g == f:
                    continue
                w = 1.0
                # Slightly upweight cross-module cochanges.
                if modules[f] != modules[g]:
                    w = 1.25
                neigh_weights[f][g] += w

    for ln in r.stdout.splitlines():
        if ln.startswith('@@@'):
            flush_commit(cur_author, cur_files)
            cur_files = []
            marker = ln[3:]
            parts = marker.split('\t', 1)
            cur_author = parts[1] if len(parts) == 2 else None
            continue
        if ln.strip() == '':
            continue
        cur_files.append(ln.strip())

    flush_commit(cur_author, cur_files)

    out: dict[str, dict] = {}
    for f in needed_files:
        author_counts = file_author_counts.get(f, Counter())
        n_auth, top_frac, ent, friction = _ownership_friction(author_counts)

        neigh = neigh_weights.get(f, Counter())
        weighted_degree = float(sum(neigh.values()))
        unique_neighbors = int(len(neigh))
        if weighted_degree <= 0:
            cross_ratio = 0.0
        else:
            fm = _top_module(f)
            cross = sum(w for g, w in neigh.items() if _top_module(g) != fm)
            cross_ratio = float(cross / weighted_degree)

        interface_stress = float(math.log1p(weighted_degree) * cross_ratio)

        if file_commit_counts.get(f, 0) == 0:
            flag = 'missing_history'
        elif unique_neighbors == 0:
            flag = 'low_signal'
        else:
            flag = 'ok'

        out[f] = {
            'commits_touching_file': int(file_commit_counts.get(f, 0)),
            'distinct_authors': int(n_auth),
            'top_author_fraction': float(top_frac),
            'author_entropy': float(ent),
            'ownership_friction': float(friction),
            'cochange_weighted_degree': weighted_degree,
            'cochange_unique_neighbors': unique_neighbors,
            'cochange_cross_module_ratio': cross_ratio,
            'interface_stress': interface_stress,
            'coverage_flag': flag,
        }

    return out


def _worker(repo: str, repo_dir: str, files: list[str], max_files_per_commit: int) -> tuple[str, dict[str, dict]]:
    metrics = _parse_repo_history(repo_dir, set(files), max_files_per_commit=max_files_per_commit)
    return repo, metrics


def _collect_db_proxy_metrics(repos: set[str], max_files_per_pr: int = 120) -> dict[tuple[str, str], dict]:
    """
    Build repo/file metrics from followups_function + prs (author + co-change),
    used as fallback when git history in local clones is too shallow.
    """
    import pg8000.native

    cfg = yaml.safe_load(open(PG_CONFIG_FILE))
    db = dict(host=cfg['ip'], port=cfg.get('port', 9999), user=cfg['user'],
              password=cfg['password'], database=cfg['database'])
    conn = pg8000.native.Connection(**db)
    rows = conn.run("""
        SELECT
            repo,
            instance_id,
            COALESCE(NULLIF(pr_author, ''), 'unknown') AS pr_author,
            file_patches
        FROM prs
        WHERE repo = ANY(:repos)
          AND file_patches IS NOT NULL
    """, repos=sorted(repos))
    conn.close()

    by_instance: dict[tuple[str, str], set[str]] = defaultdict(set)   # (repo, iid)->py files
    file_author_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    file_touch_counts: Counter = Counter()

    for repo, iid, author, file_patches in rows:
        if repo not in repos:
            continue

        patches = file_patches
        if isinstance(patches, str):
            try:
                patches = json.loads(patches)
            except Exception:
                continue
        if not isinstance(patches, list):
            continue

        files = set()
        for item in patches:
            if isinstance(item, dict):
                p = item.get('file_path') or item.get('path')
            else:
                p = None
            if not isinstance(p, str):
                continue
            if p.endswith('.py') or p.endswith('.pyi') or p.endswith('.pyx'):
                files.add(p)

        if not files:
            continue
        if len(files) > max_files_per_pr:
            # Very broad sweeps are usually generated churn; skip for signal quality.
            continue

        by_instance[(repo, iid)].update(files)
        for fpath in files:
            k = (repo, fpath)
            file_author_counts[k][author] += 1
            file_touch_counts[k] += 1

    neigh_weights: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for (repo, _iid), files in by_instance.items():
        files = list(files)
        if len(files) <= 1:
            continue
        mods = {_f: _top_module(_f) for _f in files}
        for f in files:
            for g in files:
                if g == f:
                    continue
                w = 1.25 if mods[f] != mods[g] else 1.0
                neigh_weights[(repo, f)][g] += w

    out: dict[tuple[str, str], dict] = {}
    keys = set(file_touch_counts.keys()) | set(file_author_counts.keys()) | set(neigh_weights.keys())
    for k in keys:
        repo, fpath = k
        a = file_author_counts.get(k, Counter())
        n_auth, top_frac, ent, friction = _ownership_friction(a)
        neigh = neigh_weights.get(k, Counter())
        weighted_degree = float(sum(neigh.values()))
        unique_neighbors = int(len(neigh))
        if weighted_degree <= 0:
            cross_ratio = 0.0
        else:
            fm = _top_module(fpath)
            cross = sum(w for g, w in neigh.items() if _top_module(g) != fm)
            cross_ratio = float(cross / weighted_degree)
        interface_stress = float(math.log1p(weighted_degree) * cross_ratio)
        out[k] = {
            'commits_touching_file': int(file_touch_counts.get(k, 0)),
            'distinct_authors': int(n_auth),
            'top_author_fraction': float(top_frac),
            'author_entropy': float(ent),
            'ownership_friction': float(friction),
            'cochange_weighted_degree': weighted_degree,
            'cochange_unique_neighbors': unique_neighbors,
            'cochange_cross_module_ratio': cross_ratio,
            'interface_stress': interface_stress,
            'coverage_flag': 'db_proxy',
            'metric_source': 'db_proxy',
        }
    return out


def _create_table(conn):
    for stmt in DDL.strip().split(';'):
        s = stmt.strip()
        if s:
            conn.run(s)


def _upsert_rows(conn, rows: list[dict]) -> None:
    for r in rows:
        conn.run(UPSERT, **r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigs-file', default=DEFAULT_SIGS)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--repo', type=str, default=None,
                    help='Filter to one repo slug owner/name')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-files-per-commit', type=int, default=80)
    ap.add_argument('--db-max-files-per-pr', type=int, default=120,
                    help='Drop oversized PRs when building DB co-change proxy')
    ap.add_argument('--store-db', action='store_true')
    ap.add_argument('--disable-db-proxy', action='store_true',
                    help='Do not backfill metrics from followups/prs DB proxies')
    args = ap.parse_args()

    t0 = time.time()
    print('Collecting ownership friction + interface stress metrics')
    print('=' * 60, flush=True)

    anchors = []
    with open(args.sigs_file) as f:
        for line in f:
            r = json.loads(line)
            if args.repo and r['feature_repo'] != args.repo:
                continue
            anchors.append(r)
            if args.limit and len(anchors) >= args.limit:
                break

    if not anchors:
        raise ValueError('No anchors selected.')

    print(f'Anchors: {len(anchors):,}', flush=True)

    by_repo_files: dict[str, set[str]] = defaultdict(set)
    for a in anchors:
        by_repo_files[a['feature_repo']].add(a['feature_file'])

    repo_dir_map = _repo_dir_map()
    repo_jobs = []
    missing_repo = 0
    for repo, files in by_repo_files.items():
        rd = repo_dir_map.get(repo)
        if rd is None:
            missing_repo += 1
            continue
        repo_jobs.append((repo, rd, sorted(files), args.max_files_per_commit))

    print(f'Repos with local clone: {len(repo_jobs):,} / {len(by_repo_files):,}', flush=True)
    if missing_repo:
        print(f'  Missing local clone for {missing_repo} repo(s)', flush=True)

    repo_metrics: dict[str, dict[str, dict]] = {}
    max_workers = max(1, min(args.workers, len(repo_jobs)))

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=__import__('multiprocessing').get_context('spawn')) as ex:
        futs = [ex.submit(_worker, *job) for job in repo_jobs]
        done = 0
        for fut in as_completed(futs):
            repo, metrics = fut.result()
            repo_metrics[repo] = metrics
            done += 1
            if done % 10 == 0 or done == len(futs):
                print(f'  [{done}/{len(futs)}] repos parsed', flush=True)

    db_proxy = {}
    repo_defaults: dict[str, dict] = {}
    if not args.disable_db_proxy:
        print('Building DB-proxy ownership/co-change metrics …', flush=True)
        db_proxy = _collect_db_proxy_metrics(set(by_repo_files.keys()),
                                             max_files_per_pr=args.db_max_files_per_pr)
        print(f'  DB-proxy file metrics: {len(db_proxy):,}', flush=True)
        # Repo-level fallback defaults for files not present in PR patch history.
        by_repo_vals: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for (repo, _file), m in db_proxy.items():
            for k in [
                'commits_touching_file', 'distinct_authors', 'top_author_fraction',
                'author_entropy', 'ownership_friction', 'cochange_weighted_degree',
                'cochange_unique_neighbors', 'cochange_cross_module_ratio', 'interface_stress'
            ]:
                by_repo_vals[repo][k].append(float(m[k]))
        for repo, kv in by_repo_vals.items():
            d = {}
            for k, vals in kv.items():
                vals = sorted(vals)
                med = vals[len(vals) // 2]
                if k in ('commits_touching_file', 'distinct_authors', 'cochange_unique_neighbors'):
                    d[k] = int(round(med))
                else:
                    d[k] = float(med)
            d['coverage_flag'] = 'repo_fallback'
            d['metric_source'] = 'repo_fallback'
            repo_defaults[repo] = d

    rows = []
    missing_file_metrics = 0
    source_counts = Counter()
    for a in anchors:
        fm_git = repo_metrics.get(a['feature_repo'], {}).get(a['feature_file'])
        if fm_git is not None and fm_git.get('commits_touching_file', 0) > 0:
            fm = {**fm_git, 'metric_source': 'git_history'}
        else:
            fm = db_proxy.get((a['feature_repo'], a['feature_file']))
        if fm is None:
            fallback = repo_defaults.get(a['feature_repo'])
            if fallback is not None:
                fm = fallback
            else:
                missing_file_metrics += 1
                fm = {
                    'commits_touching_file': 0,
                    'distinct_authors': 0,
                    'top_author_fraction': 0.0,
                    'author_entropy': 0.0,
                    'ownership_friction': 0.0,
                    'cochange_weighted_degree': 0.0,
                    'cochange_unique_neighbors': 0,
                    'cochange_cross_module_ratio': 0.0,
                    'interface_stress': 0.0,
                    'metric_source': 'missing',
                    'coverage_flag': 'missing_repo_or_file',
                }
        source_counts[fm['metric_source']] += 1
        rows.append({
            'feature_instance_id': a['feature_instance_id'],
            'feature_repo': a['feature_repo'],
            'feature_file': a['feature_file'],
            'feature_function': a['feature_function'],
            **fm,
        })

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    print(f'Metrics written: {len(rows):,} rows -> {args.out}', flush=True)
    print(f'Missing file-level metrics: {missing_file_metrics:,} ({100*missing_file_metrics/len(rows):.1f}%)', flush=True)
    print(f'Metric sources: {dict(source_counts)}', flush=True)

    if args.store_db:
        import pg8000.native
        cfg = yaml.safe_load(open(PG_CONFIG_FILE))
        db = dict(host=cfg['ip'], port=cfg.get('port', 9999), user=cfg['user'],
                  password=cfg['password'], database=cfg['database'])
        conn = pg8000.native.Connection(**db)
        _create_table(conn)
        for i in range(0, len(rows), 500):
            _upsert_rows(conn, rows[i:i + 500])
        conn.close()
        print('Postgres upsert complete: followup_org_metrics', flush=True)

    elapsed = (time.time() - t0) / 60.0
    print(f'Done in {elapsed:.1f} min', flush=True)


if __name__ == '__main__':
    main()
