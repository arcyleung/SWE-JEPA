#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env['GIT_TERMINAL_PROMPT'] = '0'
    env['GCM_INTERACTIVE'] = 'Never'
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)


def is_shallow(repo: str) -> bool:
    cp = run(['git', '-c', 'safe.directory=*', '-C', repo, 'rev-parse', '--is-shallow-repository'], timeout=30)
    return cp.returncode == 0 and cp.stdout.strip().lower() == 'true'


def unshallow_to_depth(repo: str, depth: int, fetch_timeout_sec: int) -> dict:
    name = os.path.basename(repo)
    if not os.path.isdir(repo):
        return {'repo': repo, 'name': name, 'status': 'error', 'reason': 'missing_dir'}
    if not os.path.isdir(os.path.join(repo, '.git')):
        return {'repo': repo, 'name': name, 'status': 'error', 'reason': 'missing_git'}

    if not is_shallow(repo):
        return {'repo': repo, 'name': name, 'status': 'skipped_not_shallow', 'reason': ''}

    # Fetch all remote heads directly so stale local refspecs like
    # refs/heads/master don't break when default branch is main.
    try:
        cp = run([
            'git', '-c', 'safe.directory=*', '-C', repo,
            'fetch', '--quiet', '--depth', str(depth), 'origin',
            '+refs/heads/*:refs/remotes/origin/*', '--tags'
        ], timeout=fetch_timeout_sec)
    except subprocess.TimeoutExpired:
        return {'repo': repo, 'name': name, 'status': 'error', 'reason': f'fetch_timeout_{fetch_timeout_sec}s'}
    if cp.returncode != 0:
        msg = (cp.stderr.strip() or cp.stdout.strip() or 'fetch_failed')[:500]
        return {'repo': repo, 'name': name, 'status': 'error', 'reason': msg}

    # verify post-state
    if is_shallow(repo):
        return {'repo': repo, 'name': name, 'status': 'still_shallow', 'reason': ''}
    return {'repo': repo, 'name': name, 'status': 'ok', 'reason': ''}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repos-list', default='/shared_workspace_mfs/arthur/coder/unshallow_repos_test/all_repos_shallow.txt')
    ap.add_argument('--depth', type=int, default=5000)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--fetch-timeout-sec', type=int, default=180)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out-json', default='/shared_workspace_mfs/arthur/coder/unshallow_repos_test/unshallow_d5000_results.json')
    ap.add_argument('--out-jsonl', default='/shared_workspace_mfs/arthur/coder/unshallow_repos_test/unshallow_d5000_results.jsonl')
    args = ap.parse_args()

    repos = [ln.strip() for ln in Path(args.repos_list).read_text().splitlines() if ln.strip()]
    if args.limit:
        repos = repos[:args.limit]

    total = len(repos)
    print(f'target repos: {total}')
    print(f'depth target: {args.depth}, workers: {args.workers}')

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    stats = {'ok': 0, 'still_shallow': 0, 'skipped_not_shallow': 0, 'error': 0}
    results = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex, out_jsonl.open('w') as jf:
        futs = [ex.submit(unshallow_to_depth, repo, args.depth, args.fetch_timeout_sec) for repo in repos]
        for idx, fut in enumerate(as_completed(futs), start=1):
            r = fut.result()
            status = r['status']
            stats[status] = stats.get(status, 0) + 1
            results.append(r)
            jf.write(json.dumps(r) + '\n')
            if idx % 100 == 0 or idx == total:
                elapsed = time.time() - start
                rate = idx / elapsed if elapsed > 0 else 0.0
                remain = (total - idx) / rate if rate > 0 else 0.0
                print(f"[{idx}/{total}] ok={stats.get('ok',0)} still={stats.get('still_shallow',0)} "
                      f"skip={stats.get('skipped_not_shallow',0)} err={stats.get('error',0)} "
                      f"eta={remain/60:.1f}m")

    elapsed = time.time() - start
    summary = {
        'timestamp_utc': int(time.time()),
        'depth': args.depth,
        'workers': args.workers,
        'total': total,
        'stats': stats,
        'elapsed_sec': elapsed,
        'results_jsonl': str(out_jsonl),
    }
    Path(args.out_json).write_text(json.dumps({'summary': summary, 'results': results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
