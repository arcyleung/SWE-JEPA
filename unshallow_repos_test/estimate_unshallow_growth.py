#!/usr/bin/env python3
"""
Estimate storage growth from deepening shallow git clones.

Workflow:
1) Sample N repos from /shared_workspace_mfs/repos
2) Measure global usage with mfsgetquota (before)
3) Deepen sampled repos by +100 commits
4) Measure global usage with mfsgetquota (d100)
5) Deepen sampled repos by +400 commits (total +500 from start)
6) Measure global usage with mfsgetquota (d500)
7) Save a quick growth plot

Notes:
- This mutates sampled repos by fetching more history.
- Uses `git -c safe.directory=*` to avoid shared ownership warnings.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

UNITS = {
    'B': 1,
    'KB': 1024,
    'MB': 1024**2,
    'GB': 1024**3,
    'TB': 1024**4,
    'PB': 1024**5,
}


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def parse_human_size(token: str) -> int:
    token = token.strip()
    m = re.match(r'^([0-9]+(?:\.[0-9]+)?)([KMGTP]?B)$', token)
    if not m:
        raise ValueError(f'Cannot parse size token: {token!r}')
    val = float(m.group(1))
    unit = m.group(2)
    return int(val * UNITS[unit])


def mfs_size_bytes(repos_root: Path) -> tuple[int, str]:
    # Use raw mode (no -H) for byte-accurate deltas.
    cp = run(['mfsgetquota', str(repos_root)])
    if cp.returncode != 0:
        raise RuntimeError(f'mfsgetquota failed: {cp.stderr.strip() or cp.stdout.strip()}')

    # Expect line like: " size     |        2913557209088 | ..."
    size_line = None
    for ln in cp.stdout.splitlines():
        if ln.strip().startswith('size'):
            size_line = ln
            break
    if size_line is None:
        raise RuntimeError(f'Could not find size line in mfsgetquota output:\n{cp.stdout}')

    parts = [p.strip() for p in size_line.split('|')]
    if len(parts) < 2:
        raise RuntimeError(f'Unexpected size line format: {size_line}')

    raw = parts[1].replace(' ', '')
    if raw.isdigit():
        return int(raw), cp.stdout
    # Fallback for unexpected formats.
    return parse_human_size(raw), cp.stdout


def write_simple_svg(plot_path: Path, labels: list[str], vals: list[int], title: str) -> None:
    w, h = 900, 500
    ml, mr, mt, mb = 70, 30, 60, 80
    pw, ph = w - ml - mr, h - mt - mb
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        vmax = vmin + 1

    def x(i: int) -> float:
        if len(vals) == 1:
            return ml + pw / 2
        return ml + i * (pw / (len(vals) - 1))

    def y(v: int) -> float:
        return mt + (vmax - v) * ph / (vmax - vmin)

    points = ' '.join(f'{x(i):.1f},{y(v):.1f}' for i, v in enumerate(vals))

    texts = []
    for i, (lab, v) in enumerate(zip(labels, vals)):
        texts.append(
            f'<circle cx=\"{x(i):.1f}\" cy=\"{y(v):.1f}\" r=\"5\" fill=\"#1f77b4\" />'
            f'<text x=\"{x(i):.1f}\" y=\"{y(v)-10:.1f}\" text-anchor=\"middle\" '
            f'font-size=\"12\">{v/(1024**4):.4f} TiB</text>'
            f'<text x=\"{x(i):.1f}\" y=\"{h-mb+25:.1f}\" text-anchor=\"middle\" '
            f'font-size=\"12\">{lab}</text>'
        )

    svg = f'''<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\">
<rect x=\"0\" y=\"0\" width=\"{w}\" height=\"{h}\" fill=\"white\"/>
<text x=\"{w/2:.1f}\" y=\"30\" text-anchor=\"middle\" font-size=\"18\" font-family=\"sans-serif\">{title}</text>
<line x1=\"{ml}\" y1=\"{h-mb}\" x2=\"{w-mr}\" y2=\"{h-mb}\" stroke=\"#333\"/>
<line x1=\"{ml}\" y1=\"{mt}\" x2=\"{ml}\" y2=\"{h-mb}\" stroke=\"#333\"/>
<polyline fill=\"none\" stroke=\"#1f77b4\" stroke-width=\"2\" points=\"{points}\"/>
{''.join(texts)}
<text x=\"{ml}\" y=\"{mt-10}\" font-size=\"12\" font-family=\"sans-serif\">bytes scale</text>
</svg>'''
    plot_path.write_text(svg)


def list_repo_dirs(repos_root: Path) -> list[Path]:
    out = []
    for p in repos_root.iterdir():
        if not p.is_dir():
            continue
        if (p / '.git').exists():
            out.append(p)
    return sorted(out)


def is_shallow(repo_dir: Path) -> bool:
    cp = run(['git', '-c', 'safe.directory=*', '-C', str(repo_dir), 'rev-parse', '--is-shallow-repository'])
    if cp.returncode != 0:
        return False
    return cp.stdout.strip() == 'true'


def deepen_repo(repo_dir: Path, deepen_by: int) -> dict:
    name = repo_dir.name
    if deepen_by <= 0:
        return {'repo': name, 'status': 'skipped', 'reason': 'non_positive_depth'}

    if not is_shallow(repo_dir):
        return {'repo': name, 'status': 'skipped', 'reason': 'not_shallow'}

    # Use origin explicitly; keep operation scoped and deterministic.
    cp = run([
        'git', '-c', 'safe.directory=*', '-C', str(repo_dir),
        'fetch', '--quiet', '--deepen', str(deepen_by), 'origin'
    ])
    if cp.returncode != 0:
        return {
            'repo': name,
            'status': 'error',
            'reason': (cp.stderr.strip() or cp.stdout.strip() or 'fetch_failed')[:400],
        }

    return {'repo': name, 'status': 'ok', 'reason': ''}


def deepen_batch(sample: list[Path], deepen_by: int, workers: int) -> list[dict]:
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(deepen_repo, p, deepen_by): p for p in sample}
        done = 0
        total = len(sample)
        for fut in as_completed(futs):
            done += 1
            results.append(fut.result())
            if done % 5 == 0 or done == total:
                print(f'  [{done}/{total}] deepen +{deepen_by} done', flush=True)
    return results


def tb(x: int) -> float:
    return x / (1024**4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repos-root', default='/shared_workspace_mfs/repos')
    ap.add_argument('--sample-size', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--targets', type=int, nargs='+',
                    default=[100, 500, 1000, 2000, 5000],
                    help='Absolute depth targets to evaluate (default: 100 500 1000 2000 5000)')
    ap.add_argument('--out-json', default='docs/unshallow_growth_sample20.json')
    ap.add_argument('--out-plot', default='docs/unshallow_growth_sample20.png')
    ap.add_argument('--sample-file', default='docs/unshallow_growth_sample20_repos.txt')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    targets = sorted(set(args.targets))
    if not targets or targets[0] <= 0:
        raise ValueError('--targets must be positive integers')

    repos_root = Path(args.repos_root)
    all_repos = list_repo_dirs(repos_root)
    if len(all_repos) < args.sample_size:
        raise ValueError(f'Only found {len(all_repos)} repos, need {args.sample_size}')

    rng = random.Random(args.seed)
    sample = rng.sample(all_repos, args.sample_size)

    Path(args.sample_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.sample_file).write_text('\n'.join(str(p) for p in sample) + '\n')

    print(f'Selected {len(sample)} sampled repos (seed={args.seed})')
    print(f'Sample list: {args.sample_file}')

    t0 = time.time()
    before_bytes, before_raw = mfs_size_bytes(repos_root)
    print(f'before: {before_bytes} bytes ({tb(before_bytes):.3f} TiB)')

    stages = {
        'before': {
            'bytes': before_bytes,
            'tib': tb(before_bytes),
            'mfsgetquota': before_raw,
        }
    }
    deepen_results: dict[str, list[dict]] = {}

    prev_target = 0
    for target in targets:
        inc = target - prev_target
        label = f'd{target}'
        if inc <= 0:
            raise ValueError(f'Non-increasing target list: {targets}')

        if not args.dry_run:
            print(f'\nDeepening sampled repos by +{inc} (to target {target}) ...')
            deepen_results[label] = deepen_batch(sample, inc, args.workers)
        else:
            deepen_results[label] = []

        cur_bytes, cur_raw = mfs_size_bytes(repos_root)
        print(f'{label}:   {cur_bytes} bytes ({tb(cur_bytes):.3f} TiB)')
        stages[label] = {'bytes': cur_bytes, 'tib': tb(cur_bytes), 'mfsgetquota': cur_raw}
        prev_target = target

    payload = {
        'timestamp_utc': int(time.time()),
        'repos_root': str(repos_root),
        'sample_size': args.sample_size,
        'seed': args.seed,
        'targets': targets,
        'sample_repos': [str(p) for p in sample],
        'stages': stages,
        'delta_bytes': {},
        'deepen_results': deepen_results,
        'elapsed_sec': time.time() - t0,
    }
    # before -> each target
    for target in targets:
        label = f'd{target}'
        payload['delta_bytes'][f'before_to_{label}'] = stages[label]['bytes'] - before_bytes
    # adjacent deltas
    prev = 'before'
    for target in targets:
        cur = f'd{target}'
        payload['delta_bytes'][f'{prev}_to_{cur}'] = stages[cur]['bytes'] - stages[prev]['bytes']
        prev = cur

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    print(f'\nWrote JSON: {out_json}')

    labels = ['before'] + [f'd{t}' for t in targets]
    vals = [stages[k]['bytes'] for k in labels]
    out_plot = Path(args.out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    title = f'Repos usage growth (sample={args.sample_size})'

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7.5, 4.5))
        plt.plot(labels, vals, marker='o', linewidth=2)
        for xk, yv in zip(labels, vals):
            plt.text(xk, yv, f'{tb(yv):.4f} TiB', fontsize=9, ha='center', va='bottom')
        plt.ylabel('Bytes')
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_plot, dpi=150)
        plt.close()
        print(f'Wrote plot: {out_plot}')
    except ModuleNotFoundError:
        svg_path = out_plot.with_suffix('.svg')
        write_simple_svg(svg_path, labels, vals, title)
        print(f'matplotlib not installed; wrote SVG plot instead: {svg_path}')


if __name__ == '__main__':
    main()
