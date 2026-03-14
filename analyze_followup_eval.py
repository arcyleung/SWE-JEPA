#!/usr/bin/env python3
"""Analyze steered vs baseline patches against function follow-up indices."""
import json, os, re
from collections import defaultdict

FILE_RE = re.compile(r'^\+\+\+ b/(.+)$', re.M)
HUNK_CTX_RE = re.compile(r'^@@[^@\n]+@@[ \t]*(.*)', re.M)
GO_FUNC_RE = re.compile(r'\bfunc\s+(?:\([^)]*\)\s+)?(\w+)\s*[({]')
PY_FUNC_RE = re.compile(r'(?:async\s+)?def\s+(\w+)\s*[\(:]|class\s+(\w+)\s*[\(:]')

def _pkg(fp):
    parts = fp.split('/'); return parts[0] if len(parts) > 1 else ''

def _parse_funcs(patch, func_re):
    ff = defaultdict(set); cur = None
    for line in patch.splitlines():
        m = FILE_RE.match(line)
        if m: cur = m.group(1); continue
        m = HUNK_CTX_RE.match(line)
        if m and cur:
            ctx = m.group(1).strip()
            fm = func_re.search(ctx)
            if fm: ff[cur].add(fm.group(1) or (fm.group(2) if fm.lastindex >= 2 else None))
    return {f: {fn for fn in fns if fn} for f, fns in ff.items()}

def _cmr(files):
    if len(files) < 2: return 0.0
    pkgs = [_pkg(f) for f in files]
    cross = sum(1 for i, a in enumerate(pkgs) for b in pkgs[i+1:] if a != b)
    total = len(pkgs) * (len(pkgs) - 1) // 2
    return cross / total if total else 0.0

def _score_funcs(ff, repo, idx):
    ri = idx.get(repo, {})
    vals = []
    for f, fns in ff.items():
        fi = ri.get(f, {})
        for fn in fns:
            vals.append(fi.get(fn, {}).get('followup_rate', 0.0))
    n = len(vals)
    if n == 0: return 0.0, 0.0, 0
    return sum(vals) / n, max(vals), n

def _parse_iid_py(iid):
    parts = iid.split('__')
    if len(parts) >= 3:
        try: return f'{parts[0]}/{parts[1]}', int(parts[-1].replace('pr', ''))
        except: pass
    return None, None

def _parse_iid_go(iid):
    s = iid.removeprefix('go_prs__')
    parts = s.split('__')
    try: return f'{parts[0]}/{parts[1]}', int(parts[-2])
    except: return None, None

def analyze(lang, patch_dir_s, patch_dir_b, func_idx, meta, parse_iid, func_re):
    rows = []
    b_files = set(os.listdir(patch_dir_b))
    s_files = set(os.listdir(patch_dir_s))
    common = sorted(b_files & s_files)
    for fname in common:
        if not fname.endswith('.patch'): continue
        iid = fname[:-6]
        repo, pr = parse_iid(iid)
        if not repo: continue
        b_patch = open(os.path.join(patch_dir_b, fname)).read()
        s_patch = open(os.path.join(patch_dir_s, fname)).read()
        b_files_list = FILE_RE.findall(b_patch)
        s_files_list = FILE_RE.findall(s_patch)
        b_ff = _parse_funcs(b_patch, func_re)
        s_ff = _parse_funcs(s_patch, func_re)
        b_fr, b_mfr, b_nf = _score_funcs(b_ff, repo, func_idx)
        s_fr, s_mfr, s_nf = _score_funcs(s_ff, repo, func_idx)
        b_cmr = _cmr(b_files_list)
        s_cmr = _cmr(s_files_list)
        m = meta.get(iid, meta.get((repo, pr), {}))
        has_fup = m.get('has_bugfix_followup', None)
        rows.append({
            'iid': iid, 'repo': repo, 'has_bugfix_followup': has_fup,
            'b_mean_fr': b_fr, 's_mean_fr': s_fr,
            'b_max_fr': b_mfr, 's_max_fr': s_mfr,
            'b_cmr': b_cmr, 's_cmr': s_cmr,
            'b_n_files': len(b_files_list), 's_n_files': len(s_files_list),
            'b_n_funcs': b_nf, 's_n_funcs': s_nf,
        })
    return rows

def report(lang, rows):
    print(f'\n{"="*70}')
    print(f' {lang.upper()} — Steered vs Baseline (function follow-up index)')
    print('=' * 70)
    pos = [r for r in rows if r['has_bugfix_followup'] is True]
    neg = [r for r in rows if r['has_bugfix_followup'] is False]
    unk = [r for r in rows if r['has_bugfix_followup'] is None]
    print(f'Total pairs: {len(rows)}  (pos={len(pos)}, neg={len(neg)}, unknown={len(unk)})')
    for label, subset in [('ALL', rows), ('HAS_FOLLOWUP (pos)', pos), ('NO_FOLLOWUP (neg)', neg)]:
        if not subset: continue
        n = len(subset)
        def avg(k): return sum(r[k] for r in subset) / n
        def win_pct(bk, sk): return sum(1 for r in subset if r[sk] < r[bk]) / n
        def tie_pct(bk, sk): return sum(1 for r in subset if r[sk] == r[bk]) / n
        print(f'\n  -- {label} ({n} pairs) --')
        print(f'  {"Metric":<28}  {"Baseline":>9}  {"Steered":>9}  {"Delta":>8}  {"S<B":>6}  {"Tie":>6}')
        print('  ' + '-' * 72)
        for lbl, bk, sk in [
            ('mean_func_followup_rate', 'b_mean_fr', 's_mean_fr'),
            ('max_func_followup_rate', 'b_max_fr', 's_max_fr'),
            ('cross_module_ratio', 'b_cmr', 's_cmr'),
            ('n_files', 'b_n_files', 's_n_files'),
            ('n_funcs', 'b_n_funcs', 's_n_funcs'),
        ]:
            ab = avg(bk); asv = avg(sk)
            wp = win_pct(bk, sk); tp = tie_pct(bk, sk)
            print(f'  {lbl:<28}  {ab:>9.4f}  {asv:>9.4f}  {asv - ab:>+8.4f}  {wp:>5.1%}  {tp:>5.1%}')

if __name__ == '__main__':
    py_func_idx = json.load(open('data/py_followup_function_index.json'))
    go_func_idx = json.load(open('data/go_ablation_v1/go_followup_function_index.json'))
    py_meta = {}
    for l in open('data/py_followup_eval/cohort_metadata.jsonl'):
        r = json.loads(l)
        py_meta[r['instance_id']] = r
        # Also key by iid__prN format used in patch filenames
        py_meta[f"{r['instance_id']}__pr{r['pull_number']}"] = r
    go_meta = {}
    for l in open('data/go_followup_eval/cohort_metadata.jsonl'):
        r = json.loads(l)
        go_meta[(r['repo'], r['pull_number'])] = r

    py_rows = analyze('python',
        'data/py_followup_eval/patches_steered', 'data/py_followup_eval/patches_baseline',
        py_func_idx, py_meta, _parse_iid_py, PY_FUNC_RE)
    go_rows = analyze('go',
        'data/go_followup_eval/patches_steered', 'data/go_followup_eval/patches_baseline',
        go_func_idx, go_meta, _parse_iid_go, GO_FUNC_RE)

    report('Python', py_rows)
    report('Go', go_rows)

    with open('data/py_followup_eval/analysis_results.jsonl', 'w') as f:
        for r in py_rows: f.write(json.dumps(r) + '\n')
    with open('data/go_followup_eval/analysis_results.jsonl', 'w') as f:
        for r in go_rows: f.write(json.dumps(r) + '\n')
    print('\nSaved analysis_results.jsonl for both languages.')
