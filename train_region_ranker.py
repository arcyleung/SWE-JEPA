"""
Experiment 4.4: SE-head stack for region-level localization.

Cheap downstream adaptation on frozen features (teacher/student embeddings +
lightweight structural/text signals), optimized for within-PR ranking.

Methods:
  - linear   : logistic probe on stacked frozen features
  - fusion   : logistic probe on [teacher_logit, student_logit, tfidf_logit, loc_logit]
  - mlp      : small MLP head on stacked frozen features
  - pairwise : in-PR pairwise ranker (logistic on feature differences)

Outputs:
  - docs/phase4_4_se_heads_localization.md
  - docs/phase4_4_se_heads_metrics.json
  - docs/phase4_4_topk_predictions.jsonl

Usage:
  python train_region_ranker.py --use-cache
  python train_region_ranker.py --use-cache --min-overlap 0.1
  python train_region_ranker.py --use-cache --models linear fusion mlp pairwise
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from probe_defect_prediction import DEFAULT_CACHE, DEFAULT_SIGS, load_sigs, repo_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PG_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'postgres_connection.yaml')

RESULTS_MD = os.path.join(os.path.dirname(__file__),
                          'docs', 'phase4_4_se_heads_localization.md')
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            'docs', 'phase4_4_se_heads_metrics.json')
TOPK_JSONL = os.path.join(os.path.dirname(__file__),
                          'docs', 'phase4_4_topk_predictions.jsonl')
RESULTS_MD_CONWAY = os.path.join(os.path.dirname(__file__),
                                 'docs', 'phase4_5_conway_localization.md')
RESULTS_JSON_CONWAY = os.path.join(os.path.dirname(__file__),
                                   'docs', 'phase4_5_conway_metrics.json')
TOPK_JSONL_CONWAY = os.path.join(os.path.dirname(__file__),
                                 'docs', 'phase4_5_topk_predictions.jsonl')


def enrich_with_static_props(records: list[dict]) -> tuple[list[dict], dict]:
    """
    Attach cyclomatic complexity from function_static_props when available.

    Returns:
      (records_with_cc, stats)
      stats = {'matched': int, 'total': int, 'coverage': float}
    """
    try:
        import pg8000.native
        cfg = yaml.safe_load(open(PG_CONFIG_FILE))
        db = dict(host=cfg['ip'], port=cfg.get('port', 9999),
                  user=cfg['user'], password=cfg['password'],
                  database=cfg['database'])
        conn = pg8000.native.Connection(**db)
    except Exception as e:
        print(f"  DB unavailable ({e}); skipping static-prop enrichment", flush=True)
        return records, {'matched': 0, 'total': len(records), 'coverage': 0.0}

    iids = list({r['feature_instance_id'] for r in records})
    try:
        rows = conn.run("""
            SELECT DISTINCT ON (ff.feature_instance_id, ff.feature_file, ff.feature_function)
                   ff.feature_instance_id,
                   ff.feature_file,
                   ff.feature_function,
                   MAX(sp.cyclomatic_complexity) OVER (
                     PARTITION BY ff.feature_instance_id, ff.feature_file, ff.feature_function
                   ) AS cyclomatic_complexity
            FROM followups_function ff
            LEFT JOIN function_static_props sp
              ON sp.instance_id = ff.feature_instance_id
             AND sp.feature_file = ff.feature_file
             AND sp.feature_function = ff.feature_function
            WHERE ff.feature_instance_id = ANY(:iids)
        """, iids=iids)
    except Exception as e:
        print(f"  Static-prop query failed ({e}); skipping enrichment", flush=True)
        conn.close()
        return records, {'matched': 0, 'total': len(records), 'coverage': 0.0}

    conn.close()
    cc_map = {(r[0], r[1], r[2]): (int(r[3]) if r[3] is not None else None)
              for r in rows}

    out = []
    for rec in records:
        key = (rec['feature_instance_id'], rec['feature_file'], rec['feature_function'])
        out.append({**rec, 'cyclomatic_complexity': cc_map.get(key)})

    n = sum(1 for r in out if r.get('cyclomatic_complexity') is not None)
    total = len(out)
    cov = (n / total) if total else 0.0
    print(f"  Enriched cyclomatic_complexity for {n:,}/{total:,} records "
          f"({100*cov:.2f}% coverage)", flush=True)
    return out, {'matched': n, 'total': total, 'coverage': cov}


def enrich_with_org_metrics(records: list[dict], org_metrics_file: str) -> tuple[list[dict], dict]:
    """
    Attach ownership/interface stress metrics from followup_org_metrics JSONL.

    Keys:
      (feature_instance_id, feature_file, feature_function)
    """
    if not os.path.exists(org_metrics_file):
        print(f"  Org metrics file missing: {org_metrics_file}; skipping org enrichment", flush=True)
        return records, {'matched': 0, 'total': len(records), 'coverage': 0.0, 'sources': {}}

    org_map = {}
    src_counts = {}
    with open(org_metrics_file) as f:
        for line in f:
            r = json.loads(line)
            key = (r['feature_instance_id'], r['feature_file'], r['feature_function'])
            org_map[key] = r
            src = r.get('metric_source', 'unknown')
            src_counts[src] = src_counts.get(src, 0) + 1

    org_keys = [
        'commits_touching_file',
        'distinct_authors',
        'top_author_fraction',
        'author_entropy',
        'ownership_friction',
        'cochange_weighted_degree',
        'cochange_unique_neighbors',
        'cochange_cross_module_ratio',
        'interface_stress',
    ]
    out = []
    n = 0
    for rec in records:
        key = (rec['feature_instance_id'], rec['feature_file'], rec['feature_function'])
        m = org_map.get(key)
        row = dict(rec)
        if m is not None:
            for k in org_keys:
                row[k] = float(m.get(k, 0.0))
            row['metric_source'] = m.get('metric_source', 'unknown')
            n += 1
        else:
            for k in org_keys:
                row[k] = 0.0
            row['metric_source'] = 'missing'
        out.append(row)

    total = len(out)
    cov = (n / total) if total else 0.0
    print(f"  Enriched org metrics for {n:,}/{total:,} records ({100*cov:.2f}% coverage)", flush=True)
    return out, {'matched': n, 'total': total, 'coverage': cov, 'sources': src_counts}


def eval_localization(records: list[dict], scores: np.ndarray,
                      label_key: str, k_values: list[int]) -> dict:
    """Compute mean Recall@K + MRR over PR groups with at least one positive."""
    groups: dict[str, list[tuple[int, int]]] = {}
    for i, rec in enumerate(records):
        groups.setdefault(rec['feature_instance_id'], []).append((i, int(rec[label_key])))

    recall_by_k = {k: [] for k in k_values}
    rr_vals: list[float] = []
    pr_sizes: list[int] = []

    for pr, items in groups.items():
        labels = [y for _, y in items]
        n_pos = sum(labels)
        if n_pos == 0:
            continue

        pr_sizes.append(len(items))
        idxs = [idx for idx, _ in items]
        s = scores[idxs]
        order = np.argsort(-s, kind='stable')
        ranked = [labels[j] for j in order]

        first_pos = next((j for j, y in enumerate(ranked) if y == 1), None)
        rr_vals.append(0.0 if first_pos is None else 1.0 / (first_pos + 1))

        for k in k_values:
            recall_by_k[k].append(sum(ranked[:k]) / n_pos)

    out = {k: (float(np.mean(v)) if v else 0.0) for k, v in recall_by_k.items()}
    out['mrr'] = float(np.mean(rr_vals)) if rr_vals else 0.0
    out['n_prs'] = len(pr_sizes)
    out['avg_size'] = float(np.mean(pr_sizes)) if pr_sizes else 0.0
    return out


def _sel(arr, idx):
    return arr[idx] if isinstance(arr, np.ndarray) else [arr[i] for i in idx]


def fit_logreg_scores(X_tv: np.ndarray, y_tv: np.ndarray, X_all: np.ndarray,
                      seed: int = 42) -> np.ndarray:
    sc = StandardScaler()
    X_tv_s = sc.fit_transform(X_tv)
    X_all_s = sc.transform(X_all)

    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced',
                             solver='lbfgs', random_state=seed)
    clf.fit(X_tv_s, y_tv)
    return clf.predict_proba(X_all_s)[:, 1]


def fit_tfidf_scores(text_tv: list[str], y_tv: np.ndarray,
                     text_all: list[str], seed: int = 42) -> np.ndarray:
    vec = TfidfVectorizer(max_features=5000, analyzer='word', sublinear_tf=True)
    X_tv = vec.fit_transform(text_tv)
    X_all = vec.transform(text_all)
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced',
                             solver='liblinear', random_state=seed)
    clf.fit(X_tv, y_tv)
    return clf.predict_proba(X_all)[:, 1]


def train_linear(X_tv: np.ndarray, y_tv: np.ndarray,
                 X_te: np.ndarray, seed: int) -> np.ndarray:
    sc = StandardScaler()
    X_tv_s = sc.fit_transform(X_tv)
    X_te_s = sc.transform(X_te)
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced',
                             solver='lbfgs', random_state=seed)
    clf.fit(X_tv_s, y_tv)
    return clf.predict_proba(X_te_s)[:, 1]


def train_mlp(X_tv: np.ndarray, y_tv: np.ndarray,
              X_te: np.ndarray, seed: int) -> np.ndarray:
    sc = StandardScaler()
    X_tv_s = sc.fit_transform(X_tv)
    X_te_s = sc.transform(X_te)

    clf = MLPClassifier(hidden_layer_sizes=(128,), activation='relu',
                        alpha=1e-4, max_iter=300, random_state=seed)
    clf.fit(X_tv_s, y_tv)
    return clf.predict_proba(X_te_s)[:, 1]


def build_pairwise_dataset(X: np.ndarray, y: np.ndarray, pr_ids: list[str],
                           seed: int, max_pairs_per_pr: int = 300) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    pr_map: dict[str, list[int]] = {}
    for i, pr in enumerate(pr_ids):
        pr_map.setdefault(pr, []).append(i)

    X_pairs = []
    y_pairs = []
    for pr, idxs in pr_map.items():
        pos = [i for i in idxs if y[i] == 1]
        neg = [i for i in idxs if y[i] == 0]
        if not pos or not neg:
            continue

        n_possible = len(pos) * len(neg)
        n_pairs = min(max_pairs_per_pr, n_possible)
        for _ in range(n_pairs):
            i = pos[rng.randint(len(pos))]
            j = neg[rng.randint(len(neg))]
            d = X[i] - X[j]
            X_pairs.append(d)
            y_pairs.append(1)
            X_pairs.append(-d)
            y_pairs.append(0)

    if not X_pairs:
        return np.zeros((0, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    return np.asarray(X_pairs, dtype=np.float32), np.asarray(y_pairs, dtype=np.int32)


def train_pairwise(X_tv: np.ndarray, y_tv: np.ndarray, pr_ids_tv: list[str],
                   X_te: np.ndarray, seed: int) -> np.ndarray:
    sc = StandardScaler()
    X_tv_s = sc.fit_transform(X_tv)
    X_te_s = sc.transform(X_te)

    X_pair, y_pair = build_pairwise_dataset(X_tv_s, y_tv, pr_ids_tv, seed)
    if len(X_pair) == 0:
        # Fallback when no usable in-PR pos/neg pairs exist.
        return train_linear(X_tv, y_tv, X_te, seed)

    clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=seed)
    clf.fit(X_pair, y_pair)

    logits = clf.decision_function(X_te_s)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))


def _mean_seed_scores(train_fn, X_tv, y_tv, X_te, seeds: list[int], pr_ids_tv=None):
    preds = []
    for s in seeds:
        if pr_ids_tv is None:
            preds.append(train_fn(X_tv, y_tv, X_te, s))
        else:
            preds.append(train_fn(X_tv, y_tv, pr_ids_tv, X_te, s))
    return np.mean(np.stack(preds, axis=0), axis=0)


def write_topk(records_test: list[dict], score_bank: dict[str, np.ndarray],
               k_values: list[int], out_path: str) -> None:
    k_max = max(k_values)
    by_pr: dict[str, list[int]] = {}
    for i, rec in enumerate(records_test):
        by_pr.setdefault(rec['feature_instance_id'], []).append(i)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_rows = 0
    with open(out_path, 'w') as f:
        for score_key, scores in score_bank.items():
            method, label_key = score_key.split('__', 1)
            for pr_id, idxs in by_pr.items():
                s = scores[idxs]
                order = np.argsort(-s, kind='stable')
                topk = []
                for rnk, j in enumerate(order[:k_max], start=1):
                    rec = records_test[idxs[j]]
                    topk.append({
                        'rank': rnk,
                        'score': float(s[j]),
                        'feature_repo': rec['feature_repo'],
                        'feature_file': rec['feature_file'],
                        'feature_function': rec['feature_function'],
                        'feature_function_start': rec.get('feature_function_start'),
                        'feature_function_end': rec.get('feature_function_end'),
                        'is_positive': int(rec[label_key]),
                    })
                payload = {
                    'method': method,
                    'label': label_key,
                    'feature_instance_id': pr_id,
                    'n_functions_in_pr': len(idxs),
                    'topk': topk,
                }
                f.write(json.dumps(payload) + '\n')
                n_rows += 1
    print(f"Top-K predictions written: {n_rows:,} rows → {out_path}", flush=True)


def write_report(metrics: dict, args, n_records: int, n_repos: int,
                 n_train: int, n_val: int, n_test: int,
                 n_bugfix_pos: int, n_feature_pos: int,
                 k_values: list[int], cc_stats: dict,
                 feature_names: list[str], out_md: str,
                 out_json: str, out_topk: str, title: str,
                 org_stats: dict | None = None) -> None:
    k10 = 10 if 10 in k_values else k_values[-1]
    k_header = ' | '.join(f'R@{k}' for k in k_values)
    k_sep = '|'.join(['-----'] * len(k_values))

    methods = ['random', 'loc', 'tfidf', 'teacher', 'student',
               'linear', 'fusion', 'mlp', 'pairwise']

    def row(label: str, key: str, label_key: str) -> str:
        m = metrics[label_key].get(key, {})
        vals = ' | '.join(f"{100*m.get(k, 0.0):.1f}%" for k in k_values)
        return f"| {label} | {vals} | {100*m.get('mrr', 0.0):.1f}% | {m.get('n_prs', 0)} |"

    bug_lines = []
    feat_lines = []
    name_map = {
        'random': 'Random',
        'loc': 'LOC',
        'tfidf': 'TF-IDF',
        'teacher': 'Teacher',
        'student': 'Student',
        'linear': 'SE-Linear',
        'fusion': 'SE-Fusion',
        'mlp': 'SE-MLP',
        'pairwise': 'SE-Pairwise',
    }
    for m in methods:
        if m in metrics['has_bugfix']:
            bug_lines.append(row(name_map[m], m, 'has_bugfix'))
        if m in metrics['has_feature']:
            feat_lines.append(row(name_map[m], m, 'has_feature'))

    pair_bug = metrics['has_bugfix'].get('pairwise', {})
    teach_bug = metrics['has_bugfix'].get('teacher', {})
    pair_feat = metrics['has_feature'].get('pairwise', {})
    rand_feat = metrics['has_feature'].get('random', {})
    delta_bug = (pair_bug.get(k10, 0.0) - teach_bug.get(k10, 0.0)) * 100
    delta_feat = (pair_feat.get(k10, 0.0) - rand_feat.get(k10, 0.0)) * 100

    org_line = ""
    if org_stats is not None:
        org_line = (f"**Org metrics coverage**: {org_stats.get('matched', 0):,}/"
                    f"{org_stats.get('total', 0):,} ({100*org_stats.get('coverage', 0.0):.2f}%)\n")

    report = f"""# {title}

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Data**: {n_records:,} function anchors, {n_repos} repos
**Split**: {n_train:,} train / {n_val:,} val / {n_test:,} test (repo split)
**Labels**: has_bugfix={n_bugfix_pos:,} ({100*n_bugfix_pos/n_records:.1f}%), has_feature={n_feature_pos:,} ({100*n_feature_pos/n_records:.1f}%)
**min_overlap**: {args.min_overlap:.2f}
**Models**: {', '.join(args.models)}
**Seeds**: {args.seeds}
**Static CC coverage**: {cc_stats.get('matched', 0):,}/{cc_stats.get('total', 0):,} ({100*cc_stats.get('coverage', 0.0):.2f}%)
{org_line}**Uses org metrics**: {'yes' if args.use_org_metrics else 'no'}
**SE feature channels**: {', '.join(feature_names)}

## Reframing summary

Frozen teacher/student embeddings are treated as a representation substrate.
Downstream SE behavior is learned via cheap task heads optimized for within-PR ranking.
No full-model SFT/RL is used.

## Task A: Bugfix Localization (`has_bugfix`)

| Method | {k_header} | MRR | n_prs |
|--------|{k_sep}|-----|------|
{chr(10).join(bug_lines)}

## Task B: Feature Localization (`has_feature`)

| Method | {k_header} | MRR | n_prs |
|--------|{k_sep}|-----|------|
{chr(10).join(feat_lines)}

## Key deltas

- Pairwise vs Teacher (bugfix R@{k10}): **{delta_bug:+.1f} pp**
- Pairwise vs Random (feature R@{k10}): **{delta_feat:+.1f} pp**

## Artifacts

- `{os.path.relpath(out_json, os.path.dirname(__file__))}`
- `{os.path.relpath(out_topk, os.path.dirname(__file__))}`
"""

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, 'w') as f:
        f.write(report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigs-file', default=DEFAULT_SIGS)
    ap.add_argument('--cache-file', default=DEFAULT_CACHE)
    ap.add_argument('--use-cache', action='store_true',
                    help='Require cached embeddings in --cache-file')
    ap.add_argument('--min-overlap', type=float, default=0.0)
    ap.add_argument('--k', type=int, nargs='+', default=[1, 3, 5, 10])
    ap.add_argument('--models', nargs='+',
                    choices=['linear', 'fusion', 'mlp', 'pairwise'],
                    default=['linear', 'fusion', 'mlp', 'pairwise'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--skip-static-enrich', action='store_true')
    ap.add_argument('--use-org-metrics', action='store_true',
                    help='Append Conway-style org metrics as feature channels')
    ap.add_argument('--org-metrics-file', default=os.path.join(os.path.dirname(__file__),
                                                               'followup_org_metrics.jsonl'))
    ap.add_argument('--results-md', default=None)
    ap.add_argument('--results-json', default=None)
    ap.add_argument('--topk-jsonl', default=None)
    args = ap.parse_args()

    if not args.use_cache:
        raise ValueError('Exp 4.4 currently requires --use-cache (frozen embedding workflow).')
    if not os.path.exists(args.cache_file):
        raise FileNotFoundError(f'Embedding cache missing: {args.cache_file}')

    t0 = time.time()
    k_values = sorted(set(args.k))
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError('--k values must be positive integers')

    print('SWE-JEPA Exp 4.4: SE-Head Stack for Region Localization')
    print('=' * 60, flush=True)

    if args.results_md:
        out_md = args.results_md
    else:
        out_md = RESULTS_MD_CONWAY if args.use_org_metrics else RESULTS_MD
    if args.results_json:
        out_json = args.results_json
    else:
        out_json = RESULTS_JSON_CONWAY if args.use_org_metrics else RESULTS_JSON
    if args.topk_jsonl:
        out_topk = args.topk_jsonl
    else:
        out_topk = TOPK_JSONL_CONWAY if args.use_org_metrics else TOPK_JSONL

    print(f"\nLoading {args.sigs_file} …", flush=True)
    records = load_sigs(args.sigs_file, args.min_overlap)
    cc_stats = {'matched': 0, 'total': len(records), 'coverage': 0.0}
    if not args.skip_static_enrich:
        print('Enriching static properties …', flush=True)
        records, cc_stats = enrich_with_static_props(records)
    org_stats = {'matched': 0, 'total': len(records), 'coverage': 0.0, 'sources': {}}
    if args.use_org_metrics:
        print('Enriching Conway org metrics …', flush=True)
        records, org_stats = enrich_with_org_metrics(records, args.org_metrics_file)

    sig_texts = [r['sig_text'] for r in records]
    repos = [r['feature_repo'] for r in records]
    pr_ids = [r['feature_instance_id'] for r in records]
    has_bugfix = np.array([r['has_bugfix'] for r in records], dtype=np.int32)
    has_feature = np.array([r['has_feature'] for r in records], dtype=np.int32)

    npz = np.load(args.cache_file)
    teacher_embs = npz['teacher_embs']
    student_embs = npz['student_embs']
    if len(teacher_embs) != len(records):
        raise ValueError(f'Cache size mismatch: {len(teacher_embs)} vs {len(records)} records')

    n_records = len(records)
    print(f"  records: {n_records:,}")
    print(f"  teacher_embs: {teacher_embs.shape}, student_embs: {student_embs.shape}")

    train_idx, val_idx, test_idx = repo_split(repos)
    tv_idx = train_idx + val_idx
    records_test = _sel(records, test_idx)

    print(f"  split: {len(train_idx):,} train / {len(val_idx):,} val / {len(test_idx):,} test")

    loc = np.array([len(t.splitlines()) for t in sig_texts], dtype=np.float32)
    cc_raw = np.array([
        np.nan if r.get('cyclomatic_complexity') is None else float(r['cyclomatic_complexity'])
        for r in records
    ], dtype=np.float32)
    use_cc = cc_stats.get('coverage', 0.0) >= 0.05
    if use_cc:
        cc_fill = np.nanmedian(cc_raw)
        if not np.isfinite(cc_fill):
            cc_fill = 1.0
        cc = np.where(np.isfinite(cc_raw), cc_raw, cc_fill).astype(np.float32)
        print(f"  Using cyclomatic_complexity channel (coverage={100*cc_stats['coverage']:.2f}%)")
    else:
        cc = None
        print("  Skipping cyclomatic_complexity channel (coverage < 5%)")
    file_func_keys = [(r['feature_file'], r['feature_function']) for r in records]
    churn_counts = {}
    for k in file_func_keys:
        churn_counts[k] = churn_counts.get(k, 0) + 1
    churn = np.array([float(churn_counts[k]) for k in file_func_keys], dtype=np.float32)
    feature_names = ['teacher_logit', 'student_logit', 'tfidf_logit', 'loc_logit']
    if use_cc:
        feature_names.append('cyclomatic_complexity')
    feature_names.append('churn_proxy')
    if args.use_org_metrics:
        feature_names.extend([
            'org_commits_touching_file',
            'org_distinct_authors',
            'org_top_author_fraction',
            'org_author_entropy',
            'org_ownership_friction',
            'org_cochange_weighted_degree',
            'org_cochange_unique_neighbors',
            'org_cochange_cross_module_ratio',
            'org_interface_stress',
        ])

    metrics = {'has_bugfix': {}, 'has_feature': {}}
    score_bank: dict[str, np.ndarray] = {}

    labels = [('has_bugfix', has_bugfix), ('has_feature', has_feature)]

    for label_key, y_all in labels:
        print(f"\n{'-'*60}\nLabel: {label_key}", flush=True)
        y_tv = y_all[tv_idx]
        y_te = y_all[test_idx]

        # Base channels
        loc_scores_all = fit_logreg_scores(loc[tv_idx].reshape(-1, 1), y_tv,
                                           loc.reshape(-1, 1), seed=args.seeds[0])
        tfidf_scores_all = fit_tfidf_scores(_sel(sig_texts, tv_idx), y_tv, sig_texts,
                                            seed=args.seeds[0])
        teacher_scores_all = fit_logreg_scores(teacher_embs[tv_idx], y_tv,
                                               teacher_embs, seed=args.seeds[0])
        student_scores_all = fit_logreg_scores(student_embs[tv_idx], y_tv,
                                               student_embs, seed=args.seeds[0])

        # Baselines in report
        rng = np.random.RandomState(args.seeds[0])
        rand_scores_te = rng.random(len(test_idx)).astype(np.float32)
        metrics[label_key]['random'] = eval_localization(records_test, rand_scores_te,
                                                         label_key, k_values)
        score_bank[f'random__{label_key}'] = rand_scores_te

        for name, scores_all in [
            ('loc', loc_scores_all),
            ('tfidf', tfidf_scores_all),
            ('teacher', teacher_scores_all),
            ('student', student_scores_all),
        ]:
            te_scores = scores_all[test_idx]
            metrics[label_key][name] = eval_localization(records_test, te_scores,
                                                         label_key, k_values)
            score_bank[f'{name}__{label_key}'] = te_scores

        # SE-head feature matrices
        cols = [
            teacher_scores_all,
            student_scores_all,
            tfidf_scores_all,
            loc_scores_all,
        ]
        if use_cc:
            cols.append(np.log1p(cc))
        cols.append(np.log1p(churn))
        if args.use_org_metrics:
            org_commits = np.array([float(r.get('commits_touching_file', 0.0)) for r in records],
                                   dtype=np.float32)
            org_authors = np.array([float(r.get('distinct_authors', 0.0)) for r in records],
                                   dtype=np.float32)
            org_top_frac = np.array([float(r.get('top_author_fraction', 0.0)) for r in records],
                                    dtype=np.float32)
            org_entropy = np.array([float(r.get('author_entropy', 0.0)) for r in records],
                                   dtype=np.float32)
            org_friction = np.array([float(r.get('ownership_friction', 0.0)) for r in records],
                                    dtype=np.float32)
            org_cc_wdeg = np.array([float(r.get('cochange_weighted_degree', 0.0)) for r in records],
                                   dtype=np.float32)
            org_cc_nei = np.array([float(r.get('cochange_unique_neighbors', 0.0)) for r in records],
                                  dtype=np.float32)
            org_cc_cross = np.array([float(r.get('cochange_cross_module_ratio', 0.0)) for r in records],
                                    dtype=np.float32)
            org_stress = np.array([float(r.get('interface_stress', 0.0)) for r in records],
                                  dtype=np.float32)
            cols.extend([
                np.log1p(org_commits),
                np.log1p(org_authors),
                org_top_frac,
                org_entropy,
                org_friction,
                np.log1p(org_cc_wdeg),
                np.log1p(org_cc_nei),
                org_cc_cross,
                org_stress,
            ])
        X_all = np.column_stack(cols).astype(np.float32)

        X_tv = X_all[tv_idx]
        X_te = X_all[test_idx]
        pr_tv = _sel(pr_ids, tv_idx)

        if 'linear' in args.models:
            pred = _mean_seed_scores(train_linear, X_tv, y_tv, X_te, args.seeds)
            metrics[label_key]['linear'] = eval_localization(records_test, pred,
                                                             label_key, k_values)
            metrics[label_key]['linear']['pr_auc'] = float(average_precision_score(y_te, pred))
            metrics[label_key]['linear']['auroc'] = float(roc_auc_score(y_te, pred))
            score_bank[f'linear__{label_key}'] = pred

        if 'fusion' in args.models:
            Xf_tv = X_tv[:, :4]
            Xf_te = X_te[:, :4]
            pred = _mean_seed_scores(train_linear, Xf_tv, y_tv, Xf_te, args.seeds)
            metrics[label_key]['fusion'] = eval_localization(records_test, pred,
                                                             label_key, k_values)
            metrics[label_key]['fusion']['pr_auc'] = float(average_precision_score(y_te, pred))
            metrics[label_key]['fusion']['auroc'] = float(roc_auc_score(y_te, pred))
            score_bank[f'fusion__{label_key}'] = pred

        if 'mlp' in args.models:
            pred = _mean_seed_scores(train_mlp, X_tv, y_tv, X_te, args.seeds)
            metrics[label_key]['mlp'] = eval_localization(records_test, pred,
                                                          label_key, k_values)
            metrics[label_key]['mlp']['pr_auc'] = float(average_precision_score(y_te, pred))
            metrics[label_key]['mlp']['auroc'] = float(roc_auc_score(y_te, pred))
            score_bank[f'mlp__{label_key}'] = pred

        if 'pairwise' in args.models:
            pred = _mean_seed_scores(train_pairwise, X_tv, y_tv, X_te, args.seeds,
                                     pr_ids_tv=pr_tv)
            metrics[label_key]['pairwise'] = eval_localization(records_test, pred,
                                                               label_key, k_values)
            metrics[label_key]['pairwise']['pr_auc'] = float(average_precision_score(y_te, pred))
            metrics[label_key]['pairwise']['auroc'] = float(roc_auc_score(y_te, pred))
            score_bank[f'pairwise__{label_key}'] = pred

        # Console summary
        for m in ['random', 'loc', 'tfidf', 'teacher', 'student'] + args.models:
            if m not in metrics[label_key]:
                continue
            r = metrics[label_key][m]
            vals = '  '.join(f"R@{k}={100*r.get(k, 0):.1f}%" for k in k_values)
            print(f"  {m:10s} {vals}  MRR={100*r.get('mrr', 0):.1f}%", flush=True)

    write_topk(records_test, score_bank, k_values, out_topk)

    payload = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'args': vars(args),
        'n_records': n_records,
        'n_repos': len(set(repos)),
        'split': {'train': len(train_idx), 'val': len(val_idx), 'test': len(test_idx)},
        'label_stats': {
            'has_bugfix_pos': int(has_bugfix.sum()),
            'has_feature_pos': int(has_feature.sum()),
        },
        'cc_stats': cc_stats,
        'org_stats': org_stats,
        'feature_names': feature_names,
        'metrics': metrics,
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)

    write_report(metrics, args, n_records, len(set(repos)),
                 len(train_idx), len(val_idx), len(test_idx),
                 int(has_bugfix.sum()), int(has_feature.sum()), k_values,
                 cc_stats, feature_names, out_md, out_json, out_topk,
                 title=('Experiment 4.5: Conway Signals for Region Localization'
                        if args.use_org_metrics else
                        'Experiment 4.4: SE-Head Stack for Region Localization'),
                 org_stats=org_stats if args.use_org_metrics else None)

    elapsed = (time.time() - t0) / 60.0
    print(f"\nDone in {elapsed:.1f} min")
    print(f"  Report: {out_md}")
    print(f"  Metrics: {out_json}")
    print(f"  Top-K: {out_topk}")


if __name__ == '__main__':
    main()
