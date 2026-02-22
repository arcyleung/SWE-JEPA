"""
Experiment 0.3: Linear probing for structural properties.

For each teacher model, trains a linear probe (Ridge regression or
LogisticRegression) on frozen mean-pooled hidden states to predict each
static property stored in function_static_props.

Outputs a markdown table of (property × model) → R² or balanced accuracy,
written to docs/phase0_3_linear_probing.md.

Static properties (from extract_static_props.py):
  Regression  : loc, cyclomatic_complexity, n_branches, n_loops,
                n_returns, n_api_calls, n_args, pr_churn (Kim et al. proxy)
  Classification: return_type_cat (4-class), has_side_effects (binary),
                  has_decorators (binary), has_docstring (binary)

pr_churn is computed on-the-fly as the number of distinct instance_ids
(i.e. PRs) that touched the same (feature_file, feature_function) pair —
a proxy for Kim et al.'s churn / developer-count metrics.

Usage:
    python linear_probe.py
    python linear_probe.py --models Qwen2.5-Coder-3B Qwen3-8B-base
    python linear_probe.py --folds 10 --no-log-transform
"""

import argparse
import os

import numpy as np
import yaml
import pg8000.native
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

PG_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'postgres_connection.yaml')
RESULTS_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'docs', 'phase0_3_linear_probing.md')

_pg_cfg = yaml.safe_load(open(PG_CONFIG_FILE))
DB = dict(
    host=_pg_cfg['ip'],
    port=_pg_cfg.get('port') or 9999,
    user=_pg_cfg['user'],
    password=_pg_cfg['password'],
    database=_pg_cfg['database'],
)

ALL_MODELS = [
    'Qwen2.5-Coder-3B',
    'Qwen2.5-Coder-3B-Instruct',
    'Qwen3-8B',
    'Qwen3-8B-base',
]

# Properties and their probe type
REGRESSION_PROPS = [
    'loc', 'cyclomatic_complexity', 'n_branches', 'n_loops',
    'n_returns', 'n_api_calls', 'n_args', 'pr_churn',
]
CLASSIFICATION_PROPS = [
    'return_type_cat',    # 4-class
    'has_side_effects',   # binary
    'has_decorators',     # binary
    'has_docstring',      # binary
]

# Log-transform these skewed regression targets before probing
LOG_TRANSFORM = {'loc', 'cyclomatic_complexity', 'n_api_calls', 'pr_churn'}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(model_name: str) -> tuple[np.ndarray, dict]:
    """
    Load embeddings + static props for one model.
    Returns (X, props) where X is (N, D) float32 and props is {name: array}.
    Also computes pr_churn: number of distinct PRs touching each (file, func).
    """
    print(f"  Loading embeddings for {model_name} …", flush=True)
    conn = pg8000.native.Connection(**DB)

    rows = conn.run("""
        SELECT
            fe.embedding,
            sp.loc,
            sp.cyclomatic_complexity,
            sp.n_branches,
            sp.n_loops,
            sp.n_returns,
            sp.n_api_calls,
            sp.n_args,
            sp.return_type_cat,
            sp.has_side_effects,
            sp.has_decorators,
            sp.has_docstring,
            fe.feature_file,
            fe.feature_function
        FROM function_embeddings fe
        JOIN function_static_props sp
          ON fe.instance_id = sp.instance_id
         AND fe.feature_file = sp.feature_file
         AND fe.feature_function = sp.feature_function
        WHERE fe.model_name = :model
        ORDER BY fe.id
    """, model=model_name)

    # Compute pr_churn: distinct PRs per (feature_file, feature_function)
    churn_rows = conn.run("""
        SELECT feature_file, feature_function,
               COUNT(DISTINCT instance_id) AS n_prs
        FROM function_embeddings
        WHERE model_name = :model
        GROUP BY feature_file, feature_function
    """, model=model_name)
    conn.close()

    churn_map = {(r[0], r[1]): r[2] for r in churn_rows}

    print(f"  Assembling {len(rows)} rows …", flush=True)
    X = np.array([r[0] for r in rows], dtype=np.float32)  # (N, D)

    props = {
        'loc':                   np.array([r[1]  for r in rows], dtype=np.float32),
        'cyclomatic_complexity': np.array([r[2]  for r in rows], dtype=np.float32),
        'n_branches':            np.array([r[3]  for r in rows], dtype=np.float32),
        'n_loops':               np.array([r[4]  for r in rows], dtype=np.float32),
        'n_returns':             np.array([r[5]  for r in rows], dtype=np.float32),
        'n_api_calls':           np.array([r[6]  for r in rows], dtype=np.float32),
        'n_args':                np.array([r[7]  for r in rows], dtype=np.float32),
        'return_type_cat':       np.array([r[8]  for r in rows]),
        'has_side_effects':      np.array([r[9]  for r in rows], dtype=np.float32),
        'has_decorators':        np.array([r[10] for r in rows], dtype=np.float32),
        'has_docstring':         np.array([r[11] for r in rows], dtype=np.float32),
        'pr_churn':              np.array(
            [churn_map.get((r[12], r[13]), 1) for r in rows], dtype=np.float32
        ),
    }
    return X, props


# ── Probing ────────────────────────────────────────────────────────────────────

def probe_regression(X: np.ndarray, y: np.ndarray,
                     prop: str, folds: int, log_transform: bool) -> float:
    """Return mean R² across k folds."""
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]

    if log_transform and prop in LOG_TRANSFORM:
        y = np.log1p(y)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    cv    = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_sc, y, cv=cv, scoring='r2')
    return float(scores.mean())


def probe_classification(X: np.ndarray, y: np.ndarray,
                         folds: int) -> float:
    """Return mean balanced accuracy across k folds."""
    mask = y != None  # noqa: E711  (handles None from postgres)
    mask &= np.array([v is not None for v in y])
    X, y = X[mask], y[mask]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model  = LogisticRegression(
        max_iter=1000, C=1.0, solver='lbfgs',
        class_weight='balanced',
    )
    cv     = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_sc, y_enc, cv=cv,
                             scoring='balanced_accuracy')
    return float(scores.mean())


def run_model(model_name: str, folds: int, log_transform: bool) -> dict[str, float]:
    """Run all probes for one model. Returns {prop: score}."""
    X, props = load_data(model_name)

    # L2-normalise (embeddings should already be near-normalised from FAISS,
    # but enforce it for numerical stability)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    X = X / norms

    results = {}

    for prop in REGRESSION_PROPS:
        y = props[prop]
        score = probe_regression(X, y, prop, folds, log_transform)
        print(f"    {prop:<28} R²  = {score:+.4f}", flush=True)
        results[prop] = score

    for prop in CLASSIFICATION_PROPS:
        y = props[prop]
        score = probe_classification(X, y, folds)
        print(f"    {prop:<28} BAcc= {score:.4f}", flush=True)
        results[prop] = score

    return results


# ── Report ─────────────────────────────────────────────────────────────────────

_PROP_META = {
    # name: (display_name, metric_label, Kim et al. reference)
    'loc':                   ('LOC',                  'R²',   'S1'),
    'cyclomatic_complexity': ('Cyclomatic complexity', 'R²',   'C3'),
    'n_branches':            ('# branches',           'R²',   '—'),
    'n_loops':               ('# loops',              'R²',   '—'),
    'n_returns':             ('# returns',            'R²',   '—'),
    'n_api_calls':           ('# API calls',          'R²',   'C5.x proxy'),
    'n_args':                ('# arguments',          'R²',   '—'),
    'pr_churn':              ('PR churn (# PRs)',      'R²',   'Kim et al. churn proxy'),
    'return_type_cat':       ('Return type category',  'BAcc', '—'),
    'has_side_effects':      ('Has side effects',      'BAcc', '—'),
    'has_decorators':        ('Has decorators',        'BAcc', '—'),
    'has_docstring':         ('Has docstring',         'BAcc', '—'),
}


def write_report(all_results: dict[str, dict[str, float]], folds: int,
                 log_transform: bool, models: list[str]):
    """Write markdown report to RESULTS_FILE."""
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    short = {m: m.replace('Qwen2.5-', '').replace('Qwen3-', 'Q3-') for m in models}

    lines = [
        '# Experiment 0.3: Linear Probing for Structural Properties',
        '',
        f'**Date**: 2026-02-22  ',
        f'**CV folds**: {folds}  ',
        f'**Log-transform applied to**: {", ".join(sorted(LOG_TRANSFORM)) if log_transform else "none"}  ',
        f'**Embeddings**: mean-pooled hidden states, layer 18, L2-normalised  ',
        f'**Probe**: Ridge regression (R²) / LogisticRegression (balanced accuracy)',
        '',
        '---',
        '',
        '## Results',
        '',
    ]

    # Table header
    col = 22
    mw  = 14
    header = f"| {'Property':<{col}} | {'Metric':<6} | {'Kim ref':<14} |"
    sep    = f"| {'-'*col} | {'-'*6} | {'-'*14} |"
    for m in models:
        header += f" {short[m]:>{mw}} |"
        sep    += f" {'-'*mw} |"
    lines += [header, sep]

    for prop, (display, metric, kim_ref) in _PROP_META.items():
        row = f"| {display:<{col}} | {metric:<6} | {kim_ref:<14} |"
        for m in models:
            score = all_results.get(m, {}).get(prop)
            if score is None:
                row += f" {'—':>{mw}} |"
            elif metric == 'R²':
                row += f" {score:>{mw}.4f} |"
            else:
                row += f" {score:>{mw}.4f} |"
        lines.append(row)

    lines += [
        '',
        '---',
        '',
        '## Interpretation',
        '',
        '**R²** (regression): fraction of variance in the property explained by a linear',
        'function of the embedding. R² ≈ 0 = not linearly encoded; R² ≈ 1 = perfectly',
        'linearly encoded. Negative R² means the probe is worse than predicting the mean.',
        '',
        '**Balanced accuracy** (classification): mean per-class recall across all classes.',
        'Random baseline is 1/n_classes (0.25 for return_type_cat, 0.50 for binary props).',
        '',
        '**pr_churn** counts how many distinct PRs in our corpus touched the same',
        '(file, function) pair — a proxy for Kim et al.\'s churn and developer-count',
        'metrics, which were among the strongest predictors of refactoring investment.',
        '',
        '---',
        '',
        '## Notes',
        '',
        '- Log-transform applied to skewed targets before probing: `log1p(y)`.',
        '- All embeddings L2-normalised before probing.',
        '- All features StandardScaler-normalised for the probe.',
        f'- {folds}-fold cross-validation; mean score reported.',
    ]

    with open(RESULTS_FILE, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"\nReport written to {RESULTS_FILE}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=ALL_MODELS,
                    help='Model names to probe (default: all 4)')
    ap.add_argument('--folds', type=int, default=5,
                    help='CV folds (default 5)')
    ap.add_argument('--no-log-transform', action='store_true',
                    help='Disable log1p transform on skewed regression targets')
    args = ap.parse_args()

    log_transform = not args.no_log_transform
    all_results: dict[str, dict[str, float]] = {}

    for model in args.models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model}")
        print('='*60)
        all_results[model] = run_model(model, args.folds, log_transform)

    # Print summary table to stdout
    print(f"\n\n{'='*60}")
    print("SUMMARY  (R² for regression, BAcc for classification)")
    print('='*60)
    short = {m: m.replace('Qwen2.5-', '').replace('Qwen3-', 'Q3-') for m in args.models}
    header = f"{'Property':<28}" + "".join(f"{short[m]:>18}" for m in args.models)
    print(header)
    print('-' * len(header))
    for prop, (display, metric, _) in _PROP_META.items():
        row = f"{display:<28}"
        for m in args.models:
            score = all_results.get(m, {}).get(prop)
            row += f"{score:>18.4f}" if score is not None else f"{'—':>18}"
        print(row)

    write_report(all_results, args.folds, log_transform, args.models)


if __name__ == '__main__':
    main()
