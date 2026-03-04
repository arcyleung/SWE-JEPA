"""
Experiment 4.3: Region-level defect and feature-extension localization.

Extends Exp 4.1 from PR-level binary classification to within-PR function ranking.
For each test-set feature PR, all function anchors are ranked by predicted
probability; we measure Recall@K (how many true-positive functions land in top-K)
for both 'has_bugfix' and 'has_feature' labels.

Ground-truth regions: followups_function.feature_function_start/end (line ranges
of each function at the feature-PR commit, enriched from the DB at run time).

GPU-hours comparison follows Exp 4.2 methodology:
  Recall@K / GPU-minutes-to-train  (normalised efficiency)

Reuses encoding infrastructure from probe_defect_prediction.py
(encode_all, load_sigs, repo_split, SigPredictorV2, TEACHER_LAYER constants).

Usage:
    # full run (encode + probe; saves embedding cache)
    python probe_region_localization.py

    # re-run probes on cached embeddings (no GPU needed)
    python probe_region_localization.py --use-cache

    # stricter bugfix overlap threshold
    python probe_region_localization.py --use-cache --min-overlap 0.1

    # custom K values
    python probe_region_localization.py --use-cache --k 1 3 5 10

    # skip DB enrichment for line ranges
    python probe_region_localization.py --use-cache --skip-enrich
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse encoding infrastructure from Exp 4.1 — no duplication of model code.
from probe_defect_prediction import (
    SigPredictorV2,     # noqa: F401  (imported so checkpoint loads cleanly)
    capture_layer,      # noqa: F401
    encode_all,
    load_sigs,
    repo_split,
    TEACHER_PATH,
    TEACHER_LAYER,
    DEFAULT_SIGS,
    DEFAULT_CKPT,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

RESULTS_FILE  = os.path.join(os.path.dirname(__file__),
                              'docs', 'phase4_3_region_localization.md')
DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), 'followup_embs.npz')
DEFAULT_PREDICTIONS = os.path.join(
    os.path.dirname(__file__), 'docs', 'phase4_3_topk_predictions.jsonl')

PG_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'postgres_connection.yaml')

# Fixed training cost for JEPA (Exp 3.0): 33 epochs × 32 GPUs × 64 batch.
JEPA_TRAIN_GPU_MINUTES = 2016
# Approximate inference GPU-minutes for ~6.7 k sigs on 1× Qwen3-8B, batch=32.
INFERENCE_GPU_MINUTES  = 2


# ── Optional DB enrichment: line ranges ───────────────────────────────────────

def enrich_with_line_ranges(records: list[dict]) -> list[dict]:
    """
    Query followups_function for feature_function_start/end per anchor and
    attach them to each record.  Falls back gracefully if the DB is unavailable
    or if the columns don't exist (e.g. when running against a stale schema).
    """
    try:
        import pg8000.native
        cfg = yaml.safe_load(open(PG_CONFIG_FILE))
        DB  = dict(host=cfg['ip'], port=cfg.get('port', 9999),
                   user=cfg['user'], password=cfg['password'],
                   database=cfg['database'])
        conn = pg8000.native.Connection(**DB)
    except Exception as e:
        print(f"  DB unavailable ({e}); skipping line-range enrichment", flush=True)
        return records

    iids = list({r['feature_instance_id'] for r in records})
    try:
        rows = conn.run("""
            SELECT DISTINCT ON (feature_instance_id, feature_file, feature_function)
                   feature_instance_id,
                   feature_file,
                   feature_function,
                   MAX(feature_function_start) OVER (
                       PARTITION BY feature_instance_id, feature_file, feature_function
                   ) AS feature_function_start,
                   MAX(feature_function_end) OVER (
                       PARTITION BY feature_instance_id, feature_file, feature_function
                   ) AS feature_function_end
            FROM followups_function
            WHERE feature_instance_id = ANY(:iids)
        """, iids=iids)
    except Exception as e:
        print(f"  Line-range query failed ({e}); skipping enrichment", flush=True)
        conn.close()
        return records

    conn.close()

    lr_map: dict[tuple, tuple] = {}
    for r in rows:
        key = (r[0], r[1], r[2])
        lr_map[key] = (r[3], r[4])

    enriched = []
    for rec in records:
        key = (rec['feature_instance_id'], rec['feature_file'],
               rec['feature_function'])
        # Preserve any existing JSONL line ranges if DB lookup misses a row.
        start, end = lr_map.get(
            key,
            (rec.get('feature_function_start'), rec.get('feature_function_end')),
        )
        enriched.append({**rec,
                         'feature_function_start': start,
                         'feature_function_end':   end})

    n_enriched = sum(1 for r in enriched
                     if r.get('feature_function_start') is not None)
    print(f"  Enriched {n_enriched:,} / {len(enriched):,} records with line ranges",
          flush=True)
    return enriched


# ── Localization evaluation ───────────────────────────────────────────────────

def eval_localization(records: list[dict], scores: np.ndarray,
                      label_key: str, k_values: list[int]) -> dict:
    """
    Group test records by PR (feature_instance_id), rank all functions within
    each PR by descending score, and compute mean Recall@K over PRs that have
    at least one positive.

    Recall@K for a PR with n_pos positives =
        (# positives among top-min(K, N_pr) ranked functions) / n_pos

    Returns a dict:  {k: mean_recall, ..., 'n_prs': count, 'avg_pr_size': float}
    """
    pr_groups: dict[str, list[tuple[int, int]]] = {}
    for i, rec in enumerate(records):
        pr = rec['feature_instance_id']
        pr_groups.setdefault(pr, []).append((i, int(rec[label_key])))

    recall_by_k: dict[int, list[float]] = {k: [] for k in k_values}
    pr_sizes: list[int] = []

    for pr, items in pr_groups.items():
        labels = [y for (_, y) in items]
        n_pos  = sum(labels)
        if n_pos == 0:
            continue  # No positives in this PR → undefined recall

        pr_sizes.append(len(items))
        pr_idxs   = [idx for (idx, _) in items]
        pr_scores = scores[pr_idxs]

        # Rank descending by score (stable sort preserves index order on ties)
        order         = np.argsort(-pr_scores, kind='stable')
        labels_ranked = [labels[j] for j in order]

        for k in k_values:
            top_k_labels = labels_ranked[:k]
            recall_by_k[k].append(sum(top_k_labels) / n_pos)

    n_prs    = len(pr_sizes)
    avg_size = float(np.mean(pr_sizes)) if pr_sizes else 0.0

    result = {k: float(np.mean(recall_by_k[k])) if recall_by_k[k] else 0.0
              for k in k_values}
    result['n_prs']    = n_prs
    result['avg_size'] = avg_size
    return result


# ── Probe: fit on train+val, return test scores for ranking ──────────────────

def train_probe_and_score(X_tr: np.ndarray, y_tr: np.ndarray,
                           X_te: np.ndarray) -> np.ndarray:
    """
    Fit balanced LogisticRegression on (X_tr, y_tr) and return
    predict_proba[:, 1] on X_te — used as ranking scores.
    """
    sc     = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced',
                             solver='lbfgs', random_state=42)
    clf.fit(X_tr_s, y_tr)
    return clf.predict_proba(X_te_s)[:, 1]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_row(name: str, res: dict, k_values: list[int]) -> None:
    vals = '  '.join(f'R@{k}={res.get(k, 0)*100:.1f}%' for k in k_values)
    print(f"  {name:30s}  {vals}  n_prs={res.get('n_prs', 0)}", flush=True)


def _sel(arr, idx):
    return arr[idx] if isinstance(arr, np.ndarray) else [arr[i] for i in idx]


def write_topk_predictions(records_test: list[dict], score_bank: dict[str, np.ndarray],
                           k_values: list[int], out_path: str) -> None:
    """
    Write per-PR top-K function rankings (with line ranges) for each
    method/label. One JSON object per PR×method×label combination.
    """
    k_max = max(k_values)
    pr_to_indices: dict[str, list[int]] = {}
    for i, rec in enumerate(records_test):
        pr_to_indices.setdefault(rec['feature_instance_id'], []).append(i)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with open(out_path, 'w') as f:
        for score_key, scores in score_bank.items():
            method, label_key = score_key.split('__', 1)
            for pr_id, pr_idxs in pr_to_indices.items():
                pr_scores = scores[pr_idxs]
                order = np.argsort(-pr_scores, kind='stable')
                top = []
                for rank, local_j in enumerate(order[:k_max], start=1):
                    idx = pr_idxs[local_j]
                    rec = records_test[idx]
                    top.append({
                        'rank': rank,
                        'score': float(pr_scores[local_j]),
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
                    'n_functions_in_pr': len(pr_idxs),
                    'topk': top,
                }
                f.write(json.dumps(payload) + '\n')
                written += 1
    print(f"Top-K predictions written: {written:,} rows → {out_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigs-file',    default=DEFAULT_SIGS)
    ap.add_argument('--cache-file',   default=DEFAULT_CACHE)
    ap.add_argument('--student-ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--use-cache',    action='store_true',
                    help='Load cached embeddings; skip teacher/student inference')
    ap.add_argument('--skip-enrich',  action='store_true',
                    help='Skip DB query for function line-range enrichment')
    ap.add_argument('--min-overlap',  type=float, default=0.0,
                    help='Min hunk_overlap_fraction to count a bugfix (default: 0.0)')
    ap.add_argument('--k',            type=int, nargs='+', default=[1, 3, 5, 10],
                    help='Recall@K values (default: 1 3 5 10)')
    ap.add_argument('--gpu',          type=int, default=0)
    ap.add_argument('--batch-size',   type=int, default=32)
    ap.add_argument('--predictions-file', default=DEFAULT_PREDICTIONS,
                    help='JSONL output for per-PR top-K function rankings')
    args = ap.parse_args()

    t0 = time.time()
    print("SWE-JEPA Exp 4.3: Region-Level Defect & Feature Localization")
    print('=' * 60, flush=True)

    k_values = sorted(set(args.k))
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("--k values must be positive integers")

    # ── Load records ──────────────────────────────────────────────────────────
    print(f"\nLoading {args.sigs_file} …", flush=True)
    records = load_sigs(args.sigs_file, args.min_overlap)
    N = len(records)
    print(f"  {N:,} records (after min_overlap={args.min_overlap:.2f} filter)")

    if not args.skip_enrich:
        print("\nEnriching with function line ranges from DB …", flush=True)
        records = enrich_with_line_ranges(records)

    sig_texts   = [r['sig_text']            for r in records]
    repos       = [r['feature_repo']        for r in records]
    has_bugfix  = np.array([r['has_bugfix']  for r in records], dtype=np.int32)
    has_feature = np.array([r['has_feature'] for r in records], dtype=np.int32)

    print(f"\n  has_bugfix:  {has_bugfix.sum():,} pos ({100*has_bugfix.mean():.1f}%)")
    print(f"  has_feature: {has_feature.sum():,} pos ({100*has_feature.mean():.1f}%)")

    # ── Embeddings ────────────────────────────────────────────────────────────
    import torch

    if args.use_cache and os.path.exists(args.cache_file):
        print(f"\nLoading cached embeddings from {args.cache_file} …", flush=True)
        npz = np.load(args.cache_file)
        teacher_embs = npz['teacher_embs']
        student_embs = npz['student_embs']
        assert len(teacher_embs) == N, \
            f"Cache size mismatch: {len(teacher_embs)} vs {N} records"
        print(f"  teacher: {teacher_embs.shape}  student: {student_embs.shape}",
              flush=True)
    else:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available()
                              else 'cpu')
        print(f"\nEncoding on {device} …", flush=True)
        teacher_embs, student_embs = encode_all(
            sig_texts, args.student_ckpt, device, args.batch_size)
        np.savez_compressed(args.cache_file,
                            teacher_embs=teacher_embs,
                            student_embs=student_embs)
        print(f"  Embeddings cached to {args.cache_file}", flush=True)

    # ── Train/val/test split (same seeded repo split as Exp 4.1) ──────────────
    train_idx, val_idx, test_idx = repo_split(repos)
    tv_idx = train_idx + val_idx
    print(f"\nSplit: {len(train_idx):,} train  {len(val_idx):,} val  "
          f"{len(test_idx):,} test  (by repo)", flush=True)

    records_test = _sel(records, test_idx)

    # ── Shared feature matrices ───────────────────────────────────────────────
    loc_feat = np.array([len(t.splitlines()) for t in sig_texts], dtype=np.float32)

    tfidf = TfidfVectorizer(max_features=5000, analyzer='word', sublinear_tf=True)
    X_tv_tfidf = tfidf.fit_transform(_sel(sig_texts, tv_idx)).toarray()
    X_te_tfidf = tfidf.transform(_sel(sig_texts, test_idx)).toarray()

    # ── Evaluate all methods × both labels ────────────────────────────────────
    results: dict[str, dict] = {}
    score_bank: dict[str, np.ndarray] = {}
    rng = np.random.RandomState(42)

    for label_key, label_arr in [('has_bugfix', has_bugfix),
                                  ('has_feature', has_feature)]:
        print(f"\n{'─'*60}")
        print(f"Label: {label_key}")
        y_tv = label_arr[tv_idx]
        y_te = label_arr[test_idx]

        # Random baseline (empirical, seed=42)
        rand_scores = rng.random(len(test_idx)).astype(np.float32)
        score_bank[f'random__{label_key}'] = rand_scores
        lk = f'random_{label_key}'
        results[lk] = eval_localization(records_test, rand_scores, label_key,
                                         k_values)
        _print_row('Random', results[lk], k_values)

        # LOC
        loc_scores = train_probe_and_score(
            loc_feat[tv_idx].reshape(-1, 1), y_tv,
            loc_feat[test_idx].reshape(-1, 1))
        score_bank[f'loc__{label_key}'] = loc_scores
        lk = f'loc_{label_key}'
        results[lk] = eval_localization(records_test, loc_scores, label_key,
                                         k_values)
        _print_row('LOC (sig lines)', results[lk], k_values)

        # TF-IDF
        tfidf_scores = train_probe_and_score(X_tv_tfidf, y_tv, X_te_tfidf)
        score_bank[f'tfidf__{label_key}'] = tfidf_scores
        lk = f'tfidf_{label_key}'
        results[lk] = eval_localization(records_test, tfidf_scores, label_key,
                                         k_values)
        _print_row('TF-IDF (5k features)', results[lk], k_values)

        # Teacher (frozen)
        teacher_scores = train_probe_and_score(
            teacher_embs[tv_idx], y_tv, teacher_embs[test_idx])
        score_bank[f'teacher__{label_key}'] = teacher_scores
        lk = f'teacher_{label_key}'
        results[lk] = eval_localization(records_test, teacher_scores, label_key,
                                         k_values)
        _print_row('Teacher emb (frozen)', results[lk], k_values)

        # Student (JEPA)
        student_scores = train_probe_and_score(
            student_embs[tv_idx], y_tv, student_embs[test_idx])
        score_bank[f'student__{label_key}'] = student_scores
        lk = f'student_{label_key}'
        results[lk] = eval_localization(records_test, student_scores, label_key,
                                         k_values)
        _print_row('Student emb (JEPA)', results[lk], k_values)

    write_topk_predictions(records_test, score_bank, k_values,
                           args.predictions_file)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min", flush=True)

    write_report(results, N, len(train_idx), len(val_idx), len(test_idx),
                 has_bugfix, has_feature, k_values, args, len(set(repos)))
    print(f"Results written to {RESULTS_FILE}")


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(results: dict, N: int, n_train: int, n_val: int, n_test: int,
                 has_bugfix: np.ndarray, has_feature: np.ndarray,
                 k_values: list[int], args, n_repos: int) -> None:

    k10      = 10 if 10 in k_values else k_values[-1]
    k_header = ' | '.join(f'R@{k}' for k in k_values)
    k_sep    = '|'.join(['-----'] * len(k_values))

    inf_gpu    = INFERENCE_GPU_MINUTES
    jepa_total = JEPA_TRAIN_GPU_MINUTES + inf_gpu

    def pct(key, k):
        return results.get(key, {}).get(k, 0.0) * 100

    def eff(key, gpu_min):
        r = pct(key, k10)
        return f'{r / gpu_min * 1000:.2f}' if gpu_min > 0 else '—'

    def md_row(label, key, gpu_min, bold=False):
        vals = ' | '.join(f'{pct(key, k):.1f}%' for k in k_values)
        b    = '**' if bold else ''
        return (f'| {b}{label}{b} | {vals} | {gpu_min:,} | '
                f'{eff(key, gpu_min)} |')

    def bugfix_row(label, key, gpu_min, bold=False):
        return md_row(label, f'{key}_has_bugfix', gpu_min, bold)

    def feature_row(label, key, gpu_min, bold=False):
        return md_row(label, f'{key}_has_feature', gpu_min, bold)

    sr_bug = results.get('student_has_bugfix',  {})
    sr_fea = results.get('student_has_feature', {})
    tr_bug = results.get('teacher_has_bugfix',  {})
    tr_fea = results.get('teacher_has_feature', {})
    ti_bug = results.get('tfidf_has_bugfix',    {})
    ti_fea = results.get('tfidf_has_feature',   {})

    delta_bug_vs_teacher  = (sr_bug.get(k10, 0) - tr_bug.get(k10, 0)) * 100
    delta_bug_vs_tfidf    = (sr_bug.get(k10, 0) - ti_bug.get(k10, 0)) * 100
    delta_feat_vs_teacher = (sr_fea.get(k10, 0) - tr_fea.get(k10, 0)) * 100
    delta_feat_vs_tfidf   = (sr_fea.get(k10, 0) - ti_fea.get(k10, 0)) * 100

    n_bugfix_pos  = int(has_bugfix.sum())
    n_feature_pos = int(has_feature.sum())

    avg_pr_size_bug  = sr_bug.get('avg_size', 0)
    avg_pr_size_feat = sr_fea.get('avg_size', 0)

    report = f"""# Experiment 4.3: Region-Level Defect & Feature Localization

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Extends**: Exp 4.1 (PR-level binary classification → within-PR function ranking)
**Data**: followups_function — {N:,} function anchors, {n_repos} repos
**Labels**: has_bugfix={n_bugfix_pos:,} pos ({100*n_bugfix_pos/N:.1f}%) | has_feature={n_feature_pos:,} pos ({100*n_feature_pos/N:.1f}%)
**Split**: {n_train:,} train / {n_val:,} val / {n_test:,} test (by repo, 80/10/10)
**Bugfix min overlap**: ≥ {args.min_overlap:.2f}

## Task description

For each test-set feature PR, all function anchors from that PR are ranked by
their predicted probability of being a bugfix/feature-extension target.

- **Predicted region**: top-K ranked functions and their line ranges
  (`feature_function_start`–`feature_function_end`) at the feature-PR commit.
- **Ground-truth region**: functions where `has_bugfix = 1` (or `has_feature = 1`).
- **Metric**: Mean Recall@K over PRs with ≥1 positive function.

Recall@K = (# true-positive functions ranked in top-K) / (# total true positives
in the PR), averaged over PRs. K is capped at the PR's function count, so
Recall@K = 1.0 when K ≥ n_functions.

## Setup

| Parameter | Value |
|-----------|-------|
| Teacher | Qwen3-8B-base, layer {TEACHER_LAYER} (frozen) |
| Student ckpt | {os.path.basename(args.student_ckpt)} |
| Embeddings | 4096-dim, mean-pool over sig tokens |
| Probe | LogisticRegression(C=1.0, class_weight='balanced') fitted on train+val |
| Ranking score | predict_proba[:, 1] |
| TF-IDF | 5000 features, word-level, sublinear_tf |
| Bugfix min hunk overlap | {args.min_overlap:.2f} |

---

## Task A: Bugfix Localization (`has_bugfix`)

Ranking functions within a feature PR by predicted bugfix-proneness.
Ground truth: functions touched in a later bugfix PR with hunk_overlap ≥ {args.min_overlap:.2f}.

- PRs with ≥1 bugfix-positive function (test set): **{sr_bug.get('n_prs', '?')}**
- Average functions per such PR: **{avg_pr_size_bug:.1f}**

| Method | {k_header} | GPU-min | R@{k10}/1k GPU-min |
|--------|{k_sep}|---------|---------------------|
{bugfix_row('Random', 'random', 0)}
{bugfix_row('LOC (sig lines)', 'loc', 0)}
{bugfix_row('TF-IDF (5k)', 'tfidf', 0)}
{bugfix_row('Teacher (frozen)', 'teacher', inf_gpu)}
{bugfix_row('Student JEPA', 'student', jepa_total, bold=True)}

Student vs Teacher  Δ R@{k10}: **{delta_bug_vs_teacher:+.1f} pp**
Student vs TF-IDF   Δ R@{k10}: **{delta_bug_vs_tfidf:+.1f} pp**

---

## Task B: Feature-Extension Localization (`has_feature`)

Ranking functions by predicted likelihood of being refactored or extended in a
later feature PR.

- PRs with ≥1 feature-positive function (test set): **{sr_fea.get('n_prs', '?')}**
- Average functions per such PR: **{avg_pr_size_feat:.1f}**

| Method | {k_header} | GPU-min | R@{k10}/1k GPU-min |
|--------|{k_sep}|---------|---------------------|
{feature_row('Random', 'random', 0)}
{feature_row('LOC (sig lines)', 'loc', 0)}
{feature_row('TF-IDF (5k)', 'tfidf', 0)}
{feature_row('Teacher (frozen)', 'teacher', inf_gpu)}
{feature_row('Student JEPA', 'student', jepa_total, bold=True)}

Student vs Teacher  Δ R@{k10}: **{delta_feat_vs_teacher:+.1f} pp**
Student vs TF-IDF   Δ R@{k10}: **{delta_feat_vs_tfidf:+.1f} pp**

---

## GPU-hours comparison (Exp 4.2 methodology)

Training cost is fixed per method; efficiency = R@{k10}(bugfix) per 1 000 GPU-min.

| Method | Training GPU-min | R@{k10} (bugfix) | R@{k10}/1k GPU-min |
|--------|-----------------|-----------------|---------------------|
| Random (no model) | 0 | {pct('random_has_bugfix', k10):.1f}% | — |
| TF-IDF (no model) | 0 | {pct('tfidf_has_bugfix', k10):.1f}% | — |
| Teacher frozen | {inf_gpu} (inf only) | {pct('teacher_has_bugfix', k10):.1f}% | {eff('teacher_has_bugfix', inf_gpu)} |
| **Student JEPA** | **{jepa_total:,} (train+inf)** | **{pct('student_has_bugfix', k10):.1f}%** | **{eff('student_has_bugfix', jepa_total)}** |

*JEPA training GPU-min: {JEPA_TRAIN_GPU_MINUTES} (Exp 3.0, 33 epochs, 32 GPUs × 64 batch).*
*Teacher inference GPU-min: ~{inf_gpu} (Qwen3-8B, 1 GPU, batch=32, {N:,} signatures).*

---

## Interpretation

### Bugfix localization

The localization task tests whether JEPA representations encode spatial information
about *which* functions within a PR are defect-prone — not just whether *any*
function in the PR will need a bugfix followup (Exp 4.1 binary task).

Exp 4.1 result: Student AUROC 0.6003 vs Teacher 0.5673 (+3.3 pp) on PR-level
binary classification.

The within-PR ranking (Recall@K) is a harder evaluation: the probe must correctly
*order* functions relative to each other, not just output a global threshold.
A probe that uniformly shifts all scores upward gains nothing in recall@K; the
signal must come from relative differences between functions in the same PR.

### Feature-extension localization

Feature-extension localization tests whether the JEPA body-structure prediction
objective captures extensibility signals — functions whose signature design will
require future refactoring or API extension — distinct from bugfix-proneness.

Feature followups have different characteristics from bugfixes: they touch
deliberately designed extension points (abstract methods, configurable parameters,
plugin interfaces). A model that only learns defect complexity proxies (cyclomatic
complexity, LOC) should score lower on feature localization than on bugfix
localization.

### Ground-truth region format

For each predicted function, the matched line range in the feature-PR source file is:
```
feature_file : feature_function_start – feature_function_end
```
These line ranges are stored in `followups_function.feature_function_start/end`
and enriched onto each record at runtime (see `--skip-enrich` flag to bypass).

---

## Files

| File | Contents |
|------|----------|
| `probe_region_localization.py` | This experiment script |
| `probe_defect_prediction.py`   | Exp 4.1 (reused for encode_all, load_sigs) |
| `followup_sigs.jsonl`          | Input: per-function anchors with labels |
| `followup_embs.npz`            | Cached embeddings (teacher + student) |
| `{os.path.basename(args.predictions_file)}` | Per-PR top-K ranked functions with line ranges |
| `docs/phase4_3_region_localization.md` | This report |
"""
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        f.write(report)


if __name__ == '__main__':
    main()
