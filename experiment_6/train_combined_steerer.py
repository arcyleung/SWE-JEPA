#!/usr/bin/env python3
"""
Train a combined Conway + followup steerer model (Experiment 6.2).

Extends the 61 Conway features with 8 function-level followup features from
postgres, trains acceptance + refactor heads, and embeds cluster-based hints
in the output model JSON.

Inputs:
    data/conway_patch_features_v2_100k.jsonl  — 61 Conway features
    postgres followups_function table          — function-level followup stats
    data/phase6_2/function_followup_clusters/super_clusters.json — cluster hints

Outputs:
    data/phase6_2/pr_steerer_model_combined.json — 69-feature model + cluster_hints
    data/phase6_2/combined_steerer_metrics.json  — CV metrics

Usage:
    python train_combined_steerer.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import pg8000.native
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

DEFAULT_CONWAY = os.path.join(PROJECT_ROOT, "data", "conway_patch_features_v2_100k.jsonl")
DEFAULT_PG_CONFIG = os.path.join(PROJECT_ROOT, "postgres_connection.yaml")
DEFAULT_CLUSTERS = os.path.join(PROJECT_ROOT, "data", "phase6_2", "function_followup_clusters", "super_clusters.json")
DEFAULT_MODEL_OUT = os.path.join(PROJECT_ROOT, "data", "phase6_2", "pr_steerer_model_combined.json")
DEFAULT_METRICS_OUT = os.path.join(PROJECT_ROOT, "data", "phase6_2", "combined_steerer_metrics.json")

EXCLUDE_FEATURES = frozenset({
    "accepted", "review_friction", "n_review_threads", "n_comments", "is_draft",
})

LOG1P_FEATURES = frozenset({
    "changed_files", "additions", "deletions", "n_langs", "n_files",
    "cross_module_spread", "imp_total_new", "imp_stdlib", "imp_relative",
    "imp_external", "imp_ext_network", "imp_ext_security", "imp_ext_infra",
    "http_missing_timeout", "n_http_calls", "new_func_defs",
    "trust_boundary_crossings",
    # followup features that benefit from log1p
    "n_bugfix", "n_feature", "n_maintenance", "n_docs", "n_total",
    "n_distinct_functions",
})

FOLLOWUP_FEATURE_NAMES = [
    "n_bugfix", "n_feature", "n_maintenance", "n_docs", "n_total",
    "max_overlap", "n_distinct_functions", "has_bugfix",
]


def _load_conway(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_followup_stats(pg_config: str) -> dict[str, dict[str, float]]:
    """Aggregate followups_function by instance_id."""
    cfg = yaml.safe_load(open(pg_config))
    conn = pg8000.native.Connection(
        host=cfg["ip"], port=int(cfg["port"]),
        user=cfg["user"], password=cfg["password"], database=cfg["database"],
    )
    rows = conn.run("""
        SELECT feature_instance_id,
               SUM(CASE WHEN followup_category = 'bugfix' THEN 1 ELSE 0 END),
               SUM(CASE WHEN followup_category = 'feature' THEN 1 ELSE 0 END),
               SUM(CASE WHEN followup_category = 'maintenance' THEN 1 ELSE 0 END),
               SUM(CASE WHEN followup_category = 'docs' THEN 1 ELSE 0 END),
               COUNT(*),
               MAX(hunk_overlap_fraction),
               COUNT(DISTINCT feature_function)
        FROM followups_function
        GROUP BY feature_instance_id
    """)
    conn.close()
    stats: dict[str, dict[str, float]] = {}
    for row in rows:
        stats[str(row[0])] = {
            "n_bugfix": float(row[1]),
            "n_feature": float(row[2]),
            "n_maintenance": float(row[3]),
            "n_docs": float(row[4]),
            "n_total": float(row[5]),
            "max_overlap": float(row[6] or 0),
            "n_distinct_functions": float(row[7]),
            "has_bugfix": float(1 if row[1] > 0 else 0),
        }
    return stats


def _feature_names_from_row(row: dict) -> list[str]:
    names = []
    for key, value in row.items():
        if key in EXCLUDE_FEATURES:
            continue
        if key in ("repo", "instance_id", "pull_number", "primary_lang"):
            continue
        if isinstance(value, (int, float)):
            names.append(key)
    return sorted(names)


def _build_vector(row: dict, feature_names: list[str], followup: dict[str, float]) -> list[float]:
    vals = []
    for name in feature_names:
        if name in FOLLOWUP_FEATURE_NAMES:
            value = float(followup.get(name, 0.0) or 0.0)
        else:
            value = float(row.get(name, 0.0) or 0.0)
        if name in LOG1P_FEATURES:
            value = float(np.log1p(max(value, 0.0)))
        vals.append(value)
    return vals


def _cv_scores(X, y, groups, n_splits=5, seed=42):
    if len(np.unique(groups)) < n_splits:
        n_splits = max(2, len(np.unique(groups)))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return []
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for tr, te in gkf.split(X, groups=groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            folds.append({"auroc": 0.5, "pr_auc": float(np.mean(y[te]))})
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                 solver="lbfgs", random_state=seed)
        clf.fit(Xtr, y[tr])
        pred = clf.predict_proba(Xte)[:, 1]
        folds.append({
            "auroc": float(roc_auc_score(y[te], pred)),
            "pr_auc": float(average_precision_score(y[te], pred)),
        })
    return folds


def _summarize_folds(folds):
    if not folds:
        return {"mean_auroc": 0.5, "mean_pr_auc": 0.0, "std_auroc": 0.0}
    return {
        "mean_auroc": float(np.mean([f["auroc"] for f in folds])),
        "mean_pr_auc": float(np.mean([f["pr_auc"] for f in folds])),
        "std_auroc": float(np.std([f["auroc"] for f in folds])),
        "folds": folds,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conway-features", default=DEFAULT_CONWAY)
    ap.add_argument("--pg-config", default=DEFAULT_PG_CONFIG)
    ap.add_argument("--clusters", default=DEFAULT_CLUSTERS)
    ap.add_argument("--model-out", default=DEFAULT_MODEL_OUT)
    ap.add_argument("--metrics-out", default=DEFAULT_METRICS_OUT)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load Conway features
    conway_rows = _load_conway(args.conway_features)
    print(f"Conway rows: {len(conway_rows):,}")

    # Get Conway feature names
    conway_feature_names = _feature_names_from_row(conway_rows[0])
    # Combined feature names = Conway + followup
    all_feature_names = conway_feature_names + [f for f in FOLLOWUP_FEATURE_NAMES
                                                 if f not in conway_feature_names]
    print(f"Conway features: {len(conway_feature_names)}, combined: {len(all_feature_names)}")

    # Load followup stats
    print("Loading followup stats from postgres...")
    followup_stats = _load_followup_stats(args.pg_config)
    print(f"PRs with followup data: {len(followup_stats):,}")

    # Build feature matrix
    X_rows = []
    y_acc = []
    y_ref = []
    groups = []
    instance_ids = []

    for row in conway_rows:
        iid = str(row.get("instance_id", ""))
        repo = row.get("repo", "")
        if not iid or not repo:
            continue
        followup = followup_stats.get(iid, {})
        vec = _build_vector(row, all_feature_names, followup)
        X_rows.append(vec)
        y_acc.append(int(row.get("accepted", 0) or 0))
        y_ref.append(int(row.get("review_friction", 0) or 0))
        groups.append(repo)
        instance_ids.append(iid)

    X = np.array(X_rows, dtype=np.float32)
    y_acc = np.array(y_acc, dtype=np.int32)
    y_ref = np.array(y_ref, dtype=np.int32)
    groups = np.array(groups)

    n_with_followup = sum(1 for iid in instance_ids if iid in followup_stats)
    print(f"Samples: {len(X):,}, with followup data: {n_with_followup:,}")
    print(f"Acceptance rate: {y_acc.mean():.3f}, refactor rate: {y_ref.mean():.3f}")

    # ── CV evaluation ─────────────────────────────────────────────────────
    # Combined model
    print("\n=== Combined (Conway + followup) ===")
    t0 = time.time()
    acc_folds = _cv_scores(X, y_acc, groups, args.cv_folds, args.seed)
    ref_folds = _cv_scores(X, y_ref, groups, args.cv_folds, args.seed + 1)
    acc_summary = _summarize_folds(acc_folds)
    ref_summary = _summarize_folds(ref_folds)
    print(f"  Acceptance AUROC: {acc_summary['mean_auroc']:.4f} (±{acc_summary['std_auroc']:.4f})")
    print(f"  Refactor  AUROC: {ref_summary['mean_auroc']:.4f} (±{ref_summary['std_auroc']:.4f})")
    print(f"  Time: {time.time() - t0:.1f}s")

    # Conway-only baseline for comparison
    n_conway = len(conway_feature_names)
    X_conway = X[:, :n_conway]
    print("\n=== Conway-only baseline ===")
    t0 = time.time()
    acc_folds_base = _cv_scores(X_conway, y_acc, groups, args.cv_folds, args.seed)
    ref_folds_base = _cv_scores(X_conway, y_ref, groups, args.cv_folds, args.seed + 1)
    acc_base = _summarize_folds(acc_folds_base)
    ref_base = _summarize_folds(ref_folds_base)
    print(f"  Acceptance AUROC: {acc_base['mean_auroc']:.4f} (±{acc_base['std_auroc']:.4f})")
    print(f"  Refactor  AUROC: {ref_base['mean_auroc']:.4f} (±{ref_base['std_auroc']:.4f})")
    print(f"  Time: {time.time() - t0:.1f}s")

    delta_acc = acc_summary["mean_auroc"] - acc_base["mean_auroc"]
    delta_ref = ref_summary["mean_auroc"] - ref_base["mean_auroc"]
    print(f"\n  Δ acceptance AUROC: {delta_acc:+.4f}")
    print(f"  Δ refactor  AUROC: {delta_ref:+.4f}")

    # ── Fit final model on all data ───────────────────────────────────────
    print("\nFitting final model on all data...")
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    clf_acc = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                 solver="lbfgs", random_state=args.seed)
    clf_acc.fit(Xs, y_acc)

    clf_ref = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                  solver="lbfgs", random_state=args.seed + 1)
    clf_ref.fit(Xs, y_ref)

    # Load cluster hints
    cluster_hints = {}
    if os.path.exists(args.clusters):
        with open(args.clusters) as f:
            clusters = json.load(f)
        for cid, info in clusters.items():
            cluster_hints[cid] = {
                "label": info.get("label", ""),
                "risk_tier": info.get("risk_tier", "MED"),
                "bugfix_rate": info.get("bugfix_rate", 0.5),
                "acceptance_rate": info.get("acceptance_rate", 0.8),
                "top_features": info.get("top_features", [])[:3],
                "count": info.get("count", 0),
            }
        print(f"Loaded {len(cluster_hints)} cluster hints")

    # ── Save model ────────────────────────────────────────────────────────
    model = {
        "version": "phase6_2_combined_v1",
        "feature_source": "conway_patch_features",
        "features": all_feature_names,
        "log1p_features": sorted(f for f in LOG1P_FEATURES if f in all_feature_names),
        "followup_features": FOLLOWUP_FEATURE_NAMES,
        "refactor_target": "review_friction",
        "acceptance": {
            "scaler_mean": sc.mean_.tolist(),
            "scaler_scale": sc.scale_.tolist(),
            "coef": clf_acc.coef_[0].tolist(),
            "intercept": float(clf_acc.intercept_[0]),
        },
        "refactor": {
            "scaler_mean": sc.mean_.tolist(),
            "scaler_scale": sc.scale_.tolist(),
            "coef": clf_ref.coef_[0].tolist(),
            "intercept": float(clf_ref.intercept_[0]),
        },
        "cluster_hints": cluster_hints,
    }

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    with open(args.model_out, "w") as f:
        json.dump(model, f, indent=2)
    print(f"Saved model → {args.model_out} ({len(all_feature_names)} features)")

    # ── Save metrics ──────────────────────────────────────────────────────
    metrics = {
        "n_samples": len(X),
        "n_features_combined": len(all_feature_names),
        "n_features_conway": n_conway,
        "n_with_followup": n_with_followup,
        "combined": {
            "acceptance": acc_summary,
            "refactor": ref_summary,
        },
        "conway_only": {
            "acceptance": acc_base,
            "refactor": ref_base,
        },
        "delta": {
            "acceptance_auroc": delta_acc,
            "refactor_auroc": delta_ref,
        },
    }
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics → {args.metrics_out}")

    print(f"\n{'='*60}")
    gate = acc_summary["mean_auroc"] >= 0.71
    print(f"Gate: acceptance AUROC {acc_summary['mean_auroc']:.4f} >= 0.71 → {'PASS' if gate else 'FAIL'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
