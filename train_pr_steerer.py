#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))


def _default_existing(*paths: str) -> str:
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


DEFAULT_DATA = _default_existing(
    os.path.join(ROOT, "data", "conway_patch_features_100k.jsonl"),
    os.path.join(ROOT, "data", "conway_patch_features.jsonl"),
)
DEFAULT_LABELS = _default_existing(
    os.path.join(ROOT, "data", "phase5_1_pr_mdp_dataset_v51_100k_conway.jsonl"),
    os.path.join(ROOT, "data", "phase5_1_pr_mdp_dataset_v51_conway.jsonl"),
)
DEFAULT_MODEL = os.path.join(ROOT, "data", "phase4_7_pr_steerer_model.json")
DEFAULT_METRICS = os.path.join(ROOT, "data", "phase4_7_pr_steerer_metrics.json")

EXCLUDE_FEATURES = frozenset(
    {
        "accepted",
        "review_friction",
        "n_review_threads",
        "n_comments",
        "is_draft",
    }
)

LOG1P_FEATURES = frozenset(
    {
        "changed_files",
        "additions",
        "deletions",
        "n_langs",
        "n_files",
        "cross_module_spread",
        "imp_total_new",
        "imp_stdlib",
        "imp_relative",
        "imp_external",
        "imp_ext_network",
        "imp_ext_security",
        "imp_ext_infra",
        "http_missing_timeout",
        "n_http_calls",
        "new_func_defs",
        "trust_boundary_crossings",
    }
)


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


def _x(row: dict, feature_names: list[str]) -> np.ndarray:
    vals = []
    for name in feature_names:
        value = float(row.get(name, 0.0) or 0.0)
        if name in LOG1P_FEATURES:
            value = float(np.log1p(max(value, 0.0)))
        vals.append(value)
    return np.asarray(vals, dtype=np.float32)


def _load_refactor_labels(path: str | None) -> tuple[dict[tuple[str, int], int], str]:
    if not path or not os.path.exists(path):
        return {}, "review_friction"
    labels: dict[tuple[str, int], int] = {}
    with open(path) as f:
        for ln in f:
            if not ln.strip():
                continue
            row = json.loads(ln)
            repo = row.get("repo")
            pull = int(row.get("pull_number") or 0)
            if not repo or pull <= 0:
                continue
            labels[(repo, pull)] = int(row.get("s_t1", {}).get("refactor_requested", 0) or 0)
    return labels, "refactor_requested"


def _load(path: str, labels_path: str | None, refactor_target: str):
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    if not rows:
        raise ValueError(f"no rows found in {path}")

    feature_names = _feature_names_from_row(rows[0])
    label_map, label_source = _load_refactor_labels(labels_path)

    X = np.stack([_x(r, feature_names) for r in rows], axis=0)
    y_acc = np.asarray([int(r.get("accepted", 0) or 0) for r in rows], dtype=np.int32)

    y_ref_list = []
    matched_ref = 0
    for r in rows:
        key = (r.get("repo"), int(r.get("pull_number") or 0))
        if refactor_target == "review_friction":
            y_ref_list.append(int(r.get("review_friction", 0) or 0))
            continue
        if key in label_map:
            y_ref_list.append(int(label_map[key]))
            matched_ref += 1
        else:
            y_ref_list.append(int(r.get("review_friction", 0) or 0))
    y_ref = np.asarray(y_ref_list, dtype=np.int32)
    groups = np.asarray([r["repo"] for r in rows])

    meta = {
        "feature_names": feature_names,
        "refactor_label_source": label_source if matched_ref else "review_friction",
        "matched_refactor_labels": matched_ref,
        "rows": len(rows),
    }
    return rows, X, y_acc, y_ref, groups, meta


def _fit_one(X, y, seed=42):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    uniq = np.unique(y)
    if len(uniq) < 2:
        prob = float(uniq[0]) if len(uniq) == 1 else 0.5
        coef = np.zeros((1, X.shape[1]), dtype=np.float32)
        if prob <= 0.0:
            inter = -20.0
        elif prob >= 1.0:
            inter = 20.0
        else:
            inter = math.log(prob / (1.0 - prob))
        class _ConstantClf:
            coef_ = coef
            intercept_ = np.asarray([inter], dtype=np.float32)
        pred = np.full(len(X), prob, dtype=np.float32)
        return sc, _ConstantClf(), pred
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=seed)
    clf.fit(Xs, y)
    pred = clf.predict_proba(Xs)[:, 1]
    return sc, clf, pred


def _cv_scores(X, y, groups, cv_folds, seed):
    out = []
    if cv_folds < 2 or len(np.unique(groups)) < cv_folds:
        return out
    gkf = GroupKFold(n_splits=cv_folds)
    idx = np.arange(len(X))
    for tr, te in gkf.split(idx, groups=groups):
        if len(np.unique(y[tr])) < 2:
            out.append({"auroc": 0.5, "pr_auc": float(np.mean(y[te]))})
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=seed)
        clf.fit(Xtr, y[tr])
        pred = clf.predict_proba(Xte)[:, 1]
        out.append(
            {
                "auroc": float(roc_auc_score(y[te], pred)) if len(set(y[te])) > 1 else 0.5,
                "pr_auc": float(average_precision_score(y[te], pred)),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--labels-data", default=DEFAULT_LABELS)
    ap.add_argument("--model-out", default=DEFAULT_MODEL)
    ap.add_argument("--metrics-out", default=DEFAULT_METRICS)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--refactor-target",
        choices=["auto", "refactor_requested", "review_friction"],
        default="auto",
        help="Second head target. 'auto' prefers refactor_requested labels when joinable.",
    )
    args = ap.parse_args()

    refactor_target = "refactor_requested" if args.refactor_target == "auto" else args.refactor_target
    rows, X, y_acc, y_ref, groups, meta = _load(args.data, args.labels_data, refactor_target)

    sc_acc, clf_acc, pred_acc = _fit_one(X, y_acc, args.seed)
    sc_ref, clf_ref, pred_ref = _fit_one(X, y_ref, args.seed + 1)

    cv = {
        "acceptance": _cv_scores(X, y_acc, groups, args.cv_folds, args.seed),
        "refactor": _cv_scores(X, y_ref, groups, args.cv_folds, args.seed + 1),
    }

    model_blob = {
        "version": "phase4_7_conway_patch_v1",
        "feature_source": "conway_patch_features",
        "features": meta["feature_names"],
        "log1p_features": sorted(LOG1P_FEATURES.intersection(meta["feature_names"])),
        "refactor_target": meta["refactor_label_source"],
        "acceptance": {
            "scaler_mean": sc_acc.mean_.tolist(),
            "scaler_scale": sc_acc.scale_.tolist(),
            "coef": clf_acc.coef_[0].tolist(),
            "intercept": float(clf_acc.intercept_[0]),
        },
        "refactor": {
            "scaler_mean": sc_ref.mean_.tolist(),
            "scaler_scale": sc_ref.scale_.tolist(),
            "coef": clf_ref.coef_[0].tolist(),
            "intercept": float(clf_ref.intercept_[0]),
        },
    }
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    with open(args.model_out, "w") as f:
        json.dump(model_blob, f, indent=2)

    metrics = {
        "rows": len(rows),
        "n_repos": int(len(np.unique(groups))),
        "n_features": len(meta["feature_names"]),
        "feature_source": args.data,
        "labels_data": args.labels_data,
        "refactor_target": meta["refactor_label_source"],
        "matched_refactor_labels": meta["matched_refactor_labels"],
        "acceptance_rate": float(np.mean(y_acc)),
        "refactor_rate": float(np.mean(y_ref)),
        "train_metrics": {
            "acceptance_auroc": float(roc_auc_score(y_acc, pred_acc)) if len(set(y_acc)) > 1 else 0.5,
            "acceptance_pr_auc": float(average_precision_score(y_acc, pred_acc)),
            "refactor_auroc": float(roc_auc_score(y_ref, pred_ref)) if len(set(y_ref)) > 1 else 0.5,
            "refactor_pr_auc": float(average_precision_score(y_ref, pred_ref)),
        },
        "cv": cv,
        "model_out": args.model_out,
    }
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
