#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(ROOT, "docs", "phase4_7_pr_mdp_dataset.jsonl")
DEFAULT_MODEL = os.path.join(ROOT, "docs", "phase4_7_pr_steerer_model.json")
DEFAULT_METRICS = os.path.join(ROOT, "docs", "phase4_7_pr_steerer_metrics.json")

FEATURES = [
    "is_draft",
    "changed_files",
    "additions",
    "deletions",
    "requested_reviewers_count",
    "has_closing_issue",
]


def _x(item: dict) -> np.ndarray:
    s = item["s_t"]
    vals = [
        float(s["is_draft"]),
        np.log1p(float(s["changed_files"])),
        np.log1p(float(s["additions"])),
        np.log1p(float(s["deletions"])),
        np.log1p(float(s["requested_reviewers_count"])),
        float(s["has_closing_issue"]),
    ]
    return np.asarray(vals, dtype=np.float32)


def _load(path: str):
    rows = [json.loads(ln) for ln in open(path)]
    X = np.stack([_x(r) for r in rows], axis=0)
    y_acc = np.asarray([int(r["s_t1"]["accepted"]) for r in rows], dtype=np.int32)
    y_ref = np.asarray([int(r["s_t1"]["refactor_requested"]) for r in rows], dtype=np.int32)
    groups = np.asarray([r["repo"] for r in rows])
    return rows, X, y_acc, y_ref, groups


def _fit_one(X, y, seed=42):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=seed)
    clf.fit(Xs, y)
    pred = clf.predict_proba(Xs)[:, 1]
    return sc, clf, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model-out", default=DEFAULT_MODEL)
    ap.add_argument("--metrics-out", default=DEFAULT_METRICS)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows, X, y_acc, y_ref, groups = _load(args.data)

    sc_acc, clf_acc, pred_acc = _fit_one(X, y_acc, args.seed)
    sc_ref, clf_ref, pred_ref = _fit_one(X, y_ref, args.seed + 1)

    cv = {"acceptance": [], "refactor": []}
    if args.cv_folds >= 2 and len(rows) >= args.cv_folds:
        gkf = GroupKFold(n_splits=args.cv_folds)
        idx = np.arange(len(rows))
        for tr, te in gkf.split(idx, groups=groups):
            sca = StandardScaler()
            Xtr = sca.fit_transform(X[tr])
            Xte = sca.transform(X[te])
            ca = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs")
            ca.fit(Xtr, y_acc[tr])
            pa = ca.predict_proba(Xte)[:, 1]
            cv["acceptance"].append({
                "auroc": float(roc_auc_score(y_acc[te], pa)) if len(set(y_acc[te])) > 1 else 0.5,
                "pr_auc": float(average_precision_score(y_acc[te], pa)),
            })

            scr = StandardScaler()
            Xtr = scr.fit_transform(X[tr])
            Xte = scr.transform(X[te])
            cr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs")
            cr.fit(Xtr, y_ref[tr])
            pr = cr.predict_proba(Xte)[:, 1]
            cv["refactor"].append({
                "auroc": float(roc_auc_score(y_ref[te], pr)) if len(set(y_ref[te])) > 1 else 0.5,
                "pr_auc": float(average_precision_score(y_ref[te], pr)),
            })

    model_blob = {
        "features": FEATURES,
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
