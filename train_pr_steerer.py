#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pg8000.native
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
PG_CONFIG_FILE = os.path.join(ROOT, "postgres_connection.yaml")


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


def _load_followup_labels_from_db() -> tuple[dict[tuple[str, int], int], str]:
    """Load binary followup labels from the followups_file Postgres table.

    Returns a map of (repo, pull_number) → 1 for PRs that have at least one
    followup PR, usable as the refactor/risk head target.
    """
    if not os.path.exists(PG_CONFIG_FILE):
        print(f"WARNING: {PG_CONFIG_FILE} not found, cannot load followup labels from DB")
        return {}, "review_friction"
    cfg = yaml.safe_load(open(PG_CONFIG_FILE))
    conn = pg8000.native.Connection(
        host=cfg["ip"], port=int(cfg["port"]),
        user=cfg["user"], password=cfg["password"], database=cfg["database"],
    )
    rows = conn.run(
        "SELECT feature_repo, feature_pr_number FROM followups_file GROUP BY feature_repo, feature_pr_number"
    )
    conn.close()
    labels = {(r[0], int(r[1])): 1 for r in rows}
    print(f"Loaded {len(labels)} followup labels from followups_file table")
    return labels, "has_followup"


def _load(path: str, labels_path: str | None, refactor_target: str, language: str | None = None):
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    if not rows:
        raise ValueError(f"no rows found in {path}")

    if language:
        lang_lower = language.lower()
        before = len(rows)
        rows = [r for r in rows if (r.get("primary_lang") or "").lower() == lang_lower]
        print(f"Language filter '{language}': {before} → {len(rows)} rows")
        if not rows:
            raise ValueError(f"no rows remain after filtering for language={language}")

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
    if len(np.unique(y)) > 2:
        pred = clf.predict_proba(Xs)  # full (n_samples, n_classes) for multi-class
    else:
        pred = clf.predict_proba(Xs)[:, 1]
    return sc, clf, pred


def _cv_scores(X, y, groups, cv_folds, seed):
    out = []
    if cv_folds < 2 or len(np.unique(groups)) < cv_folds:
        return out
    is_multiclass = len(np.unique(y)) > 2
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
        n_classes_te = len(set(y[te]))
        if is_multiclass:
            pred_proba = clf.predict_proba(Xte)
            if n_classes_te > 1:
                auroc = float(roc_auc_score(y[te], pred_proba, multi_class="ovr", average="weighted"))
            else:
                auroc = 0.5
            # For multi-class, use macro-averaged binarized PR-AUC
            from sklearn.preprocessing import label_binarize
            classes = clf.classes_
            y_bin = label_binarize(y[te], classes=classes)
            pr_auc = float(average_precision_score(y_bin, pred_proba, average="weighted"))
            out.append({"auroc": auroc, "pr_auc": pr_auc})
        else:
            pred = clf.predict_proba(Xte)[:, 1]
            out.append(
                {
                    "auroc": float(roc_auc_score(y[te], pred)) if n_classes_te > 1 else 0.5,
                    "pr_auc": float(average_precision_score(y[te], pred)),
                }
            )
    return out


def _compute_train_metrics(y_acc, pred_acc, y_ref, pred_ref):
    """Compute train metrics handling both binary and multi-class refactor targets."""
    metrics = {
        "acceptance_auroc": float(roc_auc_score(y_acc, pred_acc)) if len(set(y_acc)) > 1 else 0.5,
        "acceptance_pr_auc": float(average_precision_score(y_acc, pred_acc)),
    }
    if len(np.unique(y_ref)) > 2:
        # Multi-class (ordinal scope)
        from sklearn.preprocessing import label_binarize
        # pred_ref is predict_proba output (n_samples, n_classes)
        if pred_ref.ndim == 1:
            metrics["refactor_auroc"] = 0.5
            metrics["refactor_pr_auc"] = 0.0
        else:
            classes = np.unique(y_ref)
            y_bin = label_binarize(y_ref, classes=classes)
            metrics["refactor_auroc"] = float(roc_auc_score(y_bin, pred_ref, multi_class="ovr", average="weighted")) if len(set(y_ref)) > 1 else 0.5
            metrics["refactor_pr_auc"] = float(average_precision_score(y_bin, pred_ref, average="weighted"))
    else:
        metrics["refactor_auroc"] = float(roc_auc_score(y_ref, pred_ref)) if len(set(y_ref)) > 1 else 0.5
        metrics["refactor_pr_auc"] = float(average_precision_score(y_ref, pred_ref))
    return metrics


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
    ap.add_argument(
        "--language",
        default=None,
        help="Filter features JSONL by primary_lang (e.g. 'python', 'go'). Case-insensitive.",
    )
    ap.add_argument(
        "--followup-labels",
        action="store_true",
        help="Read refactor labels from followups_file table in Postgres instead of --labels-data file.",
    )
    ap.add_argument(
        "--llm-refactor-labels",
        default=None,
        help="Path to LLM-judged refactor labels JSONL from label_refactor_llm.py. Overrides --labels-data for refactor head.",
    )
    ap.add_argument(
        "--refactor-scope-target",
        action="store_true",
        help="Train refactor head on ordinal scope (none=0, function=1, module=2, library_component=3) instead of binary refactor_requested.",
    )
    args = ap.parse_args()

    refactor_target = "refactor_requested" if args.refactor_target == "auto" else args.refactor_target

    if args.llm_refactor_labels:
        # LLM-judged labels override --labels-data
        if args.refactor_scope_target:
            # Convert scope to ordinal labels
            _scope_ord = {"none": 0, "function": 1, "module": 2, "library_component": 3}
            import tempfile
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
            with open(args.llm_refactor_labels) as _lf:
                for ln in _lf:
                    if not ln.strip():
                        continue
                    row = json.loads(ln)
                    scope_val = _scope_ord.get(row.get("refactor_scope", "none"), 0)
                    tmp.write(json.dumps({
                        "repo": row["repo"],
                        "pull_number": row["pull_number"],
                        "s_t1": {"refactor_requested": scope_val},
                    }) + "\n")
            tmp.close()
            args.labels_data = tmp.name
        else:
            # LLM labels already have s_t1.refactor_requested — use directly
            args.labels_data = args.llm_refactor_labels
        refactor_target = "refactor_requested"
    elif args.followup_labels:
        # Override the labels_path with DB-sourced followup labels
        _followup_map, _followup_source = _load_followup_labels_from_db()
        # Write a temporary JSONL so _load_refactor_labels can consume it
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for (repo, pull), val in _followup_map.items():
            tmp.write(json.dumps({"repo": repo, "pull_number": pull, "s_t1": {"refactor_requested": val}}) + "\n")
        tmp.close()
        args.labels_data = tmp.name
        refactor_target = "refactor_requested"  # use the s_t1.refactor_requested key path

    rows, X, y_acc, y_ref, groups, meta = _load(args.data, args.labels_data, refactor_target, language=args.language)

    if args.llm_refactor_labels:
        meta["refactor_label_source"] = "llm_refactor_scope" if args.refactor_scope_target else "llm_refactor_requested"
        if args.refactor_scope_target:
            os.unlink(args.labels_data)  # clean up temp file
    elif args.followup_labels:
        meta["refactor_label_source"] = "has_followup"
        os.unlink(args.labels_data)  # clean up temp file

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
            "coef": clf_ref.coef_.tolist(),  # (1, n_features) for binary, (n_classes, n_features) for multi-class
            "intercept": clf_ref.intercept_.tolist(),
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
        "train_metrics": _compute_train_metrics(y_acc, pred_acc, y_ref, pred_ref),
        "cv": cv,
        "model_out": args.model_out,
    }
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
