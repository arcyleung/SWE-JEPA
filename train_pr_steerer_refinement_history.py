#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(ROOT, "data", "phase4_7_2_slurm_ramcache_v1", "merged.jsonl")
DEFAULT_METRICS = os.path.join(ROOT, "data", "phase4_7_2_pr_steerer_history_compare.json")
DEFAULT_HISTORY_MODEL = os.path.join(ROOT, "data", "phase4_7_2_pr_steerer_history_model.json")
DEFAULT_STATIC_MODEL = os.path.join(ROOT, "data", "phase4_7_2_pr_steerer_static_model.json")
DEFAULT_REPORT = os.path.join(ROOT, "docs", "phase4_7_2_pr_steerer_compare.md")

STRING_KEYS = {
    "repo",
    "instance_id",
    "base_sha",
    "head_sha",
    "head_branch",
    "commit_sha",
    "commit_message_headline",
    "commit_author_name",
    "commit_author_github",
    "authored_date",
    "committed_date",
    "pr_created_at",
    "pr_merged_at",
    "primary_lang",
    "snapshot_kind",
    "history_recovery_mode",
    "merge_commit_detection_mode",
}

STATIC_EXCLUDE = {
    "accepted",
    "review_friction",
    "n_review_threads",
    "n_comments",
    "review_comments_before",
    "refactor_comments_before",
    "linked_review_comments",
    "approvals_before",
    "changes_requested_before",
    "review_events_between_prev_commit",
    "submitted_reviews_between_prev_commit",
    "refactor_events_between_prev_commit",
    "is_post_review_revision",
    "is_merge_commit_sampled",
    "merge_commit_parent_count",
    "merge_commits_total",
    "non_merge_commits_total",
    "merge_commits_skipped_from_sampling",
    "sampled_commit_count",
    "sampled_commit_rank",
    "commit_idx",
    "pull_number",
    "total_commits",
}

# These are the trajectory channels we aggregate over time. They are all patch/state signals,
# not reviewer-label counters.
HISTORY_METRICS = [
    "conway_risk_proxy",
    "conway_risk_flags",
    "api_change_without_tests",
    "public_api_without_docs",
    "shared_change_isolated",
    "ownership_diffusion",
    "boundary_density",
    "operability_score",
    "security_risk_score",
    "cross_module_spread",
    "trust_boundary_crossings",
    "snapshot_changed_files",
    "test_file_ratio",
    "shared_file_ratio",
    "public_api_file_ratio",
    "topdir_entropy",
    "blame_unique_authors",
    "blame_author_entropy",
]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _load_rows(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["instance_id"])].append(row)
    for vals in groups.values():
        vals.sort(key=lambda r: int(r.get("commit_idx", 0) or 0))
    return groups


def _slice_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "all":
        return rows
    if mode == "exclude_merge_rows":
        return [r for r in rows if int(r.get("is_merge_commit_sampled", 0) or 0) == 0]
    raise ValueError(f"unsupported slice mode: {mode}")


def _static_feature_names(example_row: dict[str, Any]) -> list[str]:
    names = []
    for key, value in example_row.items():
        if key in STRING_KEYS or key in STATIC_EXCLUDE:
            continue
        if isinstance(value, bool):
            names.append(key)
        elif isinstance(value, (int, float)):
            names.append(key)
    return sorted(set(names))


def _trajectory_features(seq: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {"traj_n_snapshots": float(len(seq))}
    if not seq:
        return out
    for name in HISTORY_METRICS:
        arr = np.asarray([float(r.get(name, 0.0) or 0.0) for r in seq], dtype=np.float32)
        out[f"hist_delta__{name}"] = float(arr[-1] - arr[0])
        out[f"hist_mean__{name}"] = float(np.mean(arr))
        out[f"hist_std__{name}"] = float(np.std(arr))
        out[f"hist_min__{name}"] = float(np.min(arr))
        out[f"hist_max__{name}"] = float(np.max(arr))
    return out


def _labels_from_full_seq(full_seq: list[dict[str, Any]]) -> dict[str, int]:
    review_friction = int(max(int(r.get("review_friction", 0) or 0) for r in full_seq) > 0)
    refactor_requested = int(max(int(r.get("refactor_comments_before", 0) or 0) for r in full_seq) > 0)
    changes_requested = int(max(int(r.get("changes_requested_before", 0) or 0) for r in full_seq) > 0)
    return {
        "acceptance_proxy": 1 - review_friction,
        "review_friction": review_friction,
        "refactor_requested": refactor_requested,
        "changes_requested": changes_requested,
    }


def _build_dataset(rows: list[dict[str, Any]], slice_mode: str) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    full_groups = _group_rows(rows)
    sliced_groups = _group_rows(_slice_rows(rows, slice_mode))
    if not sliced_groups:
        raise ValueError(f"no PR rows after slice={slice_mode}")

    example_seq = next(iter(sliced_groups.values()))
    static_names = _static_feature_names(example_seq[-1])
    hist_names = sorted(_trajectory_features(example_seq).keys())

    dataset: list[dict[str, Any]] = []
    for iid, seq in sliced_groups.items():
        if not seq:
            continue
        full_seq = full_groups[iid]
        final = seq[-1]
        static = {name: float(final.get(name, 0.0) or 0.0) for name in static_names}
        hist = _trajectory_features(seq)
        labels = _labels_from_full_seq(full_seq)
        dataset.append(
            {
                "repo": str(final["repo"]),
                "instance_id": iid,
                "pull_number": int(final.get("pull_number", 0) or 0),
                "slice_mode": slice_mode,
                "n_seq_rows": len(seq),
                "static": static,
                "history": hist,
                "labels": labels,
            }
        )
    return dataset, static_names, hist_names


def _log1p_feature_names(rows: list[dict[str, Any]], feature_names: list[str], block: str) -> list[str]:
    out: list[str] = []
    for name in feature_names:
        vals = np.asarray([float(r[block].get(name, 0.0) or 0.0) for r in rows], dtype=np.float32)
        if vals.size == 0:
            continue
        if np.min(vals) < 0:
            continue
        if len(np.unique(vals)) < 8:
            continue
        if float(np.quantile(vals, 0.95)) >= 10.0:
            out.append(name)
    return sorted(out)


def _matrix(rows: list[dict[str, Any]], feature_names: list[str], blocks: list[str], log1p_features: set[str]) -> np.ndarray:
    mat = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, name in enumerate(feature_names):
            value = 0.0
            for block in blocks:
                if name in row[block]:
                    value = float(row[block].get(name, 0.0) or 0.0)
                    break
            if name in log1p_features:
                value = float(np.log1p(max(value, 0.0)))
            mat[i, j] = value
    return mat


def _fit_one(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[StandardScaler, Any, np.ndarray]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    uniq = np.unique(y)
    if len(uniq) < 2:
        prob = float(uniq[0]) if len(uniq) == 1 else 0.5
        coef = np.zeros((1, X.shape[1]), dtype=np.float32)
        if prob <= 0.0:
            intercept = -20.0
        elif prob >= 1.0:
            intercept = 20.0
        else:
            intercept = math.log(prob / (1.0 - prob))

        class _ConstantClf:
            coef_ = coef
            intercept_ = np.asarray([intercept], dtype=np.float32)

        pred = np.full(len(X), prob, dtype=np.float32)
        return scaler, _ConstantClf(), pred

    clf = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(Xs, y)
    pred = clf.predict_proba(Xs)[:, 1]
    return scaler, clf, pred


def _cv_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cv_folds: int, seed: int) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    if cv_folds < 2 or len(np.unique(groups)) < cv_folds:
        return out
    gkf = GroupKFold(n_splits=cv_folds)
    idx = np.arange(len(X))
    for tr, te in gkf.split(idx, groups=groups):
        if len(np.unique(y[tr])) < 2:
            base = float(np.mean(y[tr])) if len(y[tr]) else 0.5
            pred = np.full(len(te), base, dtype=np.float32)
        else:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xte = scaler.transform(X[te])
            clf = LogisticRegression(
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )
            clf.fit(Xtr, y[tr])
            pred = clf.predict_proba(Xte)[:, 1]
        out.append(
            {
                "auroc": float(roc_auc_score(y[te], pred)) if len(set(y[te])) > 1 else 0.5,
                "pr_auc": float(average_precision_score(y[te], pred)),
            }
        )
    return out


def _mean_metric(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(r[key]) for r in rows) / len(rows))


def _top_coefficients(feature_names: list[str], coef: np.ndarray, limit: int = 12) -> dict[str, list[list[Any]]]:
    pairs = list(zip(feature_names, coef.tolist()))
    pos = sorted(pairs, key=lambda kv: kv[1], reverse=True)[:limit]
    neg = sorted(pairs, key=lambda kv: kv[1])[:limit]
    return {
        "positive": [[name, float(weight)] for name, weight in pos],
        "negative": [[name, float(weight)] for name, weight in neg],
    }


def _head_blob(scaler: StandardScaler, clf: Any, target_name: str, feature_names: list[str]) -> dict[str, Any]:
    return {
        "target": target_name,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "top_coefficients": _top_coefficients(feature_names, clf.coef_[0]),
    }


def _train_variant(
    rows: list[dict[str, Any]],
    static_names: list[str],
    hist_names: list[str],
    use_history: bool,
    cv_folds: int,
    seed: int,
) -> dict[str, Any]:
    feature_names = list(static_names)
    if use_history:
        feature_names.extend(hist_names)
    blocks = ["static"] + (["history"] if use_history else [])
    log1p = set(_log1p_feature_names(rows, static_names, "static"))
    if use_history:
        log1p.update(_log1p_feature_names(rows, hist_names, "history"))

    X = _matrix(rows, feature_names, blocks, log1p)
    groups = np.asarray([r["repo"] for r in rows])
    targets = {}
    trained = {}
    for offset, target_name in enumerate(("acceptance_proxy", "refactor_requested")):
        y = np.asarray([int(r["labels"][target_name]) for r in rows], dtype=np.int32)
        scaler, clf, pred = _fit_one(X, y, seed + offset)
        cv = _cv_scores(X, y, groups, cv_folds, seed + offset)
        targets[target_name] = {
            "positive_rate": float(np.mean(y)),
            "train_auroc": float(roc_auc_score(y, pred)) if len(set(y)) > 1 else 0.5,
            "train_pr_auc": float(average_precision_score(y, pred)),
            "cv": cv,
            "cv_mean_auroc": _mean_metric(cv, "auroc"),
            "cv_mean_pr_auc": _mean_metric(cv, "pr_auc"),
        }
        trained[target_name] = (scaler, clf)

    model_blob = {
        "version": "phase4_7_2_refinement_history_v1",
        "feature_source": "phase4_7_2_pr_refinement_history",
        "history_enabled": bool(use_history),
        "targets": {
            "acceptance": "acceptance_proxy = 1 - review_friction",
            "refactor": "refactor_requested = max(refactor_comments_before) > 0",
        },
        "feature_blocks": {
            "static": static_names,
            "history": hist_names if use_history else [],
        },
        "features": feature_names,
        "log1p_features": sorted(log1p),
        "acceptance": _head_blob(trained["acceptance_proxy"][0], trained["acceptance_proxy"][1], "acceptance_proxy", feature_names),
        "refactor": _head_blob(trained["refactor_requested"][0], trained["refactor_requested"][1], "refactor_requested", feature_names),
    }
    return {
        "rows": len(rows),
        "n_repos": int(len(np.unique(groups))),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "log1p_features": sorted(log1p),
        "targets": targets,
        "model_blob": model_blob,
    }


def _render_report(result: dict[str, Any], out_path: str) -> None:
    def pct(v: float) -> str:
        return f"{100.0 * v:.1f}%"

    def fmt(v: float) -> str:
        return f"{v:.3f}"

    lines = [
        "# Experiment 4.7.2 — Static Patch vs Trajectory-Aware Steerer",
        "",
        "This report compares two offline steerer variants on the completed 4.7.2 merged-PR",
        "trajectory dataset:",
        "",
        "- `static`: final visible patch only",
        "- `history`: final patch plus patch-trajectory summary statistics",
        "",
        "Targets are merged-only proxies because the 4.7.2 dataset contains only merged PRs:",
        "",
        "- `acceptance_proxy = 1 - review_friction`",
        "- `refactor_requested = max(refactor_comments_before) > 0`",
        "",
        "## Cross-validation summary",
        "",
        "| Slice | Variant | Target | Pos rate | CV AUROC | CV PR-AUC |",
        "|---|---|---|---:|---:|---:|",
    ]
    for slice_name, slice_result in result["variants"].items():
        for variant_name in ("static", "history"):
            variant = slice_result[variant_name]
            for target_name in ("acceptance_proxy", "refactor_requested"):
                tgt = variant["targets"][target_name]
                lines.append(
                    f"| `{slice_name}` | `{variant_name}` | `{target_name}` | {pct(tgt['positive_rate'])} | "
                    f"{fmt(tgt['cv_mean_auroc'])} | {fmt(tgt['cv_mean_pr_auc'])} |"
                )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- Selected training slice: `{result['selected']['slice_mode']}`",
            f"- Selected model variant: `{result['selected']['variant']}`",
            f"- Selection score: `{fmt(result['selected']['selection_score'])}` (mean of target CV AUROC and PR-AUC)",
            "",
            "## Selection rule",
            "",
            "- choose the history-enabled model only if it outperforms the static baseline on the mean of `acceptance_proxy` and `refactor_requested` CV metrics",
            "- choose the merge-excluded row slice only if it beats the full row slice under the same rule",
        ]
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--metrics-out", default=DEFAULT_METRICS)
    ap.add_argument("--history-model-out", default=DEFAULT_HISTORY_MODEL)
    ap.add_argument("--static-model-out", default=DEFAULT_STATIC_MODEL)
    ap.add_argument("--report-out", default=DEFAULT_REPORT)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = _load_rows(args.input)
    variants: dict[str, dict[str, Any]] = {}
    best: dict[str, Any] | None = None
    best_history: dict[str, Any] | None = None
    best_static: dict[str, Any] | None = None

    for slice_mode in ("all", "exclude_merge_rows"):
        dataset, static_names, hist_names = _build_dataset(rows, slice_mode)
        slice_result: dict[str, Any] = {
            "n_prs": len(dataset),
            "static_features": len(static_names),
            "history_features": len(hist_names),
        }
        static_variant = _train_variant(dataset, static_names, hist_names, use_history=False, cv_folds=args.cv_folds, seed=args.seed)
        history_variant = _train_variant(dataset, static_names, hist_names, use_history=True, cv_folds=args.cv_folds, seed=args.seed)
        slice_result["static"] = static_variant
        slice_result["history"] = history_variant
        variants[slice_mode] = slice_result

        for variant_name in ("static", "history"):
            variant = slice_result[variant_name]
            score = float(
                0.25 * variant["targets"]["acceptance_proxy"]["cv_mean_auroc"]
                + 0.25 * variant["targets"]["acceptance_proxy"]["cv_mean_pr_auc"]
                + 0.25 * variant["targets"]["refactor_requested"]["cv_mean_auroc"]
                + 0.25 * variant["targets"]["refactor_requested"]["cv_mean_pr_auc"]
            )
            candidate = {
                "slice_mode": slice_mode,
                "variant": variant_name,
                "selection_score": score,
                "model_blob": variant["model_blob"],
            }
            if best is None or candidate["selection_score"] > best["selection_score"]:
                best = candidate
            if variant_name == "history":
                if best_history is None or candidate["selection_score"] > best_history["selection_score"]:
                    best_history = candidate
            else:
                if best_static is None or candidate["selection_score"] > best_static["selection_score"]:
                    best_static = candidate

    assert best is not None
    assert best_history is not None
    assert best_static is not None

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.history_model_out, "w") as f:
        json.dump(best_history["model_blob"], f, indent=2)
    with open(args.static_model_out, "w") as f:
        json.dump(best_static["model_blob"], f, indent=2)

    result = {
        "input": args.input,
        "variants": variants,
        "selected": {
            "slice_mode": best["slice_mode"],
            "variant": best["variant"],
            "selection_score": best["selection_score"],
            "history_model_out": args.history_model_out,
            "static_model_out": args.static_model_out,
        },
        "best_history": {
            "slice_mode": best_history["slice_mode"],
            "selection_score": best_history["selection_score"],
            "model_out": args.history_model_out,
        },
        "best_static": {
            "slice_mode": best_static["slice_mode"],
            "selection_score": best_static["selection_score"],
            "model_out": args.static_model_out,
        },
    }
    with open(args.metrics_out, "w") as f:
        json.dump(result, f, indent=2)
    _render_report(result, args.report_out)
    print(json.dumps(result["selected"], indent=2))


if __name__ == "__main__":
    main()
