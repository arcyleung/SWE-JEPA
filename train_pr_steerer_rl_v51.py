#!/usr/bin/env python3
"""Experiment 5.1 — Offline RL Steerer Trainer (Conway-Aware State Policy).

Two training modes:
  1. Supervised heads (backward-compatible with Exp 4.7 Steerer interface)
  2. Pairwise reward model (Bradley-Terry offline RL on same-repo preference pairs)
  3. Value regression reward model (direct reward prediction)

Serializes a JSON model blob that is a drop-in upgrade for the Steerer class in
run_phase4_7_agentic_eval_steered.py:  blob["acceptance"] / blob["refactor"] keep working
as before, and blob["reward_model"] adds the RL-trained scorer.

Usage:
  python train_pr_steerer_rl_v51.py                             # use default paths
  python train_pr_steerer_rl_v51.py --data data/phase5_1_pr_mdp_dataset_v51.jsonl
  python train_pr_steerer_rl_v51.py --pair-margin 0.4 --pair-max-per-repo 500
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA    = os.path.join(ROOT, "data", "phase5_1_pr_mdp_dataset_v51_conway.jsonl")
DEFAULT_MODEL   = os.path.join(ROOT, "data", "phase5_1_pr_steerer_model_v51_conway.json")
DEFAULT_METRICS = os.path.join(ROOT, "data", "phase5_1_pr_steerer_metrics_v51_conway.json")

# Feature ordering: original 6 + 4 PR-level v51 + 18 patch-level Conway (tree-sitter + regex)
FEATURES_V51 = [
    # ── original Exp 4.7 PR-level ──
    "is_draft",
    "changed_files",
    "additions",
    "deletions",
    "requested_reviewers_count",
    "has_closing_issue",
    # ── v51 PR-level ──
    "cross_module_spread",
    "has_tests",
    "churn_asymmetry",
    "followup_risk",
    # ── Conway patch-level (tree-sitter imports) ──
    "imp_external",          # new external package imports (trust boundary count)
    "imp_relative",          # new intra-repo relative imports
    # ── Conway patch-level (composite scores) ──
    "trust_boundary_crossings",  # external clients + dep files introduced
    "error_contract_score",      # try/catch quality (+good, -bare_except)
    "security_risk_score",       # hardcoded creds, eval, shell=True, etc.
    "operability_score",         # metrics/health-checks added minus missing
    # ── Conway patch-level (binary signals) ──
    "has_pub_func",          # new public API surface
    "has_try_catch",         # error handler added
    "has_bare_except",       # error silently swallowed
    "has_db_client",         # new DB connection initialised
    "has_http_client",       # new HTTP call added
    "ext_client_no_obs",     # external client with no metric/health-check
    "has_schema_change",     # DB schema or ORM field change
    "has_hardcoded_cred",    # credential literal in code
    "has_metric_emit",       # new observability metric emitted
    "modifies_shared_util",  # touches utils/common/shared directory
    "has_open_no_with",      # file open() without context manager
    "has_sql_fstring",       # SQL query via f-string (injection proxy)
]

# Transforms: log1p for counts, identity for binary/scores
_LOG1P_FEATS = frozenset([
    "changed_files", "additions", "deletions", "requested_reviewers_count",
    "cross_module_spread", "imp_external", "imp_relative", "trust_boundary_crossings",
    "new_func_defs", "n_http_calls",
])


def _x(item: dict) -> np.ndarray:
    s = item["s_t"]
    vals = []
    for f in FEATURES_V51:
        v = float(s.get(f, 0) or 0)
        vals.append(float(np.log1p(v)) if f in _LOG1P_FEATS else v)
    return np.asarray(vals, dtype=np.float32)


def _load(path: str):
    rows = [json.loads(ln) for ln in open(path)]
    X = np.stack([_x(r) for r in rows], axis=0)
    y_acc = np.asarray([int(r["s_t1"]["accepted"]) for r in rows], dtype=np.int32)
    y_ref = np.asarray([int(r["s_t1"]["refactor_requested"]) for r in rows], dtype=np.int32)
    rewards = np.asarray([float(r["reward"]) for r in rows], dtype=np.float32)
    groups = np.asarray([r["repo"] for r in rows])
    return rows, X, y_acc, y_ref, rewards, groups


def _fit_logreg(X, y, seed=42):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    clf = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=seed
    )
    clf.fit(Xs, y)
    pred = clf.predict_proba(Xs)[:, 1]
    return sc, clf, pred


def _build_preference_pairs(
    X: np.ndarray,
    rewards: np.ndarray,
    groups: np.ndarray,
    margin: float,
    max_per_repo: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Bradley-Terry preference pairs for offline RL.

    For same-repo pairs where reward difference > margin, adds both directions:
      (x_winner - x_loser, label=1) and (x_loser - x_winner, label=0)
    This keeps the dataset balanced (50/50) and gives sklearn 2 classes.
    The learned weights w satisfy: P(i beats j) = σ(w^T (x_i - x_j)).
    """
    diff_rows: list[np.ndarray] = []
    labels: list[int] = []
    unique_repos = np.unique(groups)
    for repo in unique_repos:
        idx = np.where(groups == repo)[0]
        if len(idx) < 2:
            continue
        r_repo = rewards[idx]
        # Unordered pairs (i, j) where |r_i - r_j| > margin
        pairs: list[tuple[int, int]] = []  # (winner_global_idx, loser_global_idx)
        for ii in range(len(idx)):
            for jj in range(ii + 1, len(idx)):
                diff = r_repo[ii] - r_repo[jj]
                if diff > margin:
                    pairs.append((idx[ii], idx[jj]))
                elif diff < -margin:
                    pairs.append((idx[jj], idx[ii]))
        if not pairs:
            continue
        if len(pairs) > max_per_repo:
            sel = rng.choice(len(pairs), max_per_repo, replace=False)
            pairs = [pairs[i] for i in sel]
        for wi, li in pairs:
            diff_rows.append(X[wi] - X[li])
            labels.append(1)
            diff_rows.append(X[li] - X[wi])
            labels.append(0)

    if not diff_rows:
        return np.empty((0, X.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int32)
    X_pairs = np.stack(diff_rows, axis=0)
    y_pairs = np.asarray(labels, dtype=np.int32)
    return X_pairs, y_pairs


def _fit_pairwise(X_pairs: np.ndarray, y_pairs: np.ndarray, seed: int):
    """Bradley-Terry logistic model on preference pair feature differences."""
    sc = StandardScaler()
    Xps = sc.fit_transform(X_pairs)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed)
    clf.fit(Xps, y_pairs)
    pred = clf.predict_proba(Xps)[:, 1]
    return sc, clf, pred


def _fit_value(X: np.ndarray, rewards: np.ndarray):
    """Direct reward regression for value-function steerer."""
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    reg = Ridge(alpha=1.0)
    reg.fit(Xs, rewards)
    pred = reg.predict(Xs)
    return sc, reg, pred


def _cv_supervised(X, y, groups, cv_folds, seed):
    results = []
    if cv_folds < 2 or len(np.unique(groups)) < cv_folds:
        return results
    gkf = GroupKFold(n_splits=cv_folds)
    for tr, te in gkf.split(X, groups=groups):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=seed
        )
        clf.fit(Xtr, y[tr])
        pa = clf.predict_proba(Xte)[:, 1]
        auroc = float(roc_auc_score(y[te], pa)) if len(set(y[te])) > 1 else 0.5
        pr_auc = float(average_precision_score(y[te], pa))
        results.append({"auroc": auroc, "pr_auc": pr_auc})
    return results


def _cv_value(X, rewards, groups, cv_folds):
    results = []
    if cv_folds < 2 or len(np.unique(groups)) < cv_folds:
        return results
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr
    gkf = GroupKFold(n_splits=cv_folds)
    for tr, te in gkf.split(X, groups=groups):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        reg = Ridge(alpha=1.0)
        reg.fit(Xtr, rewards[tr])
        pred = reg.predict(Xte)
        r2 = float(r2_score(rewards[te], pred))
        rho, _ = spearmanr(rewards[te], pred)
        results.append({"r2": r2, "spearman": float(rho)})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model-out", default=DEFAULT_MODEL)
    ap.add_argument("--metrics-out", default=DEFAULT_METRICS)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pair-margin", type=float, default=0.3,
                    help="Min reward gap to form a preference pair")
    ap.add_argument("--pair-max-per-repo", type=int, default=300,
                    help="Max pairs sampled per repo to limit dataset size")
    args = ap.parse_args()

    rows, X, y_acc, y_ref, rewards, groups = _load(args.data)
    print(f"Loaded {len(rows)} rows, {len(np.unique(groups))} repos")
    print(f"  acceptance_rate={np.mean(y_acc):.3f}  refactor_rate={np.mean(y_ref):.3f}")
    print(f"  reward  mean={np.mean(rewards):.3f}  std={np.std(rewards):.3f}")

    # ── 1. Supervised heads (backward compat) ──────────────────────────────
    sc_acc, clf_acc, pred_acc = _fit_logreg(X, y_acc, args.seed)
    sc_ref, clf_ref, pred_ref = _fit_logreg(X, y_ref, args.seed + 1)

    cv_acc = _cv_supervised(X, y_acc, groups, args.cv_folds, args.seed)
    cv_ref = _cv_supervised(X, y_ref, groups, args.cv_folds, args.seed + 1)

    # ── 2. Pairwise reward model (Bradley-Terry offline RL) ────────────────
    rng = np.random.RandomState(args.seed)
    X_pairs, y_pairs = _build_preference_pairs(X, rewards, groups, args.pair_margin, args.pair_max_per_repo, rng)
    print(f"  Preference pairs: {len(X_pairs)} (margin={args.pair_margin})")

    if len(X_pairs) >= 20:
        sc_pair, clf_pair, pred_pair = _fit_pairwise(X_pairs, y_pairs, args.seed)
        pair_acc_train = float(np.mean((pred_pair > 0.5).astype(int) == y_pairs))
    else:
        print("  WARNING: too few preference pairs — skipping pairwise model")
        sc_pair = clf_pair = None
        pair_acc_train = 0.0

    # ── 3. Value regression reward model ──────────────────────────────────
    sc_val, reg_val, pred_val = _fit_value(X, rewards)
    cv_val = _cv_value(X, rewards, groups, args.cv_folds)

    # ── Serialize ──────────────────────────────────────────────────────────
    def _supervised_blob(sc, clf):
        return {
            "scaler_mean": sc.mean_.tolist(),
            "scaler_scale": sc.scale_.tolist(),
            "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]),
        }

    def _ridge_blob(sc, reg):
        return {
            "scaler_mean": sc.mean_.tolist(),
            "scaler_scale": sc.scale_.tolist(),
            "coef": reg.coef_.tolist(),
            "intercept": float(reg.intercept_),
        }

    model_blob: dict = {
        "version": "v51_rl",
        "features": FEATURES_V51,
        # Backward-compatible supervised heads
        "acceptance": _supervised_blob(sc_acc, clf_acc),
        "refactor": _supervised_blob(sc_ref, clf_ref),
        # RL-trained value regression reward model
        "reward_model": {
            "type": "value_regression",
            **_ridge_blob(sc_val, reg_val),
        },
    }

    if clf_pair is not None:
        model_blob["pairwise_reward_model"] = {
            "type": "bradley_terry",
            **_supervised_blob(sc_pair, clf_pair),
        }

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    with open(args.model_out, "w") as f:
        json.dump(model_blob, f, indent=2)

    # ── Report ─────────────────────────────────────────────────────────────
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr

    rho_val, _ = spearmanr(rewards, pred_val)
    metrics = {
        "rows": len(rows),
        "n_repos": int(len(np.unique(groups))),
        "acceptance_rate": float(np.mean(y_acc)),
        "refactor_rate": float(np.mean(y_ref)),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "pair_margin": args.pair_margin,
        "n_pairs": len(X_pairs),
        "train_metrics": {
            "acceptance_auroc": float(roc_auc_score(y_acc, pred_acc)) if len(set(y_acc)) > 1 else 0.5,
            "acceptance_pr_auc": float(average_precision_score(y_acc, pred_acc)),
            "refactor_auroc": float(roc_auc_score(y_ref, pred_ref)) if len(set(y_ref)) > 1 else 0.5,
            "refactor_pr_auc": float(average_precision_score(y_ref, pred_ref)),
            "value_r2": float(r2_score(rewards, pred_val)),
            "value_spearman": float(rho_val),
            "pairwise_train_acc": pair_acc_train,
        },
        "cv_acceptance": cv_acc,
        "cv_refactor": cv_ref,
        "cv_value": cv_val,
        "model_out": args.model_out,
    }

    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n── Training complete ──")
    print(f"  acceptance AUROC (train): {metrics['train_metrics']['acceptance_auroc']:.4f}")
    print(f"  refactor   AUROC (train): {metrics['train_metrics']['refactor_auroc']:.4f}")
    print(f"  value      R²    (train): {metrics['train_metrics']['value_r2']:.4f}")
    print(f"  value      ρ     (train): {metrics['train_metrics']['value_spearman']:.4f}")
    if clf_pair is not None:
        print(f"  pairwise   acc  (train): {pair_acc_train:.4f}  ({len(X_pairs)} pairs)")
    if cv_acc:
        mean_auroc = np.mean([x["auroc"] for x in cv_acc])
        print(f"  acceptance CV AUROC:     {mean_auroc:.4f}")
    if cv_val:
        mean_r2 = np.mean([x["r2"] for x in cv_val])
        mean_rho = np.mean([x["spearman"] for x in cv_val])
        print(f"  value      CV R²:        {mean_r2:.4f}  ρ={mean_rho:.4f}")
    print(f"  model → {args.model_out}")
    print(f"  metrics → {args.metrics_out}")


if __name__ == "__main__":
    main()
