#!/usr/bin/env python3
"""Train a per-checkpoint value function V(s_t) -> terminal review_friction.

Data source: pr_refinement_history_checkpoints postgres table.
Each checkpoint row holds a payload_json with snapshot_rows — one dict per
commit in the PR.  The terminal labels (review_friction, refactor_requested)
come from prs_copy joined on instance_id.

Training:
  - Explode snapshot_rows so each commit is an independent training example
    (labelled with its PR's terminal outcome).
  - Ridge regressor  →  V(s_t) = E[review_friction | Conway features at step t]
  - Logistic head    →  P(refactor_requested | Conway features at step t)

These serve two roles:
  1. Offline quality signal: V(s_N) at the final commit should correlate at
     least as well as the static patch model.
  2. Potential-based dense reward (Ng et al. 1999):
       r_dense(t) = gamma * V(s_{t+1}) - V(s_t)
     This is the Phi(s_t) used in Exp 5.2's PRM.

Output model JSON is compatible with the existing Steerer class in
run_phase4_7_agentic_eval_steered.py; adds a "value" head alongside the
existing "acceptance" and "refactor" heads.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pg8000.native
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
PG_CONFIG_FILE = os.path.join(ROOT, "postgres_connection.yaml")
CHECKPOINT_TABLE = "pr_refinement_history_checkpoints"

# Default run_tag written by extract_pr_refinement_history.py
DEFAULT_RUN_TAG = "v1"

# Features used by the static model — we reuse the same set so the value head
# is directly comparable and can be added to the same model JSON.
# Loaded dynamically from the static model file if available, otherwise hardcoded.
FALLBACK_FEATURES = [
    "api_change_without_tests",
    "boundary_crossing_without_obs",
    "boundary_density",
    "conway_risk_flags",
    "conway_risk_proxy",
    "cross_module_spread",
    "dependency_change_without_tests",
    "docs_file_ratio",
    "error_contract_score",
    "ext_client_no_log",
    "ext_client_no_obs",
    "external_io_without_safety",
    "has_bare_except",
    "has_config_file",
    "has_context_mgr",
    "has_hardcoded_cred",
    "has_health_check",
    "has_http_client",
    "has_infra_file",
    "has_input_validate",
    "has_lock_prim",
    "has_log_in_except",
    "has_log_warn_err",
    "has_metric_emit",
    "has_raise_from",
    "has_try_catch",
    "imp_external",
    "modifies_auth_code",
    "modifies_shared_util",
    "n_files",
    "n_langs",
    "operability_score",
    "ownership_diffusion",
    "public_api_without_docs",
    "schema_change_without_migration",
    "security_risk_score",
    "shared_change_isolated",
    "test_file_ratio",
    "topdir_entropy",
    "trust_boundary_crossings",
    # trajectory position features added during training
    "commit_position_frac",  # t / (T-1), 0=first 1=final
    "commits_remaining_frac",  # 1 - commit_position_frac
]

LOG1P_FEATURES = frozenset([
    "n_files", "ownership_diffusion", "trust_boundary_crossings",
    "cross_module_spread", "conway_risk_proxy",
])


def _load_db() -> pg8000.native.Connection:
    cfg = yaml.safe_load(open(PG_CONFIG_FILE))
    return pg8000.native.Connection(
        host=cfg["ip"],
        port=cfg.get("port", 9999),
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
    )


def _review_friction(total_threads: int, total_comments: int, review_threads: int) -> float:
    """Same formula used in build_pr_mdp_dataset_v51.py."""
    thread_score = math.log1p(max(0, review_threads)) * 0.5
    comment_score = math.log1p(max(0, total_comments)) * 0.3
    change_score = math.log1p(max(0, total_threads)) * 0.2
    raw = thread_score + comment_score + change_score
    return float(min(1.0, raw / 10.0))


def _load_terminal_labels(
    conn: pg8000.native.Connection,
    instance_ids: list[str],
) -> dict[str, dict]:
    """Fetch terminal labels for a batch of instance_ids from prs_copy."""
    if not instance_ids:
        return {}
    literals = ", ".join("'" + iid.replace("'", "''") + "'" for iid in instance_ids)
    rows = conn.run(
        f"""
        SELECT
            instance_id,
            total_threads,
            total_comments,
            review_threads,
            review_comments_raw
        FROM prs_copy
        WHERE instance_id IN ({literals})
        """
    )
    out = {}
    for instance_id, total_threads, total_comments, review_threads, rcr in rows:
        iid = str(instance_id)
        # Compute refactor_requested from review_comments_raw (same as REFACTOR_RE logic)
        import re
        REFACTOR_RE = re.compile(
            r"\b(refactor|rewrite|restructure|reorganize|rework|cleanup|clean.up|"
            r"simplif|extract (method|class|function)|move (to|into)|split (into|out)|"
            r"separate|decouple|abstrac|consolidat|encapsulat)\b",
            re.IGNORECASE,
        )
        refactor_requested = 0
        if rcr:
            comments = rcr if isinstance(rcr, list) else json.loads(rcr)
            for c in comments:
                body = (c.get("body") or "") if isinstance(c, dict) else str(c)
                if REFACTOR_RE.search(body):
                    refactor_requested = 1
                    break

        friction = _review_friction(
            int(total_threads or 0),
            int(total_comments or 0),
            int(review_threads or 0),
        )
        out[iid] = {
            "review_friction": friction,
            "refactor_requested": refactor_requested,
            "acceptance_proxy": 1.0 - friction,
        }
    return out


def _load_checkpoints(
    conn: pg8000.native.Connection,
    run_tag: str,
    limit: int | None,
) -> list[dict]:
    """Load completed checkpoint rows from postgres."""
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    rows = conn.run(
        f"""
        SELECT
            instance_id,
            repo,
            n_rows,
            payload_json::text
        FROM {CHECKPOINT_TABLE}
        WHERE run_tag = :run_tag
          AND status = 'ok'
          AND n_rows > 0
        ORDER BY instance_id
        {limit_clause}
        """,
        run_tag=run_tag,
    )
    out = []
    for instance_id, repo, n_rows, payload_json_text in rows:
        try:
            payload = json.loads(payload_json_text or "{}")
        except Exception:
            continue
        snapshot_rows = payload.get("snapshot_rows") or []
        if not isinstance(snapshot_rows, list) or not snapshot_rows:
            continue
        out.append({
            "instance_id": str(instance_id),
            "repo": str(repo or ""),
            "snapshot_rows": snapshot_rows,
        })
    return out


def _extract_feature_vec(
    snap: dict,
    feature_names: list[str],
    commit_position_frac: float,
) -> list[float]:
    """Build a feature vector from a snapshot dict."""
    row = snap if isinstance(snap, dict) else {}
    # Flatten nested 'features' key if present
    feats = row.get("features") or row
    vec = []
    for name in feature_names:
        if name == "commit_position_frac":
            vec.append(float(commit_position_frac))
        elif name == "commits_remaining_frac":
            vec.append(float(1.0 - commit_position_frac))
        else:
            v = float(feats.get(name) or row.get(name) or 0.0)
            if name in LOG1P_FEATURES:
                v = math.log1p(max(v, 0.0))
            vec.append(v)
    return vec


def _build_dataset(
    checkpoints: list[dict],
    labels: dict[str, dict],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Returns:
        X          (N, F) float32
        y_friction (N,)   float32   — regression target
        y_refactor (N,)   int8      — binary classification target
        groups     (N,)   int32     — PR group index for GroupKFold
        iids       list[str]        — instance_id per row
    """
    Xs, y_fric, y_ref, grps, iids = [], [], [], [], []
    group_map: dict[str, int] = {}

    for cp in checkpoints:
        iid = cp["instance_id"]
        lbl = labels.get(iid)
        if lbl is None:
            continue
        snaps = cp["snapshot_rows"]
        T = len(snaps)
        if T == 0:
            continue
        if iid not in group_map:
            group_map[iid] = len(group_map)
        g = group_map[iid]

        for t, snap in enumerate(snaps):
            frac = t / max(T - 1, 1)
            vec = _extract_feature_vec(snap, feature_names, frac)
            Xs.append(vec)
            y_fric.append(lbl["review_friction"])
            y_ref.append(lbl["refactor_requested"])
            grps.append(g)
            iids.append(iid)

    X = np.array(Xs, dtype=np.float32)
    return (
        X,
        np.array(y_fric, dtype=np.float32),
        np.array(y_ref, dtype=np.int8),
        np.array(grps, dtype=np.int32),
        iids,
    )


def _cv_metrics(
    X: np.ndarray,
    y_friction: np.ndarray,
    y_refactor: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """5-fold group CV (no PR spans two folds)."""
    gkf = GroupKFold(n_splits=n_splits)
    val_r2s, val_spearman, val_refactor_auroc = [], [], []

    from scipy.stats import spearmanr

    for train_idx, val_idx in gkf.split(X, y_friction, groups):
        Xtr, Xva = X[train_idx], X[val_idx]
        ytr_f, yva_f = y_friction[train_idx], y_friction[val_idx]
        ytr_r, yva_r = y_refactor[train_idx], y_refactor[val_idx]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)

        ridge = Ridge(alpha=1.0)
        ridge.fit(Xtr_s, ytr_f)
        preds_f = ridge.predict(Xva_s)

        ss_res = float(np.sum((yva_f - preds_f) ** 2))
        ss_tot = float(np.sum((yva_f - yva_f.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        val_r2s.append(r2)
        rho, _ = spearmanr(preds_f, yva_f)
        val_spearman.append(float(rho))

        if yva_r.sum() >= 5:
            lr = LogisticRegression(C=0.5, max_iter=500, class_weight="balanced")
            lr.fit(Xtr_s, ytr_r)
            prob_r = lr.predict_proba(Xva_s)[:, 1]
            auroc = roc_auc_score(yva_r, prob_r)
            val_refactor_auroc.append(float(auroc))

    return {
        "value_cv_r2_mean": float(np.mean(val_r2s)),
        "value_cv_r2_std": float(np.std(val_r2s)),
        "value_cv_spearman_mean": float(np.mean(val_spearman)),
        "value_cv_spearman_std": float(np.std(val_spearman)),
        "refactor_early_warning_cv_auroc_mean": float(np.mean(val_refactor_auroc)) if val_refactor_auroc else None,
        "refactor_early_warning_cv_auroc_std": float(np.std(val_refactor_auroc)) if val_refactor_auroc else None,
        "n_folds": n_splits,
        "n_folds_with_refactor": len(val_refactor_auroc),
    }


def _train_final(
    X: np.ndarray,
    y_friction: np.ndarray,
    y_refactor: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Train on full dataset; return serialisable model blobs."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    ridge = Ridge(alpha=1.0)
    ridge.fit(Xs, y_friction)

    lr = LogisticRegression(C=0.5, max_iter=500, class_weight="balanced")
    lr.fit(Xs, y_refactor)

    def _head(coef, intercept):
        return {
            "coef": [float(c) for c in coef],
            "intercept": float(intercept),
            "scaler_mean": [float(m) for m in scaler.mean_],
            "scaler_scale": [float(s) for s in scaler.scale_],
        }

    return {
        "value": _head(ridge.coef_, ridge.intercept_),
        "refactor_early_warning": _head(lr.coef_[0], lr.intercept_[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    ap.add_argument("--limit", type=int, default=None,
                    help="Max number of PRs to load (default: all)")
    ap.add_argument("--static-model", default=os.path.join(ROOT, "data", "phase4_7_2_pr_steerer_static_model.json"),
                    help="Existing static model JSON to inherit feature list from")
    ap.add_argument("--out-model", default=os.path.join(ROOT, "data", "phase5_2_pr_value_function.json"))
    ap.add_argument("--out-metrics", default=os.path.join(ROOT, "data", "phase5_2_pr_value_function_metrics.json"))
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    # Feature list: prefer the static model's feature list for consistency
    feature_names = list(FALLBACK_FEATURES)
    if os.path.exists(args.static_model):
        try:
            static = json.load(open(args.static_model))
            base_features = static.get("features") or []
            if base_features:
                # Append position features that the static model doesn't have
                pos_feats = ["commit_position_frac", "commits_remaining_frac"]
                feature_names = base_features + [f for f in pos_feats if f not in base_features]
                print(f"Inherited {len(base_features)} features from {args.static_model}, "
                      f"added {len(pos_feats)} position features.")
        except Exception as e:
            print(f"Warning: could not load static model features ({e}), using fallback list.", file=sys.stderr)

    print(f"Connecting to postgres …")
    conn = _load_db()

    print(f"Loading checkpoints (run_tag={args.run_tag!r}, limit={args.limit}) …")
    checkpoints = _load_checkpoints(conn, args.run_tag, args.limit)
    print(f"  {len(checkpoints)} PRs with snapshot rows")

    instance_ids = [cp["instance_id"] for cp in checkpoints]
    print(f"Loading terminal labels for {len(instance_ids)} PRs …")
    labels = _load_terminal_labels(conn, instance_ids)
    conn.close()
    print(f"  {len(labels)} labels found")

    n_matched = sum(1 for cp in checkpoints if cp["instance_id"] in labels)
    print(f"  {n_matched}/{len(checkpoints)} PRs have labels")

    print("Building per-checkpoint dataset …")
    X, y_friction, y_refactor, groups, iids = _build_dataset(checkpoints, labels, feature_names)
    n_prs = len(set(iids))
    print(f"  {len(X)} checkpoint examples from {n_prs} PRs")
    print(f"  review_friction: mean={y_friction.mean():.3f} std={y_friction.std():.3f}")
    print(f"  refactor_requested: pos_rate={y_refactor.mean()*100:.1f}%  n_pos={y_refactor.sum()}")

    if len(X) < 50:
        print("ERROR: too few examples to train — check run_tag and postgres connection.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {args.n_splits}-fold GroupKFold CV …")
    cv = _cv_metrics(X, y_friction, y_refactor, groups, args.n_splits)
    print(f"  value head  R²={cv['value_cv_r2_mean']:.3f}±{cv['value_cv_r2_std']:.3f}"
          f"  ρ={cv['value_cv_spearman_mean']:.3f}±{cv['value_cv_spearman_std']:.3f}")
    if cv["refactor_early_warning_cv_auroc_mean"] is not None:
        print(f"  refactor early-warning AUROC={cv['refactor_early_warning_cv_auroc_mean']:.3f}"
              f"±{cv['refactor_early_warning_cv_auroc_std']:.3f}")

    print("Training final models on full dataset …")
    heads = _train_final(X, y_friction, y_refactor, feature_names)

    # Build output model JSON — same format as static model, adding value head
    model_json = {
        "version": "phase5_2_value_function_v1",
        "feature_source": "conway_patch_features",
        "features": feature_names,
        "log1p_features": sorted(LOG1P_FEATURES),
        "targets": {
            "value": "E[review_friction | Conway features at step t]  (lower = better)",
            "refactor_early_warning": "P(refactor_requested | Conway features at step t)",
        },
        "training": {
            "n_prs": n_prs,
            "n_checkpoints": int(len(X)),
            "pos_rate_refactor": float(y_refactor.mean()),
            "run_tag": args.run_tag,
        },
        **heads,
    }

    metrics_json = {
        **cv,
        "n_prs": n_prs,
        "n_checkpoints": int(len(X)),
        "feature_source": "conway_patch_features",
        "run_tag": args.run_tag,
    }

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    with open(args.out_model, "w") as f:
        json.dump(model_json, f, indent=2)
    with open(args.out_metrics, "w") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"\nSaved model  → {args.out_model}")
    print(f"Saved metrics → {args.out_metrics}")
    print("\nDone.")


if __name__ == "__main__":
    main()
