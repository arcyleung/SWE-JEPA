#!/usr/bin/env python3
"""
Stage 2 — Multi-Axis Probing Heads (Experiment 6.2).

Trains 12 independent LogReg heads on 256-dim projected embeddings, one per
Conway compound score axis.  Also trains the same heads on raw z_patch (2048)
as a comparison — contrastive projection should concentrate signal.

Includes acceptance + refactor heads for continuity with Experiment 6.1.

Inputs:
    data/phase6_2/projected_embeddings.npz         — h (N, 256), instance_ids, accepted, repos
    data/phase6_1/patch_embeddings_100k.npz         — z_patch (N, 2048) for comparison
    data/conway_patch_features_v2_100k.jsonl         — Conway compound scores

Outputs:
    data/phase6_2/multiaxis_heads.json   — 12 × {scaler_mean, scaler_scale, coef, intercept, threshold, category, hint}
    data/phase6_2/multiaxis_metrics.json — per-axis AUROC/PR-AUC on projected vs raw

Usage:
    python train_multiaxis_heads.py
    python train_multiaxis_heads.py --projected data/phase6_2/projected_embeddings.npz
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

DEFAULT_PROJECTED = os.path.join(PROJECT_ROOT, "data", "phase6_2", "projected_embeddings.npz")
DEFAULT_RAW = os.path.join(PROJECT_ROOT, "data", "phase6_1", "patch_embeddings_100k.npz")
DEFAULT_CONWAY = os.path.join(PROJECT_ROOT, "data", "conway_patch_features_v2_100k.jsonl")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "data", "phase6_2")

# ── 12 Conway compound score axes ────────────────────────────────────────────

AXES: list[dict] = [
    {"name": "trust_boundary_crossings",      "binarize": lambda v: int(v > 0),   "category": "security"},
    {"name": "error_contract_score",          "binarize": lambda v: int(v < 0),   "category": "error_handling"},
    {"name": "security_risk_score",           "binarize": lambda v: int(v > 0),   "category": "security"},
    {"name": "operability_score",             "binarize": lambda v: int(v < 0),   "category": "observability"},
    {"name": "api_change_without_tests",      "binarize": lambda v: int(v > 0),   "category": "api_surface"},
    {"name": "schema_change_without_migration", "binarize": lambda v: int(v > 0), "category": "scope"},
    {"name": "boundary_crossing_without_obs", "binarize": lambda v: int(v > 0),   "category": "observability"},
    {"name": "public_api_without_docs",       "binarize": lambda v: int(v > 0),   "category": "api_surface"},
    {"name": "dependency_change_without_tests", "binarize": lambda v: int(v > 0), "category": "scope"},
    {"name": "shared_change_isolated",        "binarize": lambda v: int(v > 0),   "category": "scope"},
    {"name": "external_io_without_safety",    "binarize": lambda v: int(v > 0),   "category": "security"},
    {"name": "ownership_diffusion",           "binarize": "p75",                  "category": "scope"},
]

AXIS_HINTS = {
    "trust_boundary_crossings": "Trust boundary crossed — add input validation or sanitization at the boundary",
    "error_contract_score": "Weak error handling — catch specific exceptions, avoid bare except/pass",
    "security_risk_score": "Security risk — parameterize queries, avoid eval/exec, no hardcoded secrets",
    "operability_score": "Low observability — add timeouts, error logging, and metrics at external calls",
    "api_change_without_tests": "Public API change without tests — add integration tests",
    "schema_change_without_migration": "Schema change without migration — add a migration file",
    "boundary_crossing_without_obs": "External call without observability — add logging at trust boundaries",
    "public_api_without_docs": "Public API without documentation — add docstrings for new interfaces",
    "dependency_change_without_tests": "Dependency change without tests — add tests covering the upgrade",
    "shared_change_isolated": "Shared utility modified in isolation — verify downstream consumers",
    "external_io_without_safety": "External I/O without safety — add timeouts and error handling",
    "ownership_diffusion": "High ownership spread — coordinate with code owners before modifying",
}


def _load_conway_scores(path: str) -> dict[str, dict[str, float]]:
    """Load Conway features keyed by instance_id → {axis_name: raw_value}."""
    records: dict[str, dict[str, float]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            iid = str(row.get("instance_id", ""))
            if not iid:
                continue
            scores = {}
            for ax in AXES:
                val = row.get(ax["name"])
                if val is not None:
                    scores[ax["name"]] = float(val)
            if scores:
                records[iid] = scores
    return records


def _binarize_labels(values: np.ndarray, rule) -> np.ndarray:
    """Apply binarization rule to an array of raw scores."""
    if rule == "p75":
        threshold = np.percentile(values, 75)
        return (values > threshold).astype(np.int32)
    return np.array([rule(v) for v in values], dtype=np.int32)


def _cv_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
               n_splits: int = 5, seed: int = 42) -> list[dict]:
    """GroupKFold CV returning per-fold AUROC + PR-AUC."""
    n_unique = len(np.unique(groups))
    if n_unique < n_splits:
        n_splits = max(2, n_unique)
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


def _summarize_folds(folds: list[dict]) -> dict:
    if not folds:
        return {"mean_auroc": 0.5, "mean_pr_auc": 0.0, "std_auroc": 0.0, "n_folds": 0}
    return {
        "mean_auroc": float(np.mean([f["auroc"] for f in folds])),
        "mean_pr_auc": float(np.mean([f["pr_auc"] for f in folds])),
        "std_auroc": float(np.std([f["auroc"] for f in folds])),
        "n_folds": len(folds),
        "folds": folds,
    }


def _fit_final(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Fit StandardScaler + LogReg on full data, return (scaler, clf)."""
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    if len(np.unique(y)) < 2:
        return sc, None
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                             solver="lbfgs", random_state=seed)
    clf.fit(Xs, y)
    return sc, clf


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: Multi-axis probing heads (Experiment 6.2)")
    ap.add_argument("--projected", default=DEFAULT_PROJECTED)
    ap.add_argument("--raw-embeddings", default=DEFAULT_RAW)
    ap.add_argument("--conway-features", default=DEFAULT_CONWAY)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load projected embeddings
    proj_data = np.load(args.projected, allow_pickle=True)
    h = proj_data["h"]  # (N, 256)
    instance_ids = proj_data["instance_ids"]
    accepted = proj_data["accepted"].astype(np.int32)
    repos = proj_data["repos"]
    print(f"Projected embeddings: {h.shape}")

    # Load raw embeddings for comparison
    raw_data = np.load(args.raw_embeddings, allow_pickle=True)
    z_patch = raw_data["z_patch"]  # (N, 2048)
    print(f"Raw embeddings: {z_patch.shape}")

    # Load Conway scores
    conway_map = _load_conway_scores(args.conway_features)
    print(f"Conway scores loaded: {len(conway_map):,} instances")

    # Join by instance_id
    id_to_idx = {str(iid): i for i, iid in enumerate(instance_ids)}
    matched_idxs = []
    matched_conway: list[dict[str, float]] = []
    for iid, scores in conway_map.items():
        if iid in id_to_idx:
            matched_idxs.append(id_to_idx[iid])
            matched_conway.append(scores)
    matched_idxs = np.array(matched_idxs)
    print(f"Joined embeddings ↔ Conway: {len(matched_idxs):,} samples")

    h_matched = h[matched_idxs]
    z_matched = z_patch[matched_idxs]
    groups_matched = np.array([str(repos[i]) for i in matched_idxs])
    accepted_matched = accepted[matched_idxs]

    # ── Per-axis heads ────────────────────────────────────────────────────
    results: dict[str, dict] = {}
    heads_out: dict[str, dict] = {}

    t0 = time.time()

    for ax in AXES:
        name = ax["name"]
        rule = ax["binarize"]
        category = ax["category"]

        # Extract raw values for this axis
        raw_vals = np.array([c.get(name, 0.0) for c in matched_conway], dtype=np.float64)
        y = _binarize_labels(raw_vals, rule)
        pos_rate = float(y.mean())
        n_pos = int(y.sum())

        print(f"\n--- {name} ({category}) ---")
        print(f"  Positive rate: {pos_rate:.3f} ({n_pos:,}/{len(y):,})")

        if n_pos < 10 or (len(y) - n_pos) < 10:
            print(f"  SKIPPED: too few positives or negatives")
            results[name] = {"category": category, "skipped": True, "n_pos": n_pos,
                             "pos_rate": pos_rate}
            continue

        # CV on projected
        proj_folds = _cv_scores(h_matched, y, groups_matched, args.cv_folds, args.seed)
        proj_summary = _summarize_folds(proj_folds)

        # CV on raw
        raw_folds = _cv_scores(z_matched, y, groups_matched, args.cv_folds, args.seed)
        raw_summary = _summarize_folds(raw_folds)

        print(f"  Projected AUROC: {proj_summary['mean_auroc']:.4f} (±{proj_summary['std_auroc']:.4f})")
        print(f"  Raw AUROC:       {raw_summary['mean_auroc']:.4f} (±{raw_summary['std_auroc']:.4f})")
        delta = proj_summary["mean_auroc"] - raw_summary["mean_auroc"]
        print(f"  Δ (proj-raw):    {delta:+.4f}")

        results[name] = {
            "category": category,
            "n_pos": n_pos,
            "pos_rate": pos_rate,
            "projected": proj_summary,
            "raw": raw_summary,
            "delta_auroc": delta,
        }

        # Fit final head on projected for deployment
        sc, clf = _fit_final(h_matched, y, args.seed)
        if clf is not None:
            # Optimal threshold at F1-max on train predictions
            pred = clf.predict_proba(sc.transform(h_matched))[:, 1]
            thresholds = np.linspace(0.1, 0.9, 81)
            best_f1, best_thr = 0.0, 0.5
            for thr in thresholds:
                tp = ((pred >= thr) & (y == 1)).sum()
                fp = ((pred >= thr) & (y == 0)).sum()
                fn = ((pred < thr) & (y == 1)).sum()
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                if f1 > best_f1:
                    best_f1, best_thr = f1, float(thr)

            heads_out[name] = {
                "scaler_mean": sc.mean_.tolist(),
                "scaler_scale": sc.scale_.tolist(),
                "coef": clf.coef_[0].tolist(),
                "intercept": float(clf.intercept_[0]),
                "threshold": best_thr,
                "category": category,
                "hint": AXIS_HINTS[name],
            }

    # ── Acceptance + refactor heads for continuity ────────────────────────
    print("\n--- acceptance (continuity with 6.1) ---")
    acc_proj_folds = _cv_scores(h_matched, accepted_matched, groups_matched, args.cv_folds, args.seed)
    acc_raw_folds = _cv_scores(z_matched, accepted_matched, groups_matched, args.cv_folds, args.seed)
    acc_proj = _summarize_folds(acc_proj_folds)
    acc_raw = _summarize_folds(acc_raw_folds)
    print(f"  Projected AUROC: {acc_proj['mean_auroc']:.4f}")
    print(f"  Raw AUROC:       {acc_raw['mean_auroc']:.4f}")
    results["acceptance"] = {
        "category": "meta",
        "projected": acc_proj,
        "raw": acc_raw,
        "delta_auroc": acc_proj["mean_auroc"] - acc_raw["mean_auroc"],
    }

    # Refactor = review_friction from Conway
    refactor_vals = np.array([c.get("review_friction", 0) for c in matched_conway], dtype=np.int32)
    if refactor_vals.sum() > 10:
        print("\n--- refactor (continuity with 6.1) ---")
        ref_proj_folds = _cv_scores(h_matched, refactor_vals, groups_matched, args.cv_folds, args.seed)
        ref_raw_folds = _cv_scores(z_matched, refactor_vals, groups_matched, args.cv_folds, args.seed)
        ref_proj = _summarize_folds(ref_proj_folds)
        ref_raw = _summarize_folds(ref_raw_folds)
        print(f"  Projected AUROC: {ref_proj['mean_auroc']:.4f}")
        print(f"  Raw AUROC:       {ref_raw['mean_auroc']:.4f}")
        results["refactor"] = {
            "category": "meta",
            "projected": ref_proj,
            "raw": ref_raw,
            "delta_auroc": ref_proj["mean_auroc"] - ref_raw["mean_auroc"],
        }

    elapsed = time.time() - t0
    print(f"\nAll heads trained in {elapsed:.1f}s")

    # ── Summary stats ─────────────────────────────────────────────────────
    axis_names = [ax["name"] for ax in AXES]
    active_axes = [n for n in axis_names if n in results and not results[n].get("skipped")]
    above_060 = [n for n in active_axes if results[n]["projected"]["mean_auroc"] > 0.60]
    proj_wins = [n for n in active_axes
                 if results[n]["projected"]["mean_auroc"] > results[n]["raw"]["mean_auroc"]]

    summary = {
        "n_axes_total": len(AXES),
        "n_axes_active": len(active_axes),
        "n_axes_above_060": len(above_060),
        "n_proj_wins": len(proj_wins),
        "axes_above_060": above_060,
        "train_time_s": round(elapsed, 1),
    }
    results["_summary"] = summary

    print(f"\n{'='*60}")
    print(f"Axes with AUROC > 0.60: {len(above_060)}/{len(active_axes)} "
          f"(target: ≥6/12)")
    print(f"Projected wins: {len(proj_wins)}/{len(active_axes)}")
    if "acceptance" in results:
        print(f"Acceptance AUROC (projected): {results['acceptance']['projected']['mean_auroc']:.4f} "
              f"(target: ≥0.90)")
    print(f"{'='*60}")

    # ── Save outputs ──────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    heads_path = os.path.join(args.out_dir, "multiaxis_heads.json")
    with open(heads_path, "w") as f:
        json.dump(heads_out, f, indent=2)
    print(f"Saved heads → {heads_path} ({len(heads_out)} axes)")

    metrics_path = os.path.join(args.out_dir, "multiaxis_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics → {metrics_path}")


if __name__ == "__main__":
    main()
