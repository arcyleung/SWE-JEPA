#!/usr/bin/env python3
"""
Train acceptance + refactor steering heads on frozen patch embeddings (Experiment 6.1).

Models:
  - LogReg-raw:    LogisticRegression on z_patch (2048)
  - LogReg-PCA256: PCA(256, whiten) + LogisticRegression
  - MLP:           2048 → 512 → 1, ReLU, BCE, 50 epochs

Evaluation: GroupKFold(5) by repo, AUROC + PR-AUC.
Phase 5.1 baselines: acceptance AUROC=0.71, refactor AUROC=0.59.

Usage:
    python train_embedding_steerer.py
    python train_embedding_steerer.py --embeddings data/phase6_1/patch_embeddings_100k.npz
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

DEFAULT_EMBEDDINGS = os.path.join(PROJECT_ROOT, "data", "phase6_1", "patch_embeddings_100k.npz")
DEFAULT_CONWAY_FEATURES = os.path.join(PROJECT_ROOT, "data", "conway_patch_features_v2_100k.jsonl")
DEFAULT_METRICS_OUT = os.path.join(PROJECT_ROOT, "data", "phase6_1", "embedding_steerer_metrics.json")

PHASE5_BASELINES = {
    "acceptance_auroc": 0.71,
    "refactor_auroc": 0.59,
}


def _load_refactor_labels(conway_path: str) -> dict[str, int]:
    """Load refactor_requested labels keyed by instance_id from Conway features JSONL."""
    labels: dict[str, int] = {}
    if not os.path.exists(conway_path):
        print(f"WARNING: {conway_path} not found, refactor labels unavailable")
        return labels
    with open(conway_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            iid = row.get("instance_id")
            if iid:
                labels[iid] = int(row.get("review_friction", 0) or 0)
    print(f"Loaded {len(labels):,} refactor labels from {conway_path}")
    return labels


def _cv_eval(X, y, groups, n_splits=5, seed=42, model_name="LogReg"):
    """Run GroupKFold CV, return list of {auroc, pr_auc} per fold."""
    if len(np.unique(groups)) < n_splits:
        n_splits = max(2, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for tr, te in gkf.split(X, groups=groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            folds.append({"auroc": 0.5, "pr_auc": float(np.mean(y[te]))})
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=seed)
        clf.fit(Xtr, y[tr])
        pred = clf.predict_proba(Xte)[:, 1]
        folds.append({
            "auroc": float(roc_auc_score(y[te], pred)),
            "pr_auc": float(average_precision_score(y[te], pred)),
        })
    return folds


def _cv_eval_mlp(X, y, groups, n_splits=5, seed=42, epochs=50, lr=1e-3, hidden=512):
    """Run GroupKFold CV with a simple MLP (PyTorch)."""
    import torch
    import torch.nn as nn

    if len(np.unique(groups)) < n_splits:
        n_splits = max(2, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    folds = []

    for tr, te in gkf.split(X, groups=groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            folds.append({"auroc": 0.5, "pr_auc": float(np.mean(y[te]))})
            continue

        sc = StandardScaler()
        Xtr = torch.tensor(sc.fit_transform(X[tr]), dtype=torch.float32)
        Xte = torch.tensor(sc.transform(X[te]), dtype=torch.float32)
        ytr = torch.tensor(y[tr], dtype=torch.float32)

        d_in = Xtr.shape[1]
        model = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)])
        )

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = model(Xtr).squeeze(-1)
            loss = loss_fn(logits, ytr)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(Xte).squeeze(-1)).numpy()

        folds.append({
            "auroc": float(roc_auc_score(y[te], pred)),
            "pr_auc": float(average_precision_score(y[te], pred)),
        })

    return folds


def _summarize_folds(folds: list[dict]) -> dict:
    if not folds:
        return {"mean_auroc": 0.5, "mean_pr_auc": 0.0, "folds": []}
    return {
        "mean_auroc": float(np.mean([f["auroc"] for f in folds])),
        "mean_pr_auc": float(np.mean([f["pr_auc"] for f in folds])),
        "std_auroc": float(np.std([f["auroc"] for f in folds])),
        "folds": folds,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS)
    ap.add_argument("--conway-features", default=DEFAULT_CONWAY_FEATURES)
    ap.add_argument("--metrics-out", default=DEFAULT_METRICS_OUT)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mlp-epochs", type=int, default=50)
    ap.add_argument("--pca-dim", type=int, default=256)
    args = ap.parse_args()

    # Load embeddings
    data = np.load(args.embeddings, allow_pickle=True)
    z_patch = data["z_patch"]  # (N, 2048)
    instance_ids = data["instance_ids"]  # (N,)
    y_acc = data["accepted"].astype(np.int32)  # (N,)
    repos = data["repos"]  # (N,)

    print(f"Loaded embeddings: {z_patch.shape}, accepted_rate={y_acc.mean():.3f}")
    assert not np.any(np.isnan(z_patch)), "NaN in embeddings!"

    # Load refactor labels
    refactor_map = _load_refactor_labels(args.conway_features)
    y_ref = np.array([refactor_map.get(str(iid), 0) for iid in instance_ids], dtype=np.int32)
    n_ref_matched = int(np.sum([1 for iid in instance_ids if str(iid) in refactor_map]))
    print(f"Refactor labels matched: {n_ref_matched:,}/{len(instance_ids):,}")

    groups = np.array([str(r) for r in repos])
    n_repos = len(np.unique(groups))
    print(f"Unique repos: {n_repos:,}")

    # Quick sanity: cosine sim merged vs closed
    merged_mask = y_acc == 1
    closed_mask = y_acc == 0
    if merged_mask.sum() > 0 and closed_mask.sum() > 0:
        merged_mean = z_patch[merged_mask].mean(axis=0)
        closed_mean = z_patch[closed_mask].mean(axis=0)
        cos_sim = float(np.dot(merged_mean, closed_mean) / (np.linalg.norm(merged_mean) * np.linalg.norm(closed_mean)))
        print(f"Cosine sim (merged centroid, closed centroid): {cos_sim:.4f}")
    else:
        cos_sim = None

    results = {"n_samples": len(y_acc), "n_repos": n_repos, "acceptance_rate": float(y_acc.mean()),
               "refactor_rate": float(y_ref.mean()), "refactor_labels_matched": n_ref_matched,
               "cosine_sim_merged_vs_closed": cos_sim, "phase5_baselines": PHASE5_BASELINES, "models": {}}

    # ── Model 1: LogReg-raw ──────────────────────────────────────────────────
    print("\n=== LogReg-raw (2048-dim) ===")
    t0 = time.time()
    acc_folds = _cv_eval(z_patch, y_acc, groups, args.cv_folds, args.seed, "LogReg-raw")
    ref_folds = _cv_eval(z_patch, y_ref, groups, args.cv_folds, args.seed + 1, "LogReg-raw")
    acc_summary = _summarize_folds(acc_folds)
    ref_summary = _summarize_folds(ref_folds)
    print(f"  Acceptance AUROC: {acc_summary['mean_auroc']:.4f} (±{acc_summary['std_auroc']:.4f})")
    print(f"  Refactor  AUROC: {ref_summary['mean_auroc']:.4f} (±{ref_summary['std_auroc']:.4f})")
    print(f"  Time: {time.time() - t0:.1f}s")
    results["models"]["logreg_raw"] = {"acceptance": acc_summary, "refactor": ref_summary}

    # ── Model 2: LogReg-PCA256 ───────────────────────────────────────────────
    print(f"\n=== LogReg-PCA{args.pca_dim} ===")
    t0 = time.time()
    pca = PCA(n_components=args.pca_dim, whiten=True, random_state=args.seed)
    z_pca = pca.fit_transform(z_patch)
    explained = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA explained variance: {explained:.4f}")
    acc_folds = _cv_eval(z_pca, y_acc, groups, args.cv_folds, args.seed, "LogReg-PCA")
    ref_folds = _cv_eval(z_pca, y_ref, groups, args.cv_folds, args.seed + 1, "LogReg-PCA")
    acc_summary = _summarize_folds(acc_folds)
    ref_summary = _summarize_folds(ref_folds)
    print(f"  Acceptance AUROC: {acc_summary['mean_auroc']:.4f} (±{acc_summary['std_auroc']:.4f})")
    print(f"  Refactor  AUROC: {ref_summary['mean_auroc']:.4f} (±{ref_summary['std_auroc']:.4f})")
    print(f"  Time: {time.time() - t0:.1f}s")
    results["models"]["logreg_pca"] = {
        "pca_dim": args.pca_dim, "pca_explained_variance": explained,
        "acceptance": acc_summary, "refactor": ref_summary,
    }

    # ── Model 3: MLP ────────────────────────────────────────────────────────
    print(f"\n=== MLP (2048→512→1, {args.mlp_epochs} epochs) ===")
    t0 = time.time()
    acc_folds = _cv_eval_mlp(z_patch, y_acc, groups, args.cv_folds, args.seed, args.mlp_epochs)
    ref_folds = _cv_eval_mlp(z_patch, y_ref, groups, args.cv_folds, args.seed + 1, args.mlp_epochs)
    acc_summary = _summarize_folds(acc_folds)
    ref_summary = _summarize_folds(ref_folds)
    print(f"  Acceptance AUROC: {acc_summary['mean_auroc']:.4f} (±{acc_summary['std_auroc']:.4f})")
    print(f"  Refactor  AUROC: {ref_summary['mean_auroc']:.4f} (±{ref_summary['std_auroc']:.4f})")
    print(f"  Time: {time.time() - t0:.1f}s")
    results["models"]["mlp"] = {"epochs": args.mlp_epochs, "acceptance": acc_summary, "refactor": ref_summary}

    # ── Gate check ───────────────────────────────────────────────────────────
    best_acc_auroc = max(
        m["acceptance"]["mean_auroc"] for m in results["models"].values()
    )
    gate_pass = best_acc_auroc >= PHASE5_BASELINES["acceptance_auroc"]
    results["gate"] = {
        "best_acceptance_auroc": best_acc_auroc,
        "threshold": PHASE5_BASELINES["acceptance_auroc"],
        "pass": gate_pass,
    }
    status = "PASS" if gate_pass else "FAIL"
    print(f"\n{'='*60}")
    print(f"Gate: best acceptance AUROC = {best_acc_auroc:.4f} vs threshold {PHASE5_BASELINES['acceptance_auroc']:.2f} → {status}")
    print(f"{'='*60}")

    # Write metrics
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics -> {args.metrics_out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
