#!/usr/bin/env python3
"""
Cluster function-level followup embeddings to find super-clusters that predict
bugfix followups, alongside acceptance and refactor risk.

Uses frozen teacher embeddings (Qwen3-8B layer-18, 4096-dim) for 6,651 function
anchors. Pipeline: UMAP → HDBSCAN → agglomerative merge → profile each
super-cluster by bugfix rate, org metrics, and function characteristics.

Inputs:
    followup_embs.npz              — teacher_embs (6651, 4096)
    followup_sigs.jsonl            — has_bugfix, n_bugfix_prs, has_feature, etc.
    followup_org_metrics.jsonl     — ownership_friction, interface_stress, etc.

Outputs (data/phase6_2/followup_clusters/):
    super_clusters.json            — per-cluster profiles with bugfix rates + top features
    super_cluster_assignments.npz  — cluster labels for all 6651 anchors
    cluster_summary.json           — summary stats

Usage:
    python cluster_followups.py
    python cluster_followups.py --n-super-clusters 15
"""
from __future__ import annotations

import argparse
import json
import os

import hdbscan
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

DEFAULT_EMBS = os.path.join(PROJECT_ROOT, "followup_embs.npz")
DEFAULT_SIGS = os.path.join(PROJECT_ROOT, "followup_sigs.jsonl")
DEFAULT_ORG = os.path.join(PROJECT_ROOT, "followup_org_metrics.jsonl")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "data", "phase6_2", "followup_clusters")

# All numeric features we'll profile clusters on
ORG_FEATURE_NAMES = [
    "commits_touching_file",
    "distinct_authors",
    "top_author_fraction",
    "author_entropy",
    "ownership_friction",
    "cochange_weighted_degree",
    "cochange_unique_neighbors",
    "cochange_cross_module_ratio",
    "interface_stress",
]

SIG_FEATURE_NAMES = [
    "has_bugfix",
    "n_bugfix_prs",
    "has_feature",
    "max_bugfix_overlap",
]

ALL_FEATURE_NAMES = SIG_FEATURE_NAMES + ORG_FEATURE_NAMES


def _load_sigs(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_org_metrics(path: str) -> dict[str, dict[str, float]]:
    """Load org metrics keyed by feature_instance_id + feature_function."""
    records: dict[str, dict[str, float]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = f"{row['feature_instance_id']}::{row['feature_function']}"
            metrics = {}
            for name in ORG_FEATURE_NAMES:
                val = row.get(name)
                if val is not None:
                    metrics[name] = float(val)
            records[key] = metrics
    return records


def _build_feature_matrix(sigs: list[dict], org_map: dict[str, dict[str, float]]) -> tuple[np.ndarray, list[str]]:
    """Build (N, F) feature matrix from sigs + org metrics for profiling."""
    rows = []
    for sig in sigs:
        key = f"{sig['feature_instance_id']}::{sig['feature_function']}"
        org = org_map.get(key, {})
        row = []
        for name in ALL_FEATURE_NAMES:
            if name in sig:
                row.append(float(sig.get(name, 0) or 0))
            elif name in org:
                row.append(float(org.get(name, 0) or 0))
            else:
                row.append(0.0)
        rows.append(row)
    return np.array(rows, dtype=np.float64), ALL_FEATURE_NAMES


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster followup embeddings → bugfix-predictive super-clusters")
    ap.add_argument("--embeddings", default=DEFAULT_EMBS)
    ap.add_argument("--sigs", default=DEFAULT_SIGS)
    ap.add_argument("--org-metrics", default=DEFAULT_ORG)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--emb-key", default="teacher_embs",
                    help="Key in NPZ: 'teacher_embs' or 'student_embs'")
    ap.add_argument("--min-cluster-size", type=int, default=30,
                    help="HDBSCAN min_cluster_size (lower than patch-level due to smaller dataset)")
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--umap-dim", type=int, default=15)
    ap.add_argument("--umap-neighbors", type=int, default=15)
    ap.add_argument("--n-super-clusters", type=int, default=15)
    ap.add_argument("--method", choices=["hdbscan", "kmeans"], default="kmeans",
                    help="Clustering method. kmeans forces even splits; hdbscan can leave mega-clusters.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load data
    emb_data = np.load(args.embeddings)
    z = emb_data[args.emb_key]  # (6651, 4096)
    print(f"Embeddings ({args.emb_key}): {z.shape}")

    sigs = _load_sigs(args.sigs)
    print(f"Signatures: {len(sigs)}")
    assert len(sigs) == z.shape[0], f"Mismatch: {len(sigs)} sigs vs {z.shape[0]} embeddings"

    org_map = _load_org_metrics(args.org_metrics)
    print(f"Org metrics: {len(org_map):,} entries")

    # Build feature matrix for profiling
    feat_matrix, feat_names = _build_feature_matrix(sigs, org_map)
    print(f"Feature matrix: {feat_matrix.shape} ({len(feat_names)} features)")

    # Repos for each anchor
    repos = np.array([s["feature_repo"] for s in sigs])
    has_bugfix = np.array([s["has_bugfix"] for s in sigs], dtype=np.int32)
    has_feature = np.array([s["has_feature"] for s in sigs], dtype=np.int32)
    n_bugfix = np.array([s["n_bugfix_prs"] for s in sigs], dtype=np.int32)

    print(f"Bugfix rate: {has_bugfix.mean():.3f}, Feature rate: {has_feature.mean():.3f}")

    # Global feature stats
    global_mean = feat_matrix.mean(axis=0)
    global_std = feat_matrix.std(axis=0)
    global_std[global_std < 1e-8] = 1.0

    # ── UMAP ──────────────────────────────────────────────────────────────
    import umap
    print(f"\nUMAP {z.shape[1]} → {args.umap_dim} (n_neighbors={args.umap_neighbors})...")
    reducer = umap.UMAP(
        n_components=args.umap_dim,
        n_neighbors=args.umap_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=args.seed,
    )
    z_reduced = reducer.fit_transform(z)
    print(f"UMAP done: {z_reduced.shape}")

    # ── Clustering ─────────────────────────────────────────────────────────
    if args.method == "hdbscan":
        print(f"\nHDBSCAN (min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples})...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            metric="euclidean",
        )
        hdb_labels = clusterer.fit_predict(z_reduced)
        n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
        n_noise = int((hdb_labels == -1).sum())
        print(f"Found {n_clusters} clusters, {n_noise:,} noise ({n_noise/len(z)*100:.1f}%)")

        if n_clusters < 2:
            print("ERROR: Not enough clusters found. Try lowering --min-cluster-size.")
            return

        # Agglomerative merge
        n_super = min(args.n_super_clusters, n_clusters)
        cids = sorted(set(hdb_labels) - {-1})
        centroids = np.array([z_reduced[hdb_labels == c].mean(axis=0) for c in cids])
        Z = linkage(centroids, method="ward")
        merge_labels = fcluster(Z, t=n_super, criterion="maxclust")
        hdb_to_super = {cid: int(sl) - 1 for cid, sl in zip(cids, merge_labels)}
        super_labels = np.full(len(z), -1, dtype=np.int32)
        for i, cl in enumerate(hdb_labels):
            if cl >= 0 and cl in hdb_to_super:
                super_labels[i] = hdb_to_super[cl]
        print(f"Merged {n_clusters} → {n_super} super-clusters")

    else:  # kmeans
        from sklearn.cluster import KMeans
        n_super = args.n_super_clusters
        print(f"\nKMeans (k={n_super}) on {args.umap_dim}-dim UMAP space...")
        km = KMeans(n_clusters=n_super, random_state=args.seed, n_init=10)
        super_labels = km.fit_predict(z_reduced)
        hdb_labels = super_labels  # no separate HDBSCAN step
        n_noise = 0
        n_clusters = n_super
        print(f"KMeans done: {n_super} clusters, all points assigned")

    # ── Profile each super-cluster ────────────────────────────────────────
    super_clusters_out: dict[str, dict] = {}

    for sid in range(n_super):
        mask = super_labels == sid
        count = int(mask.sum())
        if count == 0:
            continue

        bugfix_rate = float(has_bugfix[mask].mean())
        feature_rate = float(has_feature[mask].mean())
        mean_n_bugfix = float(n_bugfix[mask].mean())
        n_repos = len(set(repos[mask]))

        # Feature profile
        sc_mean = feat_matrix[mask].mean(axis=0)
        sc_std = feat_matrix[mask].std(axis=0)

        # Z-scores vs global
        z_scores = (sc_mean - global_mean) / global_std
        ranked = sorted(enumerate(z_scores), key=lambda x: abs(x[1]), reverse=True)

        top_features = []
        for idx, z_val in ranked[:5]:
            name = feat_names[idx]
            direction = "high" if z_val > 0 else "low"
            top_features.append({
                "feature": name,
                "z_score": round(float(z_val), 3),
                "direction": direction,
                "cluster_mean": round(float(sc_mean[idx]), 4),
                "global_mean": round(float(global_mean[idx]), 4),
            })

        # Readable label
        def _readable(feat: dict) -> str:
            short = (feat["feature"]
                     .replace("cochange_", "coch-")
                     .replace("_fraction", "")
                     .replace("_touching_file", "")
                     .replace("_entropy", "-ent"))
            arrow = "↑" if feat["direction"] == "high" else "↓"
            return f"{short}({arrow})"

        label = " / ".join(_readable(f) for f in top_features[:3])

        # Bugfix risk classification
        if bugfix_rate > has_bugfix.mean() + 0.10:
            risk_tier = "HIGH"
        elif bugfix_rate < has_bugfix.mean() - 0.10:
            risk_tier = "LOW"
        else:
            risk_tier = "MED"

        super_clusters_out[str(sid)] = {
            "label": label,
            "risk_tier": risk_tier,
            "bugfix_rate": round(bugfix_rate, 4),
            "feature_rate": round(feature_rate, 4),
            "mean_n_bugfix_prs": round(mean_n_bugfix, 3),
            "count": count,
            "n_repos": n_repos,
            "top_features": top_features,
            "mean_profile": {feat_names[i]: round(float(sc_mean[i]), 4)
                             for i in range(len(feat_names))},
        }

        print(f"  Super {sid:2d}: n={count:4d}, repos={n_repos:3d}, "
              f"bugfix={bugfix_rate:.3f} [{risk_tier:4s}], feat={feature_rate:.3f}, "
              f"label={label}")
        for feat in top_features[:3]:
            print(f"           {feat['feature']:35s}  z={feat['z_score']:+.2f}  "
                  f"cluster={feat['cluster_mean']:.3f}  global={feat['global_mean']:.3f}")

    # ── Summary stats ─────────────────────────────────────────────────────
    high_risk = [k for k, v in super_clusters_out.items() if v["risk_tier"] == "HIGH"]
    low_risk = [k for k, v in super_clusters_out.items() if v["risk_tier"] == "LOW"]

    bugfix_rates = [v["bugfix_rate"] for v in super_clusters_out.values()]
    bugfix_spread = max(bugfix_rates) - min(bugfix_rates) if bugfix_rates else 0

    print(f"\n{'='*60}")
    print(f"Global bugfix rate: {has_bugfix.mean():.3f}")
    print(f"Bugfix rate spread across super-clusters: {min(bugfix_rates):.3f} – {max(bugfix_rates):.3f} "
          f"(Δ={bugfix_spread:.3f})")
    print(f"HIGH risk clusters: {len(high_risk)}/{len(super_clusters_out)}")
    print(f"LOW risk clusters:  {len(low_risk)}/{len(super_clusters_out)}")
    print(f"{'='*60}")

    # ── Save outputs ──────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    sc_path = os.path.join(args.out_dir, "super_clusters.json")
    with open(sc_path, "w") as f:
        json.dump(super_clusters_out, f, indent=2)
    print(f"\nSaved super-clusters → {sc_path}")

    assign_path = os.path.join(args.out_dir, "super_cluster_assignments.npz")
    np.savez_compressed(assign_path,
                        super_cluster_labels=super_labels,
                        hdbscan_labels=hdb_labels,
                        instance_ids=np.array([s["feature_instance_id"] for s in sigs]),
                        functions=np.array([s["feature_function"] for s in sigs]))
    print(f"Saved assignments → {assign_path}")

    summary = {
        "n_anchors": len(sigs),
        "n_hdbscan_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": round(n_noise / len(z), 4),
        "n_super_clusters": len(super_clusters_out),
        "global_bugfix_rate": round(float(has_bugfix.mean()), 4),
        "global_feature_rate": round(float(has_feature.mean()), 4),
        "bugfix_rate_range": [round(min(bugfix_rates), 4), round(max(bugfix_rates), 4)],
        "high_risk_clusters": high_risk,
        "low_risk_clusters": low_risk,
        "emb_key": args.emb_key,
        "emb_dim": z.shape[1],
        "umap_dim": args.umap_dim,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "super_cluster_sizes": {k: v["count"] for k, v in super_clusters_out.items()},
    }
    summary_path = os.path.join(args.out_dir, "cluster_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary → {summary_path}")


if __name__ == "__main__":
    main()
