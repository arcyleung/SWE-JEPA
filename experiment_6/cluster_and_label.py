#!/usr/bin/env python3
"""
Stage 3 — HDBSCAN Clustering + Post-Hoc Conway Labeling (Experiment 6.2).

Discovers natural patch groupings in the 256-dim projected space, then labels
each cluster by its dominant Conway score profile for explainability.

Inputs:
    data/phase6_2/projected_embeddings.npz     — h (N, 256), instance_ids, accepted, repos
    data/conway_patch_features_v2_100k.jsonl     — Conway compound scores

Outputs:
    data/phase6_2/clusters.json           — {cluster_id: {label, top_axes, mean_profile, ...}}
    data/phase6_2/cluster_assignments.npz — cluster labels for all N patches

Usage:
    python cluster_and_label.py
    python cluster_and_label.py --min-cluster-size 100  # smaller clusters
"""
from __future__ import annotations

import argparse
import json
import os

import hdbscan
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

DEFAULT_PROJECTED = os.path.join(PROJECT_ROOT, "data", "phase6_2", "projected_embeddings.npz")
DEFAULT_CONWAY = os.path.join(PROJECT_ROOT, "data", "conway_patch_features_v2_100k.jsonl")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "data", "phase6_2")

AXIS_NAMES = [
    "trust_boundary_crossings",
    "error_contract_score",
    "security_risk_score",
    "operability_score",
    "api_change_without_tests",
    "schema_change_without_migration",
    "boundary_crossing_without_obs",
    "public_api_without_docs",
    "dependency_change_without_tests",
    "shared_change_isolated",
    "external_io_without_safety",
    "ownership_diffusion",
]

AXIS_CATEGORY = {
    "trust_boundary_crossings": "security",
    "security_risk_score": "security",
    "external_io_without_safety": "security",
    "error_contract_score": "error_handling",
    "operability_score": "observability",
    "boundary_crossing_without_obs": "observability",
    "api_change_without_tests": "api_surface",
    "public_api_without_docs": "api_surface",
    "schema_change_without_migration": "scope",
    "dependency_change_without_tests": "scope",
    "shared_change_isolated": "scope",
    "ownership_diffusion": "scope",
}


CONWAY_EXCLUDE = frozenset({
    "accepted", "review_friction", "n_review_threads", "n_comments", "is_draft",
    "pull_number",
})


def _load_conway_all(path: str) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Load ALL numeric Conway features keyed by instance_id.

    Returns (records, feature_names) where feature_names is the sorted list of
    numeric feature columns found in the first row.
    """
    records: dict[str, dict[str, float]] = {}
    feature_names: list[str] | None = None
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if feature_names is None:
                feature_names = sorted(
                    k for k, v in row.items()
                    if isinstance(v, (int, float))
                    and k not in CONWAY_EXCLUDE
                    and k not in ("repo", "instance_id", "primary_lang")
                )
            iid = str(row.get("instance_id", ""))
            if not iid:
                continue
            scores = {}
            for name in feature_names:
                val = row.get(name)
                if val is not None:
                    scores[name] = float(val)
            if scores:
                records[iid] = scores
    return records, feature_names or []


def _load_conway_scores(path: str) -> dict[str, dict[str, float]]:
    """Load Conway features keyed by instance_id → {axis_name: raw_value}.

    Legacy wrapper — only returns the 12 compound axes.
    """
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
            for name in AXIS_NAMES:
                val = row.get(name)
                if val is not None:
                    scores[name] = float(val)
            if scores:
                records[iid] = scores
    return records


def _label_cluster(mean_profile: dict[str, float], global_mean: dict[str, float],
                   global_std: dict[str, float]) -> tuple[str, list[str]]:
    """Generate semantic label from top-2 distinguishing axes by z-score."""
    z_scores: list[tuple[str, float]] = []
    for name in AXIS_NAMES:
        if name not in mean_profile or name not in global_mean:
            continue
        std = global_std.get(name, 1e-8)
        if std < 1e-8:
            continue
        z = (mean_profile[name] - global_mean[name]) / std
        z_scores.append((name, abs(z)))
    z_scores.sort(key=lambda x: x[1], reverse=True)
    top_axes = [name for name, _ in z_scores[:2]]

    if not top_axes:
        return "uncharacterised", []

    # Pick category with highest cumulative z-score across its axes
    cat_scores: dict[str, float] = {}
    for name, z in z_scores:
        cat = AXIS_CATEGORY.get(name, "unknown")
        cat_scores[cat] = cat_scores.get(cat, 0.0) + z
    top_cat = max(cat_scores, key=cat_scores.get) if cat_scores else "unknown"

    # Name from top-2 axes: strip common suffixes for readability
    def _short(name: str) -> str:
        return (name
                .replace("_without_", "-no-")
                .replace("_score", "")
                .replace("_crossings", "")
                .replace("_change", "")
                .replace("_isolated", "")
                .replace("_diffusion", ""))

    label = f"{top_cat}:{_short(top_axes[0])}"
    if len(top_axes) > 1:
        label += f"+{_short(top_axes[1])}"

    return label, top_axes


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3: HDBSCAN clustering + labeling (Experiment 6.2)")
    ap.add_argument("--projected", default=DEFAULT_PROJECTED)
    ap.add_argument("--conway-features", default=DEFAULT_CONWAY)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--min-cluster-size", type=int, default=200)
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--umap-dim", type=int, default=15,
                    help="UMAP reduction before HDBSCAN (256 → N). Speeds up clustering dramatically.")
    ap.add_argument("--umap-neighbors", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-super-clusters", type=int, default=20,
                    help="Agglomerative merge of HDBSCAN clusters into N super-clusters")
    args = ap.parse_args()

    # Load projected embeddings
    data = np.load(args.projected, allow_pickle=True)
    h = data["h"]  # (N, 256)
    instance_ids = data["instance_ids"]
    accepted = data["accepted"].astype(np.int32)
    repos = data["repos"]
    print(f"Projected embeddings: {h.shape}")

    # Load ALL Conway features (113 numeric columns)
    conway_map_all, all_feature_names = _load_conway_all(args.conway_features)
    print(f"Conway features loaded: {len(conway_map_all):,} instances, {len(all_feature_names)} features")

    # Build Conway lookup by embedding index
    id_to_conway: dict[int, dict[str, float]] = {}
    for i, iid in enumerate(instance_ids):
        s = str(iid)
        if s in conway_map_all:
            id_to_conway[i] = conway_map_all[s]
    print(f"Matched Conway ↔ embeddings: {len(id_to_conway):,}")

    # Compute global mean/std for ALL features
    all_scores: dict[str, list[float]] = {name: [] for name in all_feature_names}
    for scores in id_to_conway.values():
        for name in all_feature_names:
            if name in scores:
                all_scores[name].append(scores[name])
    global_mean = {name: float(np.mean(vals)) if vals else 0.0 for name, vals in all_scores.items()}
    global_std = {name: float(np.std(vals)) if len(vals) > 1 else 1.0 for name, vals in all_scores.items()}

    # Also keep the 12-axis subset for backward compat
    global_mean_12 = {k: v for k, v in global_mean.items() if k in AXIS_NAMES}
    global_std_12 = {k: v for k, v in global_std.items() if k in AXIS_NAMES}

    # ── UMAP reduction (256 → low-dim) for fast HDBSCAN ─────────────────
    import umap
    print(f"\nUMAP {h.shape[1]} → {args.umap_dim} (n_neighbors={args.umap_neighbors})...")
    reducer = umap.UMAP(
        n_components=args.umap_dim,
        n_neighbors=args.umap_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=args.seed,
        n_jobs=-1,
    )
    h_reduced = reducer.fit_transform(h)
    print(f"UMAP done: {h_reduced.shape}")

    # ── HDBSCAN ───────────────────────────────────────────────────────────
    print(f"\nRunning HDBSCAN (min_cluster_size={args.min_cluster_size}, "
          f"min_samples={args.min_samples}) on {args.umap_dim}-dim UMAP space...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
    )
    cluster_labels = clusterer.fit_predict(h_reduced)
    unique_labels = sorted(set(cluster_labels))
    n_clusters = len([l for l in unique_labels if l >= 0])
    n_noise = int((cluster_labels == -1).sum())
    print(f"Found {n_clusters} clusters, {n_noise:,} noise points ({n_noise/len(h)*100:.1f}%)")

    # ── Per-cluster profiling ─────────────────────────────────────────────
    clusters_out: dict[str, dict] = {}

    for cid in unique_labels:
        if cid < 0:
            continue
        mask = cluster_labels == cid
        count = int(mask.sum())
        acc_rate = float(accepted[mask].mean())

        # Conway profile for this cluster — all features
        cluster_scores: dict[str, list[float]] = {name: [] for name in all_feature_names}
        for i in np.where(mask)[0]:
            if i in id_to_conway:
                for name in all_feature_names:
                    if name in id_to_conway[i]:
                        cluster_scores[name].append(id_to_conway[i][name])

        mean_profile = {name: float(np.mean(vals)) if vals else 0.0
                        for name, vals in cluster_scores.items()}
        n_conway_matched = max(len(v) for v in cluster_scores.values()) if cluster_scores else 0

        # Centroid
        centroid = h[mask].mean(axis=0).tolist()

        # Label (using 12-axis subset for backward compat)
        mean_profile_12 = {k: v for k, v in mean_profile.items() if k in AXIS_NAMES}
        label, top_axes = _label_cluster(mean_profile_12, global_mean_12, global_std_12)

        clusters_out[str(cid)] = {
            "label": label,
            "top_axes": top_axes,
            "mean_profile": mean_profile,
            "acceptance_rate": acc_rate,
            "centroid": centroid,
            "count": count,
            "n_conway_matched": n_conway_matched,
        }
        print(f"  Cluster {cid:3d}: n={count:5d}, acc={acc_rate:.3f}, "
              f"label={label}, top={top_axes}")

    # ── Verify distinctness ───────────────────────────────────────────────
    all_top_axes = [c["top_axes"] for c in clusters_out.values() if c["top_axes"]]
    distinct_top = len(set(tuple(t) for t in all_top_axes))
    print(f"\nDistinct top-2 axis combinations: {distinct_top}/{len(all_top_axes)}")

    # ── Save outputs ──────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    clusters_path = os.path.join(args.out_dir, "clusters.json")
    with open(clusters_path, "w") as f:
        json.dump(clusters_out, f, indent=2)
    print(f"Saved clusters → {clusters_path}")

    assignments_path = os.path.join(args.out_dir, "cluster_assignments.npz")
    np.savez_compressed(assignments_path, cluster_labels=cluster_labels,
                        instance_ids=instance_ids)
    print(f"Saved assignments → {assignments_path}")

    # ── Agglomerative merge into super-clusters ─────────────────────────
    from scipy.cluster.hierarchy import linkage, fcluster

    n_super = min(args.n_super_clusters, n_clusters)
    if n_clusters > 1 and n_super >= 2:
        print(f"\n{'='*60}")
        print(f"Merging {n_clusters} HDBSCAN clusters → {n_super} super-clusters (Ward linkage)")

        # Build centroid matrix (in projected space) for non-noise clusters
        cids_sorted = sorted(int(k) for k in clusters_out.keys())
        centroids = np.array([clusters_out[str(c)]["centroid"] for c in cids_sorted])
        # Weight by cluster size for Ward linkage
        Z = linkage(centroids, method="ward")
        super_labels = fcluster(Z, t=n_super, criterion="maxclust")  # 1-indexed

        # Map HDBSCAN cluster → super-cluster
        hdb_to_super = {cid: int(sl) - 1 for cid, sl in zip(cids_sorted, super_labels)}

        # Assign every patch a super-cluster label (-1 for noise)
        super_assignments = np.full(len(cluster_labels), -1, dtype=np.int32)
        for i, cl in enumerate(cluster_labels):
            if cl >= 0 and cl in hdb_to_super:
                super_assignments[i] = hdb_to_super[cl]

        # Profile each super-cluster using ALL 113 Conway features
        super_clusters_out: dict[str, dict] = {}
        for sid in range(n_super):
            mask = super_assignments == sid
            count = int(mask.sum())
            if count == 0:
                continue
            acc_rate = float(accepted[mask].mean())

            # Member HDBSCAN clusters
            member_hdb = [c for c, s in hdb_to_super.items() if s == sid]

            # Full Conway feature profile
            sc_scores: dict[str, list[float]] = {name: [] for name in all_feature_names}
            for i in np.where(mask)[0]:
                if i in id_to_conway:
                    for name in all_feature_names:
                        if name in id_to_conway[i]:
                            sc_scores[name].append(id_to_conway[i][name])

            sc_mean = {name: float(np.mean(vals)) if vals else 0.0
                       for name, vals in sc_scores.items()}
            n_matched = max((len(v) for v in sc_scores.values()), default=0)

            # Rank features by |z-score| vs global — top distinguishing features
            z_ranked: list[tuple[str, float, float]] = []  # (name, z, direction)
            for name in all_feature_names:
                std = global_std.get(name, 1e-8)
                if std < 1e-8:
                    continue
                z = (sc_mean.get(name, 0.0) - global_mean.get(name, 0.0)) / std
                z_ranked.append((name, abs(z), z))
            z_ranked.sort(key=lambda x: x[1], reverse=True)

            top_features = []
            for name, abs_z, signed_z in z_ranked[:5]:
                direction = "high" if signed_z > 0 else "low"
                top_features.append({
                    "feature": name,
                    "z_score": round(signed_z, 3),
                    "direction": direction,
                    "cluster_mean": round(sc_mean.get(name, 0.0), 4),
                    "global_mean": round(global_mean.get(name, 0.0), 4),
                })

            # Generate readable label from top-2 features
            def _readable(feat: dict) -> str:
                name = feat["feature"]
                short = (name.replace("has_", "").replace("ast_delta_", "Δ")
                         .replace("ast_after_", "").replace("ast_before_", "")
                         .replace("_", "-"))
                return f"{short}({'↑' if feat['direction'] == 'high' else '↓'})"

            label_parts = [_readable(f) for f in top_features[:3]]
            label = " / ".join(label_parts) if label_parts else "uncharacterised"

            super_clusters_out[str(sid)] = {
                "label": label,
                "top_features": top_features,
                "acceptance_rate": round(acc_rate, 4),
                "count": count,
                "n_conway_matched": n_matched,
                "member_hdbscan_clusters": member_hdb,
                "mean_profile": sc_mean,
            }

            print(f"  Super {sid:2d}: n={count:6d}, acc={acc_rate:.3f}, "
                  f"label={label}")
            for feat in top_features[:3]:
                print(f"           {feat['feature']:40s}  z={feat['z_score']:+.2f}  "
                      f"cluster={feat['cluster_mean']:.3f}  global={feat['global_mean']:.3f}")

        # Save super-cluster outputs
        super_path = os.path.join(args.out_dir, "super_clusters.json")
        with open(super_path, "w") as f:
            json.dump(super_clusters_out, f, indent=2)
        print(f"\nSaved super-clusters → {super_path}")

        super_assign_path = os.path.join(args.out_dir, "super_cluster_assignments.npz")
        np.savez_compressed(super_assign_path, super_cluster_labels=super_assignments,
                            instance_ids=instance_ids)
        print(f"Saved super-cluster assignments → {super_assign_path}")

        # Verify distinctness of super-cluster labels
        super_labels_set = set(c["label"] for c in super_clusters_out.values())
        print(f"Distinct super-cluster labels: {len(super_labels_set)}/{len(super_clusters_out)}")
    else:
        print("\nSkipping super-cluster merge (not enough HDBSCAN clusters)")
        super_clusters_out = {}

    # Summary JSON
    summary = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": round(n_noise / len(h), 4),
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "umap_dim": args.umap_dim,
        "umap_neighbors": args.umap_neighbors,
        "clusters_with_200_plus": len([c for c in clusters_out.values() if c["count"] >= 200]),
        "distinct_top_axis_combos": distinct_top,
        "cluster_sizes": {k: v["count"] for k, v in clusters_out.items()},
        "n_super_clusters": len(super_clusters_out),
        "super_cluster_sizes": {k: v["count"] for k, v in super_clusters_out.items()},
    }
    summary_path = os.path.join(args.out_dir, "cluster_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary → {summary_path}")

    print(f"\n{'='*60}")
    print(f"Clusters ≥200 members: {summary['clusters_with_200_plus']} (target: ≥4)")
    print(f"Distinct top-axis combos: {distinct_top} (target: each cluster distinct)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
