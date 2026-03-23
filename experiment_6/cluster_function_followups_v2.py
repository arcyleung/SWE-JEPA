#!/usr/bin/env python3
"""
Cluster PRs by patch embeddings, profile by function-level followup outcomes.

Pulls function-level followup data directly from postgres (followups_function),
joins with patch embeddings (phase6_1), and optionally Conway features.
Clusters in UMAP space, then profiles each cluster by bugfix/feature/maintenance
followup rates.

Inputs:
    postgres followups_function table   — function-level followup hits
    data/phase6_1/patch_embeddings_100k.npz — z_patch (N, 2048)
    data/conway_patch_features_v2_100k.jsonl — Conway features (optional)

Outputs (data/phase6_2/function_followup_clusters/):
    super_clusters.json            — per-cluster profiles
    super_cluster_assignments.npz  — cluster labels
    cluster_summary.json

Usage:
    python cluster_function_followups_v2.py
    python cluster_function_followups_v2.py --n-super-clusters 20
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pg8000.native
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

DEFAULT_EMBEDDINGS = os.path.join(PROJECT_ROOT, "data", "phase6_1", "patch_embeddings_100k.npz")
DEFAULT_CONWAY = os.path.join(PROJECT_ROOT, "data", "conway_patch_features_v2_100k.jsonl")
DEFAULT_PG_CONFIG = os.path.join(PROJECT_ROOT, "postgres_connection.yaml")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "data", "phase6_2", "function_followup_clusters")

# Feature names we'll profile clusters on (from followup aggregation + Conway)
FOLLOWUP_FEATURE_NAMES = [
    "n_bugfix", "n_feature", "n_maintenance", "n_docs", "n_total",
    "max_overlap", "n_distinct_functions", "has_bugfix",
]

CONWAY_EXCLUDE = frozenset({
    "accepted", "review_friction", "n_review_threads", "n_comments", "is_draft",
    "pull_number",
})


def _load_followup_stats(pg_config: str) -> dict[str, dict[str, float]]:
    """Aggregate followups_function by instance_id → per-PR followup stats."""
    cfg = yaml.safe_load(open(pg_config))
    conn = pg8000.native.Connection(
        host=cfg["ip"], port=int(cfg["port"]),
        user=cfg["user"], password=cfg["password"], database=cfg["database"],
    )
    rows = conn.run("""
        SELECT feature_instance_id,
               SUM(CASE WHEN followup_category = 'bugfix' THEN 1 ELSE 0 END),
               SUM(CASE WHEN followup_category = 'feature' THEN 1 ELSE 0 END),
               SUM(CASE WHEN followup_category = 'maintenance' THEN 1 ELSE 0 END),
               SUM(CASE WHEN followup_category = 'docs' THEN 1 ELSE 0 END),
               COUNT(*),
               MAX(hunk_overlap_fraction),
               COUNT(DISTINCT feature_function)
        FROM followups_function
        GROUP BY feature_instance_id
    """)
    conn.close()

    stats: dict[str, dict[str, float]] = {}
    for row in rows:
        iid = str(row[0])
        stats[iid] = {
            "n_bugfix": float(row[1]),
            "n_feature": float(row[2]),
            "n_maintenance": float(row[3]),
            "n_docs": float(row[4]),
            "n_total": float(row[5]),
            "max_overlap": float(row[6] or 0),
            "n_distinct_functions": float(row[7]),
            "has_bugfix": float(1 if row[1] > 0 else 0),
        }
    return stats


def _load_conway_all(path: str) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Load all numeric Conway features keyed by instance_id."""
    records: dict[str, dict[str, float]] = {}
    feature_names: list[str] | None = None
    if not os.path.exists(path):
        return records, []
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS)
    ap.add_argument("--conway-features", default=DEFAULT_CONWAY)
    ap.add_argument("--pg-config", default=DEFAULT_PG_CONFIG)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-super-clusters", type=int, default=20)
    ap.add_argument("--umap-dim", type=int, default=15)
    ap.add_argument("--umap-neighbors", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load embeddings
    emb_data = np.load(args.embeddings, allow_pickle=True)
    z_patch = emb_data["z_patch"]
    instance_ids = emb_data["instance_ids"]
    accepted = emb_data["accepted"].astype(np.int32)
    repos = emb_data["repos"]
    id_to_idx = {str(iid): i for i, iid in enumerate(instance_ids)}
    print(f"Embeddings: {z_patch.shape}")

    # Load followup stats from postgres
    print("Loading followup stats from postgres...")
    followup_stats = _load_followup_stats(args.pg_config)
    print(f"PRs with function-level followups: {len(followup_stats):,}")

    # Load Conway features
    conway_map, conway_names = _load_conway_all(args.conway_features)
    print(f"Conway features: {len(conway_map):,} instances, {len(conway_names)} features")

    # Join: only PRs that have both embeddings and followup data
    matched_idxs = []
    matched_followup: list[dict[str, float]] = []
    matched_conway: list[dict[str, float]] = []
    for iid, fstats in followup_stats.items():
        if iid in id_to_idx:
            matched_idxs.append(id_to_idx[iid])
            matched_followup.append(fstats)
            matched_conway.append(conway_map.get(iid, {}))

    matched_idxs = np.array(matched_idxs)
    print(f"Matched (embeddings ∩ followups): {len(matched_idxs):,}")

    if len(matched_idxs) < 50:
        print("ERROR: Too few matched PRs for clustering")
        return

    z_matched = z_patch[matched_idxs]
    accepted_matched = accepted[matched_idxs]
    repos_matched = repos[matched_idxs]

    # Build feature matrix for profiling (followup + Conway)
    all_profile_names = FOLLOWUP_FEATURE_NAMES + conway_names
    feat_matrix = np.zeros((len(matched_idxs), len(all_profile_names)), dtype=np.float64)
    for i in range(len(matched_idxs)):
        for j, name in enumerate(all_profile_names):
            if name in matched_followup[i]:
                feat_matrix[i, j] = matched_followup[i][name]
            elif name in matched_conway[i]:
                feat_matrix[i, j] = matched_conway[i][name]

    global_mean = feat_matrix.mean(axis=0)
    global_std = feat_matrix.std(axis=0)
    global_std[global_std < 1e-8] = 1.0

    # Extract key label arrays
    has_bugfix = np.array([f["has_bugfix"] for f in matched_followup], dtype=np.int32)
    n_bugfix = np.array([f["n_bugfix"] for f in matched_followup], dtype=np.float64)
    n_total = np.array([f["n_total"] for f in matched_followup], dtype=np.float64)

    print(f"Bugfix rate: {has_bugfix.mean():.3f}, mean n_bugfix: {n_bugfix.mean():.2f}, "
          f"acceptance rate: {accepted_matched.mean():.3f}")

    # ── UMAP + KMeans ─────────────────────────────────────────────────────
    import umap
    from sklearn.cluster import KMeans

    print(f"\nUMAP {z_matched.shape[1]} → {args.umap_dim}...")
    reducer = umap.UMAP(
        n_components=args.umap_dim,
        n_neighbors=args.umap_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=args.seed,
    )
    z_reduced = reducer.fit_transform(z_matched)
    print(f"UMAP done: {z_reduced.shape}")

    n_super = args.n_super_clusters
    print(f"KMeans (k={n_super})...")
    km = KMeans(n_clusters=n_super, random_state=args.seed, n_init=10)
    cluster_labels = km.fit_predict(z_reduced)

    # ── Profile each cluster ──────────────────────────────────────────────
    super_clusters_out: dict[str, dict] = {}

    for sid in range(n_super):
        mask = cluster_labels == sid
        count = int(mask.sum())
        if count == 0:
            continue

        bugfix_rate = float(has_bugfix[mask].mean())
        acc_rate = float(accepted_matched[mask].mean())
        mean_n_bugfix = float(n_bugfix[mask].mean())
        mean_n_total = float(n_total[mask].mean())
        n_repos = len(set(str(r) for r in repos_matched[mask]))

        # Z-scores vs global
        sc_mean = feat_matrix[mask].mean(axis=0)
        z_scores = (sc_mean - global_mean) / global_std
        ranked = sorted(enumerate(z_scores), key=lambda x: abs(x[1]), reverse=True)

        top_features = []
        for idx, z_val in ranked[:6]:
            name = all_profile_names[idx]
            top_features.append({
                "feature": name,
                "z_score": round(float(z_val), 3),
                "direction": "high" if z_val > 0 else "low",
                "cluster_mean": round(float(sc_mean[idx]), 4),
                "global_mean": round(float(global_mean[idx]), 4),
            })

        # Risk tier
        global_bugfix_rate = float(has_bugfix.mean())
        if bugfix_rate > global_bugfix_rate + 0.10:
            risk_tier = "HIGH"
        elif bugfix_rate < global_bugfix_rate - 0.10:
            risk_tier = "LOW"
        else:
            risk_tier = "MED"

        # Readable label from top-3 features
        def _readable(feat: dict) -> str:
            short = (feat["feature"]
                     .replace("n_distinct_", "n-")
                     .replace("has_", "")
                     .replace("n_", "n-")
                     .replace("max_", "max-")
                     .replace("_", "-"))
            arrow = "↑" if feat["direction"] == "high" else "↓"
            return f"{short}({arrow})"

        label = " / ".join(_readable(f) for f in top_features[:3])

        super_clusters_out[str(sid)] = {
            "label": label,
            "risk_tier": risk_tier,
            "bugfix_rate": round(bugfix_rate, 4),
            "acceptance_rate": round(acc_rate, 4),
            "mean_n_bugfix": round(mean_n_bugfix, 3),
            "mean_n_total": round(mean_n_total, 3),
            "count": count,
            "n_repos": n_repos,
            "top_features": top_features,
        }

        print(f"  Super {sid:2d}: n={count:4d}, repos={n_repos:3d}, "
              f"bugfix={bugfix_rate:.3f} [{risk_tier:4s}], acc={acc_rate:.3f}, "
              f"label={label}")
        for feat in top_features[:3]:
            print(f"           {feat['feature']:40s}  z={feat['z_score']:+.2f}  "
                  f"cluster={feat['cluster_mean']:.3f}  global={feat['global_mean']:.3f}")

    # ── Summary ───────────────────────────────────────────────────────────
    bugfix_rates = [v["bugfix_rate"] for v in super_clusters_out.values()]
    acc_rates = [v["acceptance_rate"] for v in super_clusters_out.values()]
    high_risk = [k for k, v in super_clusters_out.items() if v["risk_tier"] == "HIGH"]
    low_risk = [k for k, v in super_clusters_out.items() if v["risk_tier"] == "LOW"]

    print(f"\n{'='*60}")
    print(f"Global bugfix rate: {has_bugfix.mean():.3f}")
    print(f"Bugfix spread: {min(bugfix_rates):.3f} – {max(bugfix_rates):.3f} "
          f"(Δ={max(bugfix_rates)-min(bugfix_rates):.3f})")
    print(f"Acceptance spread: {min(acc_rates):.3f} – {max(acc_rates):.3f}")
    print(f"HIGH risk: {len(high_risk)}/{len(super_clusters_out)}")
    print(f"LOW risk: {len(low_risk)}/{len(super_clusters_out)}")
    print(f"{'='*60}")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "super_clusters.json"), "w") as f:
        json.dump(super_clusters_out, f, indent=2)

    np.savez_compressed(
        os.path.join(args.out_dir, "super_cluster_assignments.npz"),
        cluster_labels=cluster_labels,
        instance_ids=np.array([str(instance_ids[i]) for i in matched_idxs]),
        matched_idxs=matched_idxs,
    )

    summary = {
        "n_matched": len(matched_idxs),
        "n_super_clusters": len(super_clusters_out),
        "global_bugfix_rate": round(float(has_bugfix.mean()), 4),
        "global_acceptance_rate": round(float(accepted_matched.mean()), 4),
        "bugfix_rate_range": [round(min(bugfix_rates), 4), round(max(bugfix_rates), 4)],
        "acceptance_rate_range": [round(min(acc_rates), 4), round(max(acc_rates), 4)],
        "high_risk_clusters": high_risk,
        "low_risk_clusters": low_risk,
        "n_followup_features": len(FOLLOWUP_FEATURE_NAMES),
        "n_conway_features": len(conway_names),
        "n_total_profile_features": len(all_profile_names),
    }
    with open(os.path.join(args.out_dir, "cluster_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
