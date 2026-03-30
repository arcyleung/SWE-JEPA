#!/usr/bin/env python3
"""Build a canonical PR-level teacher set for Experiment 7.1.

The original phase-6 teacher artifacts contain multiple rows per `instance_id`
because `prs_copy` stores repeated snapshots of the same PR across crawl times.
Experiment 7.1 joins patches by `instance_id`, so it needs exactly one teacher
row per PR.

This script canonicalizes the teacher to one row per PR by:

1. Picking the latest available `prs_copy` snapshot per `instance_id`.
2. Keeping all closed-table rows (already unique).
3. Aggregating duplicate phase-6 latents per `instance_id` with a mean vector.
4. Reassigning the deduped latent to the nearest existing super-cluster centroid.

This preserves the existing phase-6 representation space while producing a
stable, unique teacher set that the 7.1 preprocessing pipeline can join safely.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pg8000.native
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

DEFAULT_PROJECTED = os.path.join(PROJECT_ROOT, "data", "phase6_2", "projected_embeddings.npz")
DEFAULT_SUPER_CLUSTERS = os.path.join(PROJECT_ROOT, "data", "phase6_2", "super_cluster_assignments.npz")
DEFAULT_PG_CONFIG = os.path.join(PROJECT_ROOT, "postgres_connection.yaml")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "data", "phase6_2_canonical_latest")


def _load_db(cfg_path: str) -> pg8000.native.Connection:
    cfg = yaml.safe_load(open(cfg_path))
    return pg8000.native.Connection(
        host=cfg["ip"],
        port=cfg.get("port", 9999),
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
    )


def _fetch_canonical_rows(cfg_path: str) -> list[dict[str, Any]]:
    conn = _load_db(cfg_path)
    rows = conn.run(
        """
        WITH canonical_merged AS (
            SELECT
                instance_id,
                repo,
                pr_merged AS accepted,
                COALESCE(crawl_time, updated_at, merged_at, created_at) AS canonical_time,
                LENGTH(patch) AS patch_chars,
                ROW_NUMBER() OVER (
                    PARTITION BY instance_id
                    ORDER BY COALESCE(crawl_time, updated_at, merged_at, created_at) DESC NULLS LAST,
                             COALESCE(updated_at, merged_at, created_at) DESC NULLS LAST,
                             COALESCE(merged_at, created_at) DESC NULLS LAST,
                             created_at DESC NULLS LAST,
                             LENGTH(patch) DESC NULLS LAST,
                             id DESC NULLS LAST
                ) AS rn
            FROM prs_copy
            WHERE patch IS NOT NULL
        ),
        canonical_closed AS (
            SELECT
                REPLACE(repo, '/', '__') || '__' || pull_number AS instance_id,
                repo,
                false AS accepted,
                COALESCE(updated_at_db, updated_at, merged_at, created_at) AS canonical_time,
                LENGTH(patch) AS patch_chars
            FROM python_js_ts_rust_closed_prs
            WHERE patch IS NOT NULL
        )
        SELECT instance_id, repo, accepted, canonical_time, patch_chars, 'prs_copy' AS source
        FROM canonical_merged
        WHERE rn = 1
        UNION ALL
        SELECT instance_id, repo, accepted, canonical_time, patch_chars, 'python_js_ts_rust_closed_prs' AS source
        FROM canonical_closed
        WHERE instance_id NOT IN (
            SELECT instance_id
            FROM canonical_merged
            WHERE rn = 1
        )
        ORDER BY instance_id
        """
    )
    conn.close()
    out = []
    for instance_id, repo, accepted, canonical_time, patch_chars, source in rows:
        out.append({
            "instance_id": str(instance_id),
            "repo": str(repo),
            "accepted": int(bool(accepted)),
            "canonical_time": canonical_time.isoformat() if canonical_time is not None else None,
            "patch_chars": int(patch_chars or 0),
            "source": str(source),
        })
    return out


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return vec.astype(np.float32, copy=False)
    return (vec / norm).astype(np.float32, copy=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projected", default=DEFAULT_PROJECTED)
    ap.add_argument("--super-clusters", default=DEFAULT_SUPER_CLUSTERS)
    ap.add_argument("--pg-config", default=DEFAULT_PG_CONFIG)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    print("Loading phase-6 teacher artifacts...", flush=True)
    projected = np.load(args.projected, allow_pickle=True)
    super_clusters = np.load(args.super_clusters, allow_pickle=True)

    h = projected["h"].astype(np.float32)
    projected_ids = np.asarray([str(x) for x in projected["instance_ids"]], dtype=np.str_)
    projected_repos = np.asarray([str(x) for x in projected["repos"]], dtype=np.str_)
    projected_accepted = projected["accepted"].astype(np.int64)
    super_ids = np.asarray([str(x) for x in super_clusters["instance_ids"]], dtype=np.str_)
    super_labels = super_clusters["super_cluster_labels"].astype(np.int64)

    if len(projected_ids) != len(super_ids):
        raise RuntimeError(
            "projected_embeddings and super_cluster_assignments have different lengths: "
            f"{len(projected_ids):,} vs {len(super_ids):,}"
        )
    if not np.array_equal(projected_ids, super_ids):
        raise RuntimeError("projected_embeddings and super_cluster_assignments are not row-aligned.")

    duplicate_counter = Counter(projected_ids.tolist())
    duplicate_ids = {iid: n for iid, n in duplicate_counter.items() if n > 1}
    print(
        f"Teacher rows: {len(projected_ids):,} total, {len(duplicate_counter):,} unique, "
        f"{len(duplicate_ids):,} duplicate ids",
        flush=True,
    )

    print("Fetching canonical DB rows...", flush=True)
    canonical_rows = _fetch_canonical_rows(args.pg_config)
    print(f"Canonical DB rows: {len(canonical_rows):,}", flush=True)

    by_iid: dict[str, list[int]] = defaultdict(list)
    for idx, iid in enumerate(projected_ids.tolist()):
        by_iid[iid].append(idx)

    unique_labels = np.unique(super_labels)
    centroids = []
    centroid_labels = []
    for label in unique_labels:
        mask = super_labels == label
        centroid = _l2_normalize(h[mask].mean(axis=0))
        centroids.append(centroid)
        centroid_labels.append(int(label))
    centroid_matrix = np.stack(centroids).astype(np.float32, copy=False)
    centroid_labels_arr = np.asarray(centroid_labels, dtype=np.int64)

    out_h = []
    out_ids = []
    out_repos = []
    out_accepted = []
    out_clusters = []
    duplicate_group_sizes = []
    majority_ties = 0
    cluster_mixed = 0
    missing_in_teacher = []
    repo_mismatch = 0
    accepted_mismatch = 0

    for row in canonical_rows:
        iid = row["instance_id"]
        idxs = by_iid.get(iid)
        if not idxs:
            missing_in_teacher.append(iid)
            continue

        teacher_repos = {projected_repos[i] for i in idxs}
        teacher_accepted = {int(projected_accepted[i]) for i in idxs}
        if row["repo"] not in teacher_repos:
            repo_mismatch += 1
        if row["accepted"] not in teacher_accepted:
            accepted_mismatch += 1

        group_latents = h[idxs]
        group_labels = [int(super_labels[i]) for i in idxs]
        agg_latent = _l2_normalize(group_latents.mean(axis=0))
        sims = centroid_matrix @ agg_latent
        agg_cluster = int(centroid_labels_arr[int(np.argmax(sims))])

        label_counts = Counter(group_labels).most_common()
        if len(label_counts) > 1 and label_counts[0][1] == label_counts[1][1]:
            majority_ties += 1
        if len(set(group_labels)) > 1:
            cluster_mixed += 1

        out_h.append(agg_latent)
        out_ids.append(iid)
        out_repos.append(row["repo"])
        out_accepted.append(int(row["accepted"]))
        out_clusters.append(agg_cluster)
        duplicate_group_sizes.append(len(idxs))

    out_h_arr = np.stack(out_h).astype(np.float32, copy=False)
    out_ids_arr = np.asarray(out_ids, dtype=np.str_)
    out_repos_arr = np.asarray(out_repos, dtype=np.str_)
    out_accepted_arr = np.asarray(out_accepted, dtype=np.int64)
    out_clusters_arr = np.asarray(out_clusters, dtype=np.int64)

    os.makedirs(args.out_dir, exist_ok=True)
    projected_out = os.path.join(args.out_dir, "projected_embeddings.npz")
    super_out = os.path.join(args.out_dir, "super_cluster_assignments.npz")
    summary_out = os.path.join(args.out_dir, "canonical_teacher_summary.json")

    np.savez_compressed(
        projected_out,
        h=out_h_arr,
        instance_ids=out_ids_arr,
        accepted=out_accepted_arr,
        repos=out_repos_arr,
    )
    np.savez_compressed(
        super_out,
        super_cluster_labels=out_clusters_arr,
        instance_ids=out_ids_arr,
    )

    source_counts = Counter(row["source"] for row in canonical_rows)
    summary = {
        "input_projected": args.projected,
        "input_super_clusters": args.super_clusters,
        "output_projected": projected_out,
        "output_super_clusters": super_out,
        "n_input_teacher_rows": int(len(projected_ids)),
        "n_input_teacher_unique_ids": int(len(duplicate_counter)),
        "n_input_teacher_duplicate_ids": int(len(duplicate_ids)),
        "n_canonical_db_rows": int(len(canonical_rows)),
        "n_output_rows": int(len(out_ids_arr)),
        "n_missing_in_teacher": int(len(missing_in_teacher)),
        "missing_in_teacher_examples": missing_in_teacher[:20],
        "repo_mismatch_count": int(repo_mismatch),
        "accepted_mismatch_count": int(accepted_mismatch),
        "cluster_mixed_duplicate_ids": int(cluster_mixed),
        "majority_cluster_ties": int(majority_ties),
        "duplicate_group_size_mean": float(np.mean(duplicate_group_sizes)) if duplicate_group_sizes else 1.0,
        "duplicate_group_size_max": int(max(duplicate_group_sizes)) if duplicate_group_sizes else 1,
        "aggregation": {
            "latent": "mean_then_l2_normalize",
            "cluster": "nearest_super_cluster_centroid",
            "prs_copy_canonical_rule": (
                "latest COALESCE(crawl_time, updated_at, merged_at, created_at), "
                "then latest updated_at/merged_at/created_at, then longest patch, then highest id"
            ),
        },
        "source_counts": {k: int(v) for k, v in source_counts.items()},
    }
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nProjected -> {projected_out}")
    print(f"Clusters  -> {super_out}")
    print(f"Summary   -> {summary_out}")


if __name__ == "__main__":
    main()
