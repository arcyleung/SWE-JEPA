#!/usr/bin/env python3
"""Precompute review-state training features for one Exp 7.1 shard.

This script moves the expensive CPU-side work out of the GPU training job:

- fetch unified diffs from Postgres
- tokenize patch text
- derive deterministic review tags
- serialize compact numeric arrays to `.npz`

Each Slurm task owns one shard via `--shard-modulus/--shard-remainder`.
The training job later loads the shard cache and only does model training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pg8000.native
import yaml
from transformers import AutoTokenizer

from review_state_bridge import TAG_NAMES, detect_review_issue_flags

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

DEFAULT_PROJECTED = os.path.join(PROJECT_ROOT, "data", "phase6_2", "projected_embeddings.npz")
DEFAULT_SUPER_CLUSTERS = os.path.join(PROJECT_ROOT, "data", "phase6_2", "super_cluster_assignments.npz")
DEFAULT_PG_CONFIG = os.path.join(PROJECT_ROOT, "postgres_connection.yaml")
DEFAULT_TOKENIZER = "/home/original_models/Qwen2.5-Coder-3B"
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "data", "phase7_1", "cache", "shard_000.npz")
DEFAULT_SUMMARY_OUT = os.path.join(PROJECT_ROOT, "data", "phase7_1", "cache", "shard_000_summary.json")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_db(cfg_path: str) -> pg8000.native.Connection:
    cfg = yaml.safe_load(open(cfg_path))
    return pg8000.native.Connection(
        host=cfg["ip"],
        port=cfg.get("port", 9999),
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
    )


def _fetch_patch_chunk(
    conn: pg8000.native.Connection,
    instance_ids: list[str],
) -> dict[str, str]:
    if not instance_ids:
        return {}
    quoted = ", ".join("'" + iid.replace("'", "''") + "'" for iid in instance_ids)
    rows = conn.run(
        f"""
        SELECT instance_id_key, patch
        FROM (
            SELECT instance_id AS instance_id_key, patch
            FROM prs_copy
            WHERE patch IS NOT NULL
            UNION ALL
            SELECT
                REPLACE(repo, '/', '__') || '__' || pull_number AS instance_id_key,
                patch
            FROM python_js_ts_rust_closed_prs
            WHERE patch IS NOT NULL
        ) AS combined
        WHERE instance_id_key IN ({quoted})
        """
    )
    return {str(instance_id): str(patch) for instance_id, patch in rows}


def _load_teacher_metadata(projected_path: str, super_clusters_path: str) -> dict[str, Any]:
    projected = np.load(projected_path, allow_pickle=True)
    super_clusters = np.load(super_clusters_path, allow_pickle=True)

    h = projected["h"].astype(np.float32)
    instance_ids = projected["instance_ids"]
    accepted = projected["accepted"].astype(np.int64)
    repos = projected["repos"]
    cluster_ids = super_clusters["super_cluster_labels"].astype(np.int64)
    cluster_iids = super_clusters["instance_ids"]

    if len(instance_ids) != len(cluster_iids):
        raise RuntimeError(
            "projected_embeddings and super_cluster_assignments have different lengths: "
            f"{len(instance_ids):,} vs {len(cluster_iids):,}"
        )
    if any(str(lhs) != str(rhs) for lhs, rhs in zip(instance_ids, cluster_iids)):
        raise RuntimeError(
            "projected_embeddings and super_cluster_assignments are not row-aligned. "
            "Cannot build teacher metadata safely."
        )
    dup_counter = Counter(str(iid) for iid in instance_ids)
    n_duplicate_instance_ids = sum(1 for n in dup_counter.values() if n > 1)
    max_duplicate_instance_id_count = max(dup_counter.values(), default=0)
    rows = []
    for i, iid in enumerate(instance_ids):
        rows.append({
            "instance_id": str(iid),
            "repo": str(repos[i]),
            "accepted": int(accepted[i]),
            "latent": h[i],
            "cluster_id": int(cluster_ids[i]),
        })
    return {
        "rows": rows,
        "n_missing_cluster": 0,
        "n_duplicate_instance_ids": n_duplicate_instance_ids,
        "max_duplicate_instance_id_count": max_duplicate_instance_id_count,
        "n_unique_instance_ids": len(dup_counter),
    }


def _detect_review_issue_flags_batch(patch_texts: list[str]) -> list[dict[str, int]]:
    return [detect_review_issue_flags(patch_text) for patch_text in patch_texts]


def _extract_tag_rows(
    patch_texts: list[str],
    preprocess_workers: int,
    tag_batch_size: int,
    tag_pool: ProcessPoolExecutor | None = None,
) -> list[dict[str, int]]:
    if preprocess_workers <= 1 or len(patch_texts) <= 1:
        return [detect_review_issue_flags(patch_text) for patch_text in patch_texts]

    batch_size = max(1, int(tag_batch_size))
    patch_batches = [
        patch_texts[start:start + batch_size]
        for start in range(0, len(patch_texts), batch_size)
    ]
    owns_pool = tag_pool is None
    if owns_pool:
        ctx = mp.get_context("spawn")
        tag_pool = ProcessPoolExecutor(max_workers=preprocess_workers, mp_context=ctx)
    try:
        tag_parts = list(tag_pool.map(_detect_review_issue_flags_batch, patch_batches, chunksize=1))
    finally:
        if owns_pool and tag_pool is not None:
            tag_pool.shutdown()
    return [item for part in tag_parts for item in part]


def _preprocess_rows(
    rows: list[dict[str, Any]],
    cfg_path: str,
    tokenizer: Any,
    max_tokens: int,
    hash_vocab_size: int,
    max_patch_chars: int,
    preprocess_workers: int,
    tag_batch_size: int,
    fetch_chunk_size: int,
    progress_every: int,
    split_name: str,
) -> dict[str, Any]:
    n_rows = len(rows)
    n_chunks = max(1, math.ceil(n_rows / fetch_chunk_size))
    latent_dim = int(rows[0]["latent"].shape[0]) if rows else 0

    flat_ids_parts: list[np.ndarray] = []
    lengths: list[int] = []
    latent_parts: list[np.ndarray] = []
    cluster_ids: list[int] = []
    accepted_parts: list[float] = []
    tag_parts: list[np.ndarray] = []
    repo_parts: list[str] = []
    iid_parts: list[str] = []
    n_missing_patch = 0
    n_joined = 0

    conn = _load_db(cfg_path)
    tag_pool: ProcessPoolExecutor | None = None
    if preprocess_workers > 1:
        ctx = mp.get_context("spawn")
        tag_pool = ProcessPoolExecutor(max_workers=preprocess_workers, mp_context=ctx)
    try:
        for chunk_idx, start in enumerate(range(0, n_rows, fetch_chunk_size), start=1):
            chunk_t0 = time.time()
            chunk_rows = rows[start:start + fetch_chunk_size]
            print(
                f"{split_name}: chunk {chunk_idx}/{n_chunks} fetching "
                f"{len(chunk_rows):,} patches "
                f"(rows {start + 1:,}-{start + len(chunk_rows):,}/{n_rows:,})",
                flush=True,
            )
            fetch_t0 = time.time()
            patches_by_iid = _fetch_patch_chunk(
                conn,
                [row["instance_id"] for row in chunk_rows],
            )
            print(
                f"{split_name}: chunk {chunk_idx}/{n_chunks} fetched "
                f"{len(patches_by_iid):,} patch rows in {time.time() - fetch_t0:.1f}s",
                flush=True,
            )

            joined_rows: list[dict[str, Any]] = []
            patch_texts: list[str] = []
            for row in chunk_rows:
                patch_text = patches_by_iid.get(row["instance_id"])
                if patch_text is None:
                    n_missing_patch += 1
                    continue
                if max_patch_chars > 0:
                    patch_text = patch_text[:max_patch_chars]
                joined_rows.append(row)
                patch_texts.append(patch_text)

            if not joined_rows:
                print(
                    f"{split_name}: chunk {chunk_idx}/{n_chunks} had no matched patches "
                    f"(missing_patch={n_missing_patch:,})",
                    flush=True,
                )
                continue

            tokenize_t0 = time.time()
            print(
                f"{split_name}: chunk {chunk_idx}/{n_chunks} tokenizing "
                f"{len(joined_rows):,} patches...",
                flush=True,
            )
            encoded = tokenizer(
                patch_texts,
                truncation=True,
                max_length=max_tokens,
                padding=False,
            )
            print(
                f"{split_name}: chunk {chunk_idx}/{n_chunks} tokenized "
                f"{len(joined_rows):,} patches in {time.time() - tokenize_t0:.1f}s",
                flush=True,
            )
            tag_t0 = time.time()
            print(
                f"{split_name}: chunk {chunk_idx}/{n_chunks} tagging "
                f"{len(joined_rows):,} patches "
                f"(workers={preprocess_workers}, batch_size={tag_batch_size})...",
                flush=True,
            )
            tag_rows = _extract_tag_rows(
                patch_texts=patch_texts,
                preprocess_workers=preprocess_workers,
                tag_batch_size=tag_batch_size,
                tag_pool=tag_pool,
            )
            print(
                f"{split_name}: chunk {chunk_idx}/{n_chunks} tagged "
                f"{len(joined_rows):,} patches in {time.time() - tag_t0:.1f}s",
                flush=True,
            )

            for row, ids, issue_flags in zip(joined_rows, encoded["input_ids"], tag_rows):
                token_ids = np.asarray(ids, dtype=np.int32) % hash_vocab_size
                if token_ids.size == 0:
                    token_ids = np.zeros((1,), dtype=np.int32)
                flat_ids_parts.append(token_ids)
                lengths.append(int(token_ids.size))
                latent_parts.append(row["latent"])
                cluster_ids.append(int(row["cluster_id"]))
                accepted_parts.append(float(row["accepted"]))
                tag_parts.append(np.asarray([issue_flags[name] for name in TAG_NAMES], dtype=np.float32))
                repo_parts.append(str(row["repo"]))
                iid_parts.append(str(row["instance_id"]))

            n_joined += len(joined_rows)
            if (
                n_joined % progress_every < len(joined_rows)
                or start + fetch_chunk_size >= n_rows
            ):
                print(
                    f"{split_name}: prepared {n_joined:,}/{n_rows:,} rows "
                    f"(missing_patch={n_missing_patch:,}, "
                    f"chunk_time={time.time() - chunk_t0:.1f}s)",
                    flush=True,
                )
    finally:
        conn.close()
        if tag_pool is not None:
            tag_pool.shutdown()

    if lengths:
        lengths_arr = np.asarray(lengths, dtype=np.int32)
        offsets = np.zeros(len(lengths_arr), dtype=np.int64)
        if len(lengths_arr) > 1:
            offsets[1:] = np.cumsum(lengths_arr[:-1], dtype=np.int64)
        flat_ids = np.concatenate(flat_ids_parts).astype(np.int32, copy=False)
        latent = np.stack(latent_parts).astype(np.float32, copy=False)
        cluster_id = np.asarray(cluster_ids, dtype=np.int64)
        accepted = np.asarray(accepted_parts, dtype=np.float32)
        tags = np.stack(tag_parts).astype(np.float32, copy=False)
        repos = np.asarray(repo_parts, dtype=np.str_)
        instance_ids = np.asarray(iid_parts, dtype=np.str_)
    else:
        flat_ids = np.empty((0,), dtype=np.int32)
        offsets = np.empty((0,), dtype=np.int64)
        lengths_arr = np.empty((0,), dtype=np.int32)
        latent = np.empty((0, latent_dim), dtype=np.float32)
        cluster_id = np.empty((0,), dtype=np.int64)
        accepted = np.empty((0,), dtype=np.float32)
        tags = np.empty((0, len(TAG_NAMES)), dtype=np.float32)
        repos = np.empty((0,), dtype=np.str_)
        instance_ids = np.empty((0,), dtype=np.str_)

    return {
        "flat_ids": flat_ids,
        "offsets": offsets,
        "lengths": lengths_arr,
        "latent": latent,
        "cluster_id": cluster_id,
        "accepted": accepted,
        "tags": tags,
        "repos": repos,
        "instance_ids": instance_ids,
        "n_missing_patch": n_missing_patch,
        "n_rows": n_rows,
        "n_joined": n_joined,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projected", default=DEFAULT_PROJECTED)
    ap.add_argument("--super-clusters", default=DEFAULT_SUPER_CLUSTERS)
    ap.add_argument("--pg-config", default=DEFAULT_PG_CONFIG)
    ap.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--max-patch-chars", type=int, default=0, help="0 keeps full patch text")
    ap.add_argument("--hash-vocab-size", type=int, default=32768)
    ap.add_argument("--preprocess-workers", type=int, default=1)
    ap.add_argument("--tag-batch-size", type=int, default=64)
    ap.add_argument("--fetch-chunk-size", type=int, default=1024)
    ap.add_argument("--preprocess-progress-every", type=int, default=1024)
    ap.add_argument("--shard-modulus", type=int, default=1)
    ap.add_argument("--shard-remainder", type=int, default=0)
    args = ap.parse_args()

    _set_seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if args.preprocess_workers > 1 else "false"
    os.environ["RAYON_NUM_THREADS"] = str(max(1, args.preprocess_workers))

    print(
        f"Loading tokenizer (RAYON_NUM_THREADS={os.environ['RAYON_NUM_THREADS']}, "
        f"tag_workers={args.preprocess_workers})...",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )

    print("Loading teacher metadata...", flush=True)
    metadata = _load_teacher_metadata(args.projected, args.super_clusters)
    metadata_rows = metadata["rows"]
    if not metadata_rows:
        raise RuntimeError("No matched metadata rows found.")
    if metadata["n_duplicate_instance_ids"] > 0:
        raise RuntimeError(
            "Teacher metadata contains duplicate instance_ids "
            f"({metadata['n_duplicate_instance_ids']:,} duplicate ids across "
            f"{len(metadata_rows):,} rows; max multiplicity "
            f"{metadata['max_duplicate_instance_id_count']}). "
            "The current patch fetch path keys by instance_id, so the teacher->patch join "
            "is ambiguous. Regenerate teacher artifacts with a stable row key or build a "
            "canonical deduped teacher set before rerunning preprocessing."
        )
    if args.limit > 0 and len(metadata_rows) > args.limit:
        rng = random.Random(args.seed)
        metadata_rows = rng.sample(metadata_rows, args.limit)

    cluster_label_values = sorted({int(row["cluster_id"]) for row in metadata_rows})
    cluster_to_class = {cid: i for i, cid in enumerate(cluster_label_values)}
    for row in metadata_rows:
        row["cluster_id"] = cluster_to_class[int(row["cluster_id"])]

    if args.shard_modulus <= 0:
        raise ValueError("--shard-modulus must be > 0")
    if args.shard_remainder < 0 or args.shard_remainder >= args.shard_modulus:
        raise ValueError("--shard-remainder must satisfy 0 <= remainder < modulus")

    shard_rows = [
        row
        for i, row in enumerate(metadata_rows)
        if i % args.shard_modulus == args.shard_remainder
    ]
    print(
        f"Shard selection: remainder={args.shard_remainder}/{args.shard_modulus}, "
        f"rows={len(shard_rows):,}/{len(metadata_rows):,}",
        flush=True,
    )

    processed = _preprocess_rows(
        rows=shard_rows,
        cfg_path=args.pg_config,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        hash_vocab_size=args.hash_vocab_size,
        max_patch_chars=args.max_patch_chars,
        preprocess_workers=args.preprocess_workers,
        tag_batch_size=args.tag_batch_size,
        fetch_chunk_size=args.fetch_chunk_size,
        progress_every=args.preprocess_progress_every,
        split_name=f"shard-{args.shard_remainder:03d}",
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        flat_ids=processed["flat_ids"],
        offsets=processed["offsets"],
        lengths=processed["lengths"],
        latent=processed["latent"],
        cluster_id=processed["cluster_id"],
        accepted=processed["accepted"],
        tags=processed["tags"],
        repos=processed["repos"],
        instance_ids=processed["instance_ids"],
        cluster_label_values=np.asarray(cluster_label_values, dtype=np.int64),
    )

    summary = {
        "projected": args.projected,
        "super_clusters": args.super_clusters,
        "pg_config": args.pg_config,
        "tokenizer_path": args.tokenizer_path,
        "out": args.out,
        "n_input_rows": len(shard_rows),
        "n_joined_rows": processed["n_joined"],
        "n_missing_patch": processed["n_missing_patch"],
        "n_missing_cluster": metadata["n_missing_cluster"],
        "n_duplicate_instance_ids": metadata["n_duplicate_instance_ids"],
        "n_unique_instance_ids": metadata["n_unique_instance_ids"],
        "shard_modulus": args.shard_modulus,
        "shard_remainder": args.shard_remainder,
        "config": {
            "limit": args.limit,
            "max_tokens": args.max_tokens,
            "max_patch_chars": args.max_patch_chars,
            "hash_vocab_size": args.hash_vocab_size,
            "preprocess_workers": args.preprocess_workers,
            "tag_batch_size": args.tag_batch_size,
            "fetch_chunk_size": args.fetch_chunk_size,
            "preprocess_progress_every": args.preprocess_progress_every,
            "seed": args.seed,
        },
    }
    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Shard summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nShard NPZ -> {args.out}")
    print(f"Summary   -> {args.summary_out}")


if __name__ == "__main__":
    main()
