#!/usr/bin/env python3
"""Experiment 7.1 — minimal review-state student.

Train a compact student that reads patch diffs and predicts the teacher-side
review state used by the phase-6 steerer:

- 256-d projected patch latent `h`
- HDBSCAN super-cluster id
- acceptance label
- deterministic review issue tags

This is the first verification step for "student replaces steerer": keep the
state->prompt bridge symbolic and deterministic, and only learn the patch->state
mapping.

Usage:
    python experiment_7/train_review_state_student.py
    python experiment_7/train_review_state_student.py --limit 4096 --epochs 2 --device cpu
"""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import math
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pg8000.native
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from review_state_bridge import TAG_NAMES, detect_review_issue_flags

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

DEFAULT_PROJECTED = os.path.join(PROJECT_ROOT, "data", "phase6_2", "projected_embeddings.npz")
DEFAULT_SUPER_CLUSTERS = os.path.join(PROJECT_ROOT, "data", "phase6_2", "super_cluster_assignments.npz")
DEFAULT_STEERER = os.path.join(PROJECT_ROOT, "data", "phase6_2", "pr_steerer_model_combined.json")
DEFAULT_PG_CONFIG = os.path.join(PROJECT_ROOT, "postgres_connection.yaml")
DEFAULT_MODEL_OUT = os.path.join(PROJECT_ROOT, "data", "phase7_1", "review_state_student.pt")
DEFAULT_METRICS_OUT = os.path.join(PROJECT_ROOT, "data", "phase7_1", "review_state_student_metrics.json")
DEFAULT_TOKENIZER = "/home/original_models/Qwen2.5-Coder-3B"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    return float(average_precision_score(y_true, y_score))


def _safe_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = [i for i in range(y_true.shape[1]) if len(np.unique(y_true[:, i])) > 1]
    if not valid:
        return 0.0
    return float(f1_score(y_true[:, valid], y_pred[:, valid], average="macro", zero_division=0))


def _load_db(cfg_path: str) -> pg8000.native.Connection:
    cfg = yaml.safe_load(open(cfg_path))
    return pg8000.native.Connection(
        host=cfg["ip"],
        port=cfg.get("port", 9999),
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
    )


def _fetch_all_patches(cfg_path: str, limit: int = 0) -> dict[str, dict[str, Any]]:
    """Fetch patch texts keyed by instance_id from both PR tables."""
    conn = _load_db(cfg_path)
    limit_sql = f"LIMIT {int(limit)}" if limit > 0 else ""
    rows = conn.run(
        f"""
        SELECT * FROM (
            SELECT instance_id, repo, patch
            FROM prs_copy
            WHERE patch IS NOT NULL
            UNION ALL
            SELECT
                REPLACE(repo, '/', '__') || '__' || pull_number AS instance_id,
                repo, patch
            FROM python_js_ts_rust_closed_prs
            WHERE patch IS NOT NULL
        ) AS combined
        {limit_sql}
        """
    )
    conn.close()
    out: dict[str, dict[str, Any]] = {}
    for instance_id, repo, patch in rows:
        out[str(instance_id)] = {"repo": str(repo), "patch": str(patch)}
    return out


def _fetch_patches_for_iids(cfg_path: str, instance_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch only the requested patch texts, in manageable chunks."""
    if not instance_ids:
        return {}
    conn = _load_db(cfg_path)
    out: dict[str, dict[str, Any]] = {}
    chunk_size = 500
    for start in range(0, len(instance_ids), chunk_size):
        chunk = instance_ids[start:start + chunk_size]
        quoted = ", ".join("'" + iid.replace("'", "''") + "'" for iid in chunk)
        rows = conn.run(
            f"""
            SELECT instance_id_key, repo, patch
            FROM (
                SELECT instance_id AS instance_id_key, repo, patch
                FROM prs_copy
                WHERE patch IS NOT NULL
                UNION ALL
                SELECT
                    REPLACE(repo, '/', '__') || '__' || pull_number AS instance_id_key,
                    repo, patch
                FROM python_js_ts_rust_closed_prs
                WHERE patch IS NOT NULL
            ) AS combined
            WHERE instance_id_key IN ({quoted})
            """
        )
        for instance_id, repo, patch in rows:
            out[str(instance_id)] = {"repo": str(repo), "patch": str(patch)}
    conn.close()
    return out


def _fetch_patch_chunk(
    conn: pg8000.native.Connection,
    instance_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not instance_ids:
        return {}
    quoted = ", ".join("'" + iid.replace("'", "''") + "'" for iid in instance_ids)
    rows = conn.run(
        f"""
        SELECT instance_id_key, repo, patch
        FROM (
            SELECT instance_id AS instance_id_key, repo, patch
            FROM prs_copy
            WHERE patch IS NOT NULL
            UNION ALL
            SELECT
                REPLACE(repo, '/', '__') || '__' || pull_number AS instance_id_key,
                repo, patch
            FROM python_js_ts_rust_closed_prs
            WHERE patch IS NOT NULL
        ) AS combined
        WHERE instance_id_key IN ({quoted})
        """
    )
    out: dict[str, dict[str, Any]] = {}
    for instance_id, repo, patch in rows:
        out[str(instance_id)] = {"repo": str(repo), "patch": str(patch)}
    return out


def _load_teacher_metadata(
    projected_path: str,
    super_clusters_path: str,
    ) -> dict[str, Any]:
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


def _attach_patches_and_tags(
    metadata_rows: list[dict[str, Any]],
    patches_by_iid: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    n_missing_patch = 0
    for row in metadata_rows:
        key = row["instance_id"]
        patch_row = patches_by_iid.get(key)
        if patch_row is None:
            n_missing_patch += 1
            continue
        issue_flags = detect_review_issue_flags(patch_row["patch"])
        rows.append({
            "instance_id": key,
            "repo": row["repo"],
            "patch": patch_row["patch"],
            "accepted": row["accepted"],
            "latent": row["latent"],
            "cluster_id": row["cluster_id"],
            "tags": np.asarray([issue_flags[name] for name in TAG_NAMES], dtype=np.float32),
        })
    return {"rows": rows, "n_missing_patch": n_missing_patch}


def _prepare_split_rows(
    split_name: str,
    metadata_rows: list[dict[str, Any]],
    idx: np.ndarray,
    cfg_path: str,
    cluster_to_class: dict[int, int],
) -> dict[str, Any]:
    split_meta = [metadata_rows[i] for i in idx]
    print(f"Fetching {split_name} patches ({len(split_meta):,} ids)...", flush=True)
    patches_by_iid = _fetch_patches_for_iids(
        cfg_path,
        [row["instance_id"] for row in split_meta],
    )
    print(f"Joining {split_name} patches to teacher targets...", flush=True)
    teacher = _attach_patches_and_tags(split_meta, patches_by_iid)
    rows = teacher["rows"]
    for row in rows:
        row["cluster_id"] = cluster_to_class[int(row["cluster_id"])]
    return {
        "rows": rows,
        "n_missing_patch": teacher["n_missing_patch"],
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


def _prepare_split_dataset(
    split_name: str,
    metadata_rows: list[dict[str, Any]],
    idx: np.ndarray,
    cfg_path: str,
    cluster_to_class: dict[int, int],
    tokenizer: Any,
    max_tokens: int,
    hash_vocab_size: int,
    max_patch_chars: int,
    preprocess_workers: int,
    tag_batch_size: int,
    fetch_chunk_size: int = 512,
    progress_every: int = 5000,
) -> dict[str, Any]:
    split_meta = [metadata_rows[i] for i in idx]
    print(f"Preparing {split_name} split ({len(split_meta):,} metadata rows)...", flush=True)
    n_chunks = max(1, math.ceil(len(split_meta) / fetch_chunk_size))

    flat_ids_parts: list[np.ndarray] = []
    lengths: list[int] = []
    latent_parts: list[np.ndarray] = []
    cluster_ids: list[int] = []
    accepted_parts: list[float] = []
    tag_parts: list[np.ndarray] = []
    n_missing_patch = 0
    n_joined = 0

    conn = _load_db(cfg_path)
    tag_pool: ProcessPoolExecutor | None = None
    if preprocess_workers > 1:
        ctx = mp.get_context("spawn")
        tag_pool = ProcessPoolExecutor(max_workers=preprocess_workers, mp_context=ctx)
    try:
        for chunk_idx, start in enumerate(range(0, len(split_meta), fetch_chunk_size), start=1):
            chunk_t0 = time.time()
            chunk_meta = split_meta[start:start + fetch_chunk_size]
            print(
                f"{split_name}: chunk {chunk_idx}/{n_chunks} fetching "
                f"{len(chunk_meta):,} patches "
                f"(rows {start + 1:,}-{start + len(chunk_meta):,}/{len(split_meta):,})",
                flush=True,
            )
            fetch_t0 = time.time()
            patches_by_iid = _fetch_patch_chunk(
                conn,
                [row["instance_id"] for row in chunk_meta],
            )
            print(
                f"{split_name}: chunk {chunk_idx}/{n_chunks} fetched "
                f"{len(patches_by_iid):,} patch rows in {time.time() - fetch_t0:.1f}s",
                flush=True,
            )

            joined_meta: list[dict[str, Any]] = []
            patch_texts: list[str] = []
            for row in chunk_meta:
                patch_row = patches_by_iid.get(row["instance_id"])
                if patch_row is None:
                    n_missing_patch += 1
                    continue
                joined_meta.append(row)
                patch_text = patch_row["patch"]
                if max_patch_chars > 0:
                    patch_text = patch_text[:max_patch_chars]
                patch_texts.append(patch_text)

            if not joined_meta:
                print(
                    f"{split_name}: chunk {chunk_idx}/{n_chunks} had no matched patches "
                    f"(missing_patch={n_missing_patch:,})",
                    flush=True,
                )
                continue

            tokenize_t0 = time.time()
            print(
                f"{split_name}: chunk {chunk_idx}/{n_chunks} tokenizing "
                f"{len(joined_meta):,} patches...",
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
                f"{len(joined_meta):,} patches in {time.time() - tokenize_t0:.1f}s",
                flush=True,
            )
            tag_t0 = time.time()
            print(
                f"{split_name}: chunk {chunk_idx}/{n_chunks} tagging "
                f"{len(joined_meta):,} patches "
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
                f"{len(joined_meta):,} patches in {time.time() - tag_t0:.1f}s",
                flush=True,
            )

            for row, ids, issue_flags in zip(joined_meta, encoded["input_ids"], tag_rows):
                arr = np.asarray(ids, dtype=np.int32) % hash_vocab_size
                if arr.size == 0:
                    arr = np.zeros((1,), dtype=np.int32)
                tags = np.asarray([issue_flags[name] for name in TAG_NAMES], dtype=np.float32)
                flat_ids_parts.append(arr)
                lengths.append(int(arr.size))
                latent_parts.append(row["latent"])
                cluster_ids.append(cluster_to_class[int(row["cluster_id"])])
                accepted_parts.append(float(row["accepted"]))
                tag_parts.append(tags)

            n_joined += len(joined_meta)
            if (
                n_joined % progress_every < len(joined_meta)
                or start + fetch_chunk_size >= len(split_meta)
            ):
                print(
                    f"{split_name}: prepared {n_joined:,}/{len(split_meta):,} rows "
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
    else:
        flat_ids = np.empty((0,), dtype=np.int32)
        offsets = np.empty((0,), dtype=np.int64)
        lengths_arr = np.empty((0,), dtype=np.int32)
        latent = np.empty((0, 0), dtype=np.float32)
        cluster_id = np.empty((0,), dtype=np.int64)
        accepted = np.empty((0,), dtype=np.float32)
        tags = np.empty((0, len(TAG_NAMES)), dtype=np.float32)

    return {
        "dataset": ReviewStateDataset(
            flat_ids=flat_ids,
            offsets=offsets,
            lengths=lengths_arr,
            latent=latent,
            cluster_id=cluster_id,
            accepted=accepted,
            tags=tags,
        ),
        "n_missing_patch": n_missing_patch,
    }


def _split_by_repo(repos: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(repos)
    unique_repos = np.unique(repos)
    if len(unique_repos) < 3:
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_train = max(1, int(round(0.8 * n)))
        n_val = max(1, int(round(0.1 * n)))
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        if len(test_idx) == 0 and len(val_idx) > 1:
            test_idx = val_idx[-1:]
            val_idx = val_idx[:-1]
        return train_idx, val_idx, test_idx

    idx = np.arange(len(repos))
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
    train_idx, temp_idx = next(gss.split(idx, groups=repos))
    temp_groups = repos[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
    val_rel, test_rel = next(gss2.split(np.arange(len(temp_idx)), groups=temp_groups))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]
    return train_idx, val_idx, test_idx


def _load_preprocessed_cache(
    preprocessed_npz: str = "",
    preprocessed_dir: str = "",
) -> dict[str, Any]:
    if preprocessed_npz:
        paths = [preprocessed_npz]
    elif preprocessed_dir:
        paths = sorted(glob.glob(os.path.join(preprocessed_dir, "shard_*.npz")))
    else:
        raise ValueError("Expected either preprocessed_npz or preprocessed_dir.")

    if not paths:
        raise RuntimeError("No preprocessed shard files found.")

    flat_ids_parts: list[np.ndarray] = []
    lengths_parts: list[np.ndarray] = []
    latent_parts: list[np.ndarray] = []
    cluster_parts: list[np.ndarray] = []
    accepted_parts: list[np.ndarray] = []
    tag_parts: list[np.ndarray] = []
    repo_parts: list[np.ndarray] = []
    iid_parts: list[np.ndarray] = []
    cluster_label_values: np.ndarray | None = None

    for path in paths:
        print(f"Loading preprocessed shard: {path}", flush=True)
        data = np.load(path, allow_pickle=True)
        flat_ids_parts.append(data["flat_ids"].astype(np.int32, copy=False))
        lengths_parts.append(data["lengths"].astype(np.int32, copy=False))
        latent_parts.append(data["latent"].astype(np.float32, copy=False))
        cluster_parts.append(data["cluster_id"].astype(np.int64, copy=False))
        accepted_parts.append(data["accepted"].astype(np.float32, copy=False))
        tag_parts.append(data["tags"].astype(np.float32, copy=False))
        repo_parts.append(data["repos"].astype(np.str_))
        iid_parts.append(data["instance_ids"].astype(np.str_))
        shard_cluster_values = data["cluster_label_values"].astype(np.int64, copy=False)
        if cluster_label_values is None:
            cluster_label_values = shard_cluster_values
        elif not np.array_equal(cluster_label_values, shard_cluster_values):
            raise RuntimeError(f"cluster_label_values mismatch in {path}")

    flat_ids = np.concatenate(flat_ids_parts) if flat_ids_parts else np.empty((0,), dtype=np.int32)
    lengths = np.concatenate(lengths_parts) if lengths_parts else np.empty((0,), dtype=np.int32)
    offsets = np.zeros(len(lengths), dtype=np.int64)
    if len(lengths) > 1:
        offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64)

    latent_dim = 0
    for arr in latent_parts:
        if arr.ndim == 2 and arr.shape[1] > 0:
            latent_dim = int(arr.shape[1])
            break

    return {
        "flat_ids": flat_ids,
        "offsets": offsets,
        "lengths": lengths,
        "latent": np.concatenate(latent_parts) if latent_parts else np.empty((0, latent_dim), dtype=np.float32),
        "cluster_id": np.concatenate(cluster_parts) if cluster_parts else np.empty((0,), dtype=np.int64),
        "accepted": np.concatenate(accepted_parts) if accepted_parts else np.empty((0,), dtype=np.float32),
        "tags": np.concatenate(tag_parts) if tag_parts else np.empty((0, len(TAG_NAMES)), dtype=np.float32),
        "repos": np.concatenate(repo_parts) if repo_parts else np.empty((0,), dtype=np.str_),
        "instance_ids": np.concatenate(iid_parts) if iid_parts else np.empty((0,), dtype=np.str_),
        "cluster_label_values": cluster_label_values if cluster_label_values is not None else np.empty((0,), dtype=np.int64),
        "paths": paths,
    }


def _build_dataset_from_cache(cache: dict[str, Any], idx: np.ndarray) -> "ReviewStateDataset":
    idx = np.asarray(idx, dtype=np.int64)
    lengths = cache["lengths"][idx].astype(np.int32, copy=False)
    offsets = np.zeros(len(lengths), dtype=np.int64)
    if len(lengths) > 1:
        offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64)

    token_parts = []
    source_flat_ids = cache["flat_ids"]
    source_offsets = cache["offsets"]
    source_lengths = cache["lengths"]
    for row_idx in idx:
        start = int(source_offsets[row_idx])
        end = start + int(source_lengths[row_idx])
        token_parts.append(source_flat_ids[start:end])
    flat_ids = np.concatenate(token_parts) if token_parts else np.empty((0,), dtype=np.int32)

    return ReviewStateDataset(
        flat_ids=flat_ids,
        offsets=offsets,
        lengths=lengths,
        latent=cache["latent"][idx].astype(np.float32, copy=False),
        cluster_id=cache["cluster_id"][idx].astype(np.int64, copy=False),
        accepted=cache["accepted"][idx].astype(np.float32, copy=False),
        tags=cache["tags"][idx].astype(np.float32, copy=False),
    )


@dataclass
class Batch:
    input_ids: torch.Tensor
    offsets: torch.Tensor
    latent: torch.Tensor
    cluster_id: torch.Tensor
    accepted: torch.Tensor
    tags: torch.Tensor


class ReviewStateDataset(Dataset):
    def __init__(
        self,
        flat_ids: np.ndarray,
        offsets: np.ndarray,
        lengths: np.ndarray,
        latent: np.ndarray,
        cluster_id: np.ndarray,
        accepted: np.ndarray,
        tags: np.ndarray,
    ):
        self.flat_ids = flat_ids
        self.offsets = offsets
        self.lengths = lengths
        self.latent = latent
        self.cluster_id = cluster_id
        self.accepted = accepted
        self.tags = tags

    def __len__(self) -> int:
        return int(self.lengths.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        start = int(self.offsets[idx])
        end = start + int(self.lengths[idx])
        return {
            "token_ids": self.flat_ids[start:end],
            "latent": self.latent[idx],
            "cluster_id": int(self.cluster_id[idx]),
            "accepted": float(self.accepted[idx]),
            "tags": self.tags[idx],
        }


class EmbeddingBagStudent(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_clusters: int,
        n_tags: int,
    ):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.latent_head = nn.Linear(hidden_dim, latent_dim)
        self.cluster_head = nn.Linear(hidden_dim, n_clusters)
        self.accept_head = nn.Linear(hidden_dim, 1)
        self.tag_head = nn.Linear(hidden_dim, n_tags)

    def forward(self, input_ids: torch.Tensor, offsets: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.embedding(input_ids, offsets)
        hidden = self.trunk(pooled)
        return {
            "hidden": hidden,
            "latent": self.latent_head(hidden),
            "cluster_logits": self.cluster_head(hidden),
            "accept_logits": self.accept_head(hidden).squeeze(-1),
            "tag_logits": self.tag_head(hidden),
        }


def _make_collate():
    def _collate(rows: list[dict[str, Any]]) -> Batch:
        ids_list = []
        for row in rows:
            arr = np.asarray(row["token_ids"], dtype=np.int64)
            if arr.size == 0:
                arr = np.zeros((1,), dtype=np.int64)
            ids_list.append(torch.tensor(arr, dtype=torch.long))

        lengths = [len(ids) for ids in ids_list]
        offsets = torch.tensor(np.cumsum([0] + lengths[:-1]), dtype=torch.long)
        flat_ids = torch.cat(ids_list, dim=0)
        latent = torch.tensor(np.stack([row["latent"] for row in rows]), dtype=torch.float32)
        cluster_id = torch.tensor([row["cluster_id"] for row in rows], dtype=torch.long)
        accepted = torch.tensor([row["accepted"] for row in rows], dtype=torch.float32)
        tags = torch.tensor(np.stack([row["tags"] for row in rows]), dtype=torch.float32)
        return Batch(flat_ids, offsets, latent, cluster_id, accepted, tags)

    return _collate


def _move_batch(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        input_ids=batch.input_ids.to(device),
        offsets=batch.offsets.to(device),
        latent=batch.latent.to(device),
        cluster_id=batch.cluster_id.to(device),
        accepted=batch.accepted.to(device),
        tags=batch.tags.to(device),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    loss_weights: dict[str, float],
) -> dict[str, Any]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_items = 0
    latent_cosines: list[np.ndarray] = []
    all_cluster_true = []
    all_cluster_pred = []
    all_accept_true = []
    all_accept_score = []
    all_tags_true = []
    all_tags_score = []

    for batch in loader:
        batch = _move_batch(batch, device)
        outputs = model(batch.input_ids, batch.offsets)

        pred_latent = F.normalize(outputs["latent"], dim=-1)
        tgt_latent = F.normalize(batch.latent, dim=-1)
        loss_latent = 1.0 - (pred_latent * tgt_latent).sum(dim=-1).mean()
        loss_cluster = F.cross_entropy(outputs["cluster_logits"], batch.cluster_id)
        loss_accept = F.binary_cross_entropy_with_logits(outputs["accept_logits"], batch.accepted)
        loss_tags = F.binary_cross_entropy_with_logits(outputs["tag_logits"], batch.tags)

        loss = (
            loss_weights["latent"] * loss_latent
            + loss_weights["cluster"] * loss_cluster
            + loss_weights["accept"] * loss_accept
            + loss_weights["tags"] * loss_tags
        )

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = batch.accepted.shape[0]
        total_loss += float(loss.item()) * bs
        total_items += bs

        with torch.no_grad():
            latent_cosines.append((pred_latent * tgt_latent).sum(dim=-1).detach().cpu().numpy())
            all_cluster_true.append(batch.cluster_id.detach().cpu().numpy())
            all_cluster_pred.append(outputs["cluster_logits"].argmax(dim=-1).detach().cpu().numpy())
            all_accept_true.append(batch.accepted.detach().cpu().numpy())
            all_accept_score.append(torch.sigmoid(outputs["accept_logits"]).detach().cpu().numpy())
            all_tags_true.append(batch.tags.detach().cpu().numpy())
            all_tags_score.append(torch.sigmoid(outputs["tag_logits"]).detach().cpu().numpy())

    cluster_true = np.concatenate(all_cluster_true) if all_cluster_true else np.array([], dtype=np.int64)
    cluster_pred = np.concatenate(all_cluster_pred) if all_cluster_pred else np.array([], dtype=np.int64)
    accept_true = np.concatenate(all_accept_true) if all_accept_true else np.array([], dtype=np.float32)
    accept_score = np.concatenate(all_accept_score) if all_accept_score else np.array([], dtype=np.float32)
    tags_true = np.concatenate(all_tags_true) if all_tags_true else np.empty((0, len(TAG_NAMES)), dtype=np.float32)
    tags_score = np.concatenate(all_tags_score) if all_tags_score else np.empty((0, len(TAG_NAMES)), dtype=np.float32)
    tags_pred = (tags_score >= 0.5).astype(np.int32) if tags_score.size else tags_score

    return {
        "loss": total_loss / max(total_items, 1),
        "latent_cosine": float(np.mean(np.concatenate(latent_cosines))) if latent_cosines else 0.0,
        "cluster_acc": float(np.mean(cluster_true == cluster_pred)) if cluster_true.size else 0.0,
        "acceptance_auroc": _safe_auroc(accept_true, accept_score) if accept_true.size else 0.5,
        "acceptance_pr_auc": _safe_ap(accept_true, accept_score) if accept_true.size else 0.0,
        "tag_macro_f1": _safe_macro_f1(tags_true.astype(np.int32), tags_pred) if tags_true.size else 0.0,
        "n": int(total_items),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projected", default=DEFAULT_PROJECTED)
    ap.add_argument("--super-clusters", default=DEFAULT_SUPER_CLUSTERS)
    ap.add_argument("--steerer-model", default=DEFAULT_STEERER)
    ap.add_argument("--pg-config", default=DEFAULT_PG_CONFIG)
    ap.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER)
    ap.add_argument("--model-out", default=DEFAULT_MODEL_OUT)
    ap.add_argument("--metrics-out", default=DEFAULT_METRICS_OUT)
    ap.add_argument("--preprocessed-dir", default="", help="Directory containing shard_*.npz cache files")
    ap.add_argument("--preprocessed-npz", default="", help="Single merged preprocessed NPZ cache")
    ap.add_argument("--limit", type=int, default=0, help="Optional sample cap for smoke runs")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--max-patch-chars", type=int, default=0, help="0 keeps full patch text")
    ap.add_argument("--hash-vocab-size", type=int, default=32768)
    ap.add_argument("--embed-dim", type=int, default=192)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--preprocess-workers", type=int, default=1)
    ap.add_argument("--tag-batch-size", type=int, default=64)
    ap.add_argument("--fetch-chunk-size", type=int, default=512)
    ap.add_argument("--preprocess-progress-every", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--loss-latent", type=float, default=1.0)
    ap.add_argument("--loss-cluster", type=float, default=0.5)
    ap.add_argument("--loss-accept", type=float, default=0.25)
    ap.add_argument("--loss-tags", type=float, default=0.5)
    args = ap.parse_args()

    _set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    print(f"Device: {device}")
    use_preprocessed = bool(args.preprocessed_dir or args.preprocessed_npz)
    missing_patch = 0
    missing_cluster = 0

    if use_preprocessed:
        print("Loading preprocessed cache...", flush=True)
        cache = _load_preprocessed_cache(
            preprocessed_npz=args.preprocessed_npz,
            preprocessed_dir=args.preprocessed_dir,
        )
        selected_idx = np.arange(len(cache["repos"]), dtype=np.int64)
        if args.limit > 0 and len(selected_idx) > args.limit:
            rng = random.Random(args.seed)
            selected_idx = np.asarray(rng.sample(selected_idx.tolist(), args.limit), dtype=np.int64)
        repos = cache["repos"][selected_idx]
        cluster_label_values = [int(x) for x in cache["cluster_label_values"].tolist()]
        print("Building repo split from cached rows...", flush=True)
        train_rel, val_rel, test_rel = _split_by_repo(repos, args.seed)
        train_dataset = _build_dataset_from_cache(cache, selected_idx[train_rel])
        val_dataset = _build_dataset_from_cache(cache, selected_idx[val_rel])
        test_dataset = _build_dataset_from_cache(cache, selected_idx[test_rel])
    else:
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
            raise RuntimeError("No matched patch rows found for training.")
        if metadata["n_duplicate_instance_ids"] > 0:
            raise RuntimeError(
                "Teacher metadata contains duplicate instance_ids "
                f"({metadata['n_duplicate_instance_ids']:,} duplicate ids across "
                f"{len(metadata_rows):,} rows; max multiplicity "
                f"{metadata['max_duplicate_instance_id_count']}). "
                "The current patch fetch path keys by instance_id, so the teacher->patch join "
                "is ambiguous. Regenerate teacher artifacts with a stable row key or build a "
                "canonical deduped teacher set before rerunning training."
            )
        if args.limit > 0 and len(metadata_rows) > args.limit:
            rng = random.Random(args.seed)
            metadata_rows = rng.sample(metadata_rows, args.limit)

        cluster_label_values = sorted({int(row["cluster_id"]) for row in metadata_rows})
        cluster_to_class = {cid: i for i, cid in enumerate(cluster_label_values)}

        repos = np.asarray([row["repo"] for row in metadata_rows])
        print("Building repo split...", flush=True)
        train_idx, val_idx, test_idx = _split_by_repo(repos, args.seed)
        train_teacher = _prepare_split_dataset(
            "train",
            metadata_rows,
            train_idx,
            args.pg_config,
            cluster_to_class,
            tokenizer,
            args.max_tokens,
            args.hash_vocab_size,
            args.max_patch_chars,
            args.preprocess_workers,
            args.tag_batch_size,
            fetch_chunk_size=args.fetch_chunk_size,
            progress_every=args.preprocess_progress_every,
        )
        val_teacher = _prepare_split_dataset(
            "val",
            metadata_rows,
            val_idx,
            args.pg_config,
            cluster_to_class,
            tokenizer,
            args.max_tokens,
            args.hash_vocab_size,
            args.max_patch_chars,
            args.preprocess_workers,
            args.tag_batch_size,
            fetch_chunk_size=args.fetch_chunk_size,
            progress_every=args.preprocess_progress_every,
        )
        test_teacher = _prepare_split_dataset(
            "test",
            metadata_rows,
            test_idx,
            args.pg_config,
            cluster_to_class,
            tokenizer,
            args.max_tokens,
            args.hash_vocab_size,
            args.max_patch_chars,
            args.preprocess_workers,
            args.tag_batch_size,
            fetch_chunk_size=args.fetch_chunk_size,
            progress_every=args.preprocess_progress_every,
        )
        train_dataset = train_teacher["dataset"]
        val_dataset = val_teacher["dataset"]
        test_dataset = test_teacher["dataset"]
        missing_patch = (
            train_teacher["n_missing_patch"]
            + val_teacher["n_missing_patch"]
            + test_teacher["n_missing_patch"]
        )
        missing_cluster = metadata["n_missing_cluster"]

    n_total = len(train_dataset) + len(val_dataset) + len(test_dataset)
    if n_total == 0:
        raise RuntimeError("No rows available for training after preprocessing.")
    print(
        f"Dataset: total={n_total:,} train={len(train_dataset):,} "
        f"val={len(val_dataset):,} test={len(test_dataset):,} "
        f"missing_patch={missing_patch:,} "
        f"missing_cluster={missing_cluster:,}"
    )

    n_clusters = len(cluster_label_values)
    print(f"Clusters: {n_clusters} classes from original ids {cluster_label_values[:6]}{'...' if len(cluster_label_values) > 6 else ''}, tags: {len(TAG_NAMES)}")

    collate = _make_collate()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = EmbeddingBagStudent(
        vocab_size=args.hash_vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=int(train_dataset.latent.shape[1] or val_dataset.latent.shape[1] or test_dataset.latent.shape[1]),
        n_clusters=n_clusters,
        n_tags=len(TAG_NAMES),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_weights = {
        "latent": args.loss_latent,
        "cluster": args.loss_cluster,
        "accept": args.loss_accept,
        "tags": args.loss_tags,
    }

    history: list[dict[str, Any]] = []
    best_val = -1e9
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device, optimizer, loss_weights)
        val_metrics = _run_epoch(model, val_loader, device, None, loss_weights)
        val_score = (
            val_metrics["latent_cosine"]
            + 0.25 * val_metrics["cluster_acc"]
            + 0.25 * val_metrics["acceptance_auroc"]
            + 0.25 * val_metrics["tag_macro_f1"]
        )
        if val_score > best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        epoch_row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "val_score": float(val_score),
        }
        history.append(epoch_row)
        print(
            f"epoch {epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_cos={val_metrics['latent_cosine']:.4f} "
            f"val_cluster={val_metrics['cluster_acc']:.4f} "
            f"val_acc_auroc={val_metrics['acceptance_auroc']:.4f} "
            f"val_tag_f1={val_metrics['tag_macro_f1']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = _run_epoch(model, test_loader, device, None, loss_weights)

    cluster_hints = {}
    if os.path.exists(args.steerer_model):
        with open(args.steerer_model) as f:
            cluster_hints = json.load(f).get("cluster_hints", {})

    bundle = {
        "version": "phase7_1_review_state_student_v0",
        "arch": "embedding_bag",
        "tokenizer_path": args.tokenizer_path,
        "hash_vocab_size": args.hash_vocab_size,
        "max_tokens": args.max_tokens,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "latent_dim": int(train_dataset.latent.shape[1] or val_dataset.latent.shape[1] or test_dataset.latent.shape[1]),
        "tag_names": TAG_NAMES,
        "n_clusters": n_clusters,
        "cluster_label_values": cluster_label_values,
        "cluster_hints": cluster_hints,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    }
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    torch.save(bundle, args.model_out)

    metrics = {
        "version": bundle["version"],
        "n_total": n_total,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "n_test": len(test_dataset),
        "n_clusters": n_clusters,
        "cluster_label_values": cluster_label_values,
        "tag_names": TAG_NAMES,
        "loss_weights": loss_weights,
        "config": {
            "preprocessed_dir": args.preprocessed_dir,
            "preprocessed_npz": args.preprocessed_npz,
            "max_tokens": args.max_tokens,
            "max_patch_chars": args.max_patch_chars,
            "hash_vocab_size": args.hash_vocab_size,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "preprocess_workers": args.preprocess_workers,
            "tag_batch_size": args.tag_batch_size,
            "fetch_chunk_size": args.fetch_chunk_size,
            "preprocess_progress_every": args.preprocess_progress_every,
            "lr": args.lr,
            "device": str(device),
            "limit": args.limit,
        },
        "history": history,
        "test": test_metrics,
    }
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Test metrics ===")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nModel  -> {args.model_out}")
    print(f"Metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()
