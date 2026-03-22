#!/usr/bin/env python3
"""
Extract frozen Qwen2.5-Coder-3B layer-18 embeddings for patch diffs (Experiment 6.1).

Runs 24 model copies (3 per GPU × 8 GPUs) in parallel via ProcessPoolExecutor.
Each worker loads its own model copy and processes its chunk of patches.

Usage:
    python extract_patch_embeddings.py --out data/phase6_1/patch_embeddings_100k.npz
    python extract_patch_embeddings.py --out data/phase6_1/patch_embeddings_100k.npz --n-gpus 8 --workers-per-gpu 3
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pg8000.native
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
PG_CONFIG_FILE = os.path.join(PROJECT_ROOT, "postgres_connection.yaml")
TEACHER_PATH = "/home/original_models/Qwen2.5-Coder-3B"
TEACHER_LAYER = 18


def _load_db() -> pg8000.native.Connection:
    cfg = yaml.safe_load(open(PG_CONFIG_FILE))
    return pg8000.native.Connection(
        host=cfg["ip"],
        port=cfg.get("port", 9999),
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
    )


def _fetch_all_patches(limit: int = 0) -> list[dict]:
    """Fetch all patch texts + metadata from both tables."""
    conn = _load_db()
    limit_sql = f"LIMIT {int(limit)}" if limit > 0 else ""
    rows = conn.run(
        f"""
        SELECT * FROM (
            SELECT instance_id, repo, patch, pr_merged
            FROM prs_copy
            WHERE patch IS NOT NULL
            UNION ALL
            SELECT
                REPLACE(repo, '/', '__') || '__' || pull_number AS instance_id,
                repo, patch, false AS pr_merged
            FROM python_js_ts_rust_closed_prs
            WHERE patch IS NOT NULL
        ) AS combined
        {limit_sql}
        """
    )
    conn.close()
    records = []
    for instance_id, repo, patch, pr_merged in rows:
        records.append({
            "instance_id": instance_id,
            "repo": repo,
            "patch": patch,
            "accepted": 1 if pr_merged else 0,
        })
    return records


def _embed_worker(
    worker_id: int,
    gpu_id: int,
    texts: list[str],
    batch_size: int,
    max_tokens: int,
) -> np.ndarray:
    """
    Worker: load model on cuda:{gpu_id}, embed texts, return (N, D) float32.
    Multiple workers can share the same GPU (each loads its own copy).
    """
    device = f"cuda:{gpu_id}"
    tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_PATH, trust_remote_code=True, local_files_only=True,
    )
    full_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH,
        torch_dtype="auto",
        device_map=device,
        trust_remote_code=True,
        local_files_only=True,
    )
    full_model.eval()
    for param in full_model.parameters():
        param.requires_grad = False
    # Use base transformer (skip lm_head) to save ~4 GiB VRAM on logit computation
    model = full_model.model

    all_embs: list[np.ndarray] = []
    n_done = 0
    t0 = time.time()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
            padding=True,
        ).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        # model.model returns BaseModelOutputWithPast; hidden_states[0] is embed layer
        hs = out.hidden_states[TEACHER_LAYER]  # (B, T, D)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        embs = (hs * mask).sum(1) / mask.sum(1)  # (B, D)
        all_embs.append(embs.float().cpu().numpy())
        n_done += len(batch)
        if n_done % (batch_size * 10) == 0 or start + batch_size >= len(texts):
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            print(
                f"  [worker {worker_id:2d} GPU {gpu_id}] "
                f"{n_done:,}/{len(texts):,} ({rate:.0f} patches/s)",
                flush=True,
            )

    return np.vstack(all_embs) if all_embs else np.empty((0, 2048), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output NPZ path")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n-gpus", type=int, default=8)
    ap.add_argument("--workers-per-gpu", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()

    n_gpus = min(args.n_gpus, torch.cuda.device_count())
    n_workers = n_gpus * args.workers_per_gpu
    print(f"Config: {n_gpus} GPUs × {args.workers_per_gpu} workers/GPU = {n_workers} workers", flush=True)

    # Fetch all patches from DB
    t0 = time.time()
    records = _fetch_all_patches(args.limit)
    print(f"Fetched {len(records):,} patches from DB in {time.time() - t0:.1f}s", flush=True)

    if not records:
        print("WARNING: No patches found, writing empty NPZ", flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.savez_compressed(
            args.out,
            z_patch=np.empty((0, 2048), dtype=np.float32),
            instance_ids=np.array([], dtype="U"),
            accepted=np.array([], dtype=np.int8),
            repos=np.array([], dtype="U"),
        )
        return

    texts = [r["patch"] for r in records]
    iids = [r["instance_id"] for r in records]
    accepted = [r["accepted"] for r in records]
    repos = [r["repo"] for r in records]

    # Distribute patches across workers (round-robin)
    worker_texts: list[list[str]] = [[] for _ in range(n_workers)]
    worker_indices: list[list[int]] = [[] for _ in range(n_workers)]
    for i, text in enumerate(texts):
        w = i % n_workers
        worker_texts[w].append(text)
        worker_indices[w].append(i)

    for w in range(n_workers):
        gpu = w // args.workers_per_gpu
        print(f"  worker {w:2d} → GPU {gpu}, {len(worker_texts[w]):,} patches", flush=True)

    # Spawn workers
    t0 = time.time()
    ctx = mp.get_context("spawn")
    worker_results: dict[int, np.ndarray] = {}

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = {}
        for w in range(n_workers):
            if not worker_texts[w]:
                continue
            gpu = w // args.workers_per_gpu
            fut = executor.submit(
                _embed_worker, w, gpu, worker_texts[w], args.batch_size, args.max_tokens,
            )
            futures[fut] = w

        for future in as_completed(futures):
            w = futures[future]
            try:
                worker_results[w] = future.result()
                print(f"  worker {w:2d} done ({worker_results[w].shape[0]:,} embeddings)", flush=True)
            except Exception as e:
                print(f"ERROR worker {w}: {e}", flush=True, file=sys.stderr)
                raise

    elapsed = time.time() - t0
    print(f"\nEmbedding done in {elapsed:.1f}s ({len(texts) / elapsed:.0f} patches/s total)", flush=True)

    # Reassemble in original order
    dim = next(iter(worker_results.values())).shape[1]
    z_patch = np.empty((len(texts), dim), dtype=np.float32)
    for w, embs in worker_results.items():
        for local_i, global_i in enumerate(worker_indices[w]):
            z_patch[global_i] = embs[local_i]

    # Write NPZ
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        z_patch=z_patch,
        instance_ids=np.array(iids, dtype="U"),
        accepted=np.array(accepted, dtype=np.int8),
        repos=np.array(repos, dtype="U"),
    )
    print(f"Wrote {args.out}: z_patch={z_patch.shape}, accepted_rate={np.mean(accepted):.3f}", flush=True)

    # Sanity checks
    assert not np.any(np.isnan(z_patch)), "NaN detected in embeddings!"
    assert z_patch.shape[0] == len(iids), f"Shape mismatch: {z_patch.shape[0]} vs {len(iids)}"
    print("Sanity checks passed", flush=True)


if __name__ == "__main__":
    main()
