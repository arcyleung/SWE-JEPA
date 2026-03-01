"""
Experiment 4.2 SFT Baseline: Full fine-tune Qwen2.5-Coder-3B + InfoNCE.

Comparison vs SWE-JEPA (Exp 3.0) at fixed compute budget. Identical InfoNCE
objective and 8B body embedding targets; only the architecture differs:

  JEPA:  frozen 8B teacher + 63M trainable student (sig-only input, no mismatch)
  SFT:   3B model fine-tuned end-to-end (full function input, sig-only eval)

Training input:  sig_text + '\\n' + body_text (up to 512 tokens)
Eval input:      sig_text only  (deliberate train-test mismatch — favours JEPA)
Target:          Qwen3-8B-base layer-18 body embeddings (precomputed, then teacher freed)
Projection:      Linear(2048 → 4096) to match 8B target dimension
Quality log:     docs/sft_compute_log.jsonl — {step, epoch, gpu_minutes, val_r10} per epoch

Running:
    source .venv/bin/activate && torchrun --nproc-per-node=8 train_sft_baseline.py

If OOM:
    torchrun --nproc-per-node=8 train_sft_baseline.py --batch-size 8 --gradient-checkpointing

VRAM budget per card (80 GB, 8 GPUs):
  Qwen2.5-Coder-3B bf16           ~6 GB
  Adam fp32 states (3B params)   ~36 GB
  Activations (batch=16, 512 tok) ~2 GB
  Headroom                       ~36 GB   → batch=16 fits; batch=32 also likely fine
"""

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import faiss
import pg8000.native
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase0_1_similarity_matrix import DB

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SFT_MODEL_PATH   = '/home/original_models/Qwen2.5-Coder-3B'
TEACHER_PATH     = '/home/original_models/Qwen3-8B-base'
TEACHER_LAYER    = 18          # layer used by teacher to compute body emb targets
SFT_LAYER        = 18          # layer tapped for sig embedding at eval (consistency)
SFT_DIM          = 2048        # Qwen2.5-Coder-3B hidden_size
TEACHER_DIM      = 4096        # Qwen3-8B-base hidden_size (target dimension)
MAX_SIG_TOKENS   = 256
MAX_BODY_TOKENS  = 256
MAX_INPUT_TOKENS = 512         # sig + body concatenated

CKPT_PATH    = os.path.join(os.path.dirname(__file__), 'sft_baseline_ckpt.pt')
LOG_PATH     = os.path.join(os.path.dirname(__file__), 'docs', 'sft_compute_log.jsonl')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'docs', 'phase4_2_sft_results.md')


# ── InfoNCE loss ───────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    def __init__(self, init_temp: float = 0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))

    def forward(self, pred, targets, rank_offset=0):
        τ = self.log_temp.exp().clamp(min=1e-4)
        logits = (pred @ targets.T) / τ
        labels = torch.arange(rank_offset, rank_offset + len(pred), device=pred.device)
        return F.cross_entropy(logits, labels)


# ── Forward hook ───────────────────────────────────────────────────────────────

@contextmanager
def capture_layer(model: nn.Module, layer_idx: int):
    """Capture hidden states from model.model.layers[layer_idx]."""
    captured: list[torch.Tensor] = []

    def _hook(module, inputs, outputs):
        if isinstance(outputs, torch.Tensor):
            captured.append(outputs)
        else:
            captured.append(outputs[0])

    hook = model.model.layers[layer_idx].register_forward_hook(_hook)
    try:
        yield captured
    finally:
        hook.remove()


# ── Dataset / collate ──────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """Stores sig and body tokenised separately; collate handles concatenation."""
    def __init__(self, sig_ids, sig_masks, body_ids, body_masks, body_embs):
        self.sig_ids    = sig_ids
        self.sig_masks  = sig_masks
        self.body_ids   = body_ids
        self.body_masks = body_masks
        self.body_embs  = body_embs.astype(np.float32)

    def __len__(self):
        return len(self.body_embs)

    def __getitem__(self, idx):
        return {
            'sig_ids':    self.sig_ids[idx],
            'sig_masks':  self.sig_masks[idx],
            'body_ids':   self.body_ids[idx],
            'body_masks': self.body_masks[idx],
            'body_emb':   self.body_embs[idx],
        }


def train_collate_fn(batch):
    """Concatenate sig+body into one sequence; track body_start and body_len."""
    B = len(batch)
    # Compute truncated lengths within MAX_INPUT_TOKENS budget
    sig_lens  = [min(len(x['sig_ids']),  MAX_SIG_TOKENS)  for x in batch]
    body_lens = [min(len(x['body_ids']), MAX_BODY_TOKENS) for x in batch]
    # Further truncate body if sig+body > MAX_INPUT_TOKENS
    body_lens = [min(bl, MAX_INPUT_TOKENS - sl) for sl, bl in zip(sig_lens, body_lens)]
    body_lens = [max(bl, 0) for bl in body_lens]
    max_len   = max(sl + bl for sl, bl in zip(sig_lens, body_lens))

    ids_t        = torch.zeros(B, max_len, dtype=torch.long)
    mask_t       = torch.zeros(B, max_len, dtype=torch.long)
    body_start_t = torch.zeros(B, dtype=torch.long)
    body_len_t   = torch.zeros(B, dtype=torch.long)
    body_embs_t  = torch.stack([torch.tensor(x['body_emb']) for x in batch])

    for i, x in enumerate(batch):
        sl, bl = sig_lens[i], body_lens[i]
        ids_t[i,  :sl]      = torch.tensor(x['sig_ids'][:sl],   dtype=torch.long)
        ids_t[i,  sl:sl+bl] = torch.tensor(x['body_ids'][:bl],  dtype=torch.long)
        mask_t[i, :sl]      = torch.tensor(x['sig_masks'][:sl],  dtype=torch.long)
        mask_t[i, sl:sl+bl] = torch.tensor(x['body_masks'][:bl], dtype=torch.long)
        body_start_t[i] = sl
        body_len_t[i]   = bl

    return {
        'input_ids':      ids_t,
        'attention_mask': mask_t,
        'body_start':     body_start_t,
        'body_len':       body_len_t,
        'body_emb':       body_embs_t,
    }


def eval_collate_fn(batch):
    """Pad sig only (no body) for inference."""
    B = len(batch)
    max_len = max(len(x['sig_ids']) for x in batch)
    ids_t  = torch.zeros(B, max_len, dtype=torch.long)
    mask_t = torch.zeros(B, max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        sl = len(x['sig_ids'])
        ids_t[i,  :sl] = torch.tensor(x['sig_ids'],   dtype=torch.long)
        mask_t[i, :sl] = torch.tensor(x['sig_masks'], dtype=torch.long)
    return {'input_ids': ids_t, 'attention_mask': mask_t}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(tokenizer):
    conn = pg8000.native.Connection(**DB)
    rows = conn.run("""
        SELECT fst.function_id, fst.sig_text, fst.body_text, fe.instance_id
        FROM function_student_targets fst
        JOIN function_embeddings fe ON fe.id = fst.function_id
        WHERE fst.sig_text IS NOT NULL AND fst.body_text IS NOT NULL
        ORDER BY fst.function_id
    """)
    conn.close()

    if not rows:
        raise RuntimeError("No rows found in function_student_targets.")

    sig_texts  = [r[1] for r in rows]
    body_texts = [r[2] for r in rows]
    repos      = []
    for r in rows:
        parts = r[3].split('__')
        repos.append(f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else r[3])

    valid_mask = np.array(
        [len(s.strip()) >= 5 and len(b.strip()) >= 5
         for s, b in zip(sig_texts, body_texts)])
    valid_idx  = np.where(valid_mask)[0]
    sig_texts  = [sig_texts[i]  for i in valid_idx]
    body_texts = [body_texts[i] for i in valid_idx]
    repos      = [repos[i]      for i in valid_idx]

    sig_enc  = tokenizer(sig_texts,  truncation=True, max_length=MAX_SIG_TOKENS,
                         padding=False, return_tensors=None)
    body_enc = tokenizer(body_texts, truncation=True, max_length=MAX_BODY_TOKENS,
                         padding=False, return_tensors=None)

    return (sig_enc['input_ids'], sig_enc['attention_mask'],
            body_enc['input_ids'], body_enc['attention_mask'],
            repos)


def repo_split(repos, train_frac=0.80, val_frac=0.10, seed=42):
    rng = random.Random(seed)
    unique_repos = sorted(set(repos))
    rng.shuffle(unique_repos)
    n       = len(unique_repos)
    n_val   = max(1, int(n * val_frac))
    n_test  = max(1, n - int(n * train_frac) - n_val)
    n_train = n - n_val - n_test
    train_repos = set(unique_repos[:n_train])
    val_repos   = set(unique_repos[n_train: n_train + n_val])
    train_idx, val_idx, test_idx = [], [], []
    for i, repo in enumerate(repos):
        if repo in train_repos:
            train_idx.append(i)
        elif repo in val_repos:
            val_idx.append(i)
        else:
            test_idx.append(i)
    return train_idx, val_idx, test_idx


# ── Body embedding precomputation (8B teacher, same as Exp 3.0) ────────────────

def precompute_body_embs(frozen_qwen, sig_ids_list, sig_masks_list,
                          body_ids_list, body_masks_list, device,
                          batch_size=16, is_main=False):
    """
    Full-function context (sig+body) → layer-18 hidden states → mean-pool body positions.
    Identical to train_student_3_0.py to ensure target embeddings are the same.
    """
    N = len(body_ids_list)
    body_embs = np.zeros((N, TEACHER_DIM), dtype=np.float32)
    n_batches = (N + batch_size - 1) // batch_size
    log_every = max(1, n_batches // 10)

    if is_main:
        print(f"  Pre-computing 8B body embeddings for {N:,} functions "
              f"({n_batches} batches) …", flush=True)

    frozen_qwen.eval()
    with capture_layer(frozen_qwen, TEACHER_LAYER) as cap, torch.no_grad():
        for batch_no, start in enumerate(range(0, N, batch_size)):
            batch_idx = range(start, min(start + batch_size, N))
            bsz = len(batch_idx)
            sig_lens  = [len(sig_ids_list[i])  for i in batch_idx]
            body_lens = [len(body_ids_list[i]) for i in batch_idx]
            max_total = max(s + b for s, b in zip(sig_lens, body_lens))

            ids  = torch.zeros(bsz, max_total, dtype=torch.long, device=device)
            mask = torch.zeros(bsz, max_total, dtype=torch.long, device=device)
            body_start = []
            for j, i in enumerate(batch_idx):
                sl, bl = sig_lens[j], body_lens[j]
                ids[j,  :sl]      = torch.tensor(sig_ids_list[i],   dtype=torch.long)
                ids[j,  sl:sl+bl] = torch.tensor(body_ids_list[i],  dtype=torch.long)
                mask[j, :sl]      = torch.tensor(sig_masks_list[i], dtype=torch.long)
                mask[j, sl:sl+bl] = torch.tensor(body_masks_list[i],dtype=torch.long)
                body_start.append(sl)

            cap.clear()
            frozen_qwen(input_ids=ids, attention_mask=mask)
            hs = cap[-1].float()

            for j, i in enumerate(batch_idx):
                bs, bl = body_start[j], body_lens[j]
                body_hs = hs[j, bs:bs+bl]
                body_m  = mask[j, bs:bs+bl].float().unsqueeze(-1)
                emb     = (body_hs * body_m).sum(0) / body_m.sum(0).clamp(min=1.0)
                body_embs[i] = emb.cpu().numpy()

            if is_main and (batch_no + 1) % log_every == 0:
                print(f"    {min(start + batch_size, N)}/{N} done …", flush=True)

    body_embs = np.where(np.isfinite(body_embs), body_embs, 0.0)
    return body_embs


# ── LR schedule ────────────────────────────────────────────────────────────────

def cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr_frac=0.01):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Eval: sig-only → layer-SFT_LAYER mean-pool → proj_head → Rank@10 ──────────

def evaluate_rank10(sft_model, proj_head, sig_ids_list, sig_masks_list,
                    global_idx, body_n_corpus, device, batch_size=32):
    """
    Encode sig-only through the fine-tuned 3B at layer SFT_LAYER.
    Returns (rank1, rank5, rank10) against the full body_n_corpus.
    """
    raw_model = sft_model.module if hasattr(sft_model, 'module') else sft_model
    raw_proj  = proj_head.module  if hasattr(proj_head,  'module') else proj_head
    raw_model.eval()
    raw_proj.eval()

    preds = []
    with capture_layer(raw_model, SFT_LAYER) as cap, torch.no_grad():
        for start in range(0, len(sig_ids_list), batch_size):
            b_ids   = sig_ids_list[start:start + batch_size]
            b_masks = sig_masks_list[start:start + batch_size]
            max_len = max(len(x) for x in b_ids)
            ids_t  = torch.zeros(len(b_ids), max_len, dtype=torch.long, device=device)
            mask_t = torch.zeros(len(b_ids), max_len, dtype=torch.long, device=device)
            for i, (ids, masks) in enumerate(zip(b_ids, b_masks)):
                sl = len(ids)
                ids_t[i,  :sl] = torch.tensor(ids,   dtype=torch.long)
                mask_t[i, :sl] = torch.tensor(masks, dtype=torch.long)

            cap.clear()
            raw_model(input_ids=ids_t, attention_mask=mask_t)
            hs = cap[-1].float()

            real   = mask_t.float().unsqueeze(-1)
            pooled = (hs * real).sum(1) / real.sum(1).clamp(min=1.0)
            emb    = raw_proj(pooled.float())
            preds.append(F.normalize(emb, dim=-1).cpu().numpy())

    raw_model.train()
    raw_proj.train()

    preds = np.vstack(preds)
    preds = np.where(np.isfinite(preds), preds, 0.0)

    index = faiss.IndexFlatIP(body_n_corpus.shape[1])
    index.add(body_n_corpus.astype(np.float32))
    _, I = index.search(preds.astype(np.float32), 10)

    r1  = sum(1 for i, gi in enumerate(global_idx) if gi == I[i, 0])   / len(global_idx)
    r5  = sum(1 for i, gi in enumerate(global_idx) if gi in I[i, :5])  / len(global_idx)
    r10 = sum(1 for i, gi in enumerate(global_idx) if gi in I[i, :10]) / len(global_idx)
    return r1, r5, r10


# ── Training ───────────────────────────────────────────────────────────────────

def train(args, rank, world_size, local_rank):
    is_main = (rank == 0)
    device  = torch.device(f'cuda:{local_rank}')

    if is_main:
        print(f"SWE-JEPA Experiment 4.2: SFT Baseline (Qwen2.5-Coder-3B full fine-tune)")
        print(f"{'='*72}")
        print(f"  World size: {world_size}  |  Batch/GPU: {args.batch_size}  "
              f"|  Effective batch: {args.batch_size * world_size}", flush=True)

    # ── Tokeniser (3B) + data ─────────────────────────────────────────────────
    if is_main:
        print(f"\nLoading 3B tokeniser from {SFT_MODEL_PATH} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True)

    if is_main:
        print(f"Loading data from postgres …", flush=True)
    (sig_ids_list, sig_masks_list,
     body_ids_list, body_masks_list,
     repos) = load_data(tokenizer)
    if is_main:
        print(f"  {len(repos):,} samples from {len(set(repos))} repos", flush=True)

    train_idx, val_idx, test_idx = repo_split(repos)
    if is_main:
        print(f"  Split: {len(train_idx):,} train / "
              f"{len(val_idx):,} val / {len(test_idx):,} test", flush=True)

    # ── Load 8B teacher, precompute body embs, then free ─────────────────────
    # Optionally load precomputed body embeddings from disk to avoid recomputation.
    cache_path = args.body_emb_cache if args.body_emb_cache else None
    body_embs = None
    loaded_from_cache = False

    # Only rank 0 checks cache and computes if needed; all ranks load from cache after.
    cache_exists = cache_path and os.path.exists(cache_path)
    should_recompute = args.force_recompute_body_embs or not cache_exists

    if is_main and not should_recompute:
        try:
            if is_main:
                print(f"\nChecking body embeddings cache: {cache_path}", flush=True)
            loaded = np.load(cache_path)
            if loaded.shape[0] == len(body_ids_list) and loaded.shape[1] == TEACHER_DIM:
                loaded_from_cache = True
                if is_main:
                    print(f"  Cache valid ({loaded.shape}). All ranks will load.", flush=True)
            else:
                if is_main:
                    print(f"  Cache shape mismatch (found {loaded.shape}), will recompute.", flush=True)
                should_recompute = True
        except Exception as e:
            if is_main:
                print(f"  Failed to load cache ({cache_path}): {e}; will recompute.", flush=True)
            should_recompute = True

    if should_recompute:
        if is_main:
            print(f"\nLoading frozen Qwen3-8B-base for body emb precomputation …", flush=True)
        frozen_qwen = AutoModelForCausalLM.from_pretrained(
            TEACHER_PATH, dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device).eval()
        for p in frozen_qwen.parameters():
            p.requires_grad_(False)

        body_embs = precompute_body_embs(
            frozen_qwen, sig_ids_list, sig_masks_list,
            body_ids_list, body_masks_list, device,
            batch_size=args.batch_size, is_main=is_main,
        )
        if is_main:
            print(f"  Done: {body_embs.shape}. Freeing 8B teacher …", flush=True)
        del frozen_qwen
        torch.cuda.empty_cache()

        # Save cache (only main writes to avoid races)
        if cache_path and is_main:
            try:
                np.save(cache_path, body_embs)
                if is_main:
                    print(f"  Saved body embeddings to cache: {cache_path}", flush=True)
            except Exception as e:
                if is_main:
                    print(f"  Failed to save cache ({cache_path}): {e}", flush=True)

    # Synchronise: wait for rank 0 to finish (compute or save), then all load from cache
    dist.barrier()
    if body_embs is None:
        if cache_path and os.path.exists(cache_path):
            try:
                if is_main:
                    print(f"\nRank {rank}: Loading body embeddings from cache …", flush=True)
                loaded = np.load(cache_path)
                if loaded.shape[0] == len(body_ids_list) and loaded.shape[1] == TEACHER_DIM:
                    body_embs = loaded.astype(np.float32)
                    if is_main:
                        print(f"  Rank {rank} loaded cache: {body_embs.shape}", flush=True)
                else:
                    raise RuntimeError(
                        f"Rank {rank}: Cache shape mismatch: expected "
                        f"({len(body_ids_list)}, {TEACHER_DIM}), got {loaded.shape}")
            except Exception as e:
                raise RuntimeError(f"Rank {rank} failed to load cache ({cache_path}): {e}") from e
        else:
            raise RuntimeError(f"Rank {rank}: body_embs not loaded and no cache available.")

    # Mean-centre + L2-normalise using train-split mean
    body_mean     = body_embs[train_idx].mean(axis=0, keepdims=True)
    body_c        = body_embs - body_mean
    body_n        = body_c / (np.linalg.norm(body_c, axis=1, keepdims=True) + 1e-9)
    body_n        = np.where(np.isfinite(body_n), body_n, 0.0)
    body_n_corpus = body_n.copy()

    if is_main:
        rng      = np.random.default_rng(42)
        rand_idx = rng.integers(0, len(body_n), size=len(test_idx))
        rand_cos = float(np.mean((body_n[rand_idx] * body_n[test_idx]).sum(axis=1)))
        print(f"  Random embedding cosine (test): {rand_cos:.4f}", flush=True)

    # ── Load trainable 3B model + projection head ─────────────────────────────
    if is_main:
        print(f"\nLoading Qwen2.5-Coder-3B for fine-tuning …", flush=True)
    sft_model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)

    if args.gradient_checkpointing:
        sft_model.gradient_checkpointing_enable()
        if is_main:
            print("  Gradient checkpointing enabled.", flush=True)

    # Projection head: maps 3B dim → 8B target dim
    proj_head = nn.Linear(SFT_DIM, TEACHER_DIM, bias=False).to(device)

    n_params = (sum(p.numel() for p in sft_model.parameters()) +
                sum(p.numel() for p in proj_head.parameters()))
    if is_main:
        print(f"  Trainable: {n_params:,} params "
              f"(3B model + proj head)", flush=True)

    print(f"[rank {rank}] reached DDP barrier", flush=True)
    dist.barrier()

    # find_unused_parameters=True because layers > SFT_LAYER don't contribute to loss
    ddp_sft  = DDP(sft_model,  device_ids=[local_rank], find_unused_parameters=True)
    ddp_proj = DDP(proj_head,  device_ids=[local_rank], find_unused_parameters=False)

    criterion = InfoNCELoss(init_temp=args.init_temp).to(device)
    all_params = (list(ddp_sft.parameters()) +
                  list(ddp_proj.parameters()) +
                  list(criterion.parameters()))
    optimizer = optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)
    scheduler = cosine_schedule_with_warmup(
        optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs)

    # ── Datasets + loaders ────────────────────────────────────────────────────
    def make_ds(idx):
        return SFTDataset(
            [sig_ids_list[i]    for i in idx],
            [sig_masks_list[i]  for i in idx],
            [body_ids_list[i]   for i in idx],
            [body_masks_list[i] for i in idx],
            body_n[idx],
        )

    train_ds = make_ds(train_idx)
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        collate_fn=train_collate_fn, num_workers=4, pin_memory=True)

    # Sig-only lists for evaluate_rank10 (all ranks hold these; rank 0 uses them)
    val_sig_ids    = [sig_ids_list[i]    for i in val_idx]
    val_sig_masks  = [sig_masks_list[i]  for i in val_idx]
    test_sig_ids   = [sig_ids_list[i]    for i in test_idx]
    test_sig_masks = [sig_masks_list[i]  for i in test_idx]

    # ── Compute log ───────────────────────────────────────────────────────────
    if is_main:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        log_f = open(LOG_PATH, 'a')

    def log_checkpoint(step, epoch, gpu_minutes, val_r1, val_r5, val_r10,
                       train_loss, temp):
        if not is_main:
            return
        entry = {
            'step': step, 'epoch': epoch,
            'gpu_minutes': round(gpu_minutes, 2),
            'val_r1':  round(val_r1  * 100, 4),
            'val_r5':  round(val_r5  * 100, 4),
            'val_r10': round(val_r10 * 100, 4),
            'train_loss': round(float(train_loss), 6),
            'temp': round(float(temp), 5),
            'model': 'sft_3b',
        }
        log_f.write(json.dumps(entry) + '\n')
        log_f.flush()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_r10    = -1.0
    best_state_sft  = None
    best_state_proj = None
    patience_count  = 0
    global_step     = 0
    t0              = time.time()

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        ddp_sft.train()
        ddp_proj.train()

        epoch_losses = []

        with capture_layer(ddp_sft.module, SFT_LAYER) as captured:
            for batch in train_loader:
                input_ids  = batch['input_ids'].to(device,  non_blocking=True)
                attn_mask  = batch['attention_mask'].to(device, non_blocking=True)
                body_start = batch['body_start'].to(device, non_blocking=True)
                body_len   = batch['body_len'].to(device,   non_blocking=True)
                body_emb_t = batch['body_emb'].to(device,   non_blocking=True)

                optimizer.zero_grad()
                captured.clear()
                # Full function forward (gradients flow back to 3B params at layers ≤ SFT_LAYER).
                # We keep hold of the return value so that DistributedDataParallel can
                # validate that *all* outputs are used in the loss computation.  The
                # loss itself doesn't actually depend on the LM logits, but if the
                # return value is ignored DDP will complain (see error message about
                # "not all `forward` outputs participate in computing loss").  We
                # therefore grab the output and include a zero-weighted term in the
                # loss below.
                out = ddp_sft(input_ids=input_ids, attention_mask=attn_mask)
                # pull out a tensor we can safely reduce to a scalar
                if isinstance(out, (tuple, list)):
                    dummy_out = out[0]
                elif hasattr(out, 'logits'):
                    dummy_out = out.logits
                else:
                    dummy_out = out

                hs = captured[-1].float()  # [B, T, 2048] — NO detach: gradients needed

                # Mean-pool body-token positions to get the "predicted body embedding"
                pooled_list = []
                B = hs.shape[0]
                for j in range(B):
                    bs = body_start[j].item()
                    bl = body_len[j].item()
                    if bl > 0:
                        body_hs_j = hs[j, bs:bs + bl]                          # [bl, 2048]
                        body_m_j  = attn_mask[j, bs:bs + bl].float().unsqueeze(-1)
                        p = (body_hs_j * body_m_j).sum(0) / body_m_j.sum(0).clamp(min=1.0)
                    else:
                        # Fallback: pool full sig (body was fully truncated)
                        real = attn_mask[j].float().unsqueeze(-1)
                        p = (hs[j] * real).sum(0) / real.sum(0).clamp(min=1.0)
                    pooled_list.append(p)
                pooled = torch.stack(pooled_list)   # [B, 2048]

                pred   = ddp_proj(pooled.to(torch.float32))   # [B, 4096]
                pred_n = F.normalize(pred, dim=-1)
                tgt_n  = F.normalize(body_emb_t, dim=-1)

                # All-gather targets across GPUs for InfoNCE
                gathered = [torch.zeros_like(tgt_n) for _ in range(world_size)]
                dist.all_gather(gathered, tgt_n)
                all_tgt = torch.cat(gathered, dim=0)

                # make sure DDP sees the forward output used by tying a 0.0 * sum to
                # the loss.  This has no numerical effect but avoids the reduction
                # error seen when the output is unused.
                loss = criterion(pred_n, all_tgt, rank_offset=rank * len(pred_n))
                loss = loss + 0.0 * dummy_out.sum()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(ddp_sft.parameters()) + list(ddp_proj.parameters()), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())
                global_step += 1

        scheduler.step()
        dist.barrier()

        # ── Val Rank@10 (rank 0 only; broadcast result) ───────────────────────
        val_r1 = val_r5 = val_r10 = 0.0
        if is_main:
            val_r1, val_r5, val_r10 = evaluate_rank10(
                ddp_sft, ddp_proj,
                val_sig_ids, val_sig_masks, val_idx,
                body_n_corpus, device,
            )
            train_loss = float(np.mean(epoch_losses))
            elapsed    = time.time() - t0
            gpu_mins   = elapsed * world_size / 60.0  # total GPU-minutes consumed
            τ          = criterion.log_temp.exp().item()
            lr         = scheduler.get_last_lr()[0]

            print(
                f"  epoch {epoch:4d}/{args.epochs}  "
                f"loss={train_loss:.4f}  val_r10={val_r10*100:.2f}%  "
                f"τ={τ:.4f}  lr={lr:.2e}  gpu_min={gpu_mins:.0f}",
                flush=True,
            )
            log_checkpoint(global_step, epoch, gpu_mins,
                           val_r1, val_r5, val_r10, train_loss, τ)

        # ── Early stopping (rank 0 decides, broadcast) ────────────────────────
        val_r10_t = torch.tensor(val_r10, device=device)
        dist.broadcast(val_r10_t, src=0)
        val_r10_sync = val_r10_t.item()

        improved = torch.tensor(int(val_r10_sync > best_val_r10), device=device)
        dist.broadcast(improved, src=0)
        if improved.item():
            best_val_r10    = val_r10_sync
            best_state_sft  = {k: v.cpu().clone()
                               for k, v in ddp_sft.module.state_dict().items()}
            best_state_proj = {k: v.cpu().clone()
                               for k, v in ddp_proj.module.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if is_main and epoch % args.checkpoint_every == 0:
            periodic = CKPT_PATH.replace('.pt', f'_ep{epoch:04d}.pt')
            torch.save({
                'sft_state':  {k: v.cpu().clone()
                               for k, v in ddp_sft.module.state_dict().items()},
                'proj_state': {k: v.cpu().clone()
                               for k, v in ddp_proj.module.state_dict().items()},
                'epoch': epoch, 'args': vars(args),
            }, periodic)
            if is_main:
                print(f"  Periodic checkpoint → {periodic}", flush=True)

        if patience_count >= args.patience:
            if is_main:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no val_r10 improvement for {args.patience} epochs)", flush=True)
            break

    # ── Final evaluation (rank 0) ─────────────────────────────────────────────
    if is_main:
        ddp_sft.module.load_state_dict(best_state_sft)
        ddp_proj.module.load_state_dict(best_state_proj)

        val_r1, val_r5, val_r10 = evaluate_rank10(
            ddp_sft, ddp_proj, val_sig_ids, val_sig_masks, val_idx,
            body_n_corpus, device,
        )
        test_r1, test_r5, test_r10 = evaluate_rank10(
            ddp_sft, ddp_proj, test_sig_ids, test_sig_masks, test_idx,
            body_n_corpus, device,
        )

        N        = len(body_n)
        elapsed  = time.time() - t0
        gpu_mins = elapsed * world_size / 60.0

        print(f"\nVal  ({len(val_idx)} funcs, corpus={N:,}):  "
              f"R@1={val_r1*100:.2f}%  R@5={val_r5*100:.2f}%  "
              f"R@10={val_r10*100:.2f}%")
        print(f"Test ({len(test_idx)} funcs, corpus={N:,}):  "
              f"R@1={test_r1*100:.2f}%  R@5={test_r5*100:.2f}%  "
              f"R@10={test_r10*100:.2f}%")
        print(f"Total GPU-minutes: {gpu_mins:.0f}")

        torch.save({
            'sft_state':  best_state_sft,
            'proj_state': best_state_proj,
            'sft_dim':    SFT_DIM,
            'teacher_dim': TEACHER_DIM,
            'sft_layer':  SFT_LAYER,
            'args': vars(args),
            'val_r10': val_r10, 'test_r10': test_r10,
        }, CKPT_PATH)
        print(f"Checkpoint → {CKPT_PATH}", flush=True)

        write_results({
            'n_train': len(train_idx), 'n_val': len(val_idx),
            'n_test': len(test_idx), 'corpus': N, 'n_params': n_params,
            'val_r1': val_r1,   'val_r5': val_r5,   'val_r10': val_r10,
            'test_r1': test_r1, 'test_r5': test_r5, 'test_r10': test_r10,
            'gpu_minutes': gpu_mins, 'epochs_trained': epoch,
            'best_val_r10': best_val_r10, 'args': vars(args),
        })
        log_f.close()


# ── Results writer ─────────────────────────────────────────────────────────────

def write_results(res):
    a = res['args']
    report = f"""# Experiment 4.2 SFT Baseline Results

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Model**: Qwen2.5-Coder-3B full fine-tune + Linear(2048→4096) projection head
**Training input**: sig_text + body_text (full function, up to {MAX_INPUT_TOKENS} tokens)
**Eval input**: sig_text only (deliberate train-test mismatch — favours JEPA)
**Target**: Qwen3-8B-base layer-{TEACHER_LAYER} body embeddings (same as JEPA Exp 3.0)

## Architecture

| Component | Detail |
|-----------|--------|
| SFT model | Qwen2.5-Coder-3B ({res['n_params']:,} total params incl. proj) |
| Projection head | Linear({SFT_DIM} → {TEACHER_DIM}, no bias) |
| Eval layer | Layer {SFT_LAYER} mean-pool (sig-only at inference) |
| Loss | InfoNCE (τ learnable, same as JEPA) |
| Teacher (body targets only) | Qwen3-8B-base layer {TEACHER_LAYER} (freed after precompute) |

## Dataset

| Split | Functions |
|-------|-----------|
| Train | {res['n_train']:,} |
| Val   | {res['n_val']:,} |
| Test  | {res['n_test']:,} |
| Corpus (retrieval) | {res['corpus']:,} |

## Training

| Hyperparameter | Value |
|----------------|-------|
| Epochs trained | {res['epochs_trained']} (patience={a['patience']}) |
| Batch/GPU | {a['batch_size']} |
| LR | {a['lr']} (cosine, warmup={a['warmup_epochs']}) |
| Total GPU-minutes | {res['gpu_minutes']:.0f} |

## Results

| Split | Rank@1 | Rank@5 | Rank@10 |
|-------|--------|--------|---------|
| Val   | {res['val_r1']*100:.2f}% | {res['val_r5']*100:.2f}% | {res['val_r10']*100:.2f}% |
| Test  | {res['test_r1']*100:.2f}% | {res['test_r5']*100:.2f}% | {res['test_r10']*100:.2f}% |

## Comparison with JEPA (Exp 3.0)

| Method | Val Rank@10 | Training input | GPU-minutes |
|--------|------------|----------------|-------------|
| JEPA (Exp 3.0) | 47.52% | sig-only (no mismatch) | (see JEPA log) |
| SFT Baseline   | {res['val_r10']*100:.2f}% | sig+body → sig-only eval | {res['gpu_minutes']:.0f} |

See `docs/sft_compute_log.jsonl` for the quality-vs-GPU-minutes curve.
"""
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        f.write(report)
    print(f"Results → {RESULTS_FILE}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='SWE-JEPA Exp 4.2: SFT Baseline')
    ap.add_argument('--epochs',               type=int,   default=100)
    ap.add_argument('--batch-size',           type=int,   default=8)
    ap.add_argument('--lr',                   type=float, default=5e-5)
    ap.add_argument('--warmup-epochs',        type=int,   default=3)
    ap.add_argument('--patience',             type=int,   default=15)
    ap.add_argument('--checkpoint-every',     type=int,   default=10)
    ap.add_argument('--init-temp',            type=float, default=0.07)
    ap.add_argument('--gradient-checkpointing', action='store_true', default=False,
                    help='Enable gradient checkpointing to trade compute for VRAM.')
    ap.add_argument('--body-emb-cache', type=str,
                    default=None, help='Path to load/save precomputed body embeddings (numpy .npy).') # default=os.path.join(os.path.dirname(__file__), 'body_embs.npy'),
    ap.add_argument('--force-recompute-body-embs', action='store_true', default=False,
                    help='Force recomputation of body embeddings even if cache exists.')
    args = ap.parse_args()

    dist.init_process_group(backend='nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))

    torch.manual_seed(42 + rank)
    torch.cuda.set_device(local_rank)

    try:
        train(args, rank, world_size, local_rank)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
