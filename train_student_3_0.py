"""
Experiment 3.0: Qwen3-8B-base teacher, token-level body prediction + InfoNCE.

Upgrade from Exp 2.2 (Qwen2.5-Coder-3B teacher, 2048-dim) to Qwen3-8B-base
(4096-dim, ~4× more representational capacity).  Architecture is identical to
Exp 2.2; only the teacher model and embedding dimension change.

Key differences from Exp 2.2:
  - Teacher: Qwen3-8B-base (4096-dim, 36 layers) vs Qwen2.5-Coder-3B (2048-dim)
  - SigPredictorV2 d_in / d_out: 4096 (was 2048)
  - Body embeddings not in postgres for 8B; pre-computed at startup via teacher
  - LOCAL_RANK used for device (correct for multi-node torchrun)
  - Optional --fsdp-teacher: wraps frozen teacher with intra-node FSDP to
    reduce per-GPU memory from 16 GB → 2 GB (8-way shard).  Required when
    the other workloads on the node leave < 18 GB free VRAM per card.
  - Student: DDP across all ranks (multi-node works via torchrun + NCCL)

Single-node (8 GPUs, 80 GB each):
  source .venv/bin/activate && torchrun --nproc-per-node=8 train_student_3_0.py

Multi-node (4 nodes × 8 GPUs = 32 total, 80 GB VRAM), run on EVERY node:
  export NCCL_SOCKET_IFNAME=bond0; export NCCL_IB_HCA=mlx5; export NCCL_GID_INDEX=3
  source .venv/bin/activate && torchrun \\
    --nnodes=4 \\
    --nproc-per-node=8 \\
    --rdzv-backend=c10d \\
    --rdzv-endpoint=10.10.110.20:29500 \\
    train_student_3_0.py

If OOM with 16 GB teacher per GPU, add --fsdp-teacher:
  torchrun ... train_student_3_0.py --fsdp-teacher

VRAM budget per card (80 GB, 32 total ranks):
  Teacher Qwen3-8B bf16   ~16 GB
  Teacher activations      ~0.5 GB  (2× no_grad forward, batch=64, seq=256, dim=4096)
  Student + Adam states    ~0.4 GB
  Headroom                ~63 GB   → default batch=64; batch=128 also fits
"""

import argparse
import functools
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

TEACHER_PATH    = '/home/original_models/Qwen3-8B-base'
TEACHER_LAYER   = 18          # 0-indexed; mid-layer of 36 total (same as Exp 2.2)
MAX_SIG_TOKENS  = 256
MAX_BODY_TOKENS = 256
RESULTS_FILE    = os.path.join(os.path.dirname(__file__),
                                'docs', 'phase3_0_qwen3_8b_teacher.md')


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    def __init__(self, init_temp: float = 0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))

    def forward(self, pred, targets, rank_offset=0):
        τ = self.log_temp.exp().clamp(min=1e-4)
        logits = (pred @ targets.T) / τ
        labels = torch.arange(rank_offset, rank_offset + len(pred), device=pred.device)
        return F.cross_entropy(logits, labels)


# ── Model ─────────────────────────────────────────────────────────────────────

class _CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, query, context, query_pad_mask, context_pad_mask):
        q  = self.norm_q(query)
        kv = self.norm_kv(context)
        attn_out, _ = self.cross_attn(
            q, kv, kv, key_padding_mask=context_pad_mask, need_weights=False)
        query = query + self.drop(attn_out)
        query = query + self.drop(self.ffn(self.norm2(query)))
        return query


class _TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                key_padding_mask=pad_mask, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class SigPredictorV2(nn.Module):
    """Same architecture as Exp 2.2 but d_in/d_out=4096 for the 8B teacher.

    in_norm: LayerNorm applied to each token's 4096-dim hidden state before
    projection.  Removes per-token anisotropy (all tokens pointing in a shared
    mean direction) which is more severe for Qwen3-8B than for 3B.
    """

    def __init__(self, d_in=4096, d_model=512, nhead=8, num_layers=2,
                 d_out=4096, dropout=0.1, max_body_tokens=MAX_BODY_TOKENS):
        super().__init__()
        self.in_norm      = nn.LayerNorm(d_in)
        self.proj_in      = nn.Linear(d_in, d_model)
        self.enc_blocks   = nn.ModuleList(
            [_TransformerBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.proj_out     = nn.Linear(d_model, d_out)
        self.body_pos_enc = nn.Embedding(max_body_tokens, d_model)
        self.dec_blocks   = nn.ModuleList(
            [_CrossAttentionBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.proj_token   = nn.Linear(d_model, d_out)

    def _encode(self, hidden_states, attention_mask):
        x = self.proj_in(self.in_norm(hidden_states.float()))
        pad_mask = (attention_mask == 0)
        for block in self.enc_blocks:
            x = block(x, pad_mask)
        real_mask = attention_mask.float().unsqueeze(-1)
        pooled = (x * real_mask).sum(1) / real_mask.sum(1).clamp(min=1.0)
        return x, pooled

    def forward(self, hidden_states, attention_mask, body_len=None, body_pad_mask=None):
        enc_tokens, pooled = self._encode(hidden_states, attention_mask)
        retrieval_pred = self.proj_out(pooled)
        token_preds = None
        if body_len is not None:
            B, device = hidden_states.shape[0], hidden_states.device
            pos_ids = torch.arange(body_len, device=device)
            queries = self.body_pos_enc(pos_ids).unsqueeze(0).expand(B, -1, -1)
            sig_pad_mask = (attention_mask == 0)
            if body_pad_mask is None:
                body_pad_mask = torch.zeros(B, body_len, dtype=torch.bool, device=device)
            for block in self.dec_blocks:
                queries = block(queries, enc_tokens, body_pad_mask, sig_pad_mask)
            token_preds = self.proj_token(queries)
        return retrieval_pred, token_preds


# ── Forward hook ──────────────────────────────────────────────────────────────

@contextmanager
def capture_layer(model: nn.Module, layer_idx: int):
    """Capture hidden states from model.model.layers[layer_idx].
    Works with plain nn.Module and FSDP-wrapped models alike:
    the hook is registered on whatever object model.model.layers[layer_idx]
    resolves to at call time."""
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


# ── Dataset ───────────────────────────────────────────────────────────────────

class SigBodyDataset(Dataset):
    def __init__(self, input_ids, attention_masks, body_embs,
                 body_input_ids, body_attention_masks):
        self.input_ids            = input_ids
        self.attention_masks      = attention_masks
        self.body_embs            = body_embs.astype(np.float32)
        self.body_input_ids       = body_input_ids
        self.body_attention_masks = body_attention_masks

    def __len__(self):
        return len(self.body_embs)

    def __getitem__(self, idx):
        return {
            'input_ids':            self.input_ids[idx],
            'attention_mask':       self.attention_masks[idx],
            'body_emb':             self.body_embs[idx],
            'body_input_ids':       self.body_input_ids[idx],
            'body_attention_mask':  self.body_attention_masks[idx],
        }


def collate_fn(batch):
    B = len(batch)
    sig_max  = max(len(x['input_ids'])       for x in batch)
    body_max = max(len(x['body_input_ids'])  for x in batch)
    sig_ids_t   = torch.zeros(B, sig_max,  dtype=torch.long)
    sig_mask_t  = torch.zeros(B, sig_max,  dtype=torch.long)
    body_ids_t  = torch.zeros(B, body_max, dtype=torch.long)
    body_mask_t = torch.zeros(B, body_max, dtype=torch.long)
    body_embs_t = torch.stack([torch.tensor(x['body_emb']) for x in batch])
    for i, x in enumerate(batch):
        ns = len(x['input_ids'])
        sig_ids_t[i, :ns]  = torch.tensor(x['input_ids'],      dtype=torch.long)
        sig_mask_t[i, :ns] = torch.tensor(x['attention_mask'], dtype=torch.long)
        nb = len(x['body_input_ids'])
        body_ids_t[i, :nb]  = torch.tensor(x['body_input_ids'],      dtype=torch.long)
        body_mask_t[i, :nb] = torch.tensor(x['body_attention_mask'], dtype=torch.long)
    return {
        'input_ids':           sig_ids_t,
        'attention_mask':      sig_mask_t,
        'body_emb':            body_embs_t,
        'body_input_ids':      body_ids_t,
        'body_attention_mask': body_mask_t,
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(tokenizer):
    """Load sig/body texts from postgres. Body embeddings are NOT loaded here;
    they are computed at startup via precompute_body_embs() using the 8B teacher."""
    conn = pg8000.native.Connection(**DB)
    rows = conn.run("""
        SELECT fst.function_id,
               fst.sig_text,
               fst.body_text,
               fe.instance_id
        FROM function_student_targets fst
        JOIN function_embeddings fe
          ON fe.id = fst.function_id
        WHERE fst.sig_text IS NOT NULL
          AND fst.body_text IS NOT NULL
        ORDER BY fst.function_id
    """)
    conn.close()

    if not rows:
        raise RuntimeError("No rows found. Ensure function_embeddings has Qwen3-8B-base entries.")

    fids_raw   = [r[0] for r in rows]
    sig_texts  = [r[1] for r in rows]
    body_texts = [r[2] for r in rows]
    repos_raw  = []
    for r in rows:
        parts = r[3].split('__')
        repos_raw.append(f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else r[3])

    empty_sig  = np.array([len(t.strip()) < 5 for t in sig_texts])
    empty_body = np.array([len(t.strip()) < 5 for t in body_texts])
    valid      = ~empty_sig & ~empty_body
    if empty_sig.sum():
        print(f"  Dropped {empty_sig.sum()} empty/trivial sig_text rows", flush=True)
    if empty_body.sum():
        print(f"  Dropped {empty_body.sum()} empty/trivial body_text rows", flush=True)

    valid_idx  = np.where(valid)[0]
    fids       = [fids_raw[i]   for i in valid_idx]
    sig_texts  = [sig_texts[i]  for i in valid_idx]
    body_texts = [body_texts[i] for i in valid_idx]
    repos      = [repos_raw[i]  for i in valid_idx]

    sig_enc  = tokenizer(sig_texts,  truncation=True, max_length=MAX_SIG_TOKENS,
                         padding=False, return_tensors=None)
    body_enc = tokenizer(body_texts, truncation=True, max_length=MAX_BODY_TOKENS,
                         padding=False, return_tensors=None)

    return (sig_enc['input_ids'], sig_enc['attention_mask'],
            body_enc['input_ids'], body_enc['attention_mask'],
            repos, fids)


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


# ── Body embedding pre-computation ────────────────────────────────────────────

def precompute_body_embs(
    frozen_qwen: nn.Module,
    sig_ids_list: list,
    sig_masks_list: list,
    body_ids_list: list,
    body_masks_list: list,
    device: torch.device,
    emb_dim: int,
    batch_size: int = 16,
    is_main: bool = False,
) -> np.ndarray:
    """
    Compute mean-pooled layer-TEACHER_LAYER hidden states for all body texts,
    using sig+body concatenated as the forward-pass input.

    Feeding the full function (sig then body) gives the model the signature
    as context when producing body token representations.  This removes the
    severe anisotropy (~0.37 random cosine) that arises when the teacher sees
    body-only text: without the signature, common body-opening patterns dominate
    and all embeddings cluster in a few directions.  Full-function embeddings
    stored in the DB confirm near-isotropy (random cosine ~0.016).

    Only the body token positions are mean-pooled to form the target embedding,
    faithfully matching the JEPA objective: predict the full-context
    representation of the masked (body) region from the visible (sig) region.

    Every rank independently computes the full corpus — no cross-node collective.
    """
    N = len(body_ids_list)
    body_embs = np.zeros((N, emb_dim), dtype=np.float32)
    n_batches = (N + batch_size - 1) // batch_size
    log_every = max(1, n_batches // 10)

    if is_main:
        print(f"  Pre-computing body embeddings (sig+body context) for {N:,} functions "
              f"({n_batches} batches, logging every {log_every}) …", flush=True)

    frozen_qwen.eval()
    with capture_layer(frozen_qwen, TEACHER_LAYER) as cap, torch.no_grad():
        for batch_no, start in enumerate(range(0, N, batch_size)):
            batch_idx = range(start, min(start + batch_size, N))
            bsz = len(batch_idx)

            # Build sig+body concatenated sequences
            sig_lens  = [len(sig_ids_list[i])  for i in batch_idx]
            body_lens = [len(body_ids_list[i]) for i in batch_idx]
            max_total = max(s + b for s, b in zip(sig_lens, body_lens))

            ids  = torch.zeros(bsz, max_total, dtype=torch.long, device=device)
            mask = torch.zeros(bsz, max_total, dtype=torch.long, device=device)
            # body_start[j] = position where body tokens begin for sample j
            body_start = []
            for j, i in enumerate(batch_idx):
                sl = sig_lens[j];  bl = body_lens[j]
                ids[j,    :sl]       = torch.tensor(sig_ids_list[i],    dtype=torch.long)
                ids[j,  sl:sl+bl]   = torch.tensor(body_ids_list[i],   dtype=torch.long)
                mask[j,   :sl]       = torch.tensor(sig_masks_list[i],  dtype=torch.long)
                mask[j, sl:sl+bl]   = torch.tensor(body_masks_list[i], dtype=torch.long)
                body_start.append(sl)

            cap.clear()
            frozen_qwen(input_ids=ids, attention_mask=mask)
            hs = cap[-1].float()                              # [B, T_total, dim]

            # Mean-pool only body token positions for each sample
            for j, i in enumerate(batch_idx):
                bs = body_start[j]
                bl = body_lens[j]
                body_hs = hs[j, bs:bs+bl]                    # [bl, dim]
                body_m  = mask[j, bs:bs+bl].float().unsqueeze(-1)  # [bl, 1]
                emb = (body_hs * body_m).sum(0) / body_m.sum(0).clamp(min=1.0)
                body_embs[i] = emb.cpu().numpy()

            if is_main and (batch_no + 1) % log_every == 0:
                print(f"    body_embs {batch_no + 1}/{n_batches} batches …", flush=True)

    if is_main:
        bad = ~np.isfinite(body_embs).all(axis=1)
        if bad.any():
            print(f"  WARNING: {bad.sum()} non-finite body_emb rows; replacing with zeros",
                  flush=True)
    body_embs = np.where(np.isfinite(body_embs), body_embs, 0.0)
    return body_embs


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(predictor, frozen_qwen, loader, body_embs_corpus, test_idx,
             device, criterion, desc='eval'):
    """
    For single-rank evaluation (non-FSDP teacher) or distributed evaluation
    where all ranks in the teacher's process group call this simultaneously.
    Returns metrics dict; FAISS retrieval only when test_idx is non-empty.
    """
    predictor.eval()
    preds_list, tgts_list, losses = [], [], []

    with capture_layer(frozen_qwen, TEACHER_LAYER) as captured, torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            body_emb  = batch['body_emb'].to(device)

            captured.clear()
            frozen_qwen(input_ids=input_ids, attention_mask=attn_mask)
            hs = captured[-1]

            if hasattr(predictor, 'module'):
                pred, _ = predictor.module(hs, attn_mask)
            else:
                pred, _ = predictor(hs, attn_mask)

            pred_n = F.normalize(pred, dim=-1).float().cpu().numpy()
            tgt_n  = F.normalize(body_emb, dim=-1).float().cpu().numpy()
            preds_list.append(pred_n)
            tgts_list.append(tgt_n)

    preds = np.vstack(preds_list)
    tgts  = np.vstack(tgts_list)

    if not np.isfinite(preds).all():
        preds = np.where(np.isfinite(preds), preds, 0.0)

    cos = float((preds * tgts).sum(axis=1).mean())
    result = {'cosine': cos, 'rank1': None, 'rank5': None, 'rank10': None}

    if test_idx:
        d = body_embs_corpus.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(body_embs_corpus.astype(np.float32))
        _, I = index.search(preds.astype(np.float32), 10)
        r1  = sum(1 for i, gi in enumerate(test_idx) if gi == I[i, 0])  / len(test_idx)
        r5  = sum(1 for i, gi in enumerate(test_idx) if gi in I[i, :5]) / len(test_idx)
        r10 = sum(1 for i, gi in enumerate(test_idx) if gi in I[i, :10])/ len(test_idx)
        result.update({'rank1': r1, 'rank5': r5, 'rank10': r10})

    return result


def val_rank10_distributed(
    predictor, frozen_qwen, val_loader, val_idx, faiss_index,
    device, rank, world_size, use_fsdp_teacher,
) -> float:
    """
    Compute val Rank@10.

    - use_fsdp_teacher=False: only rank 0 runs inference (same as Exp 2.2).
    - use_fsdp_teacher=True:  all intra-node ranks participate in teacher
      forward (required for FSDP); results gathered to rank 0.
    Returns val_r10 on rank 0, 0.0 on other ranks.
    """
    raw = predictor.module if hasattr(predictor, 'module') else predictor
    raw.eval()

    if not use_fsdp_teacher and rank != 0:
        raw.train()
        return 0.0

    preds_local = []
    idx_local   = []

    if use_fsdp_teacher:
        # All ranks process their shard of val_idx
        my_val = val_idx[rank::world_size]
    else:
        my_val = val_idx  # rank 0 processes all

    # Build a minimal loader for my_val indices (reuse collate_fn)
    # We iterate val_loader and skip non-local batches by index tracking
    # Simpler: rebuild from cached hidden-state eval in loader order,
    # then keep only my_val positions.
    with capture_layer(frozen_qwen, TEACHER_LAYER) as cap, torch.no_grad():
        offset = 0
        for batch in val_loader:
            bsz = batch['input_ids'].shape[0]
            local_batch_idx = list(range(offset, offset + bsz))
            offset += bsz

            if use_fsdp_teacher:
                # Determine which items in this batch belong to my_val
                keep = [j for j, gi in enumerate(local_batch_idx)
                        if val_idx[gi] in set(my_val)]
                if not keep:
                    # Still need to run forward for FSDP collective
                    cap.clear()
                    frozen_qwen(input_ids=batch['input_ids'].to(device),
                                attention_mask=batch['attention_mask'].to(device))
                    cap.clear()
                    continue
            else:
                keep = list(range(bsz))

            cap.clear()
            frozen_qwen(input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device))
            hs = cap[-1]
            if use_fsdp_teacher and keep:
                keep_t = torch.tensor(keep, device=device)
                hs     = hs[keep_t]
                mask   = batch['attention_mask'].to(device)[keep_t]
            else:
                mask = batch['attention_mask'].to(device)

            p, _ = raw(hs, mask)
            preds_local.append(F.normalize(p, dim=-1).float().cpu().numpy())
            idx_local.extend([val_idx[local_batch_idx[j]] for j in keep]
                             if use_fsdp_teacher else list(val_idx[offset - bsz: offset]))

    raw.train()

    if use_fsdp_teacher:
        # Gather all shards to rank 0
        all_preds = [None] * world_size
        all_idxs  = [None] * world_size
        dist.all_gather_object(all_preds,
                               np.vstack(preds_local) if preds_local else np.zeros((0, 1)))
        dist.all_gather_object(all_idxs, idx_local)
        if rank != 0:
            return 0.0
        preds_all = np.vstack([p for p in all_preds if p.shape[0] > 0])
        idxs_all  = [i for lst in all_idxs for i in lst]
    else:
        preds_all = np.vstack(preds_local) if preds_local else np.zeros((0, 1))
        idxs_all  = idx_local

    if len(preds_all) == 0:
        return 0.0
    _, I = faiss_index.search(preds_all.astype(np.float32), 10)
    r10 = sum(1 for i, gi in enumerate(idxs_all) if gi in I[i, :10]) / len(idxs_all)
    return r10


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr_frac=0.01):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args, rank, world_size, local_rank):
    is_main = (rank == 0)
    device  = torch.device(f'cuda:{local_rank}')

    if is_main:
        print(f"SWE-JEPA Experiment 3.0: Qwen3-8B Teacher + Token-Level Body Prediction")
        print(f"{'='*72}")
        print(f"  World size: {world_size}  |  BF16 teacher  |  FSDP teacher: {args.fsdp_teacher}")
        print(f"  Batch/GPU:  {args.batch_size}  →  effective batch: {args.batch_size * world_size}",
              flush=True)

    # ── Tokeniser + data ──────────────────────────────────────────────────────
    if is_main:
        print(f"\nLoading tokeniser …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH, trust_remote_code=True)

    if is_main:
        print(f"Loading data from postgres …", flush=True)
    (sig_ids_list, sig_masks_list,
     body_ids_list, body_masks_list,
     repos, fids) = load_data(tokenizer)
    if is_main:
        print(f"Loaded {len(fids):,} samples from {len(set(repos))} repos", flush=True)

    train_idx, val_idx, test_idx = repo_split(repos)
    val_idx_arr  = np.array(val_idx)   # for distributed val_rank10 indexing
    if is_main:
        print(f"Split: {len(train_idx):,} train  "
              f"{len(val_idx):,} val  {len(test_idx):,} test  (by repo)", flush=True)

    # ── Load frozen teacher (full model per GPU; FSDP wrap after precompute) ──
    if is_main:
        print(f"\nLoading frozen Qwen3-8B-base on cuda:{local_rank} …", flush=True)
    frozen_qwen = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()
    for p in frozen_qwen.parameters():
        p.requires_grad_(False)

    emb_dim = frozen_qwen.config.hidden_size  # 4096

    # ── Pre-compute body embeddings (each rank independently, no collective) ──
    if is_main:
        print(f"\nPre-computing body embeddings (one-time, ~90 s/rank) …", flush=True)
    body_embs = precompute_body_embs(
        frozen_qwen, sig_ids_list, sig_masks_list,
        body_ids_list, body_masks_list,
        device, emb_dim,
        batch_size=args.batch_size, is_main=is_main,
    )
    if is_main:
        print(f"  Done. body_embs: {body_embs.shape}", flush=True)

    # Mean-centre + L2-normalise using train-split mean
    body_mean = body_embs[train_idx].mean(axis=0, keepdims=True)
    body_c    = body_embs - body_mean
    body_n    = body_c / (np.linalg.norm(body_c, axis=1, keepdims=True) + 1e-9)
    body_n    = np.where(np.isfinite(body_n), body_n, 0.0)

    # Fixed FAISS corpus (all normalised body embeddings)
    body_n_corpus = body_n.copy()   # (N, 4096)

    # Baselines on test set (rank 0 only)
    bl_cos = rand_cos = 0.0
    if is_main:
        test_body_n = body_n[test_idx]
        rng = np.random.default_rng(42)
        rand_idx = rng.integers(0, len(body_n), size=len(test_idx))
        rand_cos = float(np.mean((body_n[rand_idx] * test_body_n).sum(axis=1)))
        print(f"\nBaselines on test set ({len(test_idx):,} functions):")
        print(f"  random embedding cosine: {rand_cos:.4f}", flush=True)

    # ── Optional FSDP wrapping of frozen teacher ───────────────────────────────
    if args.fsdp_teacher:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy, MixedPrecision,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 8))
        node_id          = rank // local_world_size
        node_ranks       = list(range(node_id * local_world_size,
                                      (node_id + 1) * local_world_size))
        intra_node_pg    = dist.new_group(ranks=node_ranks)

        # Detect transformer layer class for auto-wrap policy
        layer_cls = type(frozen_qwen.model.layers[0])
        fsdp_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})

        if is_main:
            print(f"\nWrapping teacher with FSDP "
                  f"(intra-node {local_world_size}-way shard, "
                  f"{emb_dim*2//1024} GB → "
                  f"{emb_dim*2//1024//local_world_size} GB per GPU) …", flush=True)
        frozen_qwen = FSDP(
            frozen_qwen,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            auto_wrap_policy=fsdp_policy,
            device_id=device,
            process_group=intra_node_pg,
        )
        if is_main:
            print(f"  FSDP wrapping complete.", flush=True)

    # ── Datasets & loaders ────────────────────────────────────────────────────
    def make_ds(idx):
        return SigBodyDataset(
            [sig_ids_list[i]    for i in idx],
            [sig_masks_list[i]  for i in idx],
            body_n[idx],
            [body_ids_list[i]   for i in idx],
            [body_masks_list[i] for i in idx],
        )

    train_ds = make_ds(train_idx)
    val_ds   = make_ds(val_idx)
    test_ds  = make_ds(test_idx)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # ── Trainable predictor (DDP across all ranks) ────────────────────────────
    predictor = SigPredictorV2(
        d_in=emb_dim, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, d_out=emb_dim, dropout=args.dropout,
        max_body_tokens=MAX_BODY_TOKENS,
    ).to(device)
    n_params = sum(p.numel() for p in predictor.parameters())
    if is_main:
        print(f"\nSigPredictorV2: {n_params:,} trainable parameters", flush=True)

    # Explicit barrier before DDP so we can tell if a rank is slow vs NCCL failing.
    # If this hangs, set NCCL_SOCKET_IFNAME to the correct inter-node interface, e.g.:
    #   export NCCL_SOCKET_IFNAME=eth0   (run `ip route get 10.10.110.20` to find it)
    #   export NCCL_DEBUG=WARN
    print(f"[rank {rank}] reached DDP barrier", flush=True)
    dist.barrier()
    if is_main:
        print("All ranks at DDP barrier — initialising DDP …", flush=True)

    ddp_predictor = DDP(predictor, device_ids=[local_rank])

    criterion = InfoNCELoss(init_temp=args.init_temp).to(device)
    optimizer = optim.AdamW(
        list(ddp_predictor.parameters()) + list(criterion.parameters()),
        lr=args.lr, weight_decay=1e-4)
    scheduler = cosine_schedule_with_warmup(
        optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs)

    # FAISS index for per-epoch val Rank@10 (rank 0 only; None on other ranks)
    faiss_index = None
    if is_main or args.fsdp_teacher:
        faiss_index = faiss.IndexFlatIP(body_n_corpus.shape[1])
        faiss_index.add(body_n_corpus.astype(np.float32))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_r10   = -1.0
    best_state     = None
    patience_count = 0
    t0             = time.time()

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        ddp_predictor.train()

        train_losses = []
        with capture_layer(frozen_qwen, TEACHER_LAYER) as captured:
            for batch in train_loader:
                sig_ids   = batch['input_ids'].to(device, non_blocking=True)
                sig_mask  = batch['attention_mask'].to(device, non_blocking=True)
                body_emb  = batch['body_emb'].to(device, non_blocking=True)
                body_ids  = batch['body_input_ids'].to(device, non_blocking=True)
                body_mask = batch['body_attention_mask'].to(device, non_blocking=True)

                # Sig forward
                captured.clear()
                with torch.no_grad():
                    frozen_qwen(input_ids=sig_ids, attention_mask=sig_mask)
                sig_hs = captured[-1].detach()

                # Body forward (per-token targets for SmoothL1)
                captured.clear()
                with torch.no_grad():
                    frozen_qwen(input_ids=body_ids, attention_mask=body_mask)
                body_hs = captured[-1].detach().float()

                optimizer.zero_grad()
                T_body       = body_ids.shape[1]
                body_pad_mask = (body_mask == 0)
                pooled_pred, token_preds = ddp_predictor(
                    sig_hs, sig_mask, body_len=T_body, body_pad_mask=body_pad_mask)

                # InfoNCE (primary)
                pred_n   = F.normalize(pooled_pred, dim=-1)
                body_n_t = F.normalize(body_emb,    dim=-1)
                gathered = [torch.zeros_like(body_n_t) for _ in range(world_size)]
                dist.all_gather(gathered, body_n_t)
                all_body = torch.cat(gathered, dim=0)
                loss_nce = criterion(pred_n, all_body, rank_offset=rank * len(pred_n))

                # SmoothL1 token-level (auxiliary)
                real_tok = body_mask.float().unsqueeze(-1)
                loss_tok = F.smooth_l1_loss(
                    token_preds * real_tok, body_hs * real_tok, reduction='sum',
                ) / body_mask.float().sum().clamp(min=1.0)

                loss = loss_nce + args.token_loss_weight * loss_tok
                loss.backward()
                nn.utils.clip_grad_norm_(ddp_predictor.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss_nce.item())

        scheduler.step()
        dist.barrier()

        # Val Rank@10
        val_r10_now = val_rank10_distributed(
            ddp_predictor, frozen_qwen, val_loader, val_idx_arr, faiss_index,
            device, rank, world_size, args.fsdp_teacher,
        )

        if is_main:
            # Quick val cosine (rank 0 only when not fsdp_teacher)
            val_result = evaluate(
                ddp_predictor, frozen_qwen, val_loader,
                body_n_corpus, [], device, criterion, desc='val',
            ) if not args.fsdp_teacher else {'cosine': 0.0}

            train_loss = np.mean(train_losses)
            elapsed    = time.time() - t0
            lr         = scheduler.get_last_lr()[0]
            τ          = criterion.log_temp.exp().item()
            print(
                f"  epoch {epoch:4d}/{args.epochs}  "
                f"train={train_loss:.4f}  val_cos={val_result['cosine']:.4f}  "
                f"val_r10={val_r10_now*100:.2f}%  τ={τ:.4f}  "
                f"lr={lr:.2e}  elapsed={elapsed:.0f}s",
                flush=True,
            )

        # Early stopping (rank 0 decides, broadcast)
        val_r10_tensor = torch.tensor(val_r10_now, device=device)
        dist.broadcast(val_r10_tensor, src=0)
        val_r10_sync = val_r10_tensor.item()

        improved = torch.tensor(int(val_r10_sync > best_val_r10), device=device)
        dist.broadcast(improved, src=0)
        if improved.item():
            best_val_r10 = val_r10_sync
            best_state   = {k: v.cpu().clone()
                            for k, v in ddp_predictor.module.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if is_main and epoch % args.checkpoint_every == 0:
            periodic_path = os.path.join(
                os.path.dirname(__file__), f'student_3_0_epoch{epoch:04d}.pt')
            torch.save({'model_state': {k: v.cpu().clone()
                                        for k, v in ddp_predictor.module.state_dict().items()},
                        'epoch': epoch, 'args': vars(args),
                        'emb_dim': emb_dim, 'd_model': args.d_model}, periodic_path)
            print(f"  Periodic checkpoint → {periodic_path}", flush=True)

        if patience_count >= args.patience:
            if is_main:
                print(f"  Early stopping at epoch {epoch} "
                      f"(val Rank@10 no improvement for {args.patience} epochs)",
                      flush=True)
            break

    # ── Final evaluation (rank 0 — teacher forward OK since not fsdp_teacher,
    #    or all-ranks if fsdp_teacher with evaluate() coordinated below) ───────
    if is_main:
        predictor.load_state_dict(best_state)
        N = len(body_n)
        random_r1 = 1.0 / N

        val_result_final = evaluate(
            predictor, frozen_qwen, val_loader,
            body_n_corpus, val_idx, device, criterion, desc='val_final',
        )
        val_r1  = val_result_final['rank1']
        val_r5  = val_result_final['rank5']
        val_r10 = val_result_final['rank10']
        val_cos = val_result_final['cosine']
        print(f"\nVal results ({len(val_idx)} functions, corpus={N:,}):")
        print(f"  Cosine similarity:  {val_cos:.4f}")
        print(f"  Rank@1:   {val_r1*100:.2f}%  (random: {random_r1*100:.4f}%)")
        print(f"  Rank@5:   {val_r5*100:.2f}%")
        print(f"  Rank@10:  {val_r10*100:.2f}%")

        test_result = evaluate(
            predictor, frozen_qwen, test_loader,
            body_n_corpus, test_idx, device, criterion, desc='test',
        )
        r1  = test_result['rank1']
        r5  = test_result['rank5']
        r10 = test_result['rank10']
        cos = test_result['cosine']
        print(f"\nTest results ({len(test_idx)} functions, corpus={N:,}):")
        print(f"  Cosine similarity:  {cos:.4f}")
        print(f"  Rank@1:   {r1*100:.2f}%  (random: {random_r1*100:.4f}%)")
        print(f"  Rank@5:   {r5*100:.2f}%")
        print(f"  Rank@10:  {r10*100:.2f}%")

        ckpt_path = os.path.join(os.path.dirname(__file__), 'student_3_0_ckpt.pt')
        torch.save({'model_state': best_state, 'args': vars(args),
                    'emb_dim': emb_dim, 'd_model': args.d_model}, ckpt_path)
        print(f"\nCheckpoint saved to {ckpt_path}", flush=True)

        write_results({
            'n_train': len(train_idx), 'n_val': len(val_idx),
            'n_test': len(test_idx), 'corpus': N,
            'n_params': n_params, 'world_size': world_size,
            'cos': cos, 'bl_cos': 0.0, 'rand_cos': rand_cos,
            'r1': r1, 'r5': r5, 'r10': r10,
            'val_cos': val_cos, 'val_r1': val_r1, 'val_r5': val_r5, 'val_r10': val_r10,
            'best_val_r10': best_val_r10,
            'final_temp': criterion.log_temp.exp().item(),
            'random_r1': random_r1,
            'args': vars(args),
        })


# ── Results writer ────────────────────────────────────────────────────────────

def write_results(res: dict):
    a = res['args']
    eff_batch = a['batch_size'] * res['world_size']
    report = f"""# Experiment 3.0: Qwen3-8B-base Teacher + Token-Level Body Prediction

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Teacher**: Qwen3-8B-base, layer {TEACHER_LAYER} (frozen)
**Task**: Predict function body embedding from function signature token hidden states

## Architecture

```
Frozen:   Qwen3-8B-base ({TEACHER_LAYER+1}+ transformer layers, bf16, no grad)
          ├─ hook at layer {TEACHER_LAYER} on sig tokens  → sig_hs  [B, T_sig, 4096]
          └─ hook at layer {TEACHER_LAYER} on body tokens → body_hs [B, T_body, 4096] (train only)

Trainable SigPredictorV2 ({res['n_params']:,} params):
  Encoder:
    Linear(4096 → {a['d_model']})
    {a['num_layers']}× TransformerBlock(d={a['d_model']}, nhead={a['nhead']}, ffn={a['d_model']*4})
    mean-pool → Linear({a['d_model']} → 4096)  ← retrieval head (InfoNCE)
  Decoder (training only):
    body_pos_enc: Embedding({MAX_BODY_TOKENS}, {a['d_model']})
    {a['num_layers']}× CrossAttentionBlock(d={a['d_model']}, nhead={a['nhead']})
    Linear({a['d_model']} → 4096)  ← per-token prediction head (SmoothL1)
```

## Dataset

| Split | Functions |
|-------|-----------|
| Train | {res['n_train']:,} |
| Val   | {res['n_val']:,} |
| Test  | {res['n_test']:,} |

Corpus (retrieval): **{res['corpus']:,}** functions.

## Training

| Hyperparameter | Value |
|----------------|-------|
| Epochs | {a['epochs']} (early stopping patience={a['patience']}) |
| Batch/GPU | {a['batch_size']} × {res['world_size']} GPUs = {eff_batch} effective |
| Learning rate | {a['lr']} (cosine, warmup={a['warmup_epochs']} epochs) |
| d_model | {a['d_model']} |
| Transformer layers | {a['num_layers']} enc + {a['num_layers']} dec |
| Attention heads | {a['nhead']} |
| Dropout | {a['dropout']} |
| Token loss weight λ | {a['token_loss_weight']} |
| Max body tokens | {MAX_BODY_TOKENS} |
| FSDP teacher | {a['fsdp_teacher']} |
| Precision | BF16 (frozen Qwen), FP32 (predictor) |

## Results

### Val Set (in-distribution — primary metric)

| Metric | Value |
|--------|-------|
| Cosine sim | **{res['val_cos']:.4f}** |
| Rank@1 | **{res['val_r1']*100:.2f}%** |
| Rank@5 | **{res['val_r5']*100:.2f}%** |
| Rank@10 | **{res['val_r10']*100:.2f}%** |

### Test Set

| Metric | Value | Baseline |
|--------|-------|----------|
| Cosine sim | **{res['cos']:.4f}** | random: {res['rand_cos']:.4f} |
| Rank@1 | **{res['r1']*100:.2f}%** | {res['random_r1']*100:.4f}% (random) |
| Rank@5 | **{res['r5']*100:.2f}%** | — |
| Rank@10 | **{res['r10']*100:.2f}%** | — |

## Training Details

| Setting | Value |
|---------|-------|
| Loss | InfoNCE (primary) + λ·SmoothL1 token-level (auxiliary) |
| Final temperature τ | {res['final_temp']:.4f} |
| Negatives per step | {res['world_size']} GPUs × {res['args']['batch_size']} = {res['world_size']*res['args']['batch_size']} |
| Best val Rank@10 | {res['best_val_r10']*100:.2f}% |

## Success Criteria

| Criterion | Target | Achieved |
|-----------|--------|---------|
| Val Rank@1 > 5% | > 5% | {"✅" if res['val_r1'] > 0.05 else "❌"} {res['val_r1']*100:.2f}% |
| Val Rank@10 > 20% | > 20% | {"✅" if res['val_r10'] > 0.20 else "❌"} {res['val_r10']*100:.2f}% |
| Test Rank@1 > 5% | > 5% | {"✅" if res['r1'] > 0.05 else "❌"} {res['r1']*100:.2f}% |
| Test Rank@10 > 20% | > 20% | {"✅" if res['r10'] > 0.20 else "❌"} {res['r10']*100:.2f}% |
"""
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        f.write(report)
    print(f"\nResults written to {RESULTS_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs',             type=int,   default=200)
    ap.add_argument('--batch-size',         type=int,   default=64)
    ap.add_argument('--lr',                 type=float, default=1e-4)
    ap.add_argument('--warmup-epochs',      type=int,   default=5)
    ap.add_argument('--patience',           type=int,   default=15)
    ap.add_argument('--checkpoint-every',   type=int,   default=20)
    ap.add_argument('--d-model',            type=int,   default=1024)
    ap.add_argument('--nhead',              type=int,   default=8)
    ap.add_argument('--num-layers',         type=int,   default=2)
    ap.add_argument('--dropout',            type=float, default=0.1)
    ap.add_argument('--init-temp',          type=float, default=0.07)
    ap.add_argument('--token-loss-weight',  type=float, default=0.0)
    ap.add_argument('--fsdp-teacher',       action='store_true', default=False,
                    help='Wrap frozen teacher with intra-node FSDP FULL_SHARD '
                         '(reduces teacher VRAM from ~16 GB to ~2 GB per GPU). '
                         'Use if the node has < 18 GB free VRAM per card.')
    args = ap.parse_args()

    # Normalise hyphen → underscore (argparse dest uses _)
    args.warmup_epochs      = args.warmup_epochs
    args.num_layers         = args.num_layers
    args.init_temp          = args.init_temp
    args.batch_size         = args.batch_size
    args.d_model            = args.d_model
    args.token_loss_weight  = args.token_loss_weight
    args.fsdp_teacher       = args.fsdp_teacher

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
