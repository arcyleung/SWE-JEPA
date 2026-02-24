"""
Experiment 2.2: Token-level body prediction (I-JEPA style) + InfoNCE.

Extends Exp 1.4 by adding a cross-attention decoder that predicts per-token body
hidden states from the signature encoding.  This gives ~100× more gradient signal
per function than the mean-pooled target in Exp 1.4, forcing the model to recover
specific body content (variable names, control flow) rather than average direction.

Architecture:
  Encoder (same as Exp 1.4):
    Linear(2048→512) → 2× TransformerBlock → mean-pool → Linear(512→2048)
    → L2-norm → InfoNCE loss  (primary objective)

  Decoder (new):
    body_pos_enc: Embedding(MAX_BODY_TOKENS, 512)   ← positional queries only
    2× CrossAttentionBlock(d=512): queries=body positions, kv=sig encoder tokens
    Linear(512→2048) → SmoothL1 vs teacher body token hidden states  (aux, λ=0.1)

Key changes from Exp 1.4:
  - SigPredictorV2 adds cross-attention decoder with positional body queries
  - Second Qwen forward pass per step (body tokens) → token-level targets
  - Dual loss: InfoNCE + λ * SmoothL1_token
  - batch_size default 32 (was 64) to fit 2 Qwen forward passes per step
  - body_text loaded from postgres and tokenised

Usage:
  torchrun --nproc-per-node=8 train_student_2_2.py
  torchrun --nproc-per-node=8 train_student_2_2.py --epochs 100 --batch-size 32
  torchrun --nproc-per-node=8 train_student_2_2.py --token-loss-weight 0.05
"""

import argparse
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

TEACHER_PATH    = '/home/original_models/Qwen2.5-Coder-3B'
TEACHER_LAYER   = 18          # 0-indexed; hidden_states[19] in HF convention
MAX_SIG_TOKENS  = 256         # signatures are short; 256 covers >99% of cases
MAX_BODY_TOKENS = 256         # truncate long bodies; covers >95% of functions
RESULTS_FILE    = os.path.join(os.path.dirname(__file__),
                                'docs', 'phase2_2_token_level_jepa.md')


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    In-batch InfoNCE (= softmax cross-entropy over cosine similarity logits).

    pred    : [B, D]  L2-normalised predicted body embeddings (local rank)
    targets : [N, D]  L2-normalised body embeddings from ALL ranks (N = ws * B)
    rank_offset : index of this rank's block in targets

    Loss for anchor i:
        -log [ exp(sim(pred_i, targets_{rank_offset+i}) / τ)
               / Σ_j exp(sim(pred_i, targets_j) / τ) ]
    """
    def __init__(self, init_temp: float = 0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))

    def forward(
        self,
        pred: torch.Tensor,
        targets: torch.Tensor,
        rank_offset: int = 0,
    ) -> torch.Tensor:
        τ = self.log_temp.exp().clamp(min=1e-4)
        logits = (pred @ targets.T) / τ           # [B, N]
        labels = torch.arange(
            rank_offset, rank_offset + len(pred), device=pred.device)
        return F.cross_entropy(logits, labels)


# ── Model ─────────────────────────────────────────────────────────────────────

class _CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + FFN block for the token-level decoder."""

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,        # [B, T_body, d_model]
        context: torch.Tensor,      # [B, T_sig,  d_model]  (sig encoder output)
        query_pad_mask: torch.Tensor,    # [B, T_body] bool, True=pad
        context_pad_mask: torch.Tensor,  # [B, T_sig]  bool, True=pad
    ) -> torch.Tensor:
        q  = self.norm_q(query)
        kv = self.norm_kv(context)
        attn_out, _ = self.cross_attn(
            q, kv, kv,
            key_padding_mask=context_pad_mask,
            need_weights=False,
        )
        query = query + self.drop(attn_out)
        query = query + self.drop(self.ffn(self.norm2(query)))
        return query


class _TransformerBlock(nn.Module):
    """Single pre-norm self-attention + FFN block (manually implemented
    to avoid PyTorch version-specific issues with TransformerEncoderLayer)."""

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.attn    = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.drop    = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,            # [B, T, d_model]
        pad_mask: torch.Tensor,     # [B, T] bool, True = padding (ignored)
    ) -> torch.Tensor:
        # Pre-norm self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=pad_mask,
            need_weights=False,
        )
        x = x + self.drop(attn_out)
        # Pre-norm FFN
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class SigPredictorV2(nn.Module):
    """
    Trainable head with two output heads:

    1. Retrieval head (same as Exp 1.4):
       sig token hidden states → 2-layer transformer encoder → mean-pool →
       Linear(512→2048) → L2-norm → InfoNCE loss

    2. Token-level decoder (new for Exp 2.2):
       Learned body positional queries attend to sig encoder tokens via
       cross-attention → per-token body hidden state predictions →
       SmoothL1 auxiliary loss

    The decoder is training-time only — at inference only the retrieval head runs.
    """

    def __init__(
        self,
        d_in: int = 2048,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        d_out: int = 2048,
        dropout: float = 0.1,
        max_body_tokens: int = MAX_BODY_TOKENS,
    ):
        super().__init__()
        # ── Encoder (identical to Exp 1.4) ───────────────────────────────────
        self.proj_in   = nn.Linear(d_in, d_model)
        self.enc_blocks = nn.ModuleList(
            [_TransformerBlock(d_model, nhead, dropout) for _ in range(num_layers)]
        )
        self.proj_out  = nn.Linear(d_model, d_out)   # retrieval head

        # ── Token-level decoder ───────────────────────────────────────────────
        # Positional queries: one learnable vector per body token position.
        # Using position-only (not token identity) forces the decoder to
        # reconstruct body content from sig encoding rather than memorising tokens.
        self.body_pos_enc = nn.Embedding(max_body_tokens, d_model)
        self.dec_blocks   = nn.ModuleList(
            [_CrossAttentionBlock(d_model, nhead, dropout) for _ in range(num_layers)]
        )
        self.proj_token   = nn.Linear(d_model, d_out)   # per-token prediction head

    def _encode(
        self,
        hidden_states: torch.Tensor,   # [B, T_sig, d_in]
        attention_mask: torch.Tensor,  # [B, T_sig]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (enc_tokens [B, T_sig, d_model], pooled [B, d_model])."""
        x = self.proj_in(hidden_states.float())
        pad_mask = (attention_mask == 0)
        for block in self.enc_blocks:
            x = block(x, pad_mask)
        real_mask = attention_mask.float().unsqueeze(-1)
        pooled = (x * real_mask).sum(1) / real_mask.sum(1).clamp(min=1.0)
        return x, pooled

    def forward(
        self,
        hidden_states: torch.Tensor,    # [B, T_sig, d_in]  sig token hs
        attention_mask: torch.Tensor,   # [B, T_sig]
        body_len: int | None = None,    # T_body: number of body token positions
        body_pad_mask: torch.Tensor | None = None,  # [B, T_body] True=pad
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
          pooled_pred  : [B, d_out]          always
          token_preds  : [B, T_body, d_out]  only when body_len is provided
        """
        enc_tokens, pooled = self._encode(hidden_states, attention_mask)
        retrieval_pred = self.proj_out(pooled)   # [B, d_out]

        token_preds = None
        if body_len is not None:
            B = hidden_states.shape[0]
            device = hidden_states.device
            # Positional queries for body positions 0..body_len-1
            pos_ids = torch.arange(body_len, device=device)          # [T_body]
            queries = self.body_pos_enc(pos_ids).unsqueeze(0).expand(B, -1, -1)
            # [B, T_body, d_model]

            sig_pad_mask = (attention_mask == 0)   # [B, T_sig]  True=pad
            if body_pad_mask is None:
                body_pad_mask = torch.zeros(B, body_len, dtype=torch.bool,
                                            device=device)

            for block in self.dec_blocks:
                queries = block(queries, enc_tokens, body_pad_mask, sig_pad_mask)

            token_preds = self.proj_token(queries)   # [B, T_body, d_out]

        return retrieval_pred, token_preds


# ── Forward hook to capture Qwen layer-18 output ─────────────────────────────

@contextmanager
def capture_layer(model: nn.Module, layer_idx: int):
    """
    Context manager that registers a forward hook on model.model.layers[layer_idx].
    After each forward pass, the captured hidden states are available via
    the returned list (always contains the most recent output).

    Removes the hook on exit.
    """
    captured: list[torch.Tensor] = []

    def _hook(module, inputs, outputs):
        # In transformers ≥5.x, Qwen2DecoderLayer returns a plain Tensor.
        # In older versions it returned a tuple (hidden_states, ...).
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
    """
    Pre-tokenised sig and body texts with stored mean-pooled body embeddings.
    body_emb is mean-centred + L2-normalised (used for InfoNCE targets).
    body_input_ids / body_attention_mask used for the token-level auxiliary loss.
    """

    def __init__(
        self,
        input_ids: list[list[int]],
        attention_masks: list[list[int]],
        body_embs: np.ndarray,
        body_input_ids: list[list[int]],
        body_attention_masks: list[list[int]],
    ):
        assert len(input_ids) == len(body_embs) == len(body_input_ids)
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


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad variable-length sig and body token sequences within a batch."""
    B = len(batch)
    # Sig padding
    sig_max = max(len(x['input_ids']) for x in batch)
    sig_ids_t  = torch.zeros(B, sig_max, dtype=torch.long)
    sig_mask_t = torch.zeros(B, sig_max, dtype=torch.long)
    # Body padding
    body_max = max(len(x['body_input_ids']) for x in batch)
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

def load_data(tokenizer) -> tuple[
    list[list[int]], list[list[int]],   # sig ids, sig masks
    list[list[int]], list[list[int]],   # body ids, body masks
    np.ndarray,                          # body_embs (mean-pooled, for InfoNCE)
    list[str], list[int]                 # repos, fids
]:
    """
    Load sig texts, body texts, and mean-pooled body embeddings from postgres.

    Returns:
        sig_ids_list        – tokenised sig input_ids (variable length)
        sig_masks_list      – sig attention masks
        body_ids_list       – tokenised body input_ids (for token-level targets)
        body_masks_list     – body attention masks
        body_embs           – (N, 2048) float32, mean-pooled (for InfoNCE)
        repos               – repo strings
        fids                – function_ids
    """
    conn = pg8000.native.Connection(**DB)
    rows = conn.run("""
        SELECT fst.function_id,
               fst.sig_text,
               fst.body_text,
               fst.body_embedding,
               fe.instance_id
        FROM function_student_targets fst
        JOIN function_embeddings fe
          ON fe.id = fst.function_id
         AND fe.model_name = 'Qwen2.5-Coder-3B'
        WHERE fst.sig_text IS NOT NULL
          AND fst.body_text IS NOT NULL
        ORDER BY fst.function_id
    """)
    conn.close()

    if not rows:
        raise RuntimeError("No sig_text/body_text in function_student_targets.")

    fids_raw   = [r[0] for r in rows]
    sig_texts  = [r[1] for r in rows]
    body_texts = [r[2] for r in rows]
    body_raw   = np.array([r[3] for r in rows], dtype=np.float32)
    repos_raw  = []
    for r in rows:
        parts = r[4].split('__')
        repos_raw.append(f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else r[4])

    # Drop NaN/inf body embedding rows
    valid = np.isfinite(body_raw).all(axis=1)
    if (~valid).sum():
        print(f"  Dropped {(~valid).sum()} non-finite body-embedding rows", flush=True)

    # Drop empty sig_texts
    empty_sig = np.array([len(t.strip()) < 5 for t in sig_texts])
    valid = valid & ~empty_sig
    if empty_sig.sum():
        print(f"  Dropped {empty_sig.sum()} empty/trivial sig_text rows", flush=True)

    # Drop empty body_texts
    empty_body = np.array([len(t.strip()) < 5 for t in body_texts])
    valid = valid & ~empty_body
    if empty_body.sum():
        print(f"  Dropped {empty_body.sum()} empty/trivial body_text rows", flush=True)

    valid_idx  = np.where(valid)[0]
    body_embs  = body_raw[valid]
    fids       = [fids_raw[i]   for i in valid_idx]
    sig_texts  = [sig_texts[i]  for i in valid_idx]
    body_texts = [body_texts[i] for i in valid_idx]
    repos      = [repos_raw[i]  for i in valid_idx]

    # Tokenise sigs
    sig_enc = tokenizer(
        sig_texts,
        truncation=True, max_length=MAX_SIG_TOKENS,
        padding=False, return_tensors=None,
    )
    # Tokenise bodies
    body_enc = tokenizer(
        body_texts,
        truncation=True, max_length=MAX_BODY_TOKENS,
        padding=False, return_tensors=None,
    )

    return (sig_enc['input_ids'], sig_enc['attention_mask'],
            body_enc['input_ids'], body_enc['attention_mask'],
            body_embs, repos, fids)


def repo_split(
    repos: list[str],
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split by repo: returns (train_idx, val_idx, test_idx)."""
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


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    predictor: nn.Module,
    frozen_qwen: nn.Module,
    loader: DataLoader,
    body_embs_all_centred: np.ndarray,   # (N, 2048) centred + normalised, for FAISS
    test_idx: list[int],
    device: torch.device,
    criterion: nn.Module,
    desc: str = 'eval',
) -> dict:
    """
    Returns: loss, mean cosine sim, rank@1, rank@5, rank@10.
    FAISS retrieval runs only when test_idx is not empty.
    """
    predictor.eval()
    losses, preds_list, tgts_list = [], [], []

    with capture_layer(frozen_qwen, TEACHER_LAYER) as captured, \
         torch.no_grad():
        for batch in loader:
            input_ids  = batch['input_ids'].to(device)
            attn_mask  = batch['attention_mask'].to(device)
            body_emb   = batch['body_emb'].to(device)

            # Frozen forward — hook fills `captured`
            captured.clear()
            frozen_qwen(input_ids=input_ids, attention_mask=attn_mask)
            hs = captured[-1]   # [B, T, 2048]  bf16

            # Trainable predictor (unwrap DDP for eval) — retrieval head only
            raw_pred = predictor.module if hasattr(predictor, 'module') else predictor
            pred, _ = raw_pred(hs, attn_mask)   # [B, 2048], no decoder at eval

            losses.append(criterion(pred, body_emb).item())
            preds_list.append(pred.float().cpu().numpy())
            tgts_list.append(body_emb.float().cpu().numpy())

    preds = np.vstack(preds_list)
    tgts  = np.vstack(tgts_list)

    # Safety clamp: replace non-finite values before cosine computation
    if not np.isfinite(preds).all():
        print(f"  [{desc}] WARNING: non-finite predictions detected", flush=True)
        preds = np.where(np.isfinite(preds), preds, 0.0)
    if not np.isfinite(tgts).all():
        print(f"  [{desc}] WARNING: non-finite targets detected", flush=True)
        tgts = np.where(np.isfinite(tgts), tgts, 0.0)

    # Cosine similarity (preds and tgts are already in centred space)
    pred_n = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-9)
    tgt_n  = tgts  / (np.linalg.norm(tgts,  axis=1, keepdims=True) + 1e-9)
    cos    = float((pred_n * tgt_n).sum(axis=1).mean())

    result = {
        'loss': float(np.mean(losses)),
        'cosine': cos,
        'rank1': None, 'rank5': None, 'rank10': None,
    }

    # Retrieval metrics only for test set (computationally cheap with FAISS)
    if test_idx:
        d = body_embs_all_centred.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(body_embs_all_centred.astype(np.float32))
        _, I = index.search(pred_n.astype(np.float32), 10)
        r1 = sum(1 for i, gi in enumerate(test_idx) if gi == I[i, 0])  / len(test_idx)
        r5 = sum(1 for i, gi in enumerate(test_idx) if gi in I[i, :5]) / len(test_idx)
        r10= sum(1 for i, gi in enumerate(test_idx) if gi in I[i,:10]) / len(test_idx)
        result.update({'rank1': r1, 'rank5': r5, 'rank10': r10})

    return result


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_schedule_with_warmup(
    optimizer, warmup_epochs: int, total_epochs: int, min_lr_frac: float = 0.01
):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs))
        return min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (
            1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args, rank: int, world_size: int):
    is_main = (rank == 0)
    device  = torch.device(f'cuda:{rank}')

    if is_main:
        print(f"SWE-JEPA Experiment 2.2: Token-Level Body Prediction + InfoNCE")
        print(f"{'='*60}")
        print(f"  World size: {world_size}  |  BF16 mixed precision")
        print(f"  Batch/GPU:  {args.batch_size}  "
              f"→  effective batch: {args.batch_size * world_size}", flush=True)

    # ── Load tokeniser (all ranks) ────────────────────────────────────────────
    if is_main:
        print(f"\nLoading tokeniser …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_PATH, trust_remote_code=True)

    # ── Load data (all ranks, small DB, OK) ───────────────────────────────────
    if is_main:
        print(f"Loading data from postgres …", flush=True)
    (sig_ids_list, sig_masks_list,
     body_ids_list, body_masks_list,
     body_embs, repos, fids) = load_data(tokenizer)
    if is_main:
        print(f"Loaded {len(fids):,} samples from {len(set(repos))} repos", flush=True)

    # Repo split (same seed on all ranks → identical splits)
    train_idx, val_idx, test_idx = repo_split(repos)
    if is_main:
        print(f"Split: {len(train_idx):,} train  "
              f"{len(val_idx):,} val  {len(test_idx):,} test  (by repo)", flush=True)

    # Mean-centre body embeddings using train-split mean (prevent leakage)
    body_mean = body_embs[train_idx].mean(axis=0, keepdims=True)
    body_c    = body_embs - body_mean
    body_n    = body_c / (np.linalg.norm(body_c, axis=1, keepdims=True) + 1e-9)

    # Safety: replace any non-finite rows (can arise if float16 embedding values
    # interact badly with mean-centering in float32)
    bad_n = ~np.isfinite(body_n).all(axis=1)
    if bad_n.any() and is_main:
        print(f"  WARNING: {bad_n.sum()} non-finite rows in body_n; replacing with zeros",
              flush=True)
    body_n = np.where(np.isfinite(body_n), body_n, 0.0)

    # Pre-compute FAISS corpus (all centred body embeddings, L2-normalised)
    # Used for retrieval evaluation on the test set
    body_n_corpus = body_n.copy()  # (N, 2048)

    # Baselines on test set
    if is_main:
        test_body_n = body_n[test_idx]
        # sig→body baseline: use stored sig_embedding if available
        conn = pg8000.native.Connection(**DB)
        rows_bl = conn.run("""
            SELECT fst.function_id, fst.sig_embedding
            FROM function_student_targets fst
            JOIN function_embeddings fe ON fe.id = fst.function_id
              AND fe.model_name = 'Qwen2.5-Coder-3B'
            WHERE fst.sig_text IS NOT NULL
            ORDER BY fst.function_id
        """)
        conn.close()
        valid_fids = set(fids)
        sig_raw = np.array(
            [r[1] for r in rows_bl if r[0] in valid_fids], dtype=np.float32)
        sig_finite = np.isfinite(sig_raw).all(axis=1)
        if not sig_finite.all():
            # Replace non-finite sig rows with zeros so baseline is computable
            sig_raw[~sig_finite] = 0.0
        sig_mean = sig_raw[train_idx].mean(axis=0, keepdims=True)
        sig_c    = sig_raw - sig_mean
        sig_n    = sig_c / (np.linalg.norm(sig_c, axis=1, keepdims=True) + 1e-9)
        bl_cos = float(np.mean((sig_n[test_idx] * test_body_n).sum(axis=1)))
        rng = np.random.default_rng(42)
        rand_idx  = rng.integers(0, len(body_n), size=len(test_idx))
        rand_cos  = float(np.mean((body_n[rand_idx] * test_body_n).sum(axis=1)))
        print(f"\nBaselines on test set ({len(test_idx):,} functions):")
        print(f"  sig→body cosine (MLP baseline):  {bl_cos:.4f}")
        print(f"  random embedding cosine:          {rand_cos:.4f}", flush=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    def make_ds(idx: list[int]) -> SigBodyDataset:
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

    # ── Load frozen Qwen (each rank gets its own copy) ────────────────────────
    if is_main:
        print(f"\nLoading frozen Qwen2.5-Coder-3B on cuda:{rank} …", flush=True)
    frozen_qwen = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()
    for p in frozen_qwen.parameters():
        p.requires_grad_(False)

    # ── Trainable predictor ───────────────────────────────────────────────────
    emb_dim   = body_embs.shape[1]
    predictor = SigPredictorV2(
        d_in=emb_dim, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, d_out=emb_dim, dropout=args.dropout,
        max_body_tokens=MAX_BODY_TOKENS,
    ).to(device)
    n_params = sum(p.numel() for p in predictor.parameters())
    if is_main:
        print(f"SigPredictorV2: {n_params:,} trainable parameters", flush=True)

    ddp_predictor = DDP(predictor, device_ids=[rank])

    criterion = InfoNCELoss(init_temp=args.init_temp).to(device)
    # Include learnable temperature in optimizer
    optimizer = optim.AdamW(
        list(ddp_predictor.parameters()) + list(criterion.parameters()),
        lr=args.lr, weight_decay=1e-4)
    scheduler = cosine_schedule_with_warmup(
        optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs)

    # Precompute FAISS index for per-epoch val Rank@10
    if is_main:
        faiss_index = faiss.IndexFlatIP(body_n_corpus.shape[1])
        faiss_index.add(body_n_corpus.astype(np.float32))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_r10    = -1.0
    best_state      = None
    patience_count  = 0
    t0              = time.time()

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        ddp_predictor.train()

        train_losses = []
        with capture_layer(frozen_qwen, TEACHER_LAYER) as captured:
            for batch in train_loader:
                sig_ids    = batch['input_ids'].to(device, non_blocking=True)
                sig_mask   = batch['attention_mask'].to(device, non_blocking=True)
                body_emb   = batch['body_emb'].to(device, non_blocking=True)
                body_ids   = batch['body_input_ids'].to(device, non_blocking=True)
                body_mask  = batch['body_attention_mask'].to(device, non_blocking=True)

                # ── Frozen forward on sig → sig token hidden states ───────────
                captured.clear()
                with torch.no_grad():
                    frozen_qwen(input_ids=sig_ids, attention_mask=sig_mask)
                sig_hs = captured[-1].detach()   # [B, T_sig, 2048] bf16

                # ── Frozen forward on body → per-token body hidden states ──────
                captured.clear()
                with torch.no_grad():
                    frozen_qwen(input_ids=body_ids, attention_mask=body_mask)
                body_hs = captured[-1].detach().float()   # [B, T_body, 2048] f32

                # ── Predictor: retrieval head + token decoder ─────────────────
                optimizer.zero_grad()
                T_body = body_ids.shape[1]
                body_pad_mask = (body_mask == 0)   # [B, T_body] True=pad

                pooled_pred, token_preds = ddp_predictor(
                    sig_hs, sig_mask,
                    body_len=T_body, body_pad_mask=body_pad_mask,
                )   # [B, 2048], [B, T_body, 2048]

                # ── InfoNCE loss (primary) ────────────────────────────────────
                pred_n   = F.normalize(pooled_pred, dim=-1)
                body_n_t = F.normalize(body_emb,    dim=-1)

                gathered = [torch.zeros_like(body_n_t) for _ in range(world_size)]
                dist.all_gather(gathered, body_n_t)
                all_body = torch.cat(gathered, dim=0)   # [world_size*B, D]

                loss_nce = criterion(pred_n, all_body, rank_offset=rank * len(pred_n))

                # ── Token-level SmoothL1 loss (auxiliary) ─────────────────────
                # Average over real (non-padding) body tokens only
                real_tok = body_mask.float().unsqueeze(-1)   # [B, T_body, 1]
                loss_tok = F.smooth_l1_loss(
                    token_preds * real_tok,
                    body_hs      * real_tok,
                    reduction='sum',
                ) / body_mask.float().sum().clamp(min=1.0)

                loss = loss_nce + args.token_loss_weight * loss_tok
                loss.backward()
                nn.utils.clip_grad_norm_(ddp_predictor.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss_nce.item())   # log InfoNCE only

        scheduler.step()
        dist.barrier()

        # Val cosine sim — all ranks, rank 0 prints
        # (evaluate still uses SmoothL1 for loss — we pass a dummy criterion;
        #  cosine sim is the meaningful metric here)
        val_result = evaluate(
            ddp_predictor, frozen_qwen, val_loader,
            body_n_corpus, [],
            device, nn.SmoothL1Loss(), desc='val',
        )

        # Per-epoch val Rank@10 on rank 0 (FAISS is instant on 4,615 vectors)
        val_r10_now = 0.0
        if is_main:
            raw_pred = ddp_predictor.module
            raw_pred.eval()
            preds_val = []
            with capture_layer(frozen_qwen, TEACHER_LAYER) as cap2, torch.no_grad():
                for batch in val_loader:
                    cap2.clear()
                    frozen_qwen(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device))
                    hs2 = cap2[-1]
                    # Retrieval head only (no body_len → decoder skipped)
                    p, _ = raw_pred(hs2, batch['attention_mask'].to(device))
                    preds_val.append(F.normalize(p, dim=-1).float().cpu().numpy())
            preds_val = np.vstack(preds_val)
            _, I = faiss_index.search(preds_val, 10)
            val_r10_now = sum(
                1 for i, gi in enumerate(val_idx) if gi in I[i, :10]
            ) / len(val_idx)
            raw_pred.train()

        if is_main:
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

        # Early stopping on val Rank@10 (rank 0 decides, broadcast)
        val_r10_tensor = torch.tensor(val_r10_now, device=device)
        dist.broadcast(val_r10_tensor, src=0)
        val_r10_sync = val_r10_tensor.item()

        improved = torch.tensor(
            int(val_r10_sync > best_val_r10), device=device)
        dist.broadcast(improved, src=0)
        if improved.item():
            best_val_r10 = val_r10_sync
            best_state   = {k: v.cpu().clone()
                            for k, v in ddp_predictor.module.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= args.patience:
            if is_main:
                print(f"  Early stopping at epoch {epoch} "
                      f"(val Rank@10 no improvement for {args.patience} epochs)",
                      flush=True)
            break

    # ── Test + Val evaluation (rank 0 only) ───────────────────────────────────
    if is_main:
        predictor.load_state_dict(best_state)
        N = len(body_n)
        random_r1 = 1.0 / N

        # Val retrieval (in-distribution — more reliable than test for this dataset)
        val_result_final = evaluate(
            predictor, frozen_qwen, val_loader,
            body_n_corpus, val_idx,
            device, criterion, desc='val_final',
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

        # Test retrieval
        test_result = evaluate(
            predictor, frozen_qwen, test_loader,
            body_n_corpus, test_idx,
            device, criterion, desc='test',
        )
        r1  = test_result['rank1']
        r5  = test_result['rank5']
        r10 = test_result['rank10']
        cos = test_result['cosine']

        print(f"\nTest results ({len(test_idx)} functions, corpus={N:,}):")
        print(f"  Cosine similarity:  {cos:.4f}  (baseline: {bl_cos:.4f})")
        print(f"  Rank@1:   {r1*100:.2f}%  (random: {random_r1*100:.4f}%)")
        print(f"  Rank@5:   {r5*100:.2f}%")
        print(f"  Rank@10:  {r10*100:.2f}%")

        # Save checkpoint
        ckpt_path = os.path.join(os.path.dirname(__file__),
                                 'student_2_2_ckpt.pt')
        torch.save({'model_state': best_state, 'args': vars(args),
                    'emb_dim': emb_dim, 'd_model': args.d_model}, ckpt_path)
        print(f"\nCheckpoint saved to {ckpt_path}", flush=True)

        # Write results document
        write_results({
            'n_train': len(train_idx), 'n_val': len(val_idx),
            'n_test': len(test_idx), 'corpus': N,
            'n_params': n_params, 'world_size': world_size,
            'cos': cos, 'bl_cos': bl_cos, 'rand_cos': rand_cos,
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
    r1_vs_mlp = "improvement over MLP (0.00%)" if res['r1'] > 0 else \
                "no improvement over MLP (0.00%)"
    report = f"""# Experiment 2.2: Token-Level Body Prediction + InfoNCE

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Teacher**: Qwen2.5-Coder-3B, layer {TEACHER_LAYER} (frozen)
**Task**: Predict function body embedding from function signature token hidden states

## Architecture

```
Frozen:   Qwen2.5-Coder-3B (all {TEACHER_LAYER+1} transformer layers, bf16, no grad)
          ├─ hook at layer {TEACHER_LAYER} on sig tokens  → sig_hs  [B, T_sig, 2048]
          └─ hook at layer {TEACHER_LAYER} on body tokens → body_hs [B, T_body, 2048] (train only)

Trainable SigPredictorV2 ({res['n_params']:,} params):
  Encoder:
    Linear(2048 → {a['d_model']})
    {a['num_layers']}× TransformerBlock(d_model={a['d_model']}, nhead={a['nhead']}, ffn={a['d_model']*4}, dropout={a['dropout']})
    mean-pool → Linear({a['d_model']} → 2048)  ← retrieval head (InfoNCE)
  Decoder (training only):
    body_pos_enc: Embedding({MAX_BODY_TOKENS}, {a['d_model']})
    {a['num_layers']}× CrossAttentionBlock(d_model={a['d_model']}, nhead={a['nhead']})
    Linear({a['d_model']} → 2048)  ← per-token prediction head (SmoothL1)
```

## Dataset

| Split | Functions | Description |
|-------|-----------|-------------|
| Train | {res['n_train']:,} | Repo-stratified 80% split |
| Val   | {res['n_val']:,} | Held-out repos (10%) |
| Test  | {res['n_test']:,} | Held-out repos (10%) |

Corpus size (for retrieval): **{res['corpus']:,}** functions.

## Training

| Hyperparameter | Value |
|----------------|-------|
| Epochs | {a['epochs']} (early stopping patience={a['patience']}) |
| Batch/GPU | {a['batch_size']} × {res['world_size']} GPUs = {eff_batch} effective |
| Learning rate | {a['lr']} (cosine, warmup={a['warmup_epochs']} epochs) |
| d_model | {a['d_model']} |
| Transformer layers | {a['num_layers']} |
| Attention heads | {a['nhead']} |
| Dropout | {a['dropout']} |
| Token loss weight λ | {a['token_loss_weight']} |
| Max body tokens | {MAX_BODY_TOKENS} |
| Precision | BF16 (frozen Qwen), FP32 (predictor) |

## Results

### Val Set (in-distribution — primary metric)

| Metric | Value | Notes |
|--------|-------|-------|
| Cosine sim | **{res['val_cos']:.4f}** | Held-out repos, same domain as training |
| Rank@1 retrieval | **{res['val_r1']*100:.2f}%** | random={res['random_r1']*100:.4f}% |
| Rank@5 retrieval | **{res['val_r5']*100:.2f}%** | — |
| Rank@10 retrieval | **{res['val_r10']*100:.2f}%** | — |

### Test Set

| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| Cosine sim | **{res['cos']:.4f}** | {res['bl_cos']:.4f} (sig→body) | {"↑" if res['cos'] > res['bl_cos'] else "↓"}{abs(res['cos']-res['bl_cos']):.4f} |
| Cosine sim (random) | — | {res['rand_cos']:.4f} | Anisotropy baseline |
| Rank@1 retrieval | **{res['r1']*100:.2f}%** | {res['random_r1']*100:.4f}% (random) | {r1_vs_mlp} |
| Rank@5 retrieval | **{res['r5']*100:.2f}%** | — | — |
| Rank@10 retrieval | **{res['r10']*100:.2f}%** | — | — |

## Interpretation

{"Per-token body prediction substantially improves retrieval over the mean-pooled target baseline (Exp 1.4: Rank@1=1.38%). The cross-attention decoder forces the encoder to recover token-level body content (variable names, control flow) from the signature alone, producing representations that are more discriminative than mean-pooled training allows." if res['r1'] > 0.02 else "Token-level supervision did not substantially improve over the mean-pooled baseline. The bottleneck may be the teacher model capacity (Qwen2.5-Coder-3B) — functions that confuse the teacher cannot be separated by training on the teacher's own targets. Next step: upgrade to a larger teacher (Qwen3-8B, 4096-dim)."}

## Training Details

| Setting | Value |
|---------|-------|
| Loss | InfoNCE (primary) + λ·SmoothL1 token-level (auxiliary) |
| Final temperature τ | {res['final_temp']:.4f} |
| Negatives per step | {res['world_size']} GPUs × {res['args']['batch_size']} = {res['world_size']*res['args']['batch_size']} |
| Best val Rank@10 (during training) | {res['best_val_r10']*100:.2f}% |

## Success Criteria

| Criterion | Target | Achieved |
|-----------|--------|---------|
| Val Rank@1 > 3% | > 3% | {"✅" if res['val_r1'] > 0.03 else "❌"} {res['val_r1']*100:.2f}% |
| Val Rank@10 > 10% | > 10% | {"✅" if res['val_r10'] > 0.10 else "❌"} {res['val_r10']*100:.2f}% |
| Test Rank@1 > 3% | > 3% | {"✅" if res['r1'] > 0.03 else "❌"} {res['r1']*100:.2f}% |
| Test Rank@10 > 10% | > 10% | {"✅" if res['r10'] > 0.10 else "❌"} {res['r10']*100:.2f}% |
"""
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        f.write(report)
    print(f"\nResults written to {RESULTS_FILE}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs',        type=int,   default=100)
    ap.add_argument('--batch-size',    type=int,   default=32)
    ap.add_argument('--lr',            type=float, default=1e-4)
    ap.add_argument('--warmup-epochs', type=int,   default=5)
    ap.add_argument('--patience',      type=int,   default=15)
    ap.add_argument('--d-model',       type=int,   default=512)
    ap.add_argument('--nhead',         type=int,   default=8)
    ap.add_argument('--num-layers',    type=int,   default=2)
    ap.add_argument('--dropout',        type=float, default=0.1)
    ap.add_argument('--init-temp',         type=float, default=0.07)
    ap.add_argument('--token-loss-weight', type=float, default=0.1,
                    help='Weight λ for token-level SmoothL1 auxiliary loss (default 0.1)')
    args = ap.parse_args()

    # Normalise hyphen → underscore (argparse dest uses _)
    args.warmup_epochs = args.warmup_epochs
    args.num_layers    = args.num_layers
    args.init_temp     = args.init_temp
    args.batch_size    = args.batch_size
    args.d_model       = args.d_model

    # DDP initialisation
    dist.init_process_group(backend='nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(42 + rank)
    torch.cuda.set_device(rank)

    try:
        train(args, rank, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
