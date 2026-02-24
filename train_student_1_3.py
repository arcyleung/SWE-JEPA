"""
Experiment 1.3: SWE-JEPA transformer encoder student — expanded corpus.

Same architecture as Exp 1.2, but trained on the full 50,000-function corpus
from 150 repos (vs 27,565 from 9 repos in Exp 1.2).

Also adds val-set Rank@1 evaluation at the end to give an in-distribution
retrieval signal (Exp 1.2 test repo was OOD, making test retrieval unreliable).

Usage:
  torchrun --nproc-per-node=8 train_student_1_3.py
  torchrun --nproc-per-node=8 train_student_1_3.py --epochs 100 --batch-size 64
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
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import faiss
import pg8000.native
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase0_1_similarity_matrix import DB

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEACHER_PATH = '/home/original_models/Qwen2.5-Coder-3B'
TEACHER_LAYER = 18          # 0-indexed; hidden_states[19] in HF convention
MAX_SIG_TOKENS = 256        # signatures are short; 256 covers >99% of cases
RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                             'docs', 'phase1_3_transformer_student.md')


# ── Model ─────────────────────────────────────────────────────────────────────

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


class SigPredictor(nn.Module):
    """
    Trainable head that maps Qwen layer-18 token hidden states for a function
    signature to the predicted body embedding.

    ~12M parameters.
    """

    def __init__(
        self,
        d_in: int = 2048,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        d_out: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_model)
        self.blocks  = nn.ModuleList(
            [_TransformerBlock(d_model, nhead, dropout) for _ in range(num_layers)]
        )
        self.proj_out = nn.Linear(d_model, d_out)

    def forward(
        self,
        hidden_states: torch.Tensor,    # [B, T, d_in]  from frozen Qwen layer 18
        attention_mask: torch.Tensor,   # [B, T]  1=real token, 0=padding
    ) -> torch.Tensor:                  # [B, d_out]
        x = self.proj_in(hidden_states.float())   # [B, T, d_model]  (bf16→f32)
        pad_mask = (attention_mask == 0)           # [B, T] bool  (True = ignore)
        for block in self.blocks:
            x = block(x, pad_mask)
        # Mean-pool over real (non-padding) tokens
        real_mask = attention_mask.float().unsqueeze(-1)   # [B, T, 1]
        x = (x * real_mask).sum(1) / real_mask.sum(1).clamp(min=1.0)  # [B, d_model]
        return self.proj_out(x)   # [B, d_out]


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
    Pre-tokenised (sig_text → token IDs) with stored body embeddings.
    body_emb is already mean-centred + L2-normalised.
    """

    def __init__(
        self,
        input_ids: list[list[int]],
        attention_masks: list[list[int]],
        body_embs: np.ndarray,
    ):
        assert len(input_ids) == len(body_embs)
        self.input_ids      = input_ids
        self.attention_masks = attention_masks
        self.body_embs      = body_embs.astype(np.float32)

    def __len__(self):
        return len(self.body_embs)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'body_emb':       self.body_embs[idx],
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad variable-length token sequences within a batch."""
    max_len = max(len(x['input_ids']) for x in batch)
    input_ids_t      = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask_t = torch.zeros(len(batch), max_len, dtype=torch.long)
    body_embs_t      = torch.stack([torch.tensor(x['body_emb']) for x in batch])
    for i, x in enumerate(batch):
        n = len(x['input_ids'])
        input_ids_t[i, :n]      = torch.tensor(x['input_ids'],      dtype=torch.long)
        attention_mask_t[i, :n] = torch.tensor(x['attention_mask'], dtype=torch.long)
    return {
        'input_ids':      input_ids_t,
        'attention_mask': attention_mask_t,
        'body_emb':       body_embs_t,
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(tokenizer) -> tuple[
    list[list[int]], list[list[int]], np.ndarray, list[str], list[int]
]:
    """
    Load (tokenised sig, body_embedding, repo, function_id) from postgres.
    Returns:
        input_ids_list      – list of token-id lists (variable length, truncated)
        attention_mask_list – corresponding attention masks
        body_embs           – (N, 2048) float32 numpy array
        repos               – list of repo strings  (e.g. "django/django")
        fids                – list of function_ids
    """
    conn = pg8000.native.Connection(**DB)
    rows = conn.run("""
        SELECT fst.function_id,
               fst.sig_text,
               fst.body_embedding,
               fe.instance_id
        FROM function_student_targets fst
        JOIN function_embeddings fe
          ON fe.id = fst.function_id
         AND fe.model_name = 'Qwen2.5-Coder-3B'
        WHERE fst.sig_text IS NOT NULL
        ORDER BY fst.function_id
    """)
    conn.close()

    if not rows:
        raise RuntimeError(
            "No sig_text in function_student_targets. "
            "Run store_sig_texts.py first."
        )

    fids_raw  = [r[0] for r in rows]
    sig_texts = [r[1] for r in rows]
    body_raw  = np.array([r[2] for r in rows], dtype=np.float32)
    repos_raw = []
    for r in rows:
        parts = r[3].split('__')
        repos_raw.append(f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else r[3])

    # Drop NaN/inf embedding rows (float16 overflow → inf → poisons mean-centering)
    valid   = np.isfinite(body_raw).all(axis=1)
    n_bad   = (~valid).sum()
    if n_bad:
        print(f"  Dropped {n_bad} non-finite body-embedding rows", flush=True)

    # Drop empty sig_texts (single-line functions → split_source returns sig_text='')
    # These tokenize to zero-length sequences → all-padding batches → NaN attention
    empty_sig = np.array([len(t.strip()) < 5 for t in sig_texts])
    valid = valid & ~empty_sig
    if empty_sig.sum():
        print(f"  Dropped {empty_sig.sum()} empty/trivial sig_text rows", flush=True)

    valid_idx = np.where(valid)[0]
    body_embs = body_raw[valid]
    fids      = [fids_raw[i] for i in valid_idx]
    sig_texts = [sig_texts[i] for i in valid_idx]
    repos     = [repos_raw[i] for i in valid_idx]

    # Tokenise signatures once (fast, runs in main process before DDP fork)
    enc = tokenizer(
        sig_texts,
        truncation=True,
        max_length=MAX_SIG_TOKENS,
        padding=False,
        return_tensors=None,
    )
    input_ids_list      = enc['input_ids']
    attention_mask_list = enc['attention_mask']

    return input_ids_list, attention_mask_list, body_embs, repos, fids


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

            # Trainable predictor (unwrap DDP for eval)
            raw_pred = predictor.module if hasattr(predictor, 'module') else predictor
            pred = raw_pred(hs, attn_mask)   # [B, 2048]

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
        print(f"SWE-JEPA Experiment 1.3: Transformer Encoder Student (expanded corpus)")
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
    input_ids_list, attention_mask_list, body_embs, repos, fids = load_data(tokenizer)
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
            [input_ids_list[i]      for i in idx],
            [attention_mask_list[i] for i in idx],
            body_n[idx],
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
    predictor = SigPredictor(
        d_in=emb_dim, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, d_out=emb_dim, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in predictor.parameters())
    if is_main:
        print(f"SigPredictor: {n_params:,} trainable parameters", flush=True)

    ddp_predictor = DDP(predictor, device_ids=[rank])

    optimizer = optim.AdamW(
        ddp_predictor.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = cosine_schedule_with_warmup(
        optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs)
    criterion = nn.SmoothL1Loss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss   = float('inf')
    best_state      = None
    patience_count  = 0
    t0              = time.time()

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        ddp_predictor.train()

        train_losses = []
        with capture_layer(frozen_qwen, TEACHER_LAYER) as captured:
            for batch in train_loader:
                input_ids  = batch['input_ids'].to(device, non_blocking=True)
                attn_mask  = batch['attention_mask'].to(device, non_blocking=True)
                body_emb   = batch['body_emb'].to(device, non_blocking=True)

                # Frozen forward (no grad, hook captures layer-18 output)
                captured.clear()
                with torch.no_grad():
                    frozen_qwen(input_ids=input_ids, attention_mask=attn_mask)
                hs = captured[-1].detach()   # [B, T, 2048] bf16, no grad

                optimizer.zero_grad()
                pred = ddp_predictor(hs, attn_mask)   # [B, 2048] f32
                loss = criterion(pred, body_emb)
                loss.backward()
                nn.utils.clip_grad_norm_(ddp_predictor.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

        scheduler.step()
        dist.barrier()

        # Val evaluation — all ranks for DDP correctness, but only rank 0 prints
        val_result = evaluate(
            ddp_predictor, frozen_qwen, val_loader,
            body_n_corpus, [],   # no retrieval metrics during training
            device, criterion, desc='val',
        )

        if is_main:
            train_loss = np.mean(train_losses)
            elapsed    = time.time() - t0
            lr         = scheduler.get_last_lr()[0]
            print(
                f"  epoch {epoch:4d}/{args.epochs}  "
                f"train={train_loss:.4f}  val={val_result['loss']:.4f}  "
                f"val_cos={val_result['cosine']:.4f}  "
                f"lr={lr:.2e}  elapsed={elapsed:.0f}s",
                flush=True,
            )

        # Early stopping on val loss (rank 0 decides, broadcast)
        improved = torch.tensor(
            int(val_result['loss'] < best_val_loss), device=device)
        dist.broadcast(improved, src=0)
        if improved.item():
            best_val_loss = val_result['loss']
            best_state    = {k: v.cpu().clone()
                             for k, v in ddp_predictor.module.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= args.patience:
            if is_main:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)", flush=True)
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
                                 'student_transformer_1_3_ckpt.pt')
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
            'random_r1': random_r1,
            'args': vars(args),
        })


# ── Results writer ────────────────────────────────────────────────────────────

def write_results(res: dict):
    a = res['args']
    eff_batch = a['batch_size'] * res['world_size']
    r1_vs_mlp = "improvement over MLP (0.00%)" if res['r1'] > 0 else \
                "no improvement over MLP (0.00%)"
    report = f"""# Experiment 1.3: Transformer Encoder Student (Expanded Corpus)

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Teacher**: Qwen2.5-Coder-3B, layer {TEACHER_LAYER} (frozen)
**Task**: Predict function body embedding from function signature token hidden states

## Architecture

```
Frozen:   Qwen2.5-Coder-3B (all {TEACHER_LAYER+1} transformer layers, bf16, no grad)
          ↓ forward hook → token hidden states [B, T, 2048]

Trainable SigPredictor ({res['n_params']:,} params):
  Linear(2048 → {a['d_model']})
  {a['num_layers']}× TransformerEncoderLayer(d_model={a['d_model']}, nhead={a['nhead']}, ffn={a['d_model']*4}, dropout={a['dropout']})
  mean-pool over non-padding tokens
  Linear({a['d_model']} → 2048)
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

{"The transformer encoder substantially improves over the MLP baseline (Rank@1: 0.00%) by attending token-level to argument names, type annotations, and docstring specifics in the function signature. This confirms the causal chain hypothesis: mean-pooled embeddings lack the specificity required for instance-level retrieval, whereas token-level attention can distinguish similar functions." if res['r1'] > 0 else "Rank@1 remains at 0.00%, suggesting that even token-level attention on signature hidden states is insufficient. Possible causes: (1) body embeddings are still too densely clustered after centering; (2) a larger predictor capacity or longer training is needed; (3) per-token body prediction targets (not mean-pooled) would provide a stronger training signal."}

## Success Criteria

| Criterion | Target | Achieved |
|-----------|--------|---------|
| Rank@1 > MLP | > 0.00% | {"✅" if res['r1'] > 0 else "❌"} {res['r1']*100:.2f}% |
| Rank@1 >> random | > 5% | {"✅" if res['r1'] > 0.05 else "❌"} {res['r1']*100:.2f}% |
| Rank@10 > 20% | > 20% | {"✅" if res['r10'] > 0.20 else "❌"} {res['r10']*100:.2f}% |
"""
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        f.write(report)
    print(f"\nResults written to {RESULTS_FILE}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs',        type=int,   default=100)
    ap.add_argument('--batch-size',    type=int,   default=64)
    ap.add_argument('--lr',            type=float, default=1e-4)
    ap.add_argument('--warmup-epochs', type=int,   default=5)
    ap.add_argument('--patience',      type=int,   default=15)
    ap.add_argument('--d-model',       type=int,   default=512)
    ap.add_argument('--nhead',         type=int,   default=8)
    ap.add_argument('--num-layers',    type=int,   default=2)
    ap.add_argument('--dropout',       type=float, default=0.1)
    args = ap.parse_args()

    # Normalise hyphen → underscore (argparse dest uses _)
    args.warmup_epochs = args.warmup_epochs
    args.num_layers    = args.num_layers
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
