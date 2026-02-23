"""
Experiment 1.1: Train the minimal SWE-JEPA student predictor.

Architecture:
    sig_embedding (2048) → MLP predictor → predicted_body_embedding (2048)
    Loss: SmoothL1(predicted_body_embedding, actual_body_embedding)

The student learns to map function signatures to their body latent representations
using frozen teacher (Qwen2.5-Coder-3B layer-18) targets prepared by
extract_student_targets.py.

Data split: 80/10/10 by repo (functions from held-out repos never seen in training).

Evaluation metrics:
  - SmoothL1 loss (train / val / test)
  - Cosine similarity between predicted and actual body embeddings
  - Rank@1 retrieval: does predicted embedding retrieve the correct body from the
    full body embedding FAISS index? (baseline: 1/N ≈ 0.004%)

Writes results to docs/phase1_1_student_training.md.

Usage:
    python train_student.py
    python train_student.py --epochs 100 --hidden 1024 --device cuda:1
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import faiss
import pg8000.native
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase0_1_similarity_matrix import DB

RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                             'docs', 'phase1_1_student_training.md')


# ── Model ─────────────────────────────────────────────────────────────────────

class MLPPredictor(nn.Module):
    """
    Simple 4-layer bottleneck MLP: 2048 → 1024 → 512 → 1024 → 2048.
    Maps function signature embedding to predicted body embedding.
    """
    def __init__(self, emb_dim: int = 2048, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
            nn.Linear(hidden // 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """
    Load (sig_embeddings, body_embeddings, repos, function_ids) from postgres.
    Returns arrays of shape (N, 2048).
    """
    conn = pg8000.native.Connection(**DB)
    rows = conn.run("""
        SELECT fst.function_id,
               fst.sig_embedding,
               fst.body_embedding,
               fe.instance_id
        FROM function_student_targets fst
        JOIN function_embeddings fe
          ON fe.id = fst.function_id
         AND fe.model_name = 'Qwen2.5-Coder-3B'
        ORDER BY fst.function_id
    """)
    conn.close()

    if not rows:
        raise RuntimeError("No data in function_student_targets — run "
                           "extract_student_targets.py first.")

    fids_raw   = [r[0] for r in rows]
    sig_raw    = np.array([r[1] for r in rows], dtype=np.float32)
    body_raw   = np.array([r[2] for r in rows], dtype=np.float32)
    repos_raw  = []
    for r in rows:
        parts = r[3].split('__')
        repos_raw.append(f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else r[3])

    # Drop NaN rows (from float16 overflow in extraction)
    valid = ~(np.isnan(sig_raw).any(axis=1) | np.isnan(body_raw).any(axis=1))
    n_nan = (~valid).sum()
    if n_nan:
        print(f"  Dropped {n_nan} NaN embedding rows")
    fids       = [fids_raw[i] for i in np.where(valid)[0]]
    sig_embs   = sig_raw[valid]
    body_embs  = body_raw[valid]
    repos      = [repos_raw[i] for i in np.where(valid)[0]]

    print(f"Loaded {len(fids):,} (sig, body) pairs from {len(set(repos))} repos")
    return sig_embs, body_embs, repos, fids


def repo_split(
    repos: list[str],
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split indices into train/val/test ensuring entire repos stay together.
    Returns (train_idx, val_idx, test_idx).
    """
    rng = random.Random(seed)
    unique_repos = sorted(set(repos))
    rng.shuffle(unique_repos)

    n = len(unique_repos)
    # Ensure at least 1 repo per split even with small repo counts
    n_val   = max(1, int(n * val_frac))
    n_test  = max(1, n - int(n * train_frac) - n_val)
    n_train = n - n_val - n_test

    train_repos = set(unique_repos[:n_train])
    val_repos   = set(unique_repos[n_train : n_train + n_val])
    # test_repos  = remaining

    train_idx, val_idx, test_idx = [], [], []
    for i, repo in enumerate(repos):
        if repo in train_repos:
            train_idx.append(i)
        elif repo in val_repos:
            val_idx.append(i)
        else:
            test_idx.append(i)

    return train_idx, val_idx, test_idx


# ── Evaluation helpers ────────────────────────────────────────────────────────

def cosine_sim_mean(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean cosine similarity between predicted and target embeddings."""
    pred_n   = pred   / (np.linalg.norm(pred,   axis=1, keepdims=True) + 1e-9)
    target_n = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-9)
    return float((pred_n * target_n).sum(axis=1).mean())


def rank1_accuracy(
    pred: np.ndarray,
    targets_all: np.ndarray,
    test_indices: list[int],
    k: int = 10,
) -> float:
    """
    For each predicted embedding, retrieve the k nearest body embeddings from the
    full corpus index and check if the correct body (at test_indices[i]) is rank 1.

    Uses FAISS IndexFlatIP on L2-normalised vectors (cosine similarity).
    """
    # Build FAISS index over all body embeddings
    d = targets_all.shape[1]
    index = faiss.IndexFlatIP(d)
    normed = targets_all / (np.linalg.norm(targets_all, axis=1, keepdims=True) + 1e-9)
    index.add(normed.astype(np.float32))

    pred_n = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-9)
    _, I = index.search(pred_n.astype(np.float32), k)

    hits = sum(1 for i, idx in enumerate(test_indices) if I[i, 0] == idx)
    return hits / len(test_indices)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args) -> dict:
    torch.manual_seed(42)
    np.random.seed(42)
    device = args.device if torch.cuda.is_available() else 'cpu'

    # Load data
    sig_embs, body_embs, repos, fids = load_data()
    train_idx, val_idx, test_idx = repo_split(repos)
    print(f"Split: {len(train_idx):,} train  {len(val_idx):,} val  "
          f"{len(test_idx):,} test  (by repo)")

    # Mean-centre then L2-normalise embeddings.
    # Centering removes the strong anisotropic bias in LLM hidden states
    # (without it, random pairs already have cosine ~0.90 making the task trivial).
    # Centre using train-split mean only to prevent data leakage.
    sig_mean  = sig_embs[train_idx].mean(axis=0, keepdims=True)
    body_mean = body_embs[train_idx].mean(axis=0, keepdims=True)
    sig_c  = sig_embs  - sig_mean
    body_c = body_embs - body_mean
    sig_n  = sig_c  / (np.linalg.norm(sig_c,  axis=1, keepdims=True) + 1e-9)
    body_n = body_c / (np.linalg.norm(body_c, axis=1, keepdims=True) + 1e-9)

    def make_loader(idx: list[int], shuffle: bool) -> DataLoader:
        sig_t  = torch.from_numpy(sig_n[idx]).float()
        body_t = torch.from_numpy(body_n[idx]).float()
        return DataLoader(TensorDataset(sig_t, body_t),
                          batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=2, pin_memory=True)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)

    # Model, optimiser, scheduler
    emb_dim = sig_embs.shape[1]
    model   = MLPPredictor(emb_dim=emb_dim, hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters  (emb_dim={emb_dim}, hidden={args.hidden})")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.SmoothL1Loss()

    # Baselines on test set (before training)
    test_sig_n  = sig_n[test_idx]
    test_body_n = body_n[test_idx]
    baseline_cos = float(
        np.mean((test_sig_n * test_body_n).sum(axis=1))
    )
    # Random baseline: mean cosine similarity between random pairs
    rng = np.random.default_rng(42)
    rand_idx  = rng.integers(0, len(body_n), size=len(test_idx))
    rand_body = body_n[rand_idx]
    random_cos = float(np.mean((test_sig_n * rand_body).sum(axis=1)))
    print(f"\nBaselines on test set ({len(test_idx):,} functions):")
    print(f"  sig→body cosine (pre-training):  {baseline_cos:.4f}")
    print(f"  random embedding cosine:         {random_cos:.4f}")

    history = {'train_loss': [], 'val_loss': [], 'val_cos': []}
    best_val_loss = float('inf')
    best_state    = None
    t0            = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_losses = []
        for sig_b, body_b in train_loader:
            sig_b, body_b = sig_b.to(device), body_b.to(device)
            optimizer.zero_grad()
            pred = model(sig_b)
            loss = criterion(pred, body_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        val_preds_list, val_targets_list = [], []
        with torch.no_grad():
            for sig_b, body_b in val_loader:
                sig_b, body_b = sig_b.to(device), body_b.to(device)
                pred = model(sig_b)
                val_losses.append(criterion(pred, body_b).item())
                val_preds_list.append(pred.cpu().numpy())
                val_targets_list.append(body_b.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        val_preds  = np.vstack(val_preds_list)
        val_targs  = np.vstack(val_targets_list)
        val_cos    = cosine_sim_mean(val_preds, val_targs)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_cos'].append(val_cos)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(f"  epoch {epoch:4d}/{args.epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"val_cos={val_cos:.4f}  lr={lr:.2e}  "
                  f"elapsed={elapsed:.0f}s", flush=True)

    # ── Final evaluation on test set ─────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()

    test_sig_t  = torch.from_numpy(test_sig_n).float().to(device)
    with torch.no_grad():
        test_preds = model(test_sig_t).cpu().numpy()

    test_cos   = cosine_sim_mean(test_preds, test_body_n)
    test_r1    = rank1_accuracy(test_preds, body_n, test_idx)
    test_r1_random = 1.0 / len(body_n)

    print(f"\nTest results:")
    print(f"  Cosine similarity (predicted vs actual body): {test_cos:.4f}")
    print(f"    vs sig-body baseline:                       {baseline_cos:.4f}")
    print(f"    vs random baseline:                         {random_cos:.4f}")
    print(f"  Rank@1 retrieval accuracy:  {test_r1:.4f}  "
          f"({test_r1*100:.2f}%  vs random {test_r1_random*100:.4f}%)")

    # Save best checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), 'student_mlp_ckpt.pt')
    torch.save({'model_state': best_state, 'args': vars(args),
                'emb_dim': emb_dim, 'hidden': args.hidden}, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")

    return {
        'n_train': len(train_idx),
        'n_val':   len(val_idx),
        'n_test':  len(test_idx),
        'n_params': n_params,
        'baseline_cos': baseline_cos,
        'random_cos':   random_cos,
        'best_val_loss': best_val_loss,
        'test_cos':  test_cos,
        'test_r1':   test_r1,
        'test_r1_random': test_r1_random,
        'history':   history,
        'args':      vars(args),
    }


# ── Results writer ────────────────────────────────────────────────────────────

def write_results(results: dict):
    h   = results['history']
    n   = len(h['train_loss'])
    mid = n // 2

    # Sample loss curve: every 10% of epochs
    sample_epochs = sorted(set([0, mid - 1, n - 1]))
    curve_rows = '\n'.join(
        f"| {e+1} | {h['train_loss'][e]:.4f} | {h['val_loss'][e]:.4f} | "
        f"{h['val_cos'][e]:.4f} |"
        for e in sample_epochs
    )

    args   = results['args']
    report = f"""# Experiment 1.1: Minimal Student Training

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Teacher**: Qwen2.5-Coder-3B, layer 18 (frozen)
**Task**: Predict function body embedding from function signature embedding

## Architecture

```
Input:  sig_embedding  ({results['n_params']:,}-param MLP)
Hidden: {args['hidden']} → {args['hidden']//2} → {args['hidden']}
Output: predicted_body_embedding
Loss:   SmoothL1
```

## Dataset

| Split | Functions | Description |
|-------|-----------|-------------|
| Train | {results['n_train']:,} | Repo-stratified 80% split |
| Val   | {results['n_val']:,} | Held-out repos (10%) |
| Test  | {results['n_test']:,} | Held-out repos (10%) |

Split is by **repo** — no function from a held-out repo appears in training.

## Training

| Hyperparameter | Value |
|----------------|-------|
| Epochs | {args['epochs']} |
| Batch size | {args['batch_size']} |
| Learning rate | {args['lr']} (cosine decay) |
| Hidden dim | {args['hidden']} |
| Optimizer | Adam, weight_decay=1e-4 |
| Parameters | {results['n_params']:,} |

## Loss Curve (sampled)

| Epoch | Train loss | Val loss | Val cosine sim |
|-------|-----------|----------|----------------|
{curve_rows}

## Test Set Results

| Metric | Value | Baseline | Improvement |
|--------|-------|----------|-------------|
| Cosine sim (predicted vs actual body) | **{results['test_cos']:.4f}** | {results['baseline_cos']:.4f} (sig→body) | +{results['test_cos']-results['baseline_cos']:.4f} |
| Cosine sim (random body) | — | {results['random_cos']:.4f} | — |
| Rank@1 retrieval accuracy | **{results['test_r1']*100:.2f}%** | {results['test_r1_random']*100:.4f}% (random) | ×{results['test_r1']/max(results['test_r1_random'],1e-9):.0f} |

## Interpretation

The student MLP predictor maps function **signatures** to the teacher's hidden-state
representation of function **bodies**. A cosine similarity of {results['test_cos']:.4f}
between predicted and actual body embeddings {"exceeds" if results["test_cos"] > results["baseline_cos"] else "does not yet exceed"} the raw
signature→body similarity of {results['baseline_cos']:.4f} (the teacher's internal
similarity between the two parts of the same function).

The rank@1 retrieval accuracy of {results['test_r1']*100:.2f}% is
{results['test_r1']/max(results['test_r1_random'],1e-9):.0f}× higher than random
({results['test_r1_random']*100:.4f}%), demonstrating that the predicted embeddings
are **geometrically meaningful** — they can be used to retrieve functionally related
functions from the corpus.

## Success Criteria Check

| Criterion | Target | Achieved |
|-----------|--------|---------|
| Training loss decreases | Yes | {"✅" if h["train_loss"][-1] < h["train_loss"][0] else "❌"} {h["train_loss"][-1]:.4f} < {h["train_loss"][0]:.4f} |
| Predictions closer to correct than random | cos > random_cos | {"✅" if results["test_cos"] > results["random_cos"] else "❌"} {results["test_cos"]:.4f} > {results["random_cos"]:.4f} |
| Rank@1 >> random | > 0.04% | {"✅" if results["test_r1"] > 10*results["test_r1_random"] else "❌"} {results["test_r1"]*100:.2f}% |

## Next Steps (Experiment 1.2)

- Per-token hidden state targets (not mean-pooled)
- Include full file context in the encoder input (imports, class definition, sibling methods)
- Cross-attention predictor over context tokens → richer prediction
- Correlate student loss at different training checkpoints with downstream KNN retrieval quality (SALT Experiment 1.2)
"""

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        f.write(report)
    print(f"\nResults written to {RESULTS_FILE}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs',     type=int,   default=100)
    ap.add_argument('--batch-size', type=int,   default=256)
    ap.add_argument('--lr',         type=float, default=1e-3)
    ap.add_argument('--hidden',     type=int,   default=1024)
    ap.add_argument('--device',     type=str,   default='cuda:0')
    args = ap.parse_args()

    print(f"SWE-JEPA Experiment 1.1: Minimal Student Training")
    print(f"{'='*60}")

    results = train(args)
    write_results(results)


if __name__ == '__main__':
    main()
