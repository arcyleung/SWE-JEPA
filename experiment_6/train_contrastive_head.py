#!/usr/bin/env python3
"""
Stage 1 — Contrastive Projection Head (Experiment 6.2).

Learns a 256-dim "merge-readiness subspace" via InfoNCE (NT-Xent) on frozen
Qwen2.5-Coder-3B layer-18 embeddings.  Operates entirely in latent space
(JEPA-like): merged patches from the same repo should be predictable from
each other — structurally aligned changes are geometrically close.

Inputs:
    data/phase6_1/patch_embeddings_100k.npz  (z_patch, instance_ids, accepted, repos)

Outputs:
    data/phase6_2/contrastive_head.pt        — projection weights
    data/phase6_2/projected_embeddings.npz   — h (N, 256), instance_ids, accepted, repos
    data/phase6_2/contrastive_metrics.json   — loss curves, retrieval P@10

Usage:
    python train_contrastive_head.py
    python train_contrastive_head.py --epochs 10 --batch-size 256  # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

DEFAULT_EMBEDDINGS = os.path.join(PROJECT_ROOT, "data", "phase6_1", "patch_embeddings_100k.npz")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "data", "phase6_2")


# ── Model ────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """MLP projection: 2048 → d_hidden → d_out, L2-normalised output."""

    def __init__(self, d_in: int = 2048, d_hidden: int = 512, d_out: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ── Dataset ──────────────────────────────────────────────────────────────────

class RepoContrastiveDataset(Dataset):
    """Yields (anchor, positive) pairs — both merged patches from the same repo.

    Repos with < 2 merged patches are excluded from anchor selection.
    Closed patches are included in the full embedding matrix but never selected
    as anchors or positives — they act as natural hard negatives in the batch.
    """

    def __init__(self, z_patch: np.ndarray, accepted: np.ndarray, repos: np.ndarray,
                 repo_set: set[str] | None = None):
        merged_mask = accepted == 1
        # Build repo → list-of-merged-indices
        self.repo_indices: dict[str, list[int]] = {}
        for i in range(len(z_patch)):
            if not merged_mask[i]:
                continue
            r = str(repos[i])
            if repo_set is not None and r not in repo_set:
                continue
            self.repo_indices.setdefault(r, []).append(i)
        # Keep only repos with ≥2 merged patches
        self.repo_indices = {r: idxs for r, idxs in self.repo_indices.items() if len(idxs) >= 2}
        self.eligible_pairs: list[tuple[str, int]] = []
        for r, idxs in self.repo_indices.items():
            for idx in idxs:
                self.eligible_pairs.append((r, idx))

        self.z_patch = torch.tensor(z_patch, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.eligible_pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        repo, anchor_idx = self.eligible_pairs[idx]
        # Sample a different merged patch from the same repo as positive
        candidates = self.repo_indices[repo]
        pos_idx = anchor_idx
        while pos_idx == anchor_idx:
            pos_idx = candidates[torch.randint(len(candidates), (1,)).item()]
        return self.z_patch[anchor_idx], self.z_patch[pos_idx]


# ── InfoNCE loss ─────────────────────────────────────────────────────────────

def info_nce_loss(anchors: torch.Tensor, positives: torch.Tensor, tau: float) -> torch.Tensor:
    """NT-Xent with in-batch negatives.

    anchors, positives: (B, D), already L2-normalised.
    """
    B = anchors.shape[0]
    # Concatenate for the 2B pool
    z = torch.cat([anchors, positives], dim=0)  # (2B, D)
    sim = z @ z.T / tau  # (2B, 2B)
    # Mask out self-similarity
    mask = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)
    # For each anchor i, positive is at i+B; for each positive i+B, anchor is at i
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(sim.device)
    return F.cross_entropy(sim, labels)


# ── Retrieval P@10 ───────────────────────────────────────────────────────────

def retrieval_precision_at_k(embeddings: np.ndarray, accepted: np.ndarray, k: int = 10) -> float:
    """Fraction of a merged patch's k nearest neighbours that are also merged."""
    merged_mask = accepted == 1
    merged_idxs = np.where(merged_mask)[0]
    if len(merged_idxs) < k + 1:
        return 0.0
    # Subsample for speed
    rng = np.random.RandomState(42)
    query_idxs = rng.choice(merged_idxs, size=min(2000, len(merged_idxs)), replace=False)

    # Normalise for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    emb_normed = embeddings / norms

    precisions = []
    for qi in query_idxs:
        sims = emb_normed[qi] @ emb_normed.T
        sims[qi] = -1e9  # exclude self
        topk = np.argpartition(sims, -k)[-k:]
        precisions.append(float(np.mean(accepted[topk] == 1)))
    return float(np.mean(precisions))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1: Contrastive projection head (Experiment 6.2)")
    ap.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-hidden", type=int, default=512)
    ap.add_argument("--d-out", type=int, default=256)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.2, help="Fraction of repos held out for validation")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load embeddings
    data = np.load(args.embeddings, allow_pickle=True)
    z_patch = data["z_patch"]  # (N, 2048)
    instance_ids = data["instance_ids"]
    accepted = data["accepted"].astype(np.int32)
    repos = data["repos"]
    d_in = z_patch.shape[1]

    print(f"Loaded {z_patch.shape[0]:,} embeddings, dim={d_in}, "
          f"accepted_rate={accepted.mean():.3f}")

    # Repo-stratified train/val split
    all_repos = sorted(set(str(r) for r in repos))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(all_repos)
    n_val = max(1, int(len(all_repos) * args.val_frac))
    val_repos = set(all_repos[:n_val])
    train_repos = set(all_repos[n_val:])
    print(f"Repos: {len(train_repos):,} train, {len(val_repos):,} val")

    train_ds = RepoContrastiveDataset(z_patch, accepted, repos, repo_set=train_repos)
    val_ds = RepoContrastiveDataset(z_patch, accepted, repos, repo_set=val_repos)
    print(f"Train pairs: {len(train_ds):,}, Val pairs: {len(val_ds):,}")

    if len(train_ds) == 0:
        raise ValueError("No training pairs found — check that embeddings contain merged patches "
                         "with ≥2 per repo")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, drop_last=False) if len(val_ds) > 0 else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ProjectionHead(d_in=d_in, d_hidden=args.d_hidden, d_out=args.d_out).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ProjectionHead params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for anchors_raw, positives_raw in train_loader:
            anchors_raw = anchors_raw.to(device)
            positives_raw = positives_raw.to(device)
            a_proj = model(anchors_raw)
            p_proj = model(positives_raw)
            loss = info_nce_loss(a_proj, p_proj, args.tau)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # Validation
        avg_val_loss = None
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_loss_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for anchors_raw, positives_raw in val_loader:
                    anchors_raw = anchors_raw.to(device)
                    positives_raw = positives_raw.to(device)
                    a_proj = model(anchors_raw)
                    p_proj = model(positives_raw)
                    loss = info_nce_loss(a_proj, p_proj, args.tau)
                    val_loss_sum += loss.item() * anchors_raw.shape[0]
                    val_n += anchors_raw.shape[0]
            avg_val_loss = val_loss_sum / max(val_n, 1)
            val_losses.append(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            vl_str = f", val_loss={avg_val_loss:.4f}" if avg_val_loss is not None else ""
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d}/{args.epochs}: train_loss={avg_train_loss:.4f}"
                  f"{vl_str}, lr={lr_now:.2e}")

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s")

    # ── Project all embeddings ────────────────────────────────────────────
    model.eval()
    all_z = torch.tensor(z_patch, dtype=torch.float32)
    projected: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(all_z), 4096):
            batch = all_z[start:start + 4096].to(device)
            projected.append(model(batch).cpu().numpy())
    h = np.concatenate(projected, axis=0)  # (N, d_out)
    print(f"Projected embeddings: {h.shape}")

    # ── Retrieval P@10 ────────────────────────────────────────────────────
    # Build val mask for fair eval
    val_mask = np.array([str(r) in val_repos for r in repos])
    print(f"Computing retrieval P@10 on val repos ({val_mask.sum():,} patches)...")

    p10_projected_val = retrieval_precision_at_k(h[val_mask], accepted[val_mask], k=10)
    p10_raw_val = retrieval_precision_at_k(z_patch[val_mask], accepted[val_mask], k=10)
    print(f"  Retrieval P@10 (val): projected={p10_projected_val:.4f}, raw={p10_raw_val:.4f}")

    p10_projected_all = retrieval_precision_at_k(h, accepted, k=10)
    p10_raw_all = retrieval_precision_at_k(z_patch, accepted, k=10)
    print(f"  Retrieval P@10 (all): projected={p10_projected_all:.4f}, raw={p10_raw_all:.4f}")

    # ── Save outputs ──────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    # Projection weights
    model_path = os.path.join(args.out_dir, "contrastive_head.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "d_in": d_in,
        "d_hidden": args.d_hidden,
        "d_out": args.d_out,
    }, model_path)
    print(f"Saved projection head → {model_path}")

    # Projected embeddings
    emb_path = os.path.join(args.out_dir, "projected_embeddings.npz")
    np.savez_compressed(emb_path, h=h, instance_ids=instance_ids, accepted=accepted, repos=repos)
    print(f"Saved projected embeddings → {emb_path}")

    # Metrics
    metrics = {
        "n_samples": int(len(z_patch)),
        "d_in": d_in,
        "d_hidden": args.d_hidden,
        "d_out": args.d_out,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "tau": args.tau,
        "n_train_repos": len(train_repos),
        "n_val_repos": len(val_repos),
        "n_train_pairs": len(train_ds),
        "n_val_pairs": len(val_ds),
        "n_params": n_params,
        "train_time_s": round(elapsed, 1),
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
        "train_loss_curve": train_losses,
        "val_loss_curve": val_losses,
        "retrieval_p10_projected_val": p10_projected_val,
        "retrieval_p10_raw_val": p10_raw_val,
        "retrieval_p10_projected_all": p10_projected_all,
        "retrieval_p10_raw_all": p10_raw_all,
    }
    metrics_path = os.path.join(args.out_dir, "contrastive_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics → {metrics_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Retrieval P@10 (val): projected={p10_projected_val:.4f} vs raw={p10_raw_val:.4f} "
          f"(Δ={p10_projected_val - p10_raw_val:+.4f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
