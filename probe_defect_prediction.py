"""
Experiment 4.1: Defect-proneness prediction from function signatures.

Tests whether SWE-JEPA student representations encode future bugfix-proneness
better than frozen teacher representations, TF-IDF, or structural baselines.

Encodes 15k function signatures with:
  A) Frozen Qwen3-8B-base teacher (layer 18, mean-pool) → 4096-dim
  B) SWE-JEPA student (student_3_0_ckpt.pt, retrieval head) → 4096-dim

Fits linear probes (sklearn) on repo-level train/val/test split:
  - LogisticRegression → has_bugfix (AUROC, balanced accuracy, F1)
  - Ridge              → n_bugfix_prs (R², Spearman ρ)
  Baselines: majority class, TF-IDF on sig_text, LOC (sig line count)

Results → docs/phase4_1_defect_prediction.md

Usage:
    # encode + probe (first run, saves embedding cache)
    python probe_defect_prediction.py

    # re-run probes with cached embeddings (no GPU needed)
    python probe_defect_prediction.py --use-cache

    # stricter bugfix label (overlap > 10% of function)
    python probe_defect_prediction.py --min-overlap 0.1

    # custom paths
    python probe_defect_prediction.py \\
        --sigs-file followup_sigs.jsonl \\
        --cache-file followup_embs.npz \\
        --student-ckpt student_3_0_ckpt.pt \\
        --gpu 0
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEACHER_PATH   = '/home/original_models/Qwen3-8B-base'
TEACHER_LAYER  = 18
MAX_SIG_TOKENS = 256
RESULTS_FILE   = os.path.join(os.path.dirname(__file__),
                               'docs', 'phase4_1_defect_prediction.md')
DEFAULT_SIGS   = os.path.join(os.path.dirname(__file__), 'followup_sigs.jsonl')
DEFAULT_CACHE  = os.path.join(os.path.dirname(__file__), 'followup_embs.npz')
DEFAULT_CKPT   = os.path.join(os.path.dirname(__file__), 'student_3_0_ckpt.pt')


# ── Student model (copy of SigPredictorV2 from train_student_3_0.py) ──────────

class _CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.norm_q   = nn.LayerNorm(d_model)
        self.norm_kv  = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model))
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
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model))
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                key_padding_mask=pad_mask, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class SigPredictorV2(nn.Module):
    def __init__(self, d_in=4096, d_model=1024, nhead=8, num_layers=2,
                 d_out=4096, dropout=0.1, max_body_tokens=256):
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

    def forward(self, hidden_states, attention_mask):
        """Returns retrieval_pred [B, d_out] — the InfoNCE-trained head."""
        x = self.proj_in(self.in_norm(hidden_states.float()))
        pad_mask = (attention_mask == 0)
        for block in self.enc_blocks:
            x = block(x, pad_mask)
        real_mask = attention_mask.float().unsqueeze(-1)
        pooled = (x * real_mask).sum(1) / real_mask.sum(1).clamp(min=1.0)
        return self.proj_out(pooled)


# ── Forward hook ──────────────────────────────────────────────────────────────

@contextmanager
def capture_layer(model: nn.Module, layer_idx: int):
    captured: list[torch.Tensor] = []

    def _hook(module, inputs, outputs):
        if isinstance(outputs, torch.Tensor):
            captured.append(outputs.detach())
        else:
            captured.append(outputs[0].detach())

    hook = model.model.layers[layer_idx].register_forward_hook(_hook)
    try:
        yield captured
    finally:
        hook.remove()


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_all(sig_texts: list[str], student_ckpt: str, device: torch.device,
               batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode all sig_texts.
    Returns:
      teacher_embs: (N, 4096) — mean-pool of layer-18 teacher hidden states
      student_embs: (N, 4096) — student retrieval head output
    Both are float32.
    """
    N = len(sig_texts)
    print(f"  Loading tokeniser from {TEACHER_PATH} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH, trust_remote_code=True)

    print(f"  Tokenising {N:,} signatures …", flush=True)
    enc = tokenizer(sig_texts, truncation=True, max_length=MAX_SIG_TOKENS,
                    padding=False, return_tensors=None)
    input_ids_list  = enc['input_ids']
    attn_mask_list  = enc['attention_mask']

    print(f"  Loading frozen Qwen3-8B-base on {device} …", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"  Loading student from {student_ckpt} …", flush=True)
    ckpt = torch.load(student_ckpt, map_location='cpu')
    d_model = ckpt.get('d_model', 1024)
    student = SigPredictorV2(d_in=4096, d_model=d_model).to(device).eval()
    student.load_state_dict(ckpt['model_state'])
    for p in student.parameters():
        p.requires_grad_(False)

    teacher_embs = np.zeros((N, 4096), dtype=np.float32)
    student_embs = np.zeros((N, 4096), dtype=np.float32)
    n_batches    = (N + batch_size - 1) // batch_size
    log_every    = max(1, n_batches // 10)

    print(f"  Encoding {N:,} sigs ({n_batches} batches) …", flush=True)

    with capture_layer(teacher, TEACHER_LAYER) as cap, torch.no_grad():
        for bn, start in enumerate(range(0, N, batch_size)):
            end   = min(start + batch_size, N)
            bsz   = end - start
            max_l = max(len(input_ids_list[i]) for i in range(start, end))

            ids  = torch.zeros(bsz, max_l, dtype=torch.long, device=device)
            mask = torch.zeros(bsz, max_l, dtype=torch.long, device=device)
            for j, i in enumerate(range(start, end)):
                l = len(input_ids_list[i])
                ids[j, :l]  = torch.tensor(input_ids_list[i],  dtype=torch.long)
                mask[j, :l] = torch.tensor(attn_mask_list[i],  dtype=torch.long)

            cap.clear()
            teacher(input_ids=ids, attention_mask=mask)
            hs = cap[-1].float()                          # [B, T, 4096]

            # Teacher: mean-pool over non-padding positions
            real = mask.float().unsqueeze(-1)             # [B, T, 1]
            t_emb = (hs * real).sum(1) / real.sum(1).clamp(min=1.0)  # [B, 4096]
            teacher_embs[start:end] = t_emb.cpu().numpy()

            # Student: retrieval head
            s_emb = student(hs, mask)                     # [B, 4096]
            student_embs[start:end] = s_emb.float().cpu().numpy()

            if (bn + 1) % log_every == 0:
                print(f"    {bn+1}/{n_batches} batches …", flush=True)

    # Sanity: replace any NaN/Inf
    for embs, name in [(teacher_embs, 'teacher'), (student_embs, 'student')]:
        bad = ~np.isfinite(embs).all(axis=1)
        if bad.any():
            print(f"  WARNING: {bad.sum()} non-finite rows in {name}_embs → zeroed",
                  flush=True)
        np.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    return teacher_embs, student_embs


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sigs(sigs_file: str, min_overlap: float) -> list[dict]:
    """Load JSONL, apply min_overlap filter on has_bugfix label."""
    records = []
    with open(sigs_file) as f:
        for line in f:
            r = json.loads(line)
            if len(r.get('sig_text', '').strip()) < 5:
                continue
            if min_overlap > 0.0:
                # Re-label: has_bugfix only counts if overlap >= min_overlap
                if r['has_bugfix'] and r['max_bugfix_overlap'] < min_overlap:
                    r = {**r, 'has_bugfix': 0, 'n_bugfix_prs': 0}
            records.append(r)
    return records


# ── Train/val/test split ──────────────────────────────────────────────────────

def repo_split(repos: list[str], train_frac=0.80, val_frac=0.10, seed=42):
    import random
    rng = random.Random(seed)
    unique = sorted(set(repos))
    rng.shuffle(unique)
    n       = len(unique)
    n_val   = max(1, int(n * val_frac))
    n_test  = max(1, n - int(n * train_frac) - n_val)
    n_train = n - n_val - n_test
    train_r = set(unique[:n_train])
    val_r   = set(unique[n_train: n_train + n_val])
    tr, va, te = [], [], []
    for i, repo in enumerate(repos):
        if repo in train_r:   tr.append(i)
        elif repo in val_r:   va.append(i)
        else:                 te.append(i)
    return tr, va, te


# ── Probes ────────────────────────────────────────────────────────────────────

def probe_binary(X_tr, y_tr, X_te, y_te, name: str) -> dict:
    """LogisticRegression → has_bugfix. Returns dict of metrics."""
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced',
                             solver='lbfgs', random_state=42)
    clf.fit(X_tr_s, y_tr)
    probs = clf.predict_proba(X_te_s)[:, 1]
    preds = clf.predict(X_te_s)

    auroc = roc_auc_score(y_te, probs)
    bacc  = balanced_accuracy_score(y_te, preds)
    f1    = f1_score(y_te, preds, zero_division=0)

    print(f"  {name:30s}  AUROC={auroc:.4f}  BAcc={bacc:.4f}  F1={f1:.4f}")
    return {'auroc': auroc, 'bacc': bacc, 'f1': f1}


def probe_regression_count(X_tr, y_tr, X_te, y_te, name: str) -> dict:
    """Ridge → n_bugfix_prs. Returns dict of metrics."""
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    reg = Ridge(alpha=1.0)
    reg.fit(X_tr_s, np.log1p(y_tr))   # log1p for skewed count
    preds_log = reg.predict(X_te_s)
    preds     = np.expm1(np.maximum(preds_log, 0))

    ss_res = np.sum((y_te - preds) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    spear  = spearmanr(y_te, preds).statistic

    print(f"  {name:30s}  R²={r2:.4f}  Spearman ρ={spear:.4f}")
    return {'r2': r2, 'spearman': spear}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigs-file',    default=DEFAULT_SIGS)
    ap.add_argument('--cache-file',   default=DEFAULT_CACHE)
    ap.add_argument('--student-ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--use-cache',    action='store_true',
                    help='Load cached embeddings from --cache-file; skip teacher/student inference')
    ap.add_argument('--min-overlap',  type=float, default=0.0,
                    help='Min hunk_overlap_fraction to count a bugfix label (default: 0.0 = any overlap)')
    ap.add_argument('--gpu',          type=int,   default=0)
    ap.add_argument('--batch-size',   type=int,   default=32)
    args = ap.parse_args()

    t0 = time.time()
    print(f"SWE-JEPA Exp 4.1: Defect Prediction Probe")
    print(f"  sigs_file:  {args.sigs_file}")
    print(f"  min_overlap: {args.min_overlap}")
    print('=' * 60, flush=True)

    # ── Load records ──────────────────────────────────────────────────────────
    print(f"\nLoading {args.sigs_file} …", flush=True)
    records = load_sigs(args.sigs_file, args.min_overlap)
    N = len(records)
    print(f"  {N:,} records loaded")

    sig_texts = [r['sig_text']      for r in records]
    repos     = [r['feature_repo']  for r in records]
    has_bugfix  = np.array([r['has_bugfix']  for r in records], dtype=np.int32)
    n_bugfix_prs= np.array([r['n_bugfix_prs']for r in records], dtype=np.float32)

    n_pos = has_bugfix.sum()
    print(f"  has_bugfix: {n_pos:,} pos ({100*n_pos/N:.1f}%) / "
          f"{N-n_pos:,} neg ({100*(N-n_pos)/N:.1f}%)")

    # ── Embeddings ────────────────────────────────────────────────────────────
    if args.use_cache and os.path.exists(args.cache_file):
        print(f"\nLoading cached embeddings from {args.cache_file} …", flush=True)
        npz = np.load(args.cache_file)
        teacher_embs = npz['teacher_embs']
        student_embs = npz['student_embs']
        assert len(teacher_embs) == N, \
            f"Cache size mismatch: {len(teacher_embs)} vs {N} records"
        print(f"  Loaded. teacher_embs: {teacher_embs.shape}, "
              f"student_embs: {student_embs.shape}", flush=True)
    else:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available()
                              else 'cpu')
        print(f"\nEncoding on {device} …", flush=True)
        teacher_embs, student_embs = encode_all(
            sig_texts, args.student_ckpt, device, args.batch_size)
        np.savez_compressed(args.cache_file,
                            teacher_embs=teacher_embs,
                            student_embs=student_embs)
        print(f"  Embeddings cached to {args.cache_file}", flush=True)

    # ── Split ─────────────────────────────────────────────────────────────────
    train_idx, val_idx, test_idx = repo_split(repos)
    print(f"\nSplit (by repo): {len(train_idx):,} train  "
          f"{len(val_idx):,} val  {len(test_idx):,} test")

    # Combine train+val for final probe fitting (standard practice for linear probes)
    tv_idx = train_idx + val_idx

    def sel(arr, idx): return arr[idx] if isinstance(arr, np.ndarray) else [arr[i] for i in idx]

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("\n--- Baselines (has_bugfix) ---")

    # Majority class
    majority = int(has_bugfix[tv_idx].mean() > 0.5)
    maj_preds = np.full(len(test_idx), majority)
    maj_probs = np.full(len(test_idx), has_bugfix[tv_idx].mean())
    results = {
        'majority': {
            'auroc': roc_auc_score(has_bugfix[test_idx], maj_probs),
            'bacc':  balanced_accuracy_score(has_bugfix[test_idx], maj_preds),
            'f1':    f1_score(has_bugfix[test_idx], maj_preds, zero_division=0),
        }
    }
    print(f"  {'Majority class':30s}  AUROC={results['majority']['auroc']:.4f}  "
          f"BAcc={results['majority']['bacc']:.4f}  F1={results['majority']['f1']:.4f}")

    # LOC: count sig lines as structural proxy (no parsing needed)
    loc = np.array([len(t.splitlines()) for t in sig_texts], dtype=np.float32)
    results['loc'] = probe_binary(
        loc[tv_idx].reshape(-1, 1), has_bugfix[tv_idx],
        loc[test_idx].reshape(-1, 1), has_bugfix[test_idx], 'LOC (sig lines)')

    # TF-IDF on sig_text
    tfidf = TfidfVectorizer(max_features=5000, analyzer='word', sublinear_tf=True)
    X_tv_tfidf  = tfidf.fit_transform(sel(sig_texts, tv_idx)).toarray()
    X_te_tfidf  = tfidf.transform(sel(sig_texts, test_idx)).toarray()
    results['tfidf'] = probe_binary(
        X_tv_tfidf, has_bugfix[tv_idx],
        X_te_tfidf, has_bugfix[test_idx], 'TF-IDF (5k features)')

    # ── Main probes: teacher vs student ───────────────────────────────────────
    print("\n--- has_bugfix (binary classification) ---")
    results['teacher_binary'] = probe_binary(
        teacher_embs[tv_idx], has_bugfix[tv_idx],
        teacher_embs[test_idx], has_bugfix[test_idx], 'Teacher emb (frozen)')
    results['student_binary'] = probe_binary(
        student_embs[tv_idx], has_bugfix[tv_idx],
        student_embs[test_idx], has_bugfix[test_idx], 'Student emb (JEPA)')

    print("\n--- n_bugfix_prs (count regression) ---")
    results['majority_reg'] = probe_regression_count(
        np.zeros((len(tv_idx), 1)), n_bugfix_prs[tv_idx],
        np.zeros((len(test_idx), 1)), n_bugfix_prs[test_idx], 'Constant (mean)')
    results['loc_reg'] = probe_regression_count(
        loc[tv_idx].reshape(-1, 1), n_bugfix_prs[tv_idx],
        loc[test_idx].reshape(-1, 1), n_bugfix_prs[test_idx], 'LOC (sig lines)')
    results['tfidf_reg'] = probe_regression_count(
        X_tv_tfidf, n_bugfix_prs[tv_idx],
        X_te_tfidf, n_bugfix_prs[test_idx], 'TF-IDF (5k features)')
    results['teacher_reg'] = probe_regression_count(
        teacher_embs[tv_idx], n_bugfix_prs[tv_idx],
        teacher_embs[test_idx], n_bugfix_prs[test_idx], 'Teacher emb (frozen)')
    results['student_reg'] = probe_regression_count(
        student_embs[tv_idx], n_bugfix_prs[tv_idx],
        student_embs[test_idx], n_bugfix_prs[test_idx], 'Student emb (JEPA)')

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min", flush=True)

    # ── Write report ──────────────────────────────────────────────────────────
    write_report(results, N, len(train_idx), len(val_idx), len(test_idx),
                 n_pos, args)
    print(f"Results written to {RESULTS_FILE}")


def write_report(results: dict, N: int, n_train: int, n_val: int, n_test: int,
                 n_pos: int, args) -> None:
    tb  = results.get('teacher_binary', {})
    sb  = results.get('student_binary', {})
    tr  = results.get('teacher_reg',    {})
    sr  = results.get('student_reg',    {})
    tfi = results.get('tfidf',          {})
    loc = results.get('loc',            {})
    maj = results.get('majority',       {})

    student_wins_auroc = sb.get('auroc', 0) > tb.get('auroc', 0)
    student_wins_tfidf = sb.get('auroc', 0) > tfi.get('auroc', 0)
    student_useful     = sb.get('auroc', 0) > 0.60

    report = f"""# Experiment 4.1: Defect Prediction from Function Signatures

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Task**: Predict has_bugfix (binary) and n_bugfix_prs (count) from function signatures
**Data**: followups_function — {N:,} function anchors, 144 repos
**Labels**: has_bugfix={n_pos:,} pos ({100*n_pos/N:.1f}%) / {N-n_pos:,} neg
**Bugfix threshold**: min hunk_overlap_fraction ≥ {args.min_overlap:.2f}
**Split**: {n_train:,} train / {n_val:,} val / {n_test:,} test (by repo, 80/10/10)

## Task: has_bugfix — Binary Classification

| Probe | AUROC | Balanced Acc | F1 |
|-------|-------|-------------|-----|
| Majority class baseline | {maj.get('auroc', 0):.4f} | {maj.get('bacc', 0):.4f} | {maj.get('f1', 0):.4f} |
| LOC (sig line count) | {loc.get('auroc', 0):.4f} | {loc.get('bacc', 0):.4f} | {loc.get('f1', 0):.4f} |
| TF-IDF (5k features) | {tfi.get('auroc', 0):.4f} | {tfi.get('bacc', 0):.4f} | {tfi.get('f1', 0):.4f} |
| **Teacher emb (frozen)** | **{tb.get('auroc', 0):.4f}** | **{tb.get('bacc', 0):.4f}** | **{tb.get('f1', 0):.4f}** |
| **Student emb (JEPA)** | **{sb.get('auroc', 0):.4f}** | **{sb.get('bacc', 0):.4f}** | **{sb.get('f1', 0):.4f}** |

## Task: n_bugfix_prs — Count Regression (log1p-transformed)

| Probe | R² | Spearman ρ |
|-------|-----|-----------|
| Constant (mean) | {results.get('majority_reg', {}).get('r2', 0):.4f} | {results.get('majority_reg', {}).get('spearman', 0):.4f} |
| LOC (sig line count) | {results.get('loc_reg', {}).get('r2', 0):.4f} | {results.get('loc_reg', {}).get('spearman', 0):.4f} |
| TF-IDF (5k features) | {results.get('tfidf_reg', {}).get('r2', 0):.4f} | {results.get('tfidf_reg', {}).get('spearman', 0):.4f} |
| **Teacher emb (frozen)** | **{tr.get('r2', 0):.4f}** | **{tr.get('spearman', 0):.4f}** |
| **Student emb (JEPA)** | **{sr.get('r2', 0):.4f}** | **{sr.get('spearman', 0):.4f}** |

## Success Criteria

| Criterion | Target | Achieved |
|-----------|--------|----------|
| Student AUROC > Teacher AUROC | Student > Teacher | {"✅" if student_wins_auroc else "❌"} {sb.get('auroc',0):.4f} vs {tb.get('auroc',0):.4f} |
| Student AUROC > TF-IDF | Student > TF-IDF | {"✅" if student_wins_tfidf else "❌"} {sb.get('auroc',0):.4f} vs {tfi.get('auroc',0):.4f} |
| Student AUROC > 0.60 | > 0.60 | {"✅" if student_useful else "❌"} {sb.get('auroc',0):.4f} |

## Interpretation

Teacher vs Student AUROC delta: {sb.get('auroc',0) - tb.get('auroc',0):+.4f}
{"The SWE-JEPA student representations encode defect-proneness beyond the frozen teacher." if student_wins_auroc else "The student representations do not improve on the frozen teacher for defect prediction."}

Teacher vs TF-IDF AUROC delta: {tb.get('auroc',0) - tfi.get('auroc',0):+.4f}
{"Frozen teacher representations outperform TF-IDF, confirming structural encoding." if tb.get('auroc',0) > tfi.get('auroc',0) else "TF-IDF matches or exceeds frozen teacher representations."}

## Setup

| Parameter | Value |
|-----------|-------|
| Teacher | Qwen3-8B-base, layer {TEACHER_LAYER} (frozen) |
| Student ckpt | {os.path.basename(args.student_ckpt)} |
| Embeddings | 4096-dim, L2-normalised in probe |
| Probe: has_bugfix | LogisticRegression(C=1.0, class_weight='balanced') |
| Probe: n_bugfix_prs | Ridge(α=1.0) on log1p(y) |
| TF-IDF | 5000 features, word-level, sublinear_tf |
| Bugfix min overlap | {args.min_overlap:.2f} |
"""
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        f.write(report)


if __name__ == '__main__':
    main()
