"""
Extract hard negative pairs from phase0_2 LLM judge results.

A "hard negative" for InfoNCE training is a (query, neighbour) pair where:
  - cosine similarity is HIGH (>= threshold) — the teacher embedding puts them close
  - LLM judge score_c is LOW (0 or 1)  — they are functionally unrelated

These are exactly the false-positives in the embedding space: pairs the model
confuses but shouldn't. Adding them as explicit negatives in the InfoNCE
denominator forces the model to learn to distinguish them.

Output: hard_negatives.json
  {
    "<function_id>": {
      "0": [neg_id, ...],   # score_c=0 — definitely unrelated (strongest negatives)
      "1": [neg_id, ...]    # score_c=1 — marginally related
    },
    ...
  }

Both function_id and each neg_id are guaranteed to exist in function_student_targets.

Usage:
    python extract_hard_negatives.py
    python extract_hard_negatives.py --min-cosim 0.85 --max-score-c 0
"""

import argparse
import json
import sys
import os
import collections
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pg8000.native
from phase0_1_similarity_matrix import DB

JSONL_PATH   = os.path.join(os.path.dirname(__file__),
                             'docs/phase0_2_knn_results_full.jsonl')
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), 'hard_negatives.json')
TARGET_MODEL = 'Qwen2.5-Coder-3B'


def load_training_ids() -> set[int]:
    """Return set of function_ids present in function_student_targets."""
    conn = pg8000.native.Connection(**DB)
    rows = conn.run("""
        SELECT fst.function_id
        FROM function_student_targets fst
        JOIN function_embeddings fe ON fe.id = fst.function_id
        WHERE fe.model_name = :mn
    """, mn=TARGET_MODEL)
    conn.close()
    return {r[0] for r in rows}


def extract(min_cosim: float, max_score_c: int) -> dict[int, dict[str, list[int]]]:
    """
    Stream the JSONL and collect hard negative pairs.

    Returns:
        hard_negs[query_id][str(score_c)] = [neighbour_id, ...]
    """
    print(f"Loading training corpus IDs from postgres …", flush=True)
    training_ids = load_training_ids()
    print(f"  {len(training_ids):,} function IDs in training corpus", flush=True)

    hard_negs: dict[int, dict[str, list[int]]] = collections.defaultdict(
        lambda: {"0": [], "1": []}
    )

    t0 = time.time()
    n_total = n_model = n_judged = n_kept = n_both_in = 0

    print(f"\nStreaming {JSONL_PATH} …", flush=True)
    with open(JSONL_PATH) as f:
        for line in f:
            n_total += 1
            r = json.loads(line)

            if r['query_model'] != TARGET_MODEL:
                continue
            n_model += 1

            if r['judge'] is None:
                continue
            n_judged += 1

            sc = r['judge'].get('score_c')
            if sc is None or sc > max_score_c:
                continue

            if r['cosine_similarity'] < min_cosim:
                continue
            n_kept += 1

            qid = r['query_id']
            nid = r['neighbour_id']
            if qid not in training_ids or nid not in training_ids:
                continue
            n_both_in += 1

            hard_negs[qid][str(sc)].append(nid)

    elapsed = time.time() - t0
    print(f"  Scanned {n_total:,} lines in {elapsed:.1f}s", flush=True)
    print(f"  Model={TARGET_MODEL}: {n_model:,}", flush=True)
    print(f"  Judged: {n_judged:,}", flush=True)
    print(f"  score_c≤{max_score_c} AND cosim≥{min_cosim}: {n_kept:,}", flush=True)
    print(f"  Both IDs in training corpus: {n_both_in:,}", flush=True)

    return dict(hard_negs)


def report_stats(hard_negs: dict) -> None:
    n_anchors = len(hard_negs)
    counts_0 = [len(v["0"]) for v in hard_negs.values()]
    counts_1 = [len(v["1"]) for v in hard_negs.values()]
    total_0 = sum(counts_0)
    total_1 = sum(counts_1)

    print(f"\nHard negative index statistics:")
    print(f"  Anchors with any hard negative: {n_anchors:,}")
    print(f"  Pairs with score_c=0:  {total_0:,}  (avg {total_0/n_anchors:.1f}/anchor)")
    print(f"  Pairs with score_c=1:  {total_1:,}  (avg {total_1/n_anchors:.1f}/anchor)")
    print(f"  Total pairs:           {total_0+total_1:,}")

    # Distribution of hard neg count per anchor
    all_counts = [len(v["0"]) + len(v["1"]) for v in hard_negs.values()]
    buckets = collections.Counter(min(c, 10) for c in all_counts)
    print(f"\n  Hard neg count distribution per anchor (capped at 10):")
    for k in sorted(buckets):
        label = f"=={k}" if k < 10 else ">=10"
        print(f"    {label}: {buckets[k]:,} anchors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-cosim',   type=float, default=0.80,
                    help='Minimum cosine similarity for a pair to count as hard (default 0.80)')
    ap.add_argument('--max-score-c', type=int,   default=1,
                    help='Maximum LLM judge score_c (0=definitely unrelated, 1=marginally, default 1)')
    ap.add_argument('--output',      type=str,   default=OUTPUT_PATH)
    args = ap.parse_args()

    hard_negs = extract(args.min_cosim, args.max_score_c)
    report_stats(hard_negs)

    print(f"\nWriting {args.output} …", flush=True)
    with open(args.output, 'w') as f:
        json.dump(hard_negs, f)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"  Done — {size_mb:.1f} MB", flush=True)


if __name__ == '__main__':
    main()
