"""
Phase 1: Full pairwise cosine similarity matrix for all function bodies.

Best config from phase0: Coder-3B Base at layer 18 (mid-layer).
Runs all three teacher models and prints NxN pairwise matrices.

Usage:
    python phase1_similarity_matrix.py
"""

import torch
from hidden_state_extractor import load_teacher, find_function_bodies, extract_hidden_states

TEACHER_MODELS = [
    "/home/original_models/Qwen2.5-Coder-3B",
    "/home/original_models/Qwen2.5-Coder-3B-Instruct",
    "/home/original_models/Qwen3-8B",
    "/home/original_models/Qwen3-8B-base"
]

EXAMPLE_CODE = '''
import hashlib
from typing import Optional

class LRUCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, max_size: int = 128, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = {}
        self._access_order = []

    def get(self, key: str) -> Optional[str]:
        if key not in self._cache:
            return None
        entry = self._cache[key]
        if self._is_expired(entry):
            del self._cache[key]
            return None
        self._access_order.remove(key)
        self._access_order.append(key)
        return entry["value"]

    def put(self, key: str, value: str) -> None:
        if len(self._cache) >= self.max_size:
            evicted = self._access_order.pop(0)
            del self._cache[evicted]
        self._cache[key] = {"value": value, "ts": __import__("time").time()}
        self._access_order.append(key)

    def _is_expired(self, entry: dict) -> bool:
        return (__import__("time").time() - entry["ts"]) > self.ttl

    def _hash_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()
'''

FUNC_NAMES = ["__init__", "get", "put", "_is_expired", "_hash_key"]


def pairwise_sim_matrix(embeddings: list[torch.Tensor]) -> torch.Tensor:
    """Compute NxN cosine similarity matrix from list of (1, D) tensors."""
    stacked = torch.cat(embeddings, dim=0)  # (N, D)
    # L2-normalize
    normed = torch.nn.functional.normalize(stacked, dim=1)
    return normed @ normed.T  # (N, N)


def print_matrix(matrix: torch.Tensor, labels: list[str], title: str):
    col_w = 14
    name_w = 14
    print(f"\n{title}")
    print("-" * (name_w + col_w * len(labels)))
    # Header
    header = " " * name_w + "".join(f"{lb:>{col_w}}" for lb in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = f"{row_label:<{name_w}}"
        for j in range(len(labels)):
            val = matrix[i, j].item()
            row += f"{val:>{col_w}.4f}"
        print(row)


def run_model(model_path: str, layer: int = 18):
    print(f"\n{'='*60}")
    print(f"MODEL: {model_path}")
    print(f"Layer: {layer}")
    print(f"{'='*60}")

    model, tokenizer = load_teacher(model_path)
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"Hidden size: {hidden_size}, Total layers: {num_layers}")

    regions = find_function_bodies(EXAMPLE_CODE)
    print(f"Functions found: {len(regions)} -> {FUNC_NAMES[:len(regions)]}")

    # Extract mean-pooled at the chosen layer
    targets = extract_hidden_states(
        model, tokenizer, EXAMPLE_CODE, regions,
        layer=layer, pool_strategy="mean",
    )

    embeddings = [t.hidden_states for t in targets]  # list of (1, D) tensors
    matrix = pairwise_sim_matrix(embeddings)

    names = FUNC_NAMES[:len(embeddings)]
    print_matrix(matrix, names, f"Pairwise cosine similarity @ layer {layer}")

    # Also show off-diagonal stats
    n = matrix.shape[0]
    off_diag = [matrix[i, j].item() for i in range(n) for j in range(n) if i != j]
    print(f"\nOff-diagonal  min={min(off_diag):.4f}  max={max(off_diag):.4f}  "
          f"mean={sum(off_diag)/len(off_diag):.4f}")

    # Free GPU memory before loading next model
    del model
    torch.cuda.empty_cache()

    return matrix, names


if __name__ == "__main__":
    LAYER = 18  # best mid-layer from phase 0

    results = {}
    for model_path in TEACHER_MODELS:
        matrix, names = run_model(model_path, layer=LAYER)
        results[model_path] = (matrix, names)

    # Summary: off-diagonal means across models
    print(f"\n{'='*60}")
    print("SUMMARY: mean off-diagonal cosine sim @ layer 18")
    print(f"{'='*60}")
    for model_path, (matrix, names) in results.items():
        n = matrix.shape[0]
        off_diag = [matrix[i, j].item() for i in range(n) for j in range(n) if i != j]
        mean_sim = sum(off_diag) / len(off_diag)
        short = model_path.split("/")[-1]
        print(f"  {short:<35}: mean_inter_func_sim = {mean_sim:.4f}")
