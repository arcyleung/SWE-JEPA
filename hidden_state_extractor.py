"""
Code-JEPA Teacher: Extract hidden states from Qwen-2.5-Coder at masked code regions.

This is the "Level 0" prototype -- using a frozen pretrained model as the teacher
without any fine-tuning. We extract dense hidden states at structurally-masked
positions (function bodies) to serve as latent targets for the student.

Usage:
    python extract_hidden_states.py

Requirements:
    pip install torch transformers
"""

import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
from dataclasses import dataclass

TEACHER_MODELS = [
    "/home/original_models/Qwen2.5-Coder-3B",
    "/home/original_models/Qwen2.5-Coder-3B-Instruct",
    "/home/original_models/Qwen3-8B"
    "/home/original_models/Qwen3-8B-base"
]

# ---------------------------------------------------------------------------
# 1. Load model with hidden state extraction enabled
# ---------------------------------------------------------------------------

def load_teacher(
    model_name: str = TEACHER_MODELS[0],  # or 3B, 7B -- start small
    device: str = "cuda:1",
    dtype: torch.dtype | str = "auto",
):
    """Load the frozen teacher model.

    dtype defaults to "auto" so that each model loads with the dtype specified
    in its own config.json (bfloat16 for Qwen3, float16 for Qwen2.5-Coder).
    Forcing float16 on Qwen3-8B causes activation overflow → NaN hidden states.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Freeze everything -- this IS the frozen teacher
    for param in model.parameters():
        param.requires_grad = False

    # Belt-and-suspenders: normalize all parameters to the dominant dtype.
    # Qwen2 ties lm_head.weight to embed_tokens.weight, but under dtype="auto"
    # the tying can be deferred in multi-threaded loads, leaving lm_head in
    # float32 while the rest of the model is bfloat16.  We route the forward
    # pass through model.model (skipping lm_head entirely), but cast here too
    # so any other stray float32 tensors don't surface later.
    dominant = max(
        (p.dtype for p in model.parameters()),
        key=lambda d: {torch.bfloat16: 2, torch.float16: 1}.get(d, 0),
    )
    if any(p.dtype != dominant for p in model.parameters()):
        model = model.to(dominant)

    return model, tokenizer


# ---------------------------------------------------------------------------
# 2. Data structures
# ---------------------------------------------------------------------------

@dataclass
class MaskedRegion:
    """Defines a masked region in the source code."""
    start_char: int      # character offset in source
    end_char: int        # character offset in source
    region_type: str     # e.g., "function_body", "class_method", "error_handler"


@dataclass
class TeacherTargets:
    """Hidden states extracted from the teacher at masked positions."""
    hidden_states: torch.Tensor   # (num_masked_tokens, hidden_dim)
    token_ids: list               # token IDs at masked positions (for debugging)
    layer_index: int              # which layer the states came from
    region_type: str


# ---------------------------------------------------------------------------
# 3. Core extraction: get per-token hidden states at specific positions
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_hidden_states(
    model,
    tokenizer,
    source_code: str,
    masked_regions: list,
    layer: int = -1,            # which layer to extract from (-1 = last)
    pool_strategy: str = "none", # "none" (per-token), "mean", "last"
):
    """
    Run the FULL (unmasked) code through the teacher and extract hidden states
    at the token positions corresponding to masked regions.

    This is the key asymmetry: the teacher sees everything,
    the student will only see context around the masked regions.

    Args:
        model: Frozen Qwen model
        tokenizer: Corresponding tokenizer
        source_code: Full source code (unmasked)
        masked_regions: List of MaskedRegion to extract hidden states for
        layer: Which transformer layer to extract (-1 = last hidden state,
               -2 = second-to-last, etc. You can also try middle layers)
        pool_strategy: How to aggregate token-level states per region
    """

    # Tokenize the full source code
    encoding = tokenizer(
        source_code,
        return_tensors="pt",
        return_offsets_mapping=True,  # maps tokens back to char positions
        truncation=True,
        max_length=4096,
    )

    input_ids = encoding["input_ids"].to(model.device)
    offset_mapping = encoding["offset_mapping"][0]  # (seq_len, 2)

    # Use the base transformer (model.model) rather than the full causal LM so
    # that the lm_head is never executed.  This avoids the float32/bfloat16
    # dtype mismatch that occurs when lm_head.weight is newly initialised (Qwen2
    # ties it to embed_tokens but the tying can be deferred under dtype="auto"
    # in multi-threaded loads).  It also skips logit computation we don't need.
    _base = getattr(model, 'model', model)
    outputs = _base(
        input_ids=input_ids,
        output_hidden_states=True,
    )

    # outputs.hidden_states is a tuple of (num_layers + 1) tensors
    # Index 0 = embedding layer output
    # Index 1 = after transformer layer 1
    # ...
    # Index -1 = after final layer (same as last_hidden_state)
    # Each tensor shape: (batch=1, seq_len, hidden_dim)
    all_hidden_states = outputs.hidden_states
    selected_layer = all_hidden_states[layer].squeeze(0)  # (seq_len, hidden_dim)

    # Map each masked region's char span -> token indices -> hidden states
    results = []
    for region in masked_regions:
        token_indices = []
        for tok_idx, (start, end) in enumerate(offset_mapping):
            start, end = start.item(), end.item()
            if end > region.start_char and start < region.end_char:
                token_indices.append(tok_idx)

        if not token_indices:
            continue

        region_hidden = selected_layer[token_indices]  # (num_tokens, hidden_dim)
        region_token_ids = input_ids[0, token_indices].tolist()

        if pool_strategy == "mean":
            region_hidden = region_hidden.mean(dim=0, keepdim=True)
        elif pool_strategy == "last":
            region_hidden = region_hidden[-1:, :]

        results.append(TeacherTargets(
            hidden_states=region_hidden.cpu(),
            token_ids=region_token_ids,
            layer_index=layer,
            region_type=region.region_type,
        ))

    return results


# ---------------------------------------------------------------------------
# 4. AST-based structural masking (identifies WHAT to mask)
# ---------------------------------------------------------------------------

def find_function_bodies(source_code: str) -> list:
    """
    Find function body character spans using Python's ast module.
    Returns one MaskedRegion per function/method body found.
    """
    import ast as _ast

    try:
        tree = _ast.parse(source_code)
    except SyntaxError:
        return []

    # Build a map from 1-indexed line number -> character offset
    lines = source_code.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    regions = []
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            if not node.body:
                continue
            body_start_line = node.body[0].lineno        # 1-indexed
            body_end_line = node.end_lineno               # 1-indexed, inclusive
            start_char = line_offsets[body_start_line - 1]
            end_char = line_offsets[body_end_line]
            if end_char > start_char:
                regions.append(MaskedRegion(
                    start_char=start_char,
                    end_char=end_char,
                    region_type="function_body",
                ))

    return regions


# ---------------------------------------------------------------------------
# 5. Full pipeline for a single file
# ---------------------------------------------------------------------------

def process_code_file(model, tokenizer, source_code, layer=-1, pool_strategy="none"):
    """Find maskable regions and extract teacher targets."""
    masked_regions = find_function_bodies(source_code)
    if not masked_regions:
        return []
    return extract_hidden_states(
        model, tokenizer, source_code, masked_regions,
        layer=layer, pool_strategy=pool_strategy,
    )


# ---------------------------------------------------------------------------
# 6. Batch processing for dataset creation
# ---------------------------------------------------------------------------

def build_teacher_targets_dataset(
    code_files: list,
    output_path: str = "teacher_targets.pt",
    model_name: str = TEACHER_MODELS[0],
    layer: int = -1,
    pool_strategy: str = "mean",
):
    """
    Process a corpus of code files and save teacher targets to disk.
    """
    model, tokenizer = load_teacher(model_name)
    hidden_dim = model.config.hidden_size

    dataset = []
    for i, code in enumerate(code_files):
        targets = process_code_file(
            model, tokenizer, code,
            layer=layer, pool_strategy=pool_strategy,
        )
        for t in targets:
            dataset.append({
                "source_code": code,
                "region_type": t.region_type,
                "hidden_states": t.hidden_states,
                "token_ids": t.token_ids,
            })

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(code_files)} files, "
                  f"{len(dataset)} regions extracted")

    torch.save(dataset, output_path)
    print(f"Saved {len(dataset)} teacher targets to {output_path}")
    print(f"Hidden dim: {hidden_dim}, Layer: {layer}, Pool: {pool_strategy}")
    return dataset


# ---------------------------------------------------------------------------
# 7. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    example_code = '''
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
    for tmodel in TEACHER_MODELS:

        print("===== TEACHER MODEL:", tmodel, "=====")
        model, tokenizer = load_teacher(tmodel)

        print(f"\nModel hidden size: {model.config.hidden_size}")
        print(f"Number of layers: {model.config.num_hidden_layers}")

        # Find function bodies
        regions = find_function_bodies(example_code)
        print(f"\nFound {len(regions)} function bodies to mask:")
        for r in regions:
            snippet = example_code[r.start_char:r.end_char].strip()[:80]
            print(f"  [{r.region_type}] chars {r.start_char}-{r.end_char}: {snippet}...")

        # Extract per-token hidden states
        print("\n--- Per-token hidden states (last layer) ---")
        targets = extract_hidden_states(
            model, tokenizer, example_code, regions,
            layer=-1, pool_strategy="none",
        )
        print("\nTARGETS")
        for t in targets:
            print(f"  {t.region_type}: shape={t.hidden_states.shape}, "
                f"tokens={len(t.token_ids)}")

        # Extract mean-pooled hidden states
        print("\n--- Mean-pooled hidden states (last layer) ---")
        targets_pooled = extract_hidden_states(
            model, tokenizer, example_code, regions,
            layer=-1, pool_strategy="mean",
        )
        for t in targets_pooled:
            print(f"  {t.region_type}: shape={t.hidden_states.shape}")

        # Sanity check: are representations differentiated across functions?
        if len(targets_pooled) >= 2:
            cos_sim = torch.nn.functional.cosine_similarity(
                targets_pooled[0].hidden_states,
                targets_pooled[1].hidden_states,
            )
            print(f"\nCosine sim (func 0 vs func 1): {cos_sim.item():.4f}")
            print("(If ~1.0 for all pairs, try middle layers instead of last)")

        # Compare layers to find best differentiation
        print("\n--- Layer comparison (cosine sim between func 0 and func 1) ---")
        num_layers = model.config.num_hidden_layers
        for layer_idx in [1, num_layers // 4, num_layers // 2,
                        3 * num_layers // 4, -1]:
            t_layer = extract_hidden_states(
                model, tokenizer, example_code, regions,
                layer=layer_idx, pool_strategy="mean",
            )
            if len(t_layer) >= 2:
                sim = torch.nn.functional.cosine_similarity(
                    t_layer[0].hidden_states,
                    t_layer[1].hidden_states,
                ).item()
                actual_idx = (layer_idx if layer_idx >= 0
                            else num_layers + layer_idx + 1)
                print(f"  Layer {actual_idx:3d}: cosine_sim = {sim:.4f}")