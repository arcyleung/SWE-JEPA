"""
Phase 1: Extract function-body embeddings from real PRs at scale.

Best config from phase0: Coder-3B Base at layer 18 (mid-layer).
Runs all four teacher models; similarity matrix is skipped (already validated
in the 30-PR pilot run).  8 GPU threads process samples in parallel, one
model copy per GPU.

Function bodies are sourced from real PRs in postgres (prs table), reading
files at base_commit via overlayfs over the on-disk shallow clone.

Usage:
    python phase1_similarity_matrix.py --store
    python phase1_similarity_matrix.py --limit 500 --repo owner/name --store
    python phase1_similarity_matrix.py --store --index-dir ./faiss_indices --n-gpus 4
"""

import ast as _ast
import multiprocessing as mp
import os
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# Must be set before the HuggingFace tokenizers library is imported.
# The fast tokenizer (Rust-backed) maintains its own thread pool; when multiple
# Python threads call it simultaneously they deadlock fighting for the GIL.
# Disabling its internal parallelism makes each call single-threaded and safe.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import yaml
import pg8000.native
from hidden_state_extractor import load_teacher, MaskedRegion, extract_hidden_states

# ── Teacher models ─────────────────────────────────────────────────────────────
TEACHER_MODELS = [
    "/home/original_models/Qwen2.5-Coder-3B",
    "/home/original_models/Qwen2.5-Coder-3B-Instruct",
    "/home/original_models/Qwen3-8B",
    "/home/original_models/Qwen3-8B-base"
]

N_GPUS = 8   # cuda:0 … cuda:7; one model copy per GPU for parallel extraction

# ── Paths ──────────────────────────────────────────────────────────────────────
REPOS_BASE          = '/shared_workspace_mfs/repos'
OVERLAY_MERGED_BASE = '/shared_workspace_mfs/repos_tmp_overlayfs'
OVERLAY_SHM_BASE    = '/dev/shm'
PG_CONFIG_FILE      = os.path.join(os.path.dirname(__file__), 'postgres_connection.yaml')
TOKENS_FILE         = os.path.join(os.path.dirname(__file__), 'crawl_tokens.yaml')

# ── Postgres connection ────────────────────────────────────────────────────────
_pg_cfg = yaml.safe_load(open(PG_CONFIG_FILE))
DB = dict(
    host=_pg_cfg['ip'],
    port=_pg_cfg.get('port') or 9999,
    user=_pg_cfg['user'],
    password=_pg_cfg['password'],
    database=_pg_cfg['database'],
)

# ── Fallback example (used when postgres returns nothing) ──────────────────────
_EXAMPLE_CODE = '''
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

# ── AST: find functions with names ────────────────────────────────────────────

def find_functions_with_names(source_code: str) -> list[tuple[MaskedRegion, str]]:
    """
    Walk the AST and return (MaskedRegion, func_name) for every function/method.
    Mirrors hidden_state_extractor.find_function_bodies but also returns the name.
    """
    try:
        tree = _ast.parse(source_code)
    except SyntaxError:
        return []

    lines = source_code.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    results = []
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            if not node.body:
                continue
            start_char = line_offsets[node.body[0].lineno - 1]
            end_char   = line_offsets[node.end_lineno]
            if end_char > start_char:
                results.append((
                    MaskedRegion(start_char=start_char, end_char=end_char,
                                 region_type="function_body"),
                    node.name,
                ))
    return results


# ── Diff helpers ───────────────────────────────────────────────────────────────

def parse_modified_py_files(patch: str) -> list[str]:
    """Extract Python file paths modified by a unified diff patch."""
    files = []
    seen  = set()
    for line in patch.splitlines():
        m = re.match(r'^diff --git a/(.+?) b/.+$', line)
        if not m:
            m = re.match(r'^\+\+\+ b/(.+)$', line)
        if m:
            path = m.group(1)
            if (path.endswith('.py') or path.endswith('.pyx')) and path not in seen:
                files.append(path)
                seen.add(path)
    return files


# ── Git helpers ────────────────────────────────────────────────────────────────

def _sha_available(repo_dir: str, sha: str) -> bool:
    r = subprocess.run(['git', '-C', repo_dir, 'cat-file', '-e', sha],
                       capture_output=True)
    return r.returncode == 0


def _git_show(repo_dir: str, sha: str, file_path: str) -> str | None:
    r = subprocess.run(
        ['git', '-C', repo_dir, 'show', f'{sha}:{file_path}'],
        capture_output=True, timeout=30,
    )
    return r.stdout.decode(errors='replace') if r.returncode == 0 else None


def _fetch_sha(repo_dir: str, sha: str, repo_slug: str, gh_token: str | None):
    url = (f'https://{gh_token}@github.com/{repo_slug}.git' if gh_token
           else f'https://github.com/{repo_slug}.git')
    subprocess.run(['git', '-C', repo_dir, 'fetch', '--depth=1', url, sha],
                   capture_output=True, timeout=120)


# ── OverlayFS helpers ──────────────────────────────────────────────────────────

def _mount_overlay(repo_path: str, tag: str) -> tuple[str, str, str]:
    upper  = os.path.join(OVERLAY_SHM_BASE, f'ovl-upper-{tag}')
    work   = os.path.join(OVERLAY_SHM_BASE, f'ovl-work-{tag}')
    merged = os.path.join(OVERLAY_MERGED_BASE, tag)
    for d in (upper, work, merged):
        os.makedirs(d, exist_ok=True)
    subprocess.run(
        ['fuse-overlayfs',
         '-o', f'lowerdir={repo_path},upperdir={upper},workdir={work}',
         merged],
        check=True,
    )
    return merged, upper, work


def _umount_overlay(merged: str, upper: str, work: str):
    # Try normal unmount first; if the mount point is still busy (e.g. another
    # worker shares the same lowerdir), fall back to lazy unmount (-z) which
    # detaches the mount immediately and defers cleanup until all references drop.
    for cmd in (
        ['fusermount3', '-u',      merged],
        ['fusermount3', '-u', '-z', merged],
    ):
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        if r.returncode == 0:
            break
    for d in (upper, work):
        shutil.rmtree(d, ignore_errors=True)


# ── Token loading ──────────────────────────────────────────────────────────────

def _load_tokens() -> list[str]:
    """Load GitHub tokens from crawl_tokens.yaml (gh_tokens key)."""
    try:
        cfg = yaml.safe_load(open(TOKENS_FILE))
        return cfg.get('gh_tokens', [])
    except (FileNotFoundError, KeyError):
        return []


def _build_repo_id_map() -> dict[int, str]:
    """Map repo_id (int) → on-disk directory name under REPOS_BASE."""
    repo_id_to_dir: dict[int, str] = {}
    try:
        for entry in os.listdir(REPOS_BASE):
            parts = entry.split('_', 1)
            if parts[0].isdigit():
                repo_id_to_dir[int(parts[0])] = entry
    except OSError as e:
        print(f"WARNING: cannot list {REPOS_BASE}: {e}")
    return repo_id_to_dir


# ── Per-PR worker (runs in a subprocess) ──────────────────────────────────────
#
# Must be a top-level function so ProcessPoolExecutor can pickle it.
# All helpers it calls (_mount_overlay etc.) are also module-level.
#
# Each sample is a 4-tuple:
#   (source_code, regions, display_labels, meta)
#
# meta dict keys:
#   instance_id   - PR instance ID from the prs table
#   pull_number   - integer PR number
#   file_path     - full path inside the repo
#   func_names    - list[str] bare function names (parallel to regions/labels)

def _process_single_pr(
    row: tuple,
    gh_token: str | None,
    repo_id_to_dir: dict[int, str],
    max_funcs_per_file: int,
) -> list[tuple]:
    """Process one PR row; returns a list of (source, regions, labels, meta) tuples."""
    repo_slug, repo_id, pull_number, base_commit, patch, instance_id = row
    instance_id = instance_id or f"{repo_slug}#{pull_number}"

    py_files = parse_modified_py_files(patch)
    if not py_files:
        return []

    dir_name = repo_id_to_dir.get(repo_id)
    if not dir_name:
        print(f"  SKIP pr#{pull_number} ({repo_slug}): repo not on disk", flush=True)
        return []

    repo_path = os.path.join(REPOS_BASE, dir_name)
    tag = f'phase1-{repo_id}-{pull_number}'

    try:
        merged, upper, work = _mount_overlay(repo_path, tag)
    except subprocess.CalledProcessError as e:
        print(f"  SKIP pr#{pull_number} ({repo_slug}): overlayfs mount failed — {e}",
              flush=True)
        return []

    samples: list[tuple] = []
    try:
        if not _sha_available(merged, base_commit):
            _fetch_sha(merged, base_commit, repo_slug, gh_token)

        if not _sha_available(merged, base_commit):
            print(f"  SKIP pr#{pull_number} ({repo_slug}): "
                  f"could not fetch {base_commit[:12]}", flush=True)
            return samples   # empty; finally will still unmount

        for file_path in py_files:
            content = _git_show(merged, base_commit, file_path)
            if content is None:
                continue

            func_pairs = find_functions_with_names(content)[:max_funcs_per_file]
            if not func_pairs:
                continue

            regions    = [r for r, _ in func_pairs]
            func_names = [n for _, n in func_pairs]
            fname      = os.path.basename(file_path)
            labels     = [f"pr{pull_number}/{fname}/{n}" for n in func_names]

            samples.append((content, regions, labels, {
                'instance_id': instance_id,
                'pull_number': pull_number,
                'file_path':   file_path,
                'func_names':  func_names,
            }))

        n_funcs = sum(len(s[2]) for s in samples)
        print(f"  pr#{pull_number} ({repo_slug}): "
              f"{len(samples)} py file(s), {n_funcs} functions", flush=True)

    finally:
        _umount_overlay(merged, upper, work)

    return samples


# ── Main data-fetch: PRs → code samples (parallel) ────────────────────────────

def fetch_pr_code_samples(
    limit: int | None = None,      # None → fetch all rows
    repo_filter: str | None = None,
    max_funcs_per_file: int = 30,
    gh_token: str | None = None,   # optional single-token override
) -> list[tuple]:
    """
    Query the prs table, then fan out across all tokens in crawl_tokens.yaml,
    processing up to `limit` PRs in parallel (one worker per token).

    Each worker mounts its own overlayfs, fetches the base_commit if needed,
    reads modified Python files, and parses function bodies.

    Returns a flat list of 4-tuples: (source_code, regions, display_labels, meta).
    """
    # Load tokens; honour an explicit override as the first choice
    tokens = _load_tokens()
    if gh_token:
        tokens = [gh_token] + [t for t in tokens if t != gh_token]
    if not tokens:
        tokens = [None]   # unauthenticated fallback

    conn = pg8000.native.Connection(**DB)
    sql = ('SELECT repo, repo_id, pull_number, base_commit, patch, instance_id '
           'FROM prs '
           "WHERE base_commit IS NOT NULL AND base_commit != '' "
           "  AND patch      IS NOT NULL AND patch      != ''")
    params: dict = {}
    if repo_filter:
        sql += ' AND repo = :repo'
        params['repo'] = repo_filter
    if limit is not None:
        sql += f' LIMIT {limit}'

    rows = conn.run(sql, **params)
    conn.close()

    if not rows:
        print("No PRs found in postgres matching the query.")
        return []

    repo_id_to_dir = _build_repo_id_map()
    os.makedirs(OVERLAY_MERGED_BASE, exist_ok=True)

    # One worker per token, capped at the number of PRs
    max_workers = min(len(tokens), len(rows))
    print(f"Queried {len(rows)} PRs — processing with {max_workers} parallel "
          f"worker(s) ({len(tokens)} token(s))")

    all_samples: list[tuple] = []
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Round-robin token assignment by submission order
        futures = {
            executor.submit(
                _process_single_pr,
                tuple(row),
                tokens[i % len(tokens)],
                repo_id_to_dir,
                max_funcs_per_file,
            ): row
            for i, row in enumerate(rows)
        }
        for future in as_completed(futures):
            completed += 1
            row = futures[future]
            try:
                samples = future.result()
                all_samples.extend(samples)
            except Exception as e:
                print(f"  ERROR pr#{row[2]} ({row[0]}): {e}", flush=True)
            if completed % 10 == 0 or completed == len(rows):
                print(f"  [{completed}/{len(rows)}] PRs done, "
                      f"{sum(len(s[2]) for s in all_samples)} functions so far")

    return all_samples


# ── Similarity matrix helpers ──────────────────────────────────────────────────

def pairwise_sim_matrix(embeddings: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.cat([e.view(1, -1) for e in embeddings], dim=0)
    normed  = torch.nn.functional.normalize(stacked, dim=1)
    return normed @ normed.T


def print_matrix(matrix: torch.Tensor, labels: list[str], title: str):
    name_w = max(16, max(len(lb) for lb in labels) + 2)
    col_w  = 10
    print(f"\n{title}")
    print("-" * (name_w + col_w * len(labels)))
    print(" " * name_w + "".join(f"{lb[:col_w-1]:>{col_w}}" for lb in labels))
    for i, row_label in enumerate(labels):
        row = f"{row_label:<{name_w}}"
        for j in range(len(labels)):
            row += f"{matrix[i, j].item():>{col_w}.4f}"
        print(row)


# ── GPU worker (runs in a thread, one per CUDA device) ────────────────────────

def _extract_worker(
    gpu_id: int,
    model_path: str,
    samples_chunk: list[tuple],
    layer: int,
) -> list[dict]:
    """
    Load model on cuda:{gpu_id}, extract mean-pooled hidden states for every
    function in samples_chunk, return embedding dicts.

    Runs in a subprocess (ProcessPoolExecutor + spawn), so it has its own
    CUDA context, its own copy of the model, and its own tokenizer state.
    """

    device      = f"cuda:{gpu_id}"
    short_model = model_path.split('/')[-1]
    print(f"  [GPU {gpu_id}] loading {short_model} …", flush=True)

    model, tokenizer = load_teacher(model_path, device=device)

    actual_dtype = next(model.parameters()).dtype
    print(f"  [GPU {gpu_id}] ready ({actual_dtype}), "
          f"{len(samples_chunk)} file group(s)", flush=True)

    records: list[dict] = []
    for sample in samples_chunk:
        source_code, regions, labels = sample[0], sample[1], sample[2]
        meta       = sample[3] if len(sample) > 3 else {}
        func_names = meta.get('func_names', [lb.split('/')[-1] for lb in labels])

        targets = extract_hidden_states(
            model, tokenizer, source_code, regions,
            layer=layer, pool_strategy="mean",
        )

        for t, func_name in zip(targets, func_names):
            emb = t.hidden_states  # (1, D)
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                print(f"  [GPU {gpu_id}] WARNING: NaN/inf for '{func_name}' "
                      f"({actual_dtype}) — skipping", flush=True)
                continue
            records.append({
                'instance_id':      meta.get('instance_id', ''),
                'feature_file':     meta.get('file_path', ''),
                'feature_function': func_name,
                'model_name':       short_model,
                'layer_index':      layer,
                'embedding':        emb.squeeze().float().numpy(),
            })

    del model
    torch.cuda.empty_cache()
    print(f"  [GPU {gpu_id}] done — {len(records)} embeddings", flush=True)
    return records


# ── Per-model run (parallel across GPUs) ──────────────────────────────────────

def run_model(
    model_path: str,
    code_samples: list[tuple],
    layer: int = 18,
    n_gpus: int = N_GPUS,
) -> list[dict]:
    """
    Distribute code_samples round-robin across n_gpus GPU processes (spawn).
    Each subprocess loads an independent model copy on its own CUDA device and
    returns embedding dicts.  No similarity matrix is computed here.
    """
    print(f"\n{'='*60}")
    print(f"MODEL: {model_path}  |  layer={layer}  |  GPUs: cuda:0–cuda:{n_gpus-1}")
    print(f"{'='*60}")

    # Round-robin split: sample i → GPU (i % n_gpus)
    chunks = [code_samples[i::n_gpus] for i in range(n_gpus)]
    chunks = [c for c in chunks if c]   # drop empty tails when n_gpus > n_samples

    all_records: list[dict] = []
    with ProcessPoolExecutor(max_workers=len(chunks),
                             mp_context=mp.get_context('spawn')) as executor:
        futures = {
            executor.submit(_extract_worker, gpu_id, model_path, chunk, layer): gpu_id
            for gpu_id, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            gpu_id = futures[future]
            try:
                all_records.extend(future.result())
            except Exception as e:
                print(f"  ERROR on GPU {gpu_id}: {e}", flush=True)

    print(f"  {model_path.split('/')[-1]}: {len(all_records)} embeddings extracted")
    return all_records


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--limit',     type=int, default=None,
                        help='Max PRs to fetch from postgres (default: all rows)')
    parser.add_argument('--repo',      type=str, default=None,
                        help='Filter to a specific repo slug (e.g. owner/name)')
    parser.add_argument('--layer',     type=int, default=18,
                        help='Layer index to extract from (default 18)')
    parser.add_argument('--token',     type=str, default=None,
                        help='GitHub token for fetching commits (optional)')
    parser.add_argument('--store',     action='store_true',
                        help='Store embeddings to postgres and build FAISS index')
    parser.add_argument('--index-dir', type=str, default='faiss_indices',
                        help='Directory for FAISS index files (used with --store)')
    parser.add_argument('--n-gpus',    type=int, default=N_GPUS,
                        help=f'Number of GPUs to use in parallel (default {N_GPUS})')
    args = parser.parse_args()

    print("Fetching PR code samples from postgres...")
    code_samples = fetch_pr_code_samples(
        limit=args.limit,
        repo_filter=args.repo,
        gh_token=args.token,
    )

    if not code_samples:
        print("No code samples from postgres — falling back to built-in example.")
        from hidden_state_extractor import find_function_bodies
        regions    = find_function_bodies(_EXAMPLE_CODE)
        func_names = ["__init__", "get", "put", "_is_expired", "_hash_key"]
        labels     = func_names[:len(regions)]
        code_samples = [(_EXAMPLE_CODE, regions, labels, {
            'instance_id': 'example',
            'pull_number': 0,
            'file_path':   'example.py',
            'func_names':  labels,
        })]

    total_funcs = sum(len(s[2]) for s in code_samples)
    print(f"\n{len(code_samples)} file group(s), {total_funcs} functions total\n")

    all_records: list[dict] = []
    for model_path in TEACHER_MODELS:
        records = run_model(model_path, code_samples,
                            layer=args.layer, n_gpus=args.n_gpus)
        all_records.extend(records)

    print(f"\nTotal embeddings extracted: {len(all_records)}")

    if args.store:
        from store_embeddings import create_table, store_dicts, build_faiss_index
        conn = pg8000.native.Connection(**DB)
        create_table(conn)
        store_dicts(all_records, conn)
        conn.close()
        seen_tags: set[tuple] = set()
        for r in all_records:
            key = (r['model_name'], r['layer_index'])
            if key not in seen_tags:
                seen_tags.add(key)
                build_faiss_index(args.index_dir,
                                  model_name=r['model_name'],
                                  layer_index=r['layer_index'])
