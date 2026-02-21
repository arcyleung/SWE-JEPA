"""
K-nearest-neighbour retrieval from the FAISS function embedding index.

The index is built by store_embeddings.build_faiss_index() and is paired with
a companion .npy file that maps FAISS positions → postgres row IDs.

Query modes
───────────
  --id N          Look up the embedding for postgres row N and find its neighbours
  --code file.py  Extract embeddings for all functions in the file on-the-fly,
                  then query each one (uses the same teacher model used to build
                  the index; defaults to Qwen2.5-Coder-3B @ layer 18)

Examples
────────
  python knn_query.py --id 42 --k 10
  python knn_query.py --id 42 --k 10 --tag _Qwen2.5-Coder-3B_L18
  python knn_query.py --code snippet.py --k 5 --model /home/original_models/Qwen2.5-Coder-3B
"""

import argparse
import os

import numpy as np
import yaml
import pg8000.native
import faiss

PG_CONFIG_FILE    = os.path.join(os.path.dirname(__file__), 'postgres_connection.yaml')
DEFAULT_INDEX_DIR = os.path.join(os.path.dirname(__file__), 'faiss_indices')
DEFAULT_MODEL     = '/home/original_models/Qwen2.5-Coder-3B'
DEFAULT_LAYER     = 18

_pg_cfg = yaml.safe_load(open(PG_CONFIG_FILE))
DB = dict(
    host=_pg_cfg['ip'],
    port=_pg_cfg.get('port') or 9999,
    user=_pg_cfg['user'],
    password=_pg_cfg['password'],
    database=_pg_cfg['database'],
)


# ── Index I/O ─────────────────────────────────────────────────────────────────

def load_index(index_dir: str, tag: str = '') -> tuple[faiss.Index, np.ndarray]:
    """
    Load a FAISS index and its companion postgres-ID array.

    tag examples:  ''  (no filter)
                   '_Qwen2.5-Coder-3B_L18'
    """
    index_path = os.path.join(index_dir, f'function_embeddings{tag}.faiss')
    ids_path   = os.path.join(index_dir, f'function_embeddings{tag}_ids.npy')
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}\n"
            f"Run: python store_embeddings.py --build-index")
    index = faiss.read_index(index_path)
    ids   = np.load(ids_path)
    print(f"Loaded FAISS index: {index.ntotal} vectors (dim={index.d})")
    return index, ids


# ── Postgres helpers ───────────────────────────────────────────────────────────

def fetch_embedding_by_id(pg_id: int) -> np.ndarray:
    conn = pg8000.native.Connection(**DB)
    rows = conn.run(
        "SELECT embedding FROM function_embeddings WHERE id = :id", id=pg_id)
    conn.close()
    if not rows:
        raise ValueError(f"No row with id={pg_id} in function_embeddings")
    return np.array(rows[0][0], dtype=np.float32)


def fetch_metadata(pg_ids: list[int]) -> list[dict]:
    """Fetch display metadata for a list of postgres row IDs, preserving order."""
    if not pg_ids:
        return []
    conn    = pg8000.native.Connection(**DB)
    id_list = ', '.join(str(i) for i in pg_ids)
    rows    = conn.run(
        f"SELECT id, instance_id, feature_file, feature_function, model_name, layer_index "
        f"FROM function_embeddings WHERE id IN ({id_list})")
    conn.close()
    by_id = {r[0]: r for r in rows}
    return [
        {
            'id':               by_id[i][0],
            'instance_id':      by_id[i][1],
            'feature_file':     by_id[i][2],
            'feature_function': by_id[i][3],
            'model_name':       by_id[i][4],
            'layer_index':      by_id[i][5],
        }
        for i in pg_ids if i in by_id
    ]


# ── On-the-fly embedding for a code snippet ───────────────────────────────────

def embed_file(path: str, model_path: str, layer: int) -> list[tuple[np.ndarray, str]]:
    """
    Extract mean-pooled embeddings for every function in a Python file.
    Returns [(embedding_array, func_name), ...].
    """
    import torch
    from hidden_state_extractor import load_teacher, extract_hidden_states
    from phase1_similarity_matrix import find_functions_with_names

    source = open(path).read()
    pairs  = find_functions_with_names(source)
    if not pairs:
        print(f"No functions found in {path}")
        return []

    regions    = [r for r, _ in pairs]
    func_names = [n for _, n in pairs]

    model, tokenizer = load_teacher(model_path)
    targets = extract_hidden_states(
        model, tokenizer, source, regions, layer=layer, pool_strategy="mean")
    del model
    torch.cuda.empty_cache()

    return [(t.hidden_states.squeeze().float().numpy(), name)
            for t, name in zip(targets, func_names)]


# ── Core search ───────────────────────────────────────────────────────────────

def search(query_emb: np.ndarray, index: faiss.Index, ids: np.ndarray,
           k: int) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalise the query and run inner-product search. Returns (sims, pg_ids)."""
    q              = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    sims, positions = index.search(q.reshape(1, -1).astype(np.float32), k)
    return sims[0], ids[positions[0]]


# ── Display ───────────────────────────────────────────────────────────────────

def print_results(query_label: str, sims: np.ndarray, metadata: list[dict]):
    print(f"\nNearest neighbours for: {query_label}")
    print(f"{'Rank':<5} {'Sim':>7}  {'Function':<30} {'File':<30} {'instance_id'}")
    print("─" * 100)
    for rank, (sim, meta) in enumerate(zip(sims, metadata), 1):
        func  = meta['feature_function'][:28]
        ffile = os.path.basename(meta['feature_file'])[:28]
        iid   = meta['instance_id'][:35]
        print(f"{rank:<5} {sim:>7.4f}  {func:<30} {ffile:<30} {iid}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="K-NN query against function embedding index")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--id',   type=int, help='Query by postgres row ID')
    src.add_argument('--code', type=str, help='Path to Python file (embed on-the-fly)')
    ap.add_argument('--k',          type=int, default=10,
                    help='Number of neighbours to return (default 10)')
    ap.add_argument('--index-dir',  type=str, default=DEFAULT_INDEX_DIR)
    ap.add_argument('--tag',        type=str, default='',
                    help='Index tag suffix, e.g. _Qwen2.5-Coder-3B_L18')
    ap.add_argument('--model',      type=str, default=DEFAULT_MODEL,
                    help='Teacher model path (only used with --code)')
    ap.add_argument('--layer',      type=int, default=DEFAULT_LAYER,
                    help='Extraction layer (only used with --code)')
    args = ap.parse_args()

    index, ids = load_index(args.index_dir, args.tag)

    if args.id is not None:
        query_emb  = fetch_embedding_by_id(args.id)
        # Fetch one extra so we can drop the query itself (sim == 1.0)
        sims, nbr_ids = search(query_emb, index, ids, args.k + 1)
        mask       = nbr_ids != args.id
        sims       = sims[mask][:args.k]
        nbr_ids    = nbr_ids[mask][:args.k]
        metadata   = fetch_metadata(nbr_ids.tolist())
        print_results(f"id={args.id}", sims, metadata)

    else:
        snippets = embed_file(args.code, args.model, args.layer)
        if not snippets:
            return
        for emb, func_name in snippets:
            sims, nbr_ids = search(emb, index, ids, args.k)
            metadata = fetch_metadata(nbr_ids.tolist())
            print_results(f"{os.path.basename(args.code)}::{func_name}", sims, metadata)


if __name__ == '__main__':
    main()
