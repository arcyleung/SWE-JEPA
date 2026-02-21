"""
Persist function embeddings to postgres and build a FAISS index.

Postgres table schema
─────────────────────
  function_embeddings(
      id               SERIAL PRIMARY KEY,
      instance_id      TEXT,    -- PR instance ID
      feature_file     TEXT,    -- file path inside the repo
      feature_function TEXT,    -- bare function name
      model_name       TEXT,    -- e.g. "Qwen2.5-Coder-3B"
      layer_index      INTEGER, -- extraction layer
      embedding        REAL[]   -- mean-pooled hidden state (float32)
  )

FAISS index
───────────
  One IndexFlatIP index per (model_name, layer_index) pair, saved as:
    {index_dir}/function_embeddings_{model}_{layer}.faiss
    {index_dir}/function_embeddings_{model}_{layer}_ids.npy  ← postgres row IDs

  Embeddings are L2-normalised before insertion so inner product == cosine sim.

Standalone usage
────────────────
  python store_embeddings.py --create-table
  python store_embeddings.py --build-index --model Qwen2.5-Coder-3B --layer 18
  python store_embeddings.py --create-table --build-index
"""

import os

import numpy as np
import yaml
import pg8000.native

PG_CONFIG_FILE    = os.path.join(os.path.dirname(__file__), 'postgres_connection.yaml')
DEFAULT_INDEX_DIR = os.path.join(os.path.dirname(__file__), 'faiss_indices')

_pg_cfg = yaml.safe_load(open(PG_CONFIG_FILE))
DB = dict(
    host=_pg_cfg['ip'],
    port=_pg_cfg.get('port') or 9999,
    user=_pg_cfg['user'],
    password=_pg_cfg['password'],
    database=_pg_cfg['database'],
)

# ── DDL ───────────────────────────────────────────────────────────────────────

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS function_embeddings (
    id               SERIAL  PRIMARY KEY,
    instance_id      TEXT    NOT NULL,
    feature_file     TEXT    NOT NULL,
    feature_function TEXT    NOT NULL,
    model_name       TEXT    NOT NULL,
    layer_index      INTEGER NOT NULL,
    embedding        REAL[]  NOT NULL,
    UNIQUE (instance_id, feature_file, feature_function, model_name, layer_index)
)
"""

_CREATE_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_fe_model_layer "
    "ON function_embeddings (model_name, layer_index)"
)

_UPSERT_SQL = """
INSERT INTO function_embeddings
    (instance_id, feature_file, feature_function, model_name, layer_index, embedding)
VALUES
    (:instance_id, :feature_file, :feature_function, :model_name, :layer_index, :embedding)
ON CONFLICT (instance_id, feature_file, feature_function, model_name, layer_index)
DO UPDATE SET embedding = EXCLUDED.embedding
"""


def create_table(conn=None):
    """Create the function_embeddings table and index (idempotent)."""
    own = conn is None
    if own:
        conn = pg8000.native.Connection(**DB)
    conn.run(_CREATE_TABLE_SQL)
    conn.run(_CREATE_IDX_SQL)
    if own:
        conn.close()


def store_dicts(records: list[dict], conn=None):
    """
    Upsert a list of embedding dicts into postgres.

    Each dict must have keys:
        instance_id, feature_file, feature_function,
        model_name, layer_index, embedding (np.ndarray shape (D,))
    """
    if not records:
        return
    own = conn is None
    if own:
        conn = pg8000.native.Connection(**DB)
    for rec in records:
        conn.run(
            _UPSERT_SQL,
            instance_id=rec['instance_id'],
            feature_file=rec['feature_file'],
            feature_function=rec['feature_function'],
            model_name=rec['model_name'],
            layer_index=rec['layer_index'],
            embedding=rec['embedding'].tolist(),
        )
    if own:
        conn.close()
    print(f"Stored {len(records)} embedding records to postgres")


# ── FAISS index build ─────────────────────────────────────────────────────────

def build_faiss_index(
    output_dir: str = DEFAULT_INDEX_DIR,
    model_name: str | None = None,
    layer_index: int | None = None,
) -> str:
    """
    Load embeddings from postgres for a given (model_name, layer_index),
    L2-normalise them, build a FAISS IndexFlatIP (inner product = cosine sim
    on normalised vectors), and write two files to output_dir:

        function_embeddings_{tag}.faiss   — the FAISS index
        function_embeddings_{tag}_ids.npy — int64 array of postgres row IDs
                                            in the same order as the index

    Returns the path to the .faiss file, or '' if no embeddings were found.
    """
    import faiss  # lazy import so the module is usable without faiss installed

    conn = pg8000.native.Connection(**DB)
    sql  = ("SELECT id, embedding FROM function_embeddings"
            " WHERE TRUE")
    params: dict = {}
    if model_name:
        sql += " AND model_name = :model_name"
        params['model_name'] = model_name
    if layer_index is not None:
        sql += " AND layer_index = :layer_index"
        params['layer_index'] = layer_index
    sql += " ORDER BY id"

    rows = conn.run(sql, **params)
    conn.close()

    if not rows:
        print("No embeddings found in postgres — nothing to index")
        return ''

    ids  = np.array([r[0] for r in rows], dtype=np.int64)
    embs = np.array([r[1] for r in rows], dtype=np.float32)   # (N, D)

    # L2-normalise so inner product equals cosine similarity
    norms     = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_norm = embs / (norms + 1e-9)

    D     = embs_norm.shape[1]
    index = faiss.IndexFlatIP(D)
    index.add(embs_norm)

    # Build a file-name tag that encodes (model, layer)
    parts: list[str] = []
    if model_name:
        parts.append(model_name.replace('/', '_'))
    if layer_index is not None:
        parts.append(f"L{layer_index}")
    tag = ('_' + '_'.join(parts)) if parts else ''

    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, f'function_embeddings{tag}.faiss')
    ids_path   = os.path.join(output_dir, f'function_embeddings{tag}_ids.npy')

    faiss.write_index(index, index_path)
    np.save(ids_path, ids)

    print(f"FAISS index [{index.ntotal} vectors, dim={D}] → {index_path}")
    return index_path


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Create function_embeddings table and/or rebuild FAISS index")
    ap.add_argument('--create-table', action='store_true',
                    help='Create (or verify) the postgres table')
    ap.add_argument('--build-index', action='store_true',
                    help='Build FAISS index from rows currently in postgres')
    ap.add_argument('--model',     type=str, default=None,
                    help='Filter by model_name when building index')
    ap.add_argument('--layer',     type=int, default=None,
                    help='Filter by layer_index when building index')
    ap.add_argument('--index-dir', type=str, default=DEFAULT_INDEX_DIR,
                    help=f'Output directory for FAISS files (default: {DEFAULT_INDEX_DIR})')
    args = ap.parse_args()

    if not args.create_table and not args.build_index:
        ap.print_help()
    if args.create_table:
        create_table()
        print("Table created (or already exists)")
    if args.build_index:
        build_faiss_index(args.index_dir, model_name=args.model,
                          layer_index=args.layer)
