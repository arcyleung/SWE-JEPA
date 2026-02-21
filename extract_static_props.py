"""
Experiment 0.3: Extract static structural properties for every function
already indexed in function_embeddings, and store them in a new postgres
table `function_static_props`.

Properties extracted via AST (no model required):
  loc                  Lines of code (function body only)
  cyclomatic_complexity  McCabe complexity (decision-point count + 1)
  n_branches           if / elif / match-case / except blocks
  n_loops              for / while statements
  n_returns            return statements
  return_type_cat      'none' | 'primitive' | 'collection' | 'custom'
  has_side_effects     True if the body contains I/O, mutation, or subprocess calls
  n_api_calls          Number of attribute-access calls (obj.method(...))
  n_args               Number of parameters (excluding self/cls)
  has_decorators       Function has at least one decorator
  has_docstring        First statement is a string literal

The script reuses the overlayfs + git-show fetch pipeline from
phase1_similarity_matrix.py so no source code needs to be stored separately.

Usage:
    python extract_static_props.py                 # all PRs
    python extract_static_props.py --limit 500
    python extract_static_props.py --repo owner/name
"""

import ast as _ast
import os
import sys

import pg8000.native
import yaml

# Reuse the PR-fetch helpers from phase 0.1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase0_1_similarity_matrix import (
    DB,
    OVERLAY_MERGED_BASE,
    _build_repo_id_map,
    _load_tokens,
    _mount_overlay,
    _umount_overlay,
    _sha_available,
    _fetch_sha,
    _git_show,
    parse_modified_py_files,
    find_functions_with_names,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PG_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'postgres_connection.yaml')

# ── DDL ────────────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS function_static_props (
    id                   SERIAL PRIMARY KEY,
    instance_id          TEXT    NOT NULL,
    feature_file         TEXT    NOT NULL,
    feature_function     TEXT    NOT NULL,

    loc                  INTEGER,
    cyclomatic_complexity INTEGER,
    n_branches           INTEGER,
    n_loops              INTEGER,
    n_returns            INTEGER,
    return_type_cat      TEXT,
    has_side_effects     BOOLEAN,
    n_api_calls          INTEGER,
    n_args               INTEGER,
    has_decorators       BOOLEAN,
    has_docstring        BOOLEAN,

    UNIQUE (instance_id, feature_file, feature_function)
);
CREATE INDEX IF NOT EXISTS fsp_instance_idx ON function_static_props (instance_id);
CREATE INDEX IF NOT EXISTS fsp_func_idx     ON function_static_props (feature_function);
"""


def create_table(conn=None):
    close = conn is None
    if close:
        conn = pg8000.native.Connection(**DB)
    for stmt in DDL.strip().split(';'):
        stmt = stmt.strip()
        if stmt:
            conn.run(stmt)
    if close:
        conn.close()


# ── AST static analysis ────────────────────────────────────────────────────────

# Names that indicate side effects when called
_SIDE_EFFECT_NAMES = {
    # I/O
    'open', 'print', 'write', 'read', 'readline', 'readlines',
    'flush', 'close', 'seek', 'truncate',
    # subprocess / OS
    'system', 'popen', 'run', 'call', 'check_call', 'check_output', 'Popen',
    'exec', 'execv', 'execve', 'execvp', 'fork', 'spawn',
    # network
    'send', 'sendall', 'recv', 'recvfrom', 'connect', 'bind', 'listen',
    'accept', 'get', 'post', 'put', 'delete', 'patch', 'request',
    # logging / events
    'log', 'debug', 'info', 'warning', 'error', 'critical', 'exception',
    'emit', 'publish', 'push',
    # DB
    'execute', 'executemany', 'commit', 'rollback', 'insert', 'update', 'delete',
}

# Primitive python types for return classification
_PRIMITIVE_TYPES = {'int', 'float', 'str', 'bool', 'bytes', 'complex',
                    'NoneType', 'None', 'True', 'False'}
_COLLECTION_TYPES = {'list', 'dict', 'set', 'tuple', 'frozenset',
                     'List', 'Dict', 'Set', 'Tuple', 'FrozenSet',
                     'Sequence', 'Mapping', 'Iterable', 'Iterator',
                     'Generator', 'deque', 'OrderedDict', 'defaultdict'}


def _cyclomatic(node: _ast.AST) -> int:
    """McCabe cyclomatic complexity: 1 + number of decision points."""
    complexity = 1
    for n in _ast.walk(node):
        if isinstance(n, (_ast.If, _ast.For, _ast.While,
                          _ast.ExceptHandler, _ast.With,
                          _ast.Assert, _ast.IfExp)):
            complexity += 1
        elif isinstance(n, _ast.BoolOp):
            # each 'and'/'or' adds one branch
            complexity += len(n.values) - 1
        elif isinstance(n, (_ast.ListComp, _ast.SetComp,
                             _ast.DictComp, _ast.GeneratorExp)):
            complexity += len(n.generators)
        elif isinstance(n, _ast.Match):
            # match adds one branch per case (minus the default)
            complexity += len(n.cases)
    return complexity


def _count_branches(node: _ast.AST) -> int:
    count = 0
    for n in _ast.walk(node):
        if isinstance(n, _ast.If):
            count += 1 + (1 if n.orelse else 0)
        elif isinstance(n, _ast.ExceptHandler):
            count += 1
        elif isinstance(n, _ast.Match):
            count += len(n.cases)
    return count


def _count_loops(node: _ast.AST) -> int:
    return sum(1 for n in _ast.walk(node)
               if isinstance(n, (_ast.For, _ast.While,
                                  _ast.ListComp, _ast.SetComp,
                                  _ast.DictComp, _ast.GeneratorExp)))


def _count_returns(node: _ast.AST) -> int:
    return sum(1 for n in _ast.walk(node) if isinstance(n, _ast.Return))


def _return_type_cat(node: _ast.FunctionDef | _ast.AsyncFunctionDef) -> str:
    """Classify the return type from annotation or first return statement."""
    ann = node.returns
    if ann is not None:
        name = None
        if isinstance(ann, _ast.Name):
            name = ann.id
        elif isinstance(ann, _ast.Constant) and ann.value is None:
            return 'none'
        elif isinstance(ann, _ast.Attribute):
            name = ann.attr
        if name:
            if name in ('None', 'NoneType') or name == 'None':
                return 'none'
            if name in _PRIMITIVE_TYPES:
                return 'primitive'
            if name in _COLLECTION_TYPES:
                return 'collection'
            return 'custom'

    # Fall back to first return value
    for n in _ast.walk(node):
        if isinstance(n, _ast.Return):
            if n.value is None:
                return 'none'
            if isinstance(n.value, _ast.Constant):
                v = n.value.value
                if v is None:
                    return 'none'
                if isinstance(v, (int, float, str, bool, bytes, complex)):
                    return 'primitive'
            if isinstance(n.value, (_ast.List, _ast.Dict, _ast.Set, _ast.Tuple)):
                return 'collection'
            if isinstance(n.value, _ast.Name):
                if n.value.id in _PRIMITIVE_TYPES:
                    return 'primitive'
                if n.value.id in _COLLECTION_TYPES:
                    return 'collection'
            return 'custom'

    return 'none'  # no return statement → implicitly None


def _has_side_effects(node: _ast.AST) -> bool:
    for n in _ast.walk(node):
        if isinstance(n, _ast.Call):
            # Direct call: open(...), print(...)
            if isinstance(n.func, _ast.Name) and n.func.id in _SIDE_EFFECT_NAMES:
                return True
            # Attribute call: self.write(...), os.system(...), requests.get(...)
            if isinstance(n.func, _ast.Attribute) and n.func.attr in _SIDE_EFFECT_NAMES:
                return True
    return False


def _count_api_calls(node: _ast.AST) -> int:
    """Count calls via attribute access (obj.method()) as proxy for external API use."""
    return sum(
        1 for n in _ast.walk(node)
        if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
    )


def _has_docstring(node: _ast.FunctionDef | _ast.AsyncFunctionDef) -> bool:
    return (bool(node.body)
            and isinstance(node.body[0], _ast.Expr)
            and isinstance(node.body[0].value, _ast.Constant)
            and isinstance(node.body[0].value.value, str))


def _n_args(node: _ast.FunctionDef | _ast.AsyncFunctionDef) -> int:
    args = node.args
    all_args = args.args + args.posonlyargs + args.kwonlyargs
    if args.vararg:
        all_args.append(args.vararg)
    if args.kwarg:
        all_args.append(args.kwarg)
    # strip self / cls
    return sum(1 for a in all_args if a.arg not in ('self', 'cls'))


def analyse_function(
    func_node: _ast.FunctionDef | _ast.AsyncFunctionDef,
    source_lines: list[str],
) -> dict:
    start = func_node.lineno - 1   # 0-indexed
    end   = func_node.end_lineno   # exclusive
    loc   = end - start

    return {
        'loc':                   loc,
        'cyclomatic_complexity': _cyclomatic(func_node),
        'n_branches':            _count_branches(func_node),
        'n_loops':               _count_loops(func_node),
        'n_returns':             _count_returns(func_node),
        'return_type_cat':       _return_type_cat(func_node),
        'has_side_effects':      _has_side_effects(func_node),
        'n_api_calls':           _count_api_calls(func_node),
        'n_args':                _n_args(func_node),
        'has_decorators':        bool(func_node.decorator_list),
        'has_docstring':         _has_docstring(func_node),
    }


def analyse_source(source: str) -> dict[str, dict]:
    """
    Parse source and return {func_name: props_dict} for every function.
    For overloaded names, the last definition wins (consistent with
    find_functions_with_names ordering).
    """
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return {}

    lines = source.splitlines()
    results = {}
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            results[node.name] = analyse_function(node, lines)
    return results


# ── Per-PR worker ──────────────────────────────────────────────────────────────

def _process_pr_props(
    row: tuple,
    gh_token: str | None,
    repo_id_to_dir: dict[int, str],
    target_funcs: set[tuple],   # set of (instance_id, feature_file, feature_function)
) -> list[dict]:
    """
    Mount overlayfs for this PR, read modified .py files, run static analysis,
    return list of property dicts ready for upsert.
    """
    repo_slug, repo_id, pull_number, base_commit, patch, instance_id = row
    instance_id = instance_id or f"{repo_slug}#{pull_number}"

    # Only process this PR if we have functions to analyse for it
    relevant = {(f, fn) for (iid, f, fn) in target_funcs if iid == instance_id}
    if not relevant:
        return []

    py_files = parse_modified_py_files(patch)
    if not py_files:
        return []

    dir_name = repo_id_to_dir.get(repo_id)
    if not dir_name:
        return []

    from phase0_1_similarity_matrix import (
        REPOS_BASE, _mount_overlay, _umount_overlay,
        _sha_available, _fetch_sha, _git_show,
    )

    repo_path = os.path.join(REPOS_BASE, dir_name)
    tag = f'props-{repo_id}-{pull_number}'

    try:
        merged, upper, work = _mount_overlay(repo_path, tag)
    except Exception:
        return []

    records: list[dict] = []
    try:
        if not _sha_available(merged, base_commit):
            _fetch_sha(merged, base_commit, repo_slug, gh_token)
        if not _sha_available(merged, base_commit):
            return records

        for file_path in py_files:
            # Only analyse files we actually need
            needed_funcs = {fn for (fp, fn) in relevant if fp == file_path}
            if not needed_funcs:
                continue

            content = _git_show(merged, base_commit, file_path)
            if content is None:
                continue

            props_by_name = analyse_source(content)
            for func_name, props in props_by_name.items():
                if func_name not in needed_funcs:
                    continue
                records.append({
                    'instance_id':      instance_id,
                    'feature_file':     file_path,
                    'feature_function': func_name,
                    **props,
                })

        if records:
            print(f"  pr#{pull_number} ({repo_slug}): {len(records)} functions analysed",
                  flush=True)
    finally:
        _umount_overlay(merged, upper, work)

    return records


# ── Postgres upsert ────────────────────────────────────────────────────────────

UPSERT_SQL = """
INSERT INTO function_static_props
    (instance_id, feature_file, feature_function,
     loc, cyclomatic_complexity, n_branches, n_loops, n_returns,
     return_type_cat, has_side_effects, n_api_calls,
     n_args, has_decorators, has_docstring)
VALUES
    (:instance_id, :feature_file, :feature_function,
     :loc, :cyclomatic_complexity, :n_branches, :n_loops, :n_returns,
     :return_type_cat, :has_side_effects, :n_api_calls,
     :n_args, :has_decorators, :has_docstring)
ON CONFLICT (instance_id, feature_file, feature_function) DO UPDATE SET
    loc                   = EXCLUDED.loc,
    cyclomatic_complexity = EXCLUDED.cyclomatic_complexity,
    n_branches            = EXCLUDED.n_branches,
    n_loops               = EXCLUDED.n_loops,
    n_returns             = EXCLUDED.n_returns,
    return_type_cat       = EXCLUDED.return_type_cat,
    has_side_effects      = EXCLUDED.has_side_effects,
    n_api_calls           = EXCLUDED.n_api_calls,
    n_args                = EXCLUDED.n_args,
    has_decorators        = EXCLUDED.has_decorators,
    has_docstring         = EXCLUDED.has_docstring
"""


def store_props(records: list[dict], conn=None):
    if not records:
        return
    close = conn is None
    if close:
        conn = pg8000.native.Connection(**DB)
    for r in records:
        conn.run(UPSERT_SQL, **r)
    if close:
        conn.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit',      type=int,  default=None,
                    help='Max PRs to process (default: all)')
    ap.add_argument('--repo',       type=str,  default=None,
                    help='Filter to a specific repo slug')
    ap.add_argument('--create-table', action='store_true',
                    help='Create the table before inserting')
    ap.add_argument('--batch',      type=int,  default=200,
                    help='Rows to upsert per transaction (default 200)')
    args = ap.parse_args()

    conn = pg8000.native.Connection(**DB)

    if args.create_table:
        print("Creating table function_static_props …")
        create_table(conn)
        print("Done.")

    # ── 1. Find all unique (instance_id, feature_file, feature_function) we need
    print("Fetching target functions from function_embeddings …")
    target_rows = conn.run("""
        SELECT DISTINCT instance_id, feature_file, feature_function
        FROM function_embeddings
        WHERE model_name = 'Qwen2.5-Coder-3B'
    """)
    target_funcs: set[tuple] = {tuple(r) for r in target_rows}
    print(f"  {len(target_funcs)} unique functions to analyse")

    # ── 2. Skip already-computed ones
    done_rows = conn.run(
        "SELECT instance_id, feature_file, feature_function FROM function_static_props")
    done_funcs: set[tuple] = {tuple(r) for r in done_rows}
    target_funcs -= done_funcs
    print(f"  {len(done_funcs)} already done → {len(target_funcs)} remaining")

    if not target_funcs:
        print("Nothing to do.")
        conn.close()
        return

    # ── 3. Fetch PR rows for those instance_ids
    needed_instances = tuple({iid for (iid, _, _) in target_funcs})
    sql = (
        "SELECT repo, repo_id, pull_number, base_commit, patch, instance_id "
        "FROM prs "
        "WHERE base_commit IS NOT NULL AND base_commit != '' "
        "  AND patch IS NOT NULL AND patch != '' "
        "  AND instance_id = ANY(:iids)"
    )
    params: dict = {'iids': list(needed_instances)}
    if args.repo:
        sql += ' AND repo = :repo'
        params['repo'] = args.repo
    if args.limit:
        sql += f' LIMIT {args.limit}'

    rows = conn.run(sql, **params)
    conn.close()
    print(f"  {len(rows)} PR rows to process")

    if not rows:
        print("No matching PRs found.")
        return

    tokens = _load_tokens()
    if not tokens:
        tokens = [None]
    repo_id_to_dir = _build_repo_id_map()
    os.makedirs(OVERLAY_MERGED_BASE, exist_ok=True)

    max_workers = min(len(tokens), len(rows), 14)
    print(f"Processing with {max_workers} parallel worker(s) …\n")

    all_records: list[dict] = []
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers,
                             mp_context=mp.get_context('spawn')) as executor:
        futures = {
            executor.submit(
                _process_pr_props,
                tuple(row),
                tokens[i % len(tokens)],
                repo_id_to_dir,
                target_funcs,
            ): row
            for i, row in enumerate(rows)
        }
        conn = pg8000.native.Connection(**DB)
        buf: list[dict] = []

        for future in as_completed(futures):
            completed += 1
            try:
                records = future.result()
                buf.extend(records)
            except Exception as e:
                row = futures[future]
                print(f"  ERROR pr#{row[2]} ({row[0]}): {e}", flush=True)

            # Flush buffer periodically
            if len(buf) >= args.batch or completed == len(rows):
                if buf:
                    store_props(buf, conn)
                    all_records.extend(buf)
                    buf = []

            if completed % 50 == 0 or completed == len(rows):
                print(f"  [{completed}/{len(rows)}] PRs done, "
                      f"{len(all_records)} functions stored so far")

        conn.close()

    print(f"\nDone. {len(all_records)} function_static_props rows written.")


if __name__ == '__main__':
    main()
