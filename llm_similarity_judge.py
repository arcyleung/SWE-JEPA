"""
LLM judge for functional similarity between two Python functions.

Three scoring levels are computed independently, each using a different view
of the functions:

  (a) Signature / naming semantics
        → only the def-line + docstring snippet; implementation hidden
        → tests whether the public contract / naming implies similar purpose

  (b) Body / algorithmic semantics
        → full implementation with the function name replaced by a generic label
          and the docstring stripped
        → tests whether the underlying algorithm is the same, ignoring naming

  (c) Full function (both together)
        → tests overall functional equivalence

Disconnect detection
────────────────────
When |score_a - score_b| ≥ 2 the judge flags a disconnect and categorises it:

  "name_misleads":    score_a is LOW but score_b is HIGH
                      → same algorithm, poor or misleading name
  "name_overclaims":  score_a is HIGH but score_b is LOW
                      → similar-sounding names / signatures, very different code

This is the interesting research signal: it reveals cases where developers
named a function one thing but implemented something different (or vice-versa).

Score scale (0–3)
─────────────────
  0 – unrelated
  1 – weakly related (same broad domain or category)
  2 – moderately similar (same purpose or similar algorithm, notable differences)
  3 – functionally equivalent / same algorithm

Usage
─────
  python llm_similarity_judge.py --func-a a.py --func-b b.py
  python llm_similarity_judge.py --func-a a.py --func-b b.py \\
      --endpoint http://localhost:8000/v1 --model qwen2.5-coder-7b-instruct
  python llm_similarity_judge.py --id-a 42 --id-b 107   # fetch source from postgres
"""

import ast
import argparse
import json
import os
import re
import textwrap
from dataclasses import dataclass

import yaml
import pg8000.native
from openai import OpenAI

PG_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'postgres_connection.yaml')

_pg_cfg = yaml.safe_load(open(PG_CONFIG_FILE))
DB = dict(
    host=_pg_cfg['ip'],
    port=_pg_cfg.get('port') or 9999,
    user=_pg_cfg['user'],
    password=_pg_cfg['password'],
    database=_pg_cfg['database'],
)

SCORE_RUBRIC = """\
Score scale:
  3 – functionally equivalent / same algorithm
  2 – moderately similar (same purpose or similar approach, notable differences)
  1 – weakly related (same broad domain or category)
  0 – unrelated"""


# ── Function source extraction ────────────────────────────────────────────────

def _parse_first_function(source: str):
    """Return the first FunctionDef/AsyncFunctionDef node from source, or None."""
    src = textwrap.dedent(source)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None, src
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node, src
    return None, src


def extract_signature(source: str) -> str:
    """
    Return a stub containing only the def-line and (truncated) docstring,
    with the body replaced by '...'.
    Used for level-a (naming/contract) scoring.
    """
    node, src = _parse_first_function(source)
    if node is None:
        return source.split('\n')[0] + '\n    ...'

    lines = src.splitlines()
    def_line = lines[node.lineno - 1]
    doc = ast.get_docstring(node) or ''
    if doc:
        first_line = doc.split('\n')[0].strip()[:120]
        return f'{def_line}\n    """{first_line}"""\n    ...'
    return f'{def_line}\n    ...'


def redact_for_body_scoring(source: str, anon_name: str = 'func_ANON') -> str:
    """
    Replace the function name with anon_name, strip the docstring.
    Used for level-b (algorithmic) scoring so the LLM cannot use the name.
    """
    node, src = _parse_first_function(source)
    if node is None:
        return re.sub(r'\bdef \w+', f'def {anon_name}', source)

    lines = src.splitlines()

    # def-line with name redacted
    def_line = re.sub(r'\bdef \w+', f'def {anon_name}', lines[node.lineno - 1])

    # Skip docstring if the first body statement is a string literal
    body = node.body
    if (body and isinstance(body[0], ast.Expr) and
            isinstance(body[0].value, ast.Constant) and
            isinstance(body[0].value.value, str)):
        body = body[1:]

    if not body:
        return f'{def_line}\n    pass'

    body_start = body[0].lineno - 1  # 0-indexed line
    body_lines = '\n'.join(lines[body_start:node.end_lineno])
    return f'{def_line}\n{body_lines}'


# ── LLM interaction ───────────────────────────────────────────────────────────

_SYSTEM = "You are a senior software engineer evaluating Python function similarity."


def _build_prompt_a(sig_a: str, sig_b: str) -> str:
    return f"""\
Evaluate the functional similarity of these two Python functions based ONLY on \
their names, parameters, return types, and docstrings — do NOT infer anything \
about their implementations.

Function A (signature only):
```python
{sig_a}
```

Function B (signature only):
```python
{sig_b}
```

{SCORE_RUBRIC}

Respond with a single JSON object:
{{"score": <0-3>, "reasoning": "<one concise sentence>"}}"""


def _build_prompt_b(body_a: str, body_b: str) -> str:
    return f"""\
Evaluate the algorithmic similarity of these two Python functions. The function \
names have been replaced with generic labels so you cannot use naming as a signal. \
Judge based ONLY on what the code actually does.

Function A (name redacted):
```python
{body_a}
```

Function B (name redacted):
```python
{body_b}
```

{SCORE_RUBRIC}

Respond with a single JSON object:
{{"score": <0-3>, "reasoning": "<one concise sentence>"}}"""


def _build_prompt_c(full_a: str, full_b: str) -> str:
    return f"""\
Evaluate the overall functional similarity of these two Python functions, \
considering both their intent (names, docstrings) and their implementation.

Function A:
```python
{full_a}
```

Function B:
```python
{full_b}
```

{SCORE_RUBRIC}

Respond with a single JSON object:
{{"score": <0-3>, "reasoning": "<one concise sentence>"}}"""


def _call_llm(client: OpenAI, model: str, prompt: str) -> dict:
    """Call the LLM and parse the JSON response. Returns {'score': int, 'reasoning': str}."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    raw = response.choices[0].message.content.strip()

    # Tolerate LLMs that wrap JSON in a markdown code fence
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if not m:
        raise ValueError(f"LLM response did not contain JSON: {raw!r}")
    data = json.loads(m.group(0))
    return {'score': int(data['score']), 'reasoning': data.get('reasoning', '')}


# ── Main judge ────────────────────────────────────────────────────────────────

@dataclass
class SimilarityResult:
    score_a:    int    # signature / naming  (0-3)
    score_b:    int    # body / algorithmic  (0-3)
    score_c:    int    # full function       (0-3)
    reasoning_a: str
    reasoning_b: str
    reasoning_c: str
    disconnect:       bool   # |score_a - score_b| >= 2
    disconnect_type:  str    # "name_misleads" | "name_overclaims" | ""
    disconnect_note:  str    # human-readable explanation


def _disconnect_analysis(score_a: int, score_b: int) -> tuple[bool, str, str]:
    gap = abs(score_a - score_b)
    if gap < 2:
        return False, '', ''
    if score_b > score_a:
        dtype = 'name_misleads'
        note  = (f"Algorithms are similar (body score={score_b}) but the names "
                 f"suggest different purposes (signature score={score_a}). "
                 f"Classic symptom of poor or misleading function naming.")
    else:
        dtype = 'name_overclaims'
        note  = (f"Names/signatures suggest similar purpose (signature score={score_a}) "
                 f"but the implementations differ significantly (body score={score_b}). "
                 f"The name may be overloaded or reused for different behaviour.")
    return True, dtype, note


def judge_pair(
    func_a: str,
    func_b: str,
    client: OpenAI,
    model: str,
) -> SimilarityResult:
    """Score a pair of functions at all three levels of granularity."""
    sig_a  = extract_signature(func_a)
    sig_b  = extract_signature(func_b)
    body_a = redact_for_body_scoring(func_a, 'func_A')
    body_b = redact_for_body_scoring(func_b, 'func_B')

    res_a = _call_llm(client, model, _build_prompt_a(sig_a,   sig_b))
    res_b = _call_llm(client, model, _build_prompt_b(body_a,  body_b))
    res_c = _call_llm(client, model, _build_prompt_c(func_a,  func_b))

    disconnect, dtype, note = _disconnect_analysis(res_a['score'], res_b['score'])

    return SimilarityResult(
        score_a=res_a['score'],  reasoning_a=res_a['reasoning'],
        score_b=res_b['score'],  reasoning_b=res_b['reasoning'],
        score_c=res_c['score'],  reasoning_c=res_c['reasoning'],
        disconnect=disconnect,
        disconnect_type=dtype,
        disconnect_note=note,
    )


def print_result(result: SimilarityResult, label_a: str = 'A', label_b: str = 'B'):
    bar = '─' * 64
    print(f"\n{bar}")
    print(f"  Similarity: {label_a!r}  vs  {label_b!r}")
    print(bar)
    print(f"  (a) Signature / naming  : {result.score_a}/3  — {result.reasoning_a}")
    print(f"  (b) Body / algorithmic  : {result.score_b}/3  — {result.reasoning_b}")
    print(f"  (c) Overall (both)      : {result.score_c}/3  — {result.reasoning_c}")
    if result.disconnect:
        print(f"\n  ⚠  DISCONNECT [{result.disconnect_type}]")
        print(f"     {result.disconnect_note}")
    print(bar)


# ── Postgres source retrieval (for --id-a / --id-b mode) ─────────────────────

def fetch_source_by_id(pg_id: int) -> tuple[str, str]:
    """
    Retrieve the source code for a function stored in function_embeddings.
    We re-read it from the repo via git show (requires the repo to be accessible).
    Falls back to returning a placeholder stub so the script doesn't crash.

    Returns (source_code, display_label).
    """
    conn = pg8000.native.Connection(**DB)
    rows = conn.run(
        "SELECT instance_id, feature_file, feature_function "
        "FROM function_embeddings WHERE id = :id", id=pg_id)
    conn.close()
    if not rows:
        raise ValueError(f"No row with id={pg_id}")
    instance_id, feature_file, feature_function = rows[0]
    label = f"{feature_function} ({os.path.basename(feature_file)})"

    # Try to retrieve from the prs table and git
    try:
        conn2 = pg8000.native.Connection(**DB)
        pr_rows = conn2.run(
            "SELECT repo, repo_id, base_commit FROM prs WHERE instance_id = :iid",
            iid=instance_id)
        conn2.close()
        if pr_rows:
            import subprocess, os as _os
            repo_slug, repo_id, base_commit = pr_rows[0]
            repos_base = '/shared_workspace_mfs/repos'
            for entry in _os.listdir(repos_base):
                if entry.startswith(str(repo_id) + '_'):
                    repo_path = _os.join(repos_base, entry)
                    r = subprocess.run(
                        ['git', '-C', repo_path, 'show', f'{base_commit}:{feature_file}'],
                        capture_output=True, timeout=30)
                    if r.returncode == 0:
                        full_source = r.stdout.decode(errors='replace')
                        # Extract just the specific function
                        src = _extract_named_function(full_source, feature_function)
                        if src:
                            return src, label
    except Exception:
        pass

    return f"# Could not retrieve source for {label}\ndef {feature_function}(): ...", label


def _extract_named_function(file_source: str, func_name: str) -> str | None:
    """Extract the source of a named function from a file's full source."""
    try:
        tree = ast.parse(file_source)
    except SyntaxError:
        return None
    lines = file_source.splitlines()
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == func_name):
            return '\n'.join(lines[node.lineno - 1: node.end_lineno])
    return None


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Score functional similarity between two Python functions at "
                    "three levels of granularity (signature, body, both)")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--func-a', type=str,
                     help='Path to a .py file containing function A')
    src.add_argument('--id-a',   type=int,
                     help='Postgres row ID of function A in function_embeddings')

    src2 = ap.add_mutually_exclusive_group(required=True)
    src2.add_argument('--func-b', type=str,
                      help='Path to a .py file containing function B')
    src2.add_argument('--id-b',   type=int,
                      help='Postgres row ID of function B in function_embeddings')

    ap.add_argument('--endpoint', type=str, default='https://api.openai.com/v1',
                    help='OpenAI-compatible API base URL '
                         '(default: https://api.openai.com/v1)')
    ap.add_argument('--api-key',  type=str, default=os.environ.get('OPENAI_API_KEY', 'EMPTY'),
                    help='API key (default: $OPENAI_API_KEY or "EMPTY" for local endpoints)')
    ap.add_argument('--model',    type=str, default='gpt-4o-mini',
                    help='Model name to use for judging (default: gpt-4o-mini)')
    ap.add_argument('--json',     action='store_true',
                    help='Output results as JSON instead of human-readable text')
    args = ap.parse_args()

    # Load function sources
    if args.func_a:
        source_a = open(args.func_a).read()
        label_a  = os.path.basename(args.func_a)
    else:
        source_a, label_a = fetch_source_by_id(args.id_a)

    if args.func_b:
        source_b = open(args.func_b).read()
        label_b  = os.path.basename(args.func_b)
    else:
        source_b, label_b = fetch_source_by_id(args.id_b)

    # Build client
    client = OpenAI(base_url=args.endpoint, api_key=args.api_key)

    # Run judge
    result = judge_pair(source_a, source_b, client, args.model)

    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        print_result(result, label_a, label_b)

        # Show what the LLM saw for each level (useful for debugging prompts)
        print("\n── What the LLM saw at each level ──────────────────────────────")
        print("\n[a] Signatures shown to judge:")
        print("  A:", extract_signature(source_a))
        print("  B:", extract_signature(source_b))
        print("\n[b] Redacted bodies shown to judge:")
        print("  A:", redact_for_body_scoring(source_a, 'func_A')[:120], "...")
        print("  B:", redact_for_body_scoring(source_b, 'func_B')[:120], "...")


if __name__ == '__main__':
    main()
