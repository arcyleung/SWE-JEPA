#!/usr/bin/env python3
"""
Extract function-level follow-up signal from prs_copy (Python PRs).

For each PR in prs_copy WHERE primary_language='Python', extracts the set of
Python functions/classes touched (from `@@ ... @@` hunk context headers), then
computes per-function follow-up metrics within each repo:

  churn_count              — distinct PRs touching this function
  followup_rate            — fraction followed by another PR on the same function
                             within `window_days` (default 90)
  bugfix_followup_rate     — same but restricted to follow-up being bugfix/maintenance
  mean_days_to_followup

Python hunk headers look like:
  @@ -10,5 +10,8 @@ def my_function(self, x):
  @@ -10,5 +10,8 @@ class MyClass(Base):
  @@ -10,5 +10,8 @@                         ← no context (module-level code)

Usage:
    source .venv/bin/activate
    python extract_py_followup_function.py \\
        --language Python \\
        --out data/py_followup_function_index.json \\
        --window-days 90

    # Restrict to specific repos via a JSONL with instance_id field:
    python extract_py_followup_function.py \\
        --repos-from data/phase4_7_agentic_eval_results_feature_sl80.jsonl \\
        --out data/py_followup_function_index_evalrepos.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone

import psycopg2
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))

FILE_RE        = re.compile(r'^\+\+\+ b/(.+)$', re.M)
HUNK_CTX_RE    = re.compile(r'^@@[^@\n]+@@[ \t]*(.*)', re.M)
# Match Python def / async def / class from hunk context
PY_FUNC_RE     = re.compile(r'(?:async\s+)?def\s+(\w+)\s*[\(:]|class\s+(\w+)\s*[\(:]')
# Python package = top-level directory of path (mirrors Go _go_package logic)


def _pg_connect(pg_yaml: str):
    cfg = yaml.safe_load(open(pg_yaml))
    return psycopg2.connect(
        host=cfg["ip"], port=cfg["port"],
        user=cfg["user"], password=cfg["password"],
        database=cfg["database"],
    )


def _py_package(filepath: str) -> str:
    parts = filepath.split("/")
    return parts[0] if len(parts) > 1 else ""


def _extract_func_name(ctx: str) -> str | None:
    ctx = ctx.strip()
    if not ctx:
        return None
    m = PY_FUNC_RE.search(ctx)
    if m:
        return m.group(1) or m.group(2)
    # Fallback: truncated context (module-level code, decorator lines, etc.)
    clean = re.sub(r'\s+', ' ', ctx)[:80]
    return clean if clean else None


def _parse_patch_functions(patch: str) -> dict[str, set[str]]:
    """Return {file: set_of_function_or_class_names} from a unified diff."""
    file_funcs: dict[str, set[str]] = defaultdict(set)
    current_file: str | None = None
    for line in patch.splitlines():
        fm = FILE_RE.match(line)
        if fm:
            current_file = fm.group(1)
            continue
        hm = HUNK_CTX_RE.match(line)
        if hm and current_file:
            name = _extract_func_name(hm.group(1))
            if name:
                file_funcs[current_file].add(name)
    return dict(file_funcs)


def _parse_instance_id(instance_id: str) -> tuple[str, int] | None:
    """
    Convert instance_id → (repo, pull_number).
    Formats:
      'owner__repo__pr_number'            → ('owner/repo', pr_number)
      'owner__repo__pr_number__prNNNN'    → ('owner/repo', pr_number)
    """
    parts = instance_id.split("__")
    if len(parts) < 3:
        return None
    repo = f"{parts[0]}/{parts[1]}"
    # Try parts[2] first (the numeric PR number), then parts[-1] as fallback
    for candidate in (parts[2], parts[-1]):
        cleaned = candidate.lstrip("pr")
        try:
            return repo, int(cleaned)
        except ValueError:
            continue
    return None


def _fetch_repo_prs(conn, repo: str, language: str) -> list[dict]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT pull_number, patch, merged_at,
               COALESCE(pr_category, 'unknown')       AS pr_category,
               COALESCE(pr_author,   'unknown')       AS pr_author
        FROM prs_copy
        WHERE repo = %s
          AND primary_language = %s
          AND patch IS NOT NULL
        ORDER BY pull_number
        """,
        (repo, language),
    )
    rows = []
    for pull_number, patch, merged_at, category, author in cur.fetchall():
        file_funcs = _parse_patch_functions(patch)
        if not file_funcs:
            continue
        rows.append({
            "pull_number": pull_number,
            "file_funcs":  file_funcs,
            "merged_at":   merged_at,
            "pr_category": category,
            "pr_author":   author,
        })
    return rows


def _compute_repo_function_index(prs: list[dict], window_days: int) -> dict:
    """
    Returns {file: {func: {churn_count, followup_rate, bugfix_followup_rate,
                            mean_days_to_followup}}}
    """
    func_touches: dict[tuple[str, str], list[tuple[int, datetime | None, str]]] = defaultdict(list)
    for pr in prs:
        for f, funcs in pr["file_funcs"].items():
            for fn in funcs:
                func_touches[(f, fn)].append(
                    (pr["pull_number"], pr["merged_at"], pr["pr_category"])
                )

    index: dict[str, dict[str, dict]] = defaultdict(dict)
    for (f, fn), touches in func_touches.items():
        churn_count = len(touches)
        if churn_count < 2:
            followup_rate = bugfix_followup_rate = 0.0
            mean_days = None
        else:
            touches_sorted = sorted(touches, key=lambda x: x[0])
            followup_gaps: list[float] = []
            has_followup = has_bugfix = 0
            for i, (pr_num, merged_at, _) in enumerate(touches_sorted[:-1]):
                next_pr_num, next_merged_at, next_cat = touches_sorted[i + 1]
                if merged_at is not None and next_merged_at is not None:
                    t0 = merged_at    if merged_at.tzinfo    else merged_at.replace(tzinfo=timezone.utc)
                    t1 = next_merged_at if next_merged_at.tzinfo else next_merged_at.replace(tzinfo=timezone.utc)
                    gap = (t1 - t0).total_seconds() / 86400
                    if gap <= window_days:
                        has_followup += 1
                        followup_gaps.append(gap)
                        if next_cat in ("bugfix", "maintenance"):
                            has_bugfix += 1
                else:
                    if next_pr_num - pr_num <= 20:
                        has_followup += 1
                        if next_cat in ("bugfix", "maintenance"):
                            has_bugfix += 1
            denom = churn_count - 1
            followup_rate      = has_followup / denom if denom else 0.0
            bugfix_followup_rate = has_bugfix  / denom if denom else 0.0
            mean_days = sum(followup_gaps) / len(followup_gaps) if followup_gaps else None

        index[f][fn] = {
            "churn_count":           churn_count,
            "followup_rate":         round(followup_rate, 4),
            "bugfix_followup_rate":  round(bugfix_followup_rate, 4),
            "mean_days_to_followup": round(mean_days, 1) if mean_days is not None else None,
        }
    return dict(index)


def _score_patch_functions(patch: str, repo: str, index: dict) -> dict:
    file_funcs = _parse_patch_functions(patch)
    if not file_funcs:
        return {"n_files": 0, "n_funcs": 0,
                "mean_func_followup_rate": 0.0,
                "mean_func_bugfix_followup_rate": 0.0,
                "mean_func_churn": 0.0,
                "max_func_followup_rate": 0.0}

    repo_idx = index.get(repo, {})
    fr_vals = bfr_vals = ch_vals = []
    fr_vals, bfr_vals, ch_vals = [], [], []
    per_func = []
    for f, funcs in file_funcs.items():
        file_idx = repo_idx.get(f, {})
        for fn in funcs:
            fm = file_idx.get(fn, {})
            fr  = fm.get("followup_rate", 0.0)
            bfr = fm.get("bugfix_followup_rate", 0.0)
            ch  = fm.get("churn_count", 0)
            fr_vals.append(fr); bfr_vals.append(bfr); ch_vals.append(ch)
            per_func.append({"file": f, "func": fn,
                              "followup_rate": fr, "bugfix_followup_rate": bfr,
                              "churn_count": ch})

    n = len(fr_vals)
    n_files = len(file_funcs)
    if n == 0:
        return {"n_files": n_files, "n_funcs": 0,
                "mean_func_followup_rate": 0.0,
                "mean_func_bugfix_followup_rate": 0.0,
                "mean_func_churn": 0.0,
                "max_func_followup_rate": 0.0}

    return {
        "n_files":  n_files, "n_funcs": n,
        "mean_func_followup_rate":       round(sum(fr_vals)  / n, 4),
        "mean_func_bugfix_followup_rate": round(sum(bfr_vals) / n, 4),
        "mean_func_churn":               round(sum(ch_vals)  / n, 2),
        "max_func_followup_rate":        round(max(fr_vals), 4),
        "per_func": per_func,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--language",    default="Python",
                    help="primary_language filter in prs_copy")
    ap.add_argument("--repos-from",  default=None,
                    help="JSONL with instance_id field to restrict repos; "
                         "omit to use ALL repos for the given language")
    ap.add_argument("--out",         required=True,
                    help="Output JSON: repo → file → func → metrics")
    ap.add_argument("--window-days", type=int, default=90)
    ap.add_argument("--pg-yaml",     default=os.path.join(ROOT, "postgres_connection.yaml"))
    ap.add_argument("--eval-patches-baseline", default=None)
    ap.add_argument("--eval-patches-steered",  default=None)
    ap.add_argument("--iid-list",    default=None,
                    help="JSONL with instance_id column to restrict eval pairs")
    args = ap.parse_args()

    conn = _pg_connect(args.pg_yaml)
    cur  = conn.cursor()

    # ── Determine target repos ───────────────────────────────────────────────
    if args.repos_from:
        ref_rows = [json.loads(l) for l in open(args.repos_from) if l.strip()]
        iids     = [r["instance_id"] for r in ref_rows if "instance_id" in r]
        repo_set: set[str] = set()
        for iid in iids:
            parsed = _parse_instance_id(iid)
            if parsed:
                repo_set.add(parsed[0])
        repos = sorted(repo_set)
        print(f"Target repos from {args.repos_from}: {len(repos)}", flush=True)
    else:
        cur.execute(
            "SELECT DISTINCT repo FROM prs_copy WHERE primary_language=%s ORDER BY repo",
            (args.language,),
        )
        repos = [r[0] for r in cur.fetchall()]
        print(f"All {args.language} repos in prs_copy: {len(repos)}", flush=True)

    # ── Build function follow-up index ───────────────────────────────────────
    index: dict[str, dict] = {}
    for i, repo in enumerate(repos):
        prs = _fetch_repo_prs(conn, repo, args.language)
        if not prs:
            print(f"  [{i+1}/{len(repos)}] {repo}: 0 PRs — skipping", flush=True)
            continue
        repo_index = _compute_repo_function_index(prs, args.window_days)
        index[repo] = repo_index
        n_files  = len(repo_index)
        n_funcs  = sum(len(fd) for fd in repo_index.values())
        n_fol    = sum(1 for fd in repo_index.values()
                       for fm in fd.values() if fm["followup_rate"] > 0)
        print(
            f"  [{i+1}/{len(repos)}] {repo}: {len(prs)} PRs, "
            f"{n_files} files, {n_funcs} funcs, {n_fol} with follow-up signal",
            flush=True,
        )

    conn.close()

    with open(args.out, "w") as f:
        json.dump(index, f, indent=2)
    n_total = sum(len(fd) for ri in index.values() for fd in ri.values())
    print(f"\nIndex saved → {args.out}")
    print(f"Repos: {len(index)}, functions tracked: {n_total:,}")

    # ── Optional: eval steered vs baseline ──────────────────────────────────
    if args.eval_patches_baseline and args.eval_patches_steered:
        print("\n── Scoring steered vs baseline (function-level) ──")
        iid_filter = None
        if args.iid_list:
            iid_filter = set()
            for l in open(args.iid_list):
                if l.strip():
                    r = json.loads(l)
                    if "instance_id" in r:
                        iid_filter.add(r["instance_id"])

        results = []
        for fname in sorted(os.listdir(args.eval_patches_baseline)):
            if not fname.endswith(".patch"):
                continue
            iid = fname[:-6]
            if iid_filter and iid not in iid_filter:
                continue
            parsed = _parse_instance_id(iid)
            if not parsed:
                continue
            repo, _ = parsed
            if repo not in index:
                continue

            b_patch = open(os.path.join(args.eval_patches_baseline, fname)).read()
            s_path  = os.path.join(args.eval_patches_steered, fname)
            if not os.path.exists(s_path):
                continue
            s_patch = open(s_path).read()

            b = _score_patch_functions(b_patch, repo, index)
            s = _score_patch_functions(s_patch, repo, index)
            results.append({
                "instance_id": iid, "repo": repo,
                "b_mean_func_followup_rate":        b["mean_func_followup_rate"],
                "s_mean_func_followup_rate":        s["mean_func_followup_rate"],
                "b_mean_func_bugfix_followup_rate": b["mean_func_bugfix_followup_rate"],
                "s_mean_func_bugfix_followup_rate": s["mean_func_bugfix_followup_rate"],
                "b_mean_func_churn":  b["mean_func_churn"],
                "s_mean_func_churn":  s["mean_func_churn"],
                "steered_lower_followup":       s["mean_func_followup_rate"]        < b["mean_func_followup_rate"],
                "steered_lower_bugfix_followup": s["mean_func_bugfix_followup_rate"] < b["mean_func_bugfix_followup_rate"],
                "steered_lower_churn":          s["mean_func_churn"]                < b["mean_func_churn"],
            })

        if results:
            n = len(results)
            def _avg(key): return sum(r[key] for r in results) / n
            print(f"\n  Pairs: {n}")
            print(f"  {'Metric':<38}  {'Baseline':>9}  {'Steered':>9}  {'Steered ↓%':>10}")
            print("  " + "-"*72)
            for bk, sk, wk, lbl in [
                ("b_mean_func_followup_rate",        "s_mean_func_followup_rate",        "steered_lower_followup",       "mean_func_followup_rate"),
                ("b_mean_func_bugfix_followup_rate", "s_mean_func_bugfix_followup_rate", "steered_lower_bugfix_followup","mean_func_bugfix_followup_rate"),
                ("b_mean_func_churn",                "s_mean_func_churn",                "steered_lower_churn",          "mean_func_churn"),
            ]:
                wr = sum(1 for r in results if r[wk]) / n
                print(f"  {lbl:<38}  {_avg(bk):>9.4f}  {_avg(sk):>9.4f}  {wr:>9.1%}")
            out_eval = args.out.replace(".json", "_eval.jsonl")
            with open(out_eval, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            print(f"\n  Eval → {out_eval}")
        else:
            print("  No matching pairs found.")


if __name__ == "__main__":
    main()
