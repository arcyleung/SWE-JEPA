#!/usr/bin/env python3
"""
Extract function-level follow-up signal from go_prs patches.

For each PR in go_prs, extracts the set of Go functions touched (from `@@ ... @@`
hunk context headers, which contain the enclosing function signature).

Then for each repo, computes per-function follow-up metrics:
  - churn_count       — number of distinct PRs touching this function
  - followup_rate     — fraction of those PRs followed by another PR touching
                        the same function within `window_days`
  - is_bugfix_followup_rate — same but restricted to the follow-up being a bugfix/maintenance PR
  - mean_days_to_followup

Writes the index to a JSON file, and optionally scores agent patches against it.

Go hunk headers look like:
  @@ -17,8 +24,12 @@ func (s *Stream) dispatchEvent(event interface{}) {
  @@ -1,5 +1,5 @@ package foo
  ^^ the function context comes after the final @@

Usage:
    source .venv/bin/activate
    python extract_go_followup_function.py \\
        --repos-from data/go_ablation_v1/judge_panel/gpt-5-codex.jsonl \\
        --out        data/go_ablation_v1/go_followup_function_index.json \\
        --window-days 90

    # Also score steered vs baseline patches:
    python extract_go_followup_function.py \\
        --repos-from data/go_ablation_v1/judge_panel/gpt-5-codex.jsonl \\
        --out        data/go_ablation_v1/go_followup_function_index.json \\
        --eval-patches-baseline data/go_ablation_v1/patches_baseline \\
        --eval-patches-steered  data/go_ablation_v1/patches_steered_specific \\
        --iid-pr-list           data/go_ablation_v1/judge_panel/gpt-5-codex.jsonl
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

# Matches the context fragment after the final @@ in a unified diff hunk header.
# Examples:
#   @@ -3,6 +3,8 @@ func (s *State) Reset() {
#   @@ -1,5 +1,5 @@ package foo
#   @@ -10,4 +10,4 @@
HUNK_CONTEXT_RE = re.compile(r'^@@[^@\n]+@@[ \t]*(.*)', re.M)

# Extract file from unified diff
FILE_RE = re.compile(r'^\+\+\+ b/(.+)$', re.M)

# Normalise Go function context: strip to just the func signature stem.
# e.g. " func (s *Stream) dispatchEvent(event interface{}) {" → "dispatchEvent"
FUNC_NAME_RE = re.compile(
    r'\bfunc\s+(?:\([^)]*\)\s+)?(\w+)\s*[({]',
)


def _pg_connect(pg_yaml: str):
    cfg = yaml.safe_load(open(pg_yaml))
    return psycopg2.connect(
        host=cfg["ip"], port=cfg["port"],
        user=cfg["user"], password=cfg["password"],
        database=cfg["database"],
    )


def _parse_iid(iid_pr: str) -> tuple[str, int]:
    s = iid_pr.removeprefix("go_prs__")
    parts = s.split("__")
    return f"{parts[0]}/{parts[1]}", int(parts[-2])


def _extract_func_context(hunk_context: str) -> str | None:
    """Return a normalised function name from a hunk context string, or None."""
    ctx = hunk_context.strip()
    if not ctx:
        return None
    m = FUNC_NAME_RE.search(ctx)
    if m:
        return m.group(1)
    # Fallback: keep up to 80 chars of context as identifier
    # (covers cases like anonymous funcs, init blocks, type declarations)
    clean = re.sub(r'\s+', ' ', ctx)[:80]
    return clean if clean else None


def _parse_patch_functions(patch: str) -> dict[str, set[str]]:
    """
    Parse a unified diff patch and return {file: set_of_function_names}.
    Function names are extracted from hunk context headers.
    """
    file_funcs: dict[str, set[str]] = defaultdict(set)
    current_file: str | None = None

    for line in patch.splitlines():
        fm = FILE_RE.match(line)
        if fm:
            current_file = fm.group(1)
            continue
        hm = HUNK_CONTEXT_RE.match(line)
        if hm and current_file:
            ctx = hm.group(1)
            func_name = _extract_func_context(ctx)
            if func_name:
                file_funcs[current_file].add(func_name)

    return dict(file_funcs)


def _fetch_repo_prs(conn, repo: str) -> list[dict]:
    """Fetch all PRs for a repo with patch, merged_at, and pr_category."""
    cur = conn.cursor()
    # pr_category may not exist or may be all NULL — use coalesce
    cur.execute(
        """
        SELECT pull_number, non_test_patch, merged_at,
               COALESCE(pr_category, 'unknown') AS pr_category,
               COALESCE(pr_category_confidence, 0) AS pr_category_confidence
        FROM go_prs
        WHERE repo = %s AND non_test_patch IS NOT NULL
        ORDER BY pull_number
        """,
        (repo,),
    )
    rows = []
    for pull_number, patch, merged_at, category, confidence in cur.fetchall():
        file_funcs = _parse_patch_functions(patch)
        if not file_funcs:
            continue
        rows.append({
            "pull_number":  pull_number,
            "file_funcs":   file_funcs,
            "merged_at":    merged_at,
            "pr_category":  category,
            "pr_category_confidence": confidence,
        })
    return rows


def _is_followup_category(pr: dict) -> bool:
    """True if this PR is a bugfix or maintenance (likely a follow-up fix)."""
    return pr["pr_category"] in ("bugfix", "maintenance")


def _compute_repo_function_index(prs: list[dict], window_days: int) -> dict:
    """
    For a single repo, compute per-function follow-up metrics.

    Returns:
      {
        file: {
          func: {
            churn_count,
            followup_rate,
            bugfix_followup_rate,
            mean_days_to_followup,
          }
        }
      }
    """
    # (file, func) → list of (pull_number, merged_at, pr_category)
    func_touches: dict[tuple[str, str], list[tuple[int, datetime | None, str]]] = defaultdict(list)
    for pr in prs:
        for f, funcs in pr["file_funcs"].items():
            for fn in funcs:
                func_touches[(f, fn)].append(
                    (pr["pull_number"], pr["merged_at"], pr["pr_category"])
                )

    # Nested dict: file → func → metrics
    index: dict[str, dict[str, dict]] = defaultdict(dict)

    for (f, fn), touches in func_touches.items():
        churn_count = len(touches)
        if churn_count < 2:
            followup_rate = 0.0
            bugfix_followup_rate = 0.0
            mean_days = None
        else:
            touches_sorted = sorted(touches, key=lambda x: x[0])
            followup_gaps = []
            has_followup = 0
            has_bugfix_followup = 0
            for i, (pr_num, merged_at, _) in enumerate(touches_sorted[:-1]):
                next_pr_num, next_merged_at, next_cat = touches_sorted[i + 1]
                if merged_at is not None and next_merged_at is not None:
                    t0 = merged_at if merged_at.tzinfo else merged_at.replace(tzinfo=timezone.utc)
                    t1 = next_merged_at if next_merged_at.tzinfo else next_merged_at.replace(tzinfo=timezone.utc)
                    days_gap = (t1 - t0).total_seconds() / 86400
                    if days_gap <= window_days:
                        has_followup += 1
                        followup_gaps.append(days_gap)
                        if next_cat in ("bugfix", "maintenance"):
                            has_bugfix_followup += 1
                else:
                    if next_pr_num - pr_num <= 20:
                        has_followup += 1
                        if next_cat in ("bugfix", "maintenance"):
                            has_bugfix_followup += 1

            denom = churn_count - 1
            followup_rate      = has_followup       / denom if denom > 0 else 0.0
            bugfix_followup_rate = has_bugfix_followup / denom if denom > 0 else 0.0
            mean_days = sum(followup_gaps) / len(followup_gaps) if followup_gaps else None

        index[f][fn] = {
            "churn_count":          churn_count,
            "followup_rate":        round(followup_rate, 4),
            "bugfix_followup_rate": round(bugfix_followup_rate, 4),
            "mean_days_to_followup": round(mean_days, 1) if mean_days is not None else None,
        }

    return dict(index)


def _score_patch_functions(
    patch: str,
    repo: str,
    index: dict,
) -> dict:
    """
    Score a patch against the function-level follow-up index.
    Returns aggregated risk scores over all (file, func) pairs touched.
    """
    file_funcs = _parse_patch_functions(patch)
    if not file_funcs:
        return {
            "n_files": 0, "n_funcs": 0,
            "mean_func_followup_rate": 0.0,
            "mean_func_bugfix_followup_rate": 0.0,
            "mean_func_churn": 0.0,
            "max_func_followup_rate": 0.0,
        }

    repo_idx = index.get(repo, {})
    followup_vals = []
    bugfix_vals = []
    churn_vals = []
    per_func = []

    for f, funcs in file_funcs.items():
        file_idx = repo_idx.get(f, {})
        for fn in funcs:
            fm = file_idx.get(fn, {})
            fr = fm.get("followup_rate", 0.0)
            bfr = fm.get("bugfix_followup_rate", 0.0)
            ch = fm.get("churn_count", 0)
            followup_vals.append(fr)
            bugfix_vals.append(bfr)
            churn_vals.append(ch)
            per_func.append({"file": f, "func": fn,
                              "followup_rate": fr, "bugfix_followup_rate": bfr,
                              "churn_count": ch})

    n_funcs = len(followup_vals)
    n_files = len(file_funcs)

    if n_funcs == 0:
        return {
            "n_files": n_files, "n_funcs": 0,
            "mean_func_followup_rate": 0.0,
            "mean_func_bugfix_followup_rate": 0.0,
            "mean_func_churn": 0.0,
            "max_func_followup_rate": 0.0,
        }

    return {
        "n_files":                      n_files,
        "n_funcs":                      n_funcs,
        "mean_func_followup_rate":      round(sum(followup_vals) / n_funcs, 4),
        "mean_func_bugfix_followup_rate": round(sum(bugfix_vals) / n_funcs, 4),
        "mean_func_churn":              round(sum(churn_vals) / n_funcs, 2),
        "max_func_followup_rate":       round(max(followup_vals), 4),
        "per_func":                     per_func,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos-from",  required=True,
                    help="JSONL with iid_pr field to determine target repos")
    ap.add_argument("--out",         required=True,
                    help="Output JSON: repo → file → func → metrics")
    ap.add_argument("--window-days", type=int, default=90)
    ap.add_argument("--pg-yaml",     default=os.path.join(ROOT, "postgres_connection.yaml"))
    ap.add_argument("--eval-patches-baseline", default=None)
    ap.add_argument("--eval-patches-steered",  default=None)
    ap.add_argument("--iid-pr-list", default=None,
                    help="JSONL with iid_pr column to restrict eval pairs")
    args = ap.parse_args()

    # ── Determine target repos ───────────────────────────────────────────────
    ref_rows = [json.loads(l) for l in open(args.repos_from) if l.strip()]
    repo_pr_map: dict[str, int] = {}
    for r in ref_rows:
        if "error" in r:
            continue
        if "iid_pr" in r:
            repo, pr = _parse_iid(r["iid_pr"])
        elif "repo" in r and "pull_number" in r:
            repo, pr = r["repo"], int(r["pull_number"])
        elif "instance_id" in r:
            repo, pr = _parse_iid(r["instance_id"])
        else:
            continue
        repo_pr_map[repo] = pr
    repos = sorted(repo_pr_map.keys())
    print(f"Target repos: {len(repos)}", flush=True)

    # ── Build function follow-up index ───────────────────────────────────────
    conn = _pg_connect(args.pg_yaml)
    index: dict[str, dict] = {}

    for i, repo in enumerate(repos):
        prs = _fetch_repo_prs(conn, repo)
        if not prs:
            print(f"  [{i+1}/{len(repos)}] {repo}: 0 PRs — skipping", flush=True)
            continue
        repo_index = _compute_repo_function_index(prs, args.window_days)
        index[repo] = repo_index
        n_files = len(repo_index)
        n_funcs = sum(len(fd) for fd in repo_index.values())
        n_followup = sum(
            1 for fd in repo_index.values()
            for fm in fd.values()
            if fm["followup_rate"] > 0
        )
        print(
            f"  [{i+1}/{len(repos)}] {repo}: {len(prs)} PRs, "
            f"{n_files} files, {n_funcs} funcs, {n_followup} with follow-up signal",
            flush=True,
        )

    conn.close()

    with open(args.out, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nIndex saved → {args.out}")
    n_total_funcs = sum(len(fd) for ri in index.values() for fd in ri.values())
    print(f"Repos indexed: {len(index)}, total functions tracked: {n_total_funcs}")

    # ── Optional: score steered vs baseline patches ──────────────────────────
    if args.eval_patches_baseline and args.eval_patches_steered:
        print("\n── Scoring steered vs baseline (function-level) ──")
        iid_filter = None
        if args.iid_pr_list:
            iid_filter = set()
            for l in open(args.iid_pr_list):
                if not l.strip():
                    continue
                r = json.loads(l)
                if "error" in r:
                    continue
                iid_filter.add(r.get("iid_pr") or r.get("instance_id", ""))

        results = []
        for fname in sorted(os.listdir(args.eval_patches_baseline)):
            if not fname.endswith(".patch"):
                continue
            iid_pr = fname[:-6]
            if iid_filter and iid_pr not in iid_filter:
                continue
            repo, _ = _parse_iid(iid_pr)
            if repo not in index:
                continue

            baseline_patch = open(
                os.path.join(args.eval_patches_baseline, fname)
            ).read()
            steered_path = os.path.join(args.eval_patches_steered, fname)
            if not os.path.exists(steered_path):
                continue
            steered_patch = open(steered_path).read()

            b = _score_patch_functions(baseline_patch, repo, index)
            s = _score_patch_functions(steered_patch,  repo, index)

            results.append({
                "iid_pr": iid_pr,
                "repo":   repo,
                "b_n_funcs":   b["n_funcs"],
                "s_n_funcs":   s["n_funcs"],
                "b_mean_func_followup_rate":       b["mean_func_followup_rate"],
                "s_mean_func_followup_rate":       s["mean_func_followup_rate"],
                "b_mean_func_bugfix_followup_rate": b["mean_func_bugfix_followup_rate"],
                "s_mean_func_bugfix_followup_rate": s["mean_func_bugfix_followup_rate"],
                "b_mean_func_churn": b["mean_func_churn"],
                "s_mean_func_churn": s["mean_func_churn"],
                "steered_lower_followup": s["mean_func_followup_rate"] < b["mean_func_followup_rate"],
                "steered_lower_bugfix_followup": s["mean_func_bugfix_followup_rate"] < b["mean_func_bugfix_followup_rate"],
                "steered_lower_churn": s["mean_func_churn"] < b["mean_func_churn"],
            })

        if results:
            n = len(results)
            avg_b_fup = sum(r["b_mean_func_followup_rate"] for r in results) / n
            avg_s_fup = sum(r["s_mean_func_followup_rate"] for r in results) / n
            avg_b_bfup = sum(r["b_mean_func_bugfix_followup_rate"] for r in results) / n
            avg_s_bfup = sum(r["s_mean_func_bugfix_followup_rate"] for r in results) / n
            avg_b_ch = sum(r["b_mean_func_churn"] for r in results) / n
            avg_s_ch = sum(r["s_mean_func_churn"] for r in results) / n
            lf  = sum(1 for r in results if r["steered_lower_followup"]) / n
            lbf = sum(1 for r in results if r["steered_lower_bugfix_followup"]) / n
            lch = sum(1 for r in results if r["steered_lower_churn"]) / n

            print(f"\n  Pairs scored: {n}")
            print(f"  {'Metric':<38}  {'Baseline':>9}  {'Steered':>9}  {'Steered ↓%':>10}")
            print("  " + "-" * 72)
            print(f"  {'mean_func_followup_rate':<38}  {avg_b_fup:>9.4f}  {avg_s_fup:>9.4f}  {lf:>9.1%}")
            print(f"  {'mean_func_bugfix_followup_rate':<38}  {avg_b_bfup:>9.4f}  {avg_s_bfup:>9.4f}  {lbf:>9.1%}")
            print(f"  {'mean_func_churn':<38}  {avg_b_ch:>9.2f}  {avg_s_ch:>9.2f}  {lch:>9.1%}")

            out_eval = args.out.replace(".json", "_eval.jsonl")
            with open(out_eval, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            print(f"\n  Eval results → {out_eval}")
        else:
            print("  No matching pairs found.")


if __name__ == "__main__":
    main()
