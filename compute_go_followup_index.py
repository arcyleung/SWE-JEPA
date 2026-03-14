#!/usr/bin/env python3
"""
Compute file-level follow-up risk index from go_prs for a set of repos.

For each (repo, file), computes:
  churn_count      — distinct PRs that touched this file
  followup_rate    — fraction of those PRs followed by another PR touching
                     the same file within `window_days` (default 90)
  mean_days_to_followup — mean gap to next touch (for PRs that have a follow-up)
  cochange_partners — set of other files that frequently co-appear with this one
  cochange_degree  — number of distinct co-change partners

These are file-level proxies for the org-level Conway risk metrics used by the
steerer (scope_drift, trust_boundary_crossings, ownership_friction).

Usage:
    source .venv/bin/activate
    python compute_go_followup_index.py \\
        --repos-from data/go_ablation_v1/judge_panel/gpt-5-codex.jsonl \\
        --out data/go_ablation_v1/go_followup_index.json \\
        --window-days 90

Produces a JSON file keyed by repo → file → metrics.
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

FILE_RE = re.compile(r'^\+\+\+ b/(.+)$', re.M)
HUNK_RE = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', re.M)


def _pg_connect(pg_yaml: str):
    cfg = yaml.safe_load(open(pg_yaml))
    return psycopg2.connect(
        host=cfg["ip"], port=cfg["port"],
        user=cfg["user"], password=cfg["password"],
        database=cfg["database"],
    )


def _parse_patch_files(patch: str) -> list[str]:
    return FILE_RE.findall(patch)


def _parse_iid(iid_pr: str) -> tuple[str, int]:
    s = iid_pr.removeprefix("go_prs__")
    parts = s.split("__")
    return f"{parts[0]}/{parts[1]}", int(parts[-2])


def _fetch_repo_prs(conn, repo: str) -> list[dict]:
    """Fetch all PRs for a repo sorted by pull_number, with files extracted."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT pull_number, non_test_patch, merged_at
        FROM go_prs
        WHERE repo = %s AND non_test_patch IS NOT NULL
        ORDER BY pull_number
        """,
        (repo,),
    )
    rows = []
    for pull_number, patch, merged_at in cur.fetchall():
        files = _parse_patch_files(patch)
        if not files:
            continue
        rows.append({
            "pull_number": pull_number,
            "files": files,
            "merged_at": merged_at,
        })
    return rows


def _compute_repo_index(prs: list[dict], window_days: int) -> dict[str, dict]:
    """
    For a single repo, compute per-file metrics.
    Returns {file: {churn_count, followup_rate, mean_days_to_followup, cochange_degree}}
    """
    # file → list of (pull_number, merged_at)
    file_prs: dict[str, list[tuple[int, datetime | None]]] = defaultdict(list)
    for pr in prs:
        for f in pr["files"]:
            file_prs[f].append((pr["pull_number"], pr["merged_at"]))

    # For each file, compute follow-up rate within window_days
    file_metrics: dict[str, dict] = {}
    for f, touches in file_prs.items():
        churn_count = len(touches)
        if churn_count < 2:
            followup_rate = 0.0
            mean_days = None
        else:
            # Sort by pull_number (already sorted but to be safe)
            touches_sorted = sorted(touches, key=lambda x: x[0])
            followup_gaps = []
            has_followup = 0
            for i, (pr_num, merged_at) in enumerate(touches_sorted[:-1]):
                next_pr_num, next_merged_at = touches_sorted[i + 1]
                if merged_at is not None and next_merged_at is not None:
                    # Make timezone-aware comparison
                    t0 = merged_at if merged_at.tzinfo else merged_at.replace(tzinfo=timezone.utc)
                    t1 = next_merged_at if next_merged_at.tzinfo else next_merged_at.replace(tzinfo=timezone.utc)
                    days_gap = (t1 - t0).total_seconds() / 86400
                    if days_gap <= window_days:
                        has_followup += 1
                        followup_gaps.append(days_gap)
                else:
                    # No date info — fallback to pr_number proximity (within 20 PRs)
                    if next_pr_num - pr_num <= 20:
                        has_followup += 1

            followup_rate = has_followup / (churn_count - 1) if churn_count > 1 else 0.0
            mean_days = sum(followup_gaps) / len(followup_gaps) if followup_gaps else None

        file_metrics[f] = {
            "churn_count":          churn_count,
            "followup_rate":        round(followup_rate, 4),
            "mean_days_to_followup": round(mean_days, 1) if mean_days is not None else None,
        }

    # Co-change: count how often pairs of files appear in the same PR
    # cochange_degree = number of distinct files that co-changed with this file
    cochange: dict[str, set] = defaultdict(set)
    for pr in prs:
        files = pr["files"]
        if len(files) < 2:
            continue
        for i, fa in enumerate(files):
            for fb in files[i + 1:]:
                cochange[fa].add(fb)
                cochange[fb].add(fa)
    for f in file_metrics:
        file_metrics[f]["cochange_degree"] = len(cochange.get(f, set()))

    return file_metrics


def _score_patch(
    patch: str,
    repo: str,
    index: dict[str, dict[str, dict]],
) -> dict:
    """
    Score a patch against the follow-up risk index.
    Returns per-file metrics + aggregated risk scores.
    """
    files = _parse_patch_files(patch)
    if not files:
        return {"files": [], "n_files": 0, "mean_churn": 0.0, "mean_followup_rate": 0.0, "mean_cochange_degree": 0.0}

    repo_idx = index.get(repo, {})
    churn_vals = []
    followup_vals = []
    cochange_vals = []
    per_file = []

    for f in files:
        fm = repo_idx.get(f, {})
        churn = fm.get("churn_count", 0)
        followup = fm.get("followup_rate", 0.0)
        cochange = fm.get("cochange_degree", 0)
        churn_vals.append(churn)
        followup_vals.append(followup)
        cochange_vals.append(cochange)
        per_file.append({"file": f, "churn_count": churn, "followup_rate": followup, "cochange_degree": cochange})

    n = len(files)
    return {
        "files":                 files,
        "n_files":               n,
        "mean_churn":            round(sum(churn_vals) / n, 2),
        "mean_followup_rate":    round(sum(followup_vals) / n, 4),
        "mean_cochange_degree":  round(sum(cochange_vals) / n, 2),
        "max_followup_rate":     round(max(followup_vals), 4),
        "per_file":              per_file,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos-from",   required=True,
                    help="JSONL with iid_pr field to determine target repos")
    ap.add_argument("--out",          required=True,
                    help="Output JSON file: repo → file → metrics")
    ap.add_argument("--window-days",  type=int, default=90)
    ap.add_argument("--pg-yaml",      default=os.path.join(ROOT, "postgres_connection.yaml"))
    ap.add_argument("--eval-patches-baseline", default=None,
                    help="If provided, also score and compare baseline + steered patches")
    ap.add_argument("--eval-patches-steered",  default=None)
    ap.add_argument("--iid-pr-list",  default=None,
                    help="JSONL with iid_pr column to restrict eval pairs")
    args = ap.parse_args()

    # ── Determine target repos ──────────────────────────────────────────────
    ref_rows = [json.loads(l) for l in open(args.repos_from) if l.strip()]
    iid_prs  = [r["iid_pr"] for r in ref_rows if "error" not in r]
    repo_pr_map: dict[str, int] = {}
    for iid in iid_prs:
        repo, pr = _parse_iid(iid)
        repo_pr_map[repo] = pr   # just to know the eval PR number for each repo
    repos = list(repo_pr_map.keys())
    print(f"Target repos: {len(repos)}")

    # ── Build follow-up index ───────────────────────────────────────────────
    conn = _pg_connect(args.pg_yaml)
    index: dict[str, dict[str, dict]] = {}

    for i, repo in enumerate(sorted(repos)):
        prs = _fetch_repo_prs(conn, repo)
        if not prs:
            print(f"  [{i+1}/{len(repos)}] {repo}: 0 PRs — skipping")
            continue
        repo_index = _compute_repo_index(prs, args.window_days)
        index[repo] = repo_index
        n_files = len(repo_index)
        n_followup = sum(1 for m in repo_index.values() if m["followup_rate"] > 0)
        print(f"  [{i+1}/{len(repos)}] {repo}: {len(prs)} PRs, "
              f"{n_files} files, {n_followup} with follow-up signal")

    conn.close()

    with open(args.out, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nIndex saved → {args.out}")
    print(f"Repos indexed: {len(index)}")

    # ── Optional: evaluate steered vs baseline patches ─────────────────────
    if args.eval_patches_baseline and args.eval_patches_steered:
        print("\n── Scoring steered vs baseline patches ──")
        iid_filter = None
        if args.iid_pr_list:
            iid_filter = {
                json.loads(l)["iid_pr"]
                for l in open(args.iid_pr_list) if l.strip() and "error" not in json.loads(l)
            }

        baseline_dir = args.eval_patches_baseline
        steered_dir  = args.eval_patches_steered

        results = []
        for fname in sorted(os.listdir(baseline_dir)):
            if not fname.endswith(".patch"):
                continue
            iid_pr = fname[:-6]
            if iid_filter and iid_pr not in iid_filter:
                continue
            repo, pr_num = _parse_iid(iid_pr)
            if repo not in index:
                continue

            baseline_patch = open(os.path.join(baseline_dir, fname)).read()
            steered_path = os.path.join(steered_dir, fname)
            if not os.path.exists(steered_path):
                continue
            steered_patch = open(steered_path).read()

            b_scores = _score_patch(baseline_patch, repo, index)
            s_scores = _score_patch(steered_patch, repo, index)

            results.append({
                "iid_pr":                     iid_pr,
                "repo":                       repo,
                "baseline_n_files":           b_scores["n_files"],
                "steered_n_files":            s_scores["n_files"],
                "baseline_mean_churn":        b_scores["mean_churn"],
                "steered_mean_churn":         s_scores["mean_churn"],
                "baseline_mean_followup_rate": b_scores["mean_followup_rate"],
                "steered_mean_followup_rate": s_scores["mean_followup_rate"],
                "baseline_mean_cochange":     b_scores["mean_cochange_degree"],
                "steered_mean_cochange":      s_scores["mean_cochange_degree"],
                "steered_lower_followup":     s_scores["mean_followup_rate"] < b_scores["mean_followup_rate"],
                "steered_lower_churn":        s_scores["mean_churn"] < b_scores["mean_churn"],
                "steered_lower_cochange":     s_scores["mean_cochange_degree"] < b_scores["mean_cochange_degree"],
            })

        if results:
            n = len(results)
            lower_followup = sum(1 for r in results if r["steered_lower_followup"])
            lower_churn    = sum(1 for r in results if r["steered_lower_churn"])
            lower_cochange = sum(1 for r in results if r["steered_lower_cochange"])
            avg_b_fup  = sum(r["baseline_mean_followup_rate"] for r in results) / n
            avg_s_fup  = sum(r["steered_mean_followup_rate"]  for r in results) / n
            avg_b_churn = sum(r["baseline_mean_churn"] for r in results) / n
            avg_s_churn = sum(r["steered_mean_churn"]  for r in results) / n

            print(f"\n  Pairs scored: {n}")
            print(f"  {'Metric':<30}  {'Baseline':>9}  {'Steered':>9}  {'Steered wins':>12}")
            print("  " + "-" * 66)
            print(f"  {'mean_followup_rate':<30}  {avg_b_fup:>9.4f}  {avg_s_fup:>9.4f}  {lower_followup/n:>11.1%}")
            print(f"  {'mean_churn (PRs touching file)':<30}  {avg_b_churn:>9.2f}  {avg_s_churn:>9.2f}  {lower_churn/n:>11.1%}")

            out_eval = args.out.replace(".json", "_eval.jsonl")
            with open(out_eval, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            print(f"\n  Eval results → {out_eval}")
        else:
            print("  No matching pairs found.")


if __name__ == "__main__":
    main()
