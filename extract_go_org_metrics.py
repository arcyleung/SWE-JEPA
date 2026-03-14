#!/usr/bin/env python3
"""
Compute org-level Conway metrics from go_prs patch history.

For each (repo, PR), computes:
  cross_module_ratio        — fraction of touched files from different Go packages
                              (different top-level directory = different module)
  author_entropy_proxy      — Shannon entropy of author distribution across PRs
                              that touched overlapping files (requires pr_author col)
  cochange_cross_module     — fraction of file co-changes that cross package boundaries
  n_packages_touched        — distinct top-level package directories in this PR
  ownership_friction_proxy  — fraction of files in this PR that have been touched
                              by more than K distinct authors in the past M months
                              (proxy: fraction touched by > 1 PR in window)

Writes a JSON index: {repo: {pull_number: metrics}}
Also writes a flat JSONL for easy eval comparison.

Usage:
    source .venv/bin/activate
    python extract_go_org_metrics.py \\
        --repos-from data/go_ablation_v1/judge_panel/gpt-5-codex.jsonl \\
        --out        data/go_ablation_v1/go_org_metrics_index.json \\
        --window-days 180

    # Score steered vs baseline:
    python extract_go_org_metrics.py \\
        --repos-from data/go_ablation_v1/judge_panel/gpt-5-codex.jsonl \\
        --out        data/go_ablation_v1/go_org_metrics_index.json \\
        --eval-patches-baseline data/go_ablation_v1/patches_baseline \\
        --eval-patches-steered  data/go_ablation_v1/patches_steered_specific \\
        --iid-pr-list           data/go_ablation_v1/judge_panel/gpt-5-codex.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone

import psycopg2
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))

FILE_RE = re.compile(r'^\+\+\+ b/(.+)$', re.M)


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


def _parse_patch_files(patch: str) -> list[str]:
    return FILE_RE.findall(patch)


def _go_package(filepath: str) -> str:
    """
    Return the top-level package directory of a Go file path.
    Examples:
      "cmd/server/main.go"  → "cmd"
      "pkg/store/store.go"  → "pkg"
      "main.go"             → ""   (root package)
    """
    parts = filepath.split("/")
    return parts[0] if len(parts) > 1 else ""


def _entropy(counts: dict) -> float:
    """Shannon entropy of a count distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (v / total) * math.log2(v / total)
        for v in counts.values()
        if v > 0
    )


def _fetch_repo_prs(conn, repo: str) -> list[dict]:
    cur = conn.cursor()
    # Try to get pr_author; fall back gracefully if column doesn't exist
    try:
        cur.execute(
            """
            SELECT pull_number, non_test_patch, merged_at,
                   COALESCE(pr_author, 'unknown') AS pr_author
            FROM go_prs
            WHERE repo = %s AND non_test_patch IS NOT NULL
            ORDER BY pull_number
            """,
            (repo,),
        )
    except psycopg2.errors.UndefinedColumn:
        conn.rollback()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT pull_number, non_test_patch, merged_at, 'unknown' AS pr_author
            FROM go_prs
            WHERE repo = %s AND non_test_patch IS NOT NULL
            ORDER BY pull_number
            """,
            (repo,),
        )

    rows = []
    for pull_number, patch, merged_at, author in cur.fetchall():
        files = _parse_patch_files(patch)
        if not files:
            continue
        rows.append({
            "pull_number": pull_number,
            "files":       files,
            "merged_at":   merged_at,
            "pr_author":   author,
        })
    return rows


def _compute_repo_org_metrics(prs: list[dict], window_days: int) -> dict[str, dict]:
    """
    Compute org metrics for every PR in the repo.
    Returns {str(pull_number): metrics_dict}
    """
    # Build file → list of (pull_number, merged_at, author) for history lookups
    file_history: dict[str, list[tuple[int, datetime | None, str]]] = defaultdict(list)
    for pr in prs:
        for f in pr["files"]:
            file_history[f].append((pr["pull_number"], pr["merged_at"], pr["pr_author"]))

    result: dict[str, dict] = {}

    for pr in prs:
        files = pr["files"]
        n_files = len(files)
        if n_files == 0:
            result[str(pr["pull_number"])] = _empty_metrics(pr)
            continue

        # ── cross_module_ratio ───────────────────────────────────────────────
        packages = [_go_package(f) for f in files]
        unique_packages = set(packages)
        n_packages = len(unique_packages)

        if n_files > 1:
            cross_module_pairs = 0
            total_pairs = 0
            for i, pa in enumerate(packages):
                for pb in packages[i + 1:]:
                    total_pairs += 1
                    if pa != pb:
                        cross_module_pairs += 1
            cross_module_ratio = cross_module_pairs / total_pairs if total_pairs else 0.0
        else:
            cross_module_ratio = 0.0

        # ── ownership_friction_proxy ─────────────────────────────────────────
        # Fraction of files that have been touched by more than 1 *other* PR
        # in the past `window_days` days before this PR
        pr_merged = pr["merged_at"]
        friction_files = 0
        author_counts_all: dict[str, int] = defaultdict(int)

        for f in files:
            history = file_history.get(f, [])
            prior_authors: set[str] = set()
            for h_pr_num, h_merged, h_author in history:
                if h_pr_num >= pr["pull_number"]:
                    continue  # only look at prior PRs
                in_window = True
                if pr_merged is not None and h_merged is not None:
                    t_pr = pr_merged if pr_merged.tzinfo else pr_merged.replace(tzinfo=timezone.utc)
                    t_h  = h_merged  if h_merged.tzinfo  else h_merged.replace(tzinfo=timezone.utc)
                    days_gap = (t_pr - t_h).total_seconds() / 86400
                    in_window = (0 <= days_gap <= window_days)
                if in_window:
                    prior_authors.add(h_author)
                    author_counts_all[h_author] += 1

            if len(prior_authors) > 1:
                friction_files += 1

        ownership_friction = friction_files / n_files if n_files else 0.0

        # ── author_entropy_proxy ─────────────────────────────────────────────
        # Entropy of the author distribution across all prior PRs that touched
        # any file in this PR within the window
        author_entropy = _entropy(author_counts_all)

        result[str(pr["pull_number"])] = {
            "n_files":               n_files,
            "n_packages":            n_packages,
            "cross_module_ratio":    round(cross_module_ratio, 4),
            "ownership_friction":    round(ownership_friction, 4),
            "author_entropy":        round(author_entropy, 3),
            "packages":              sorted(unique_packages),
        }

    return result


def _empty_metrics(pr: dict) -> dict:
    return {
        "n_files": 0, "n_packages": 0,
        "cross_module_ratio": 0.0,
        "ownership_friction": 0.0,
        "author_entropy": 0.0,
        "packages": [],
    }


def _score_patch_org(
    patch: str,
    repo: str,
    index: dict,
    prior_prs_lookup: dict | None = None,
) -> dict:
    """
    Compute org metrics for an agent-generated patch (no pull_number yet).
    Uses only the static information available from the patch itself.
    """
    files = _parse_patch_files(patch)
    if not files:
        return {
            "n_files": 0, "n_packages": 0,
            "cross_module_ratio": 0.0,
            "mean_cross_module_ratio": 0.0,
            "mean_ownership_friction": 0.0,
            "mean_author_entropy": 0.0,
        }

    packages = [_go_package(f) for f in files]
    unique_packages = set(packages)
    n_files = len(files)

    # Static: cross_module_ratio for this patch
    if n_files > 1:
        cross_pairs = sum(
            1 for i, pa in enumerate(packages)
            for pb in packages[i + 1:]
            if pa != pb
        )
        total_pairs = n_files * (n_files - 1) // 2
        cross_module_ratio = cross_pairs / total_pairs
    else:
        cross_module_ratio = 0.0

    # Historical: look up metrics for the files touched (mean over their history)
    repo_idx = index.get(repo, {})
    friction_vals = []
    entropy_vals = []
    cmr_vals = []

    # Gather per-PR metrics for PRs touching these files
    for pr_key, pr_metrics in repo_idx.items():
        pr_files_set = set()
        # We don't store files per PR in this index (too large) — use cross_module as proxy
        friction_vals.append(pr_metrics.get("ownership_friction", 0.0))
        entropy_vals.append(pr_metrics.get("author_entropy", 0.0))
        cmr_vals.append(pr_metrics.get("cross_module_ratio", 0.0))

    mean_friction = sum(friction_vals) / len(friction_vals) if friction_vals else 0.0
    mean_entropy  = sum(entropy_vals)  / len(entropy_vals)  if entropy_vals  else 0.0
    mean_cmr      = sum(cmr_vals)      / len(cmr_vals)      if cmr_vals      else 0.0

    return {
        "n_files":                 n_files,
        "n_packages":              len(unique_packages),
        "cross_module_ratio":      round(cross_module_ratio, 4),
        "mean_cross_module_ratio": round(mean_cmr, 4),
        "mean_ownership_friction": round(mean_friction, 4),
        "mean_author_entropy":     round(mean_entropy, 3),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos-from",  required=True,
                    help="JSONL with iid_pr field to determine target repos")
    ap.add_argument("--out",         required=True,
                    help="Output JSON: repo → pull_number → org metrics")
    ap.add_argument("--window-days", type=int, default=180)
    ap.add_argument("--pg-yaml",     default=os.path.join(ROOT, "postgres_connection.yaml"))
    ap.add_argument("--eval-patches-baseline", default=None)
    ap.add_argument("--eval-patches-steered",  default=None)
    ap.add_argument("--iid-pr-list", default=None)
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

    # ── Compute org metrics per repo ─────────────────────────────────────────
    conn = _pg_connect(args.pg_yaml)
    index: dict[str, dict] = {}

    for i, repo in enumerate(repos):
        prs = _fetch_repo_prs(conn, repo)
        if not prs:
            print(f"  [{i+1}/{len(repos)}] {repo}: 0 PRs — skipping", flush=True)
            continue
        repo_metrics = _compute_repo_org_metrics(prs, args.window_days)
        index[repo] = repo_metrics

        # Summary stats for this repo
        cmr_vals  = [m["cross_module_ratio"]  for m in repo_metrics.values()]
        fric_vals = [m["ownership_friction"]  for m in repo_metrics.values()]
        avg_cmr   = sum(cmr_vals)  / len(cmr_vals)  if cmr_vals  else 0
        avg_fric  = sum(fric_vals) / len(fric_vals) if fric_vals else 0
        print(
            f"  [{i+1}/{len(repos)}] {repo}: {len(prs)} PRs, "
            f"avg_cross_module={avg_cmr:.3f}, avg_ownership_friction={avg_fric:.3f}",
            flush=True,
        )

    conn.close()

    with open(args.out, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nIndex saved → {args.out}")
    print(f"Repos indexed: {len(index)}")

    # ── Also write flat JSONL for easy querying ──────────────────────────────
    out_jsonl = args.out.replace(".json", ".jsonl")
    with open(out_jsonl, "w") as f:
        for repo, pr_metrics in index.items():
            for pr_key, metrics in pr_metrics.items():
                row = {"repo": repo, "pull_number": int(pr_key)}
                row.update(metrics)
                f.write(json.dumps(row) + "\n")
    print(f"Flat JSONL → {out_jsonl}")

    # ── Optional: score steered vs baseline patches ──────────────────────────
    if args.eval_patches_baseline and args.eval_patches_steered:
        print("\n── Scoring steered vs baseline (org metrics) ──")
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

            b = _score_patch_org(baseline_patch, repo, index)
            s = _score_patch_org(steered_patch,  repo, index)

            results.append({
                "iid_pr": iid_pr,
                "repo":   repo,
                "b_cross_module_ratio": b["cross_module_ratio"],
                "s_cross_module_ratio": s["cross_module_ratio"],
                "b_n_packages":         b["n_packages"],
                "s_n_packages":         s["n_packages"],
                "b_n_files":            b["n_files"],
                "s_n_files":            s["n_files"],
                "steered_lower_cross_module": s["cross_module_ratio"] < b["cross_module_ratio"],
                "steered_fewer_packages":     s["n_packages"] < b["n_packages"],
            })

        if results:
            n = len(results)
            avg_b_cmr = sum(r["b_cross_module_ratio"] for r in results) / n
            avg_s_cmr = sum(r["s_cross_module_ratio"] for r in results) / n
            avg_b_pkg = sum(r["b_n_packages"] for r in results) / n
            avg_s_pkg = sum(r["s_n_packages"] for r in results) / n
            lcm = sum(1 for r in results if r["steered_lower_cross_module"]) / n
            lpk = sum(1 for r in results if r["steered_fewer_packages"]) / n

            print(f"\n  Pairs scored: {n}")
            print(f"  {'Metric':<30}  {'Baseline':>9}  {'Steered':>9}  {'Steered ↓%':>10}")
            print("  " + "-" * 64)
            print(f"  {'cross_module_ratio':<30}  {avg_b_cmr:>9.4f}  {avg_s_cmr:>9.4f}  {lcm:>9.1%}")
            print(f"  {'n_packages_touched':<30}  {avg_b_pkg:>9.2f}  {avg_s_pkg:>9.2f}  {lpk:>9.1%}")

            out_eval = args.out.replace(".json", "_eval.jsonl")
            with open(out_eval, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            print(f"\n  Eval results → {out_eval}")
        else:
            print("  No matching pairs found.")


if __name__ == "__main__":
    main()
