#!/usr/bin/env python3
"""
Compute org-level Conway metrics from prs_copy (Python PRs).

Mirrors extract_go_org_metrics.py but reads from prs_copy:
  cross_module_ratio      — fraction of file-pairs from different top-level dirs
  ownership_friction      — fraction of files touched by >1 author in window
  author_entropy          — Shannon entropy of author distribution on touched files
  n_packages_touched      — distinct top-level directories in this PR

Output:
  JSON index: {repo: {pull_number: metrics}}
  Flat JSONL for easy querying / feature join

Usage:
    source .venv/bin/activate
    python extract_py_org_metrics.py \\
        --language Python \\
        --out data/py_org_metrics_index.json \\
        --window-days 180

    # Restrict to repos from a JSONL:
    python extract_py_org_metrics.py \\
        --repos-from data/phase4_7_agentic_eval_results_feature_sl80.jsonl \\
        --out data/py_org_metrics_evalrepos.json
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


def _top_dir(filepath: str) -> str:
    parts = filepath.split("/")
    return parts[0] if len(parts) > 1 else ""


def _entropy(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in counts.values() if v > 0)


def _parse_patch_files(patch: str) -> list[str]:
    return FILE_RE.findall(patch)


def _parse_instance_id(instance_id: str) -> tuple[str, int] | None:
    parts = instance_id.split("__")
    if len(parts) < 3:
        return None
    repo = f"{parts[0]}/{parts[1]}"
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
               COALESCE(pr_author, 'unknown') AS pr_author
        FROM prs_copy
        WHERE repo = %s AND primary_language = %s AND patch IS NOT NULL
        ORDER BY pull_number
        """,
        (repo, language),
    )
    rows = []
    for pull_number, patch, merged_at, author in cur.fetchall():
        files = _parse_patch_files(patch)
        if not files:
            continue
        rows.append({"pull_number": pull_number, "files": files,
                     "merged_at": merged_at, "pr_author": author})
    return rows


def _compute_repo_org_metrics(prs: list[dict], window_days: int) -> dict[str, dict]:
    file_history: dict[str, list[tuple[int, datetime | None, str]]] = defaultdict(list)
    for pr in prs:
        for f in pr["files"]:
            file_history[f].append((pr["pull_number"], pr["merged_at"], pr["pr_author"]))

    result: dict[str, dict] = {}
    for pr in prs:
        files = pr["files"]
        n_files = len(files)
        if n_files == 0:
            result[str(pr["pull_number"])] = {
                "n_files": 0, "n_packages": 0, "cross_module_ratio": 0.0,
                "ownership_friction": 0.0, "author_entropy": 0.0, "packages": []}
            continue

        packages = [_top_dir(f) for f in files]
        unique_packages = set(packages)
        n_packages = len(unique_packages)

        if n_files > 1:
            cross = sum(1 for i, pa in enumerate(packages)
                        for pb in packages[i+1:] if pa != pb)
            total = n_files * (n_files - 1) // 2
            cross_module_ratio = cross / total
        else:
            cross_module_ratio = 0.0

        pr_merged = pr["merged_at"]
        friction_files = 0
        author_counts: dict[str, int] = defaultdict(int)
        for f in files:
            prior_authors: set[str] = set()
            for h_pn, h_m, h_a in file_history.get(f, []):
                if h_pn >= pr["pull_number"]:
                    continue
                in_window = True
                if pr_merged is not None and h_m is not None:
                    t0 = pr_merged if pr_merged.tzinfo else pr_merged.replace(tzinfo=timezone.utc)
                    t1 = h_m       if h_m.tzinfo       else h_m.replace(tzinfo=timezone.utc)
                    in_window = 0 <= (t0 - t1).total_seconds() / 86400 <= window_days
                if in_window:
                    prior_authors.add(h_a)
                    author_counts[h_a] += 1
            if len(prior_authors) > 1:
                friction_files += 1

        result[str(pr["pull_number"])] = {
            "n_files":            n_files,
            "n_packages":         n_packages,
            "cross_module_ratio": round(cross_module_ratio, 4),
            "ownership_friction": round(friction_files / n_files, 4),
            "author_entropy":     round(_entropy(author_counts), 3),
            "packages":           sorted(unique_packages),
        }
    return result


def _score_patch_org(patch: str, repo: str, index: dict) -> dict:
    files = _parse_patch_files(patch)
    if not files:
        return {"n_files": 0, "n_packages": 0, "cross_module_ratio": 0.0}
    packages = [_top_dir(f) for f in files]
    unique_packages = set(packages)
    n_files = len(files)
    if n_files > 1:
        cross = sum(1 for i, pa in enumerate(packages)
                    for pb in packages[i+1:] if pa != pb)
        total = n_files * (n_files - 1) // 2
        cross_module_ratio = cross / total
    else:
        cross_module_ratio = 0.0
    return {
        "n_files": n_files, "n_packages": len(unique_packages),
        "cross_module_ratio": round(cross_module_ratio, 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--language",    default="Python")
    ap.add_argument("--repos-from",  default=None,
                    help="JSONL with instance_id to restrict repos; omit = all")
    ap.add_argument("--out",         required=True)
    ap.add_argument("--window-days", type=int, default=180)
    ap.add_argument("--pg-yaml",     default=os.path.join(ROOT, "postgres_connection.yaml"))
    ap.add_argument("--eval-patches-baseline", default=None)
    ap.add_argument("--eval-patches-steered",  default=None)
    ap.add_argument("--iid-list",    default=None)
    args = ap.parse_args()

    conn = _pg_connect(args.pg_yaml)
    cur  = conn.cursor()

    # ── Determine target repos ───────────────────────────────────────────────
    if args.repos_from:
        ref_rows = [json.loads(l) for l in open(args.repos_from) if l.strip()]
        repo_set: set[str] = set()
        for r in ref_rows:
            iid = r.get("instance_id") or r.get("iid_pr", "")
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

    # ── Compute org metrics ──────────────────────────────────────────────────
    index: dict[str, dict] = {}
    for i, repo in enumerate(repos):
        prs = _fetch_repo_prs(conn, repo, args.language)
        if not prs:
            print(f"  [{i+1}/{len(repos)}] {repo}: 0 PRs — skipping", flush=True)
            continue
        metrics = _compute_repo_org_metrics(prs, args.window_days)
        index[repo] = metrics
        cmr_vals  = [m["cross_module_ratio"] for m in metrics.values()]
        fric_vals = [m["ownership_friction"] for m in metrics.values()]
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
    print(f"Repos: {len(index)}")

    out_jsonl = args.out.replace(".json", ".jsonl")
    with open(out_jsonl, "w") as f:
        for repo, pr_metrics in index.items():
            for pr_key, m in pr_metrics.items():
                row = {"repo": repo, "pull_number": int(pr_key)}
                row.update({k: v for k, v in m.items() if k != "packages"})
                f.write(json.dumps(row) + "\n")
    print(f"Flat JSONL → {out_jsonl}")

    # ── Optional eval ────────────────────────────────────────────────────────
    if args.eval_patches_baseline and args.eval_patches_steered:
        print("\n── Scoring steered vs baseline (org metrics) ──")
        iid_filter = None
        if args.iid_list:
            iid_filter = {json.loads(l)["instance_id"]
                          for l in open(args.iid_list) if l.strip() and "instance_id" in json.loads(l)}
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
            b = _score_patch_org(b_patch, repo, index)
            s = _score_patch_org(open(s_path).read(), repo, index)
            results.append({
                "instance_id": iid, "repo": repo,
                "b_cross_module_ratio": b["cross_module_ratio"],
                "s_cross_module_ratio": s["cross_module_ratio"],
                "b_n_packages": b["n_packages"], "s_n_packages": s["n_packages"],
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
            print(f"\n  Pairs: {n}")
            print(f"  {'cross_module_ratio':<25}  B={avg_b_cmr:.4f}  S={avg_s_cmr:.4f}  steered↓={lcm:.1%}")
            print(f"  {'n_packages':<25}  B={avg_b_pkg:.2f}  S={avg_s_pkg:.2f}  steered↓={lpk:.1%}")
            out_eval = args.out.replace(".json", "_eval.jsonl")
            with open(out_eval, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            print(f"  Eval → {out_eval}")

if __name__ == "__main__":
    main()
