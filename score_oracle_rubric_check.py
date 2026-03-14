#!/usr/bin/env python3
"""
Oracle rubric sanity check for Experiment 4.7.

Compares ground-truth merged patches (from go_prs Postgres table) against
the baseline agent patches using the same judge panel and rubric as the steered
eval.  If the rubric is sound, judges should consistently prefer oracle > baseline.

Usage:
    source .venv/bin/activate
    python score_oracle_rubric_check.py \\
        --baseline-patch-dir data/go_ablation_v1/patches_baseline \\
        --baseline-traj-dir  data/go_ablation_v1/trajs_baseline \\
        --existing-panel-dir data/go_ablation_v1/judge_panel \\
        --out-dir            data/go_ablation_v1/oracle_rubric_check \\
        --judge-models gpt-5-codex qwen3.5_397b_a17b_judge claude_opus_4_6 glm_5 \\
        --concurrency 16
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_JUDGES = [
    "gpt-5-codex",
    "qwen3.5_397b_a17b_judge",
    "claude_opus_4_6",
    "glm_5",
]


def _load_scorer():
    spec = importlib.util.spec_from_file_location(
        "score_patch_judge_4_7",
        os.path.join(ROOT, "score_patch_judge_4_7.py"),
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _pg_connect(pg_yaml: str):
    cfg = yaml.safe_load(open(pg_yaml))
    return psycopg2.connect(
        host=cfg["ip"], port=cfg["port"],
        user=cfg["user"], password=cfg["password"],
        database=cfg["database"],
    )


def _parse_iid(iid_pr: str) -> tuple[str, int]:
    """go_prs__owner__repo__NNNN__prNNNN → (owner/repo, pull_number)."""
    s = iid_pr.removeprefix("go_prs__")
    parts = s.split("__")
    pull_number = int(parts[-2])
    owner = parts[0]
    repo_name = parts[1]
    return f"{owner}/{repo_name}", pull_number


def _fetch_oracle_patches(
    conn,
    iid_prs: list[str],
) -> dict[str, str]:
    """Return {iid_pr: non_test_patch_text} for all found rows."""
    cur = conn.cursor()
    result: dict[str, str] = {}
    missing = []
    for iid in iid_prs:
        repo, pr = _parse_iid(iid)
        cur.execute(
            "SELECT non_test_patch FROM go_prs WHERE repo=%s AND pull_number=%s",
            (repo, pr),
        )
        row = cur.fetchone()
        if row and row[0] and len(row[0].strip()) >= 50:
            result[iid] = row[0]
        else:
            missing.append(iid)
    if missing:
        print(f"  WARNING: {len(missing)} iid_prs not found / empty in go_prs: {missing[:3]}")
    return result


def _build_oracle_pairs(
    iid_prs: list[str],
    oracle_patches: dict[str, str],
    baseline_patch_dir: str,
    baseline_traj_dir: str,
    scorer,
) -> list[dict]:
    """Build (oracle_patch, baseline_patch) pairs for the judge to evaluate."""
    pairs = []
    for iid in iid_prs:
        oracle_patch = oracle_patches.get(iid)
        if not oracle_patch:
            continue

        baseline_path = os.path.join(baseline_patch_dir, f"{iid}.patch")
        if not os.path.exists(baseline_path):
            continue
        baseline_patch = open(baseline_path).read()
        if len(baseline_patch.strip()) < 50:
            continue

        # Get problem statement from baseline traj (fallback to empty string)
        problem = ""
        traj_path = os.path.join(baseline_traj_dir, f"{iid}__a0.traj.json")
        if os.path.exists(traj_path):
            problem = scorer._extract_problem_from_traj(traj_path)
        # If no baseline traj, try any file matching the iid prefix
        if not problem:
            for fname in os.listdir(baseline_traj_dir):
                if fname.startswith(iid) and fname.endswith(".traj.json"):
                    problem = scorer._extract_problem_from_traj(
                        os.path.join(baseline_traj_dir, fname)
                    )
                    break

        pairs.append({
            "iid_pr":         iid,
            # Use the same field names as _score_pair expects
            "steered_patch":  oracle_patch,   # oracle takes the "steered" slot
            "baseline_patch": baseline_patch,
            "problem":        problem,
        })
    return pairs


def _run_judge(
    model_name: str,
    pairs: list[dict],
    model_cfg: dict,
    out_path: str,
    concurrency: int,
    seed: int,
    scorer,
) -> list[dict]:
    """Score all pairs with one judge; write incrementally to out_path."""
    # Idempotent: skip if already fully scored
    if os.path.exists(out_path):
        done = [json.loads(l) for l in open(out_path) if l.strip()]
        if len(done) >= len(pairs):
            print(f"  [{model_name}] Already done ({len(done)} rows) — loading")
            return done

    results: list[dict] = []
    with open(out_path, "w") as fout:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(scorer._score_pair, p, model_cfg, seed): p["iid_pr"]
                for p in pairs
            }
            done_n = 0
            for fut in as_completed(futures):
                iid_pr = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {"iid_pr": iid_pr, "error": str(exc)}
                results.append(result)
                fout.write(json.dumps(result) + "\n")
                fout.flush()
                done_n += 1
                if done_n % 10 == 0 or done_n == len(pairs):
                    wins = sum(1 for r in results if r.get("steered_wins"))  # oracle_wins
                    errs = sum(1 for r in results if "error" in r)
                    valid_n = done_n - errs
                    rate = wins / valid_n if valid_n else 0.0
                    print(f"    [{done_n}/{len(pairs)}] oracle_win_rate={rate:.1%}  errors={errs}")
    return results


def _summarise(all_results: dict[str, list[dict]]) -> dict:
    summary = {}
    for model, rows in all_results.items():
        valid = [r for r in rows if "error" not in r]
        errs  = len(rows) - len(valid)
        wins  = sum(1 for r in valid if r.get("steered_wins"))  # oracle wins
        n     = len(valid)
        summary[model] = {
            "n_valid":          n,
            "n_errors":         errs,
            "oracle_wins":      wins,
            "baseline_wins":    n - wins,
            "oracle_win_rate":  round(wins / n, 4) if n else 0.0,
        }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-patch-dir",  required=True)
    ap.add_argument("--baseline-traj-dir",   required=True)
    ap.add_argument("--existing-panel-dir",  required=True,
                    help="Dir with per-judge .jsonl from the steered eval (to get the 52 iid_prs)")
    ap.add_argument("--existing-model",      default="gpt-5-codex")
    ap.add_argument("--out-dir",             required=True)
    ap.add_argument("--judge-models",        nargs="+", default=DEFAULT_JUDGES)
    ap.add_argument("--models-yaml",         default=os.path.join(ROOT, "models.yaml"))
    ap.add_argument("--pg-yaml",             default=os.path.join(ROOT, "postgres_connection.yaml"))
    ap.add_argument("--concurrency",         type=int, default=16)
    ap.add_argument("--seed",                type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scorer = _load_scorer()

    # ── Load the 52 iid_prs from the steered-eval panel ───────────────────
    ref_path = os.path.join(args.existing_panel_dir, f"{args.existing_model}.jsonl")
    ref_rows  = [json.loads(l) for l in open(ref_path) if l.strip()]
    iid_prs   = [r["iid_pr"] for r in ref_rows if "error" not in r]
    print(f"Using {len(iid_prs)} iid_prs from {args.existing_model} steered-eval scores")

    # ── Fetch oracle patches from Postgres ────────────────────────────────
    print("Fetching oracle patches from Postgres (go_prs.non_test_patch)...")
    conn = _pg_connect(args.pg_yaml)
    oracle_patches = _fetch_oracle_patches(conn, iid_prs)
    conn.close()
    print(f"  Found oracle patches for {len(oracle_patches)}/{len(iid_prs)} pairs")

    # ── Build oracle-vs-baseline pairs ────────────────────────────────────
    pairs = _build_oracle_pairs(
        iid_prs, oracle_patches,
        args.baseline_patch_dir,
        args.baseline_traj_dir,
        scorer,
    )
    print(f"Built {len(pairs)} oracle-vs-baseline pairs")

    # ── Load model configs ────────────────────────────────────────────────
    cfg_by_name = {
        m["model_name"]: m
        for m in yaml.safe_load(open(args.models_yaml))["model_list"]
    }

    # ── Run each judge ────────────────────────────────────────────────────
    all_results: dict[str, list[dict]] = {}
    for model_name in args.judge_models:
        model_cfg = cfg_by_name.get(model_name)
        if not model_cfg:
            print(f"  WARN: {model_name} not in models.yaml — skipping")
            continue
        out_path = os.path.join(args.out_dir, f"{model_name}.jsonl")
        print(f"\n[{model_name}] Scoring {len(pairs)} oracle-vs-baseline pairs "
              f"(concurrency={args.concurrency})...")
        rows = _run_judge(model_name, pairs, model_cfg, out_path,
                          args.concurrency, args.seed, scorer)
        all_results[model_name] = rows

    # ── Summary ───────────────────────────────────────────────────────────
    summary = _summarise(all_results)
    summary_path = os.path.join(args.out_dir, "oracle_rubric_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("Oracle (ground-truth) vs Baseline — Rubric Sanity Check")
    print(f"{'='*60}")
    print(f"  Pairs: {len(pairs)}")
    print()
    print(f"  {'Judge':<32}  {'Oracle win%':>11}  {'n_valid':>7}  {'errors':>6}")
    print("  " + "-" * 60)
    for model, s in sorted(summary.items(), key=lambda x: -x[1]["oracle_win_rate"]):
        print(f"  {model:<32}  {s['oracle_win_rate']:>10.1%}  "
              f"{s['n_valid']:>7}  {s['n_errors']:>6}")

    # Rubric quality verdict
    print()
    valid_judges = {m: s for m, s in summary.items() if s["n_valid"] >= 10}
    if valid_judges:
        avg_oracle_wr = sum(s["oracle_win_rate"] for s in valid_judges.values()) / len(valid_judges)
        print(f"  Avg oracle win rate across {len(valid_judges)} judges: {avg_oracle_wr:.1%}")
        if avg_oracle_wr >= 0.70:
            verdict = "GOOD — rubric reliably identifies higher-quality patches"
        elif avg_oracle_wr >= 0.55:
            verdict = "MARGINAL — rubric shows some signal but may need refinement"
        else:
            verdict = "POOR — rubric cannot distinguish oracle from random agent output; needs rework"
        print(f"  Rubric verdict: {verdict}")
    print()
    print(f"  Summary → {summary_path}")


if __name__ == "__main__":
    main()
