#!/usr/bin/env python3
"""
Conway-aligned judge for Experiment 4.7 steerer eval.

Uses a slimmed-down 4-criterion rubric that mirrors the steerer's training signal
(scope_drift, trust_boundary_crossings, conway_risk_proxy → bug / follow-up risk):

  1. bug_risk        — likely to introduce bugs
  2. followup_risk   — likely to require immediate follow-up PRs
  3. scope_discipline — minimal, focused, no scope drift
  4. overall_quality  — holistic merge readiness

Inverted scale: lower score = BETTER for bug_risk and followup_risk
(score 1 = very low risk, score 5 = very high risk).
scope_discipline and overall_quality use the normal higher-is-better scale.

Usage (oracle rubric check):
    python score_patch_judge_conway.py \\
        --mode oracle \\
        --baseline-patch-dir data/go_ablation_v1/patches_baseline \\
        --baseline-traj-dir  data/go_ablation_v1/trajs_baseline \\
        --oracle-iid-prs-from data/go_ablation_v1/judge_panel/gpt-5-codex.jsonl \\
        --out-dir  data/go_ablation_v1/conway_judge \\
        --judge-models gpt-5-codex qwen3.5_397b_a17b_judge claude_opus_4_6 glm_5

Usage (steered vs baseline):
    python score_patch_judge_conway.py \\
        --mode steered \\
        --baseline-patch-dir data/go_ablation_v1/patches_baseline \\
        --steered-traj-dir   data/go_ablation_v1/trajs_steered_specific \\
        --out-dir  data/go_ablation_v1/conway_judge \\
        --judge-models gpt-5-codex qwen3.5_397b_a17b_judge claude_opus_4_6 glm_5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_JUDGES = [
    "gpt-5-codex",
    "qwen3.5_397b_a17b_judge",
    "claude_opus_4_6",
    "glm_5",
]

# ── Conway rubric ──────────────────────────────────────────────────────────

CRITERIA = [
    ("bug_risk",
     "Bug Risk (lower is better, 1=very low, 5=very high): "
     "How likely is this patch to introduce bugs — brittle APIs, shared state access, crossing "
     "module/package boundaries, dependency on external systems, naming ambiguity, "
     "edge cases and recursion, complex control flow, etc."),
    ("followup_risk",
     "Follow-up Risk (lower is better, 1=very low, 5=very high): "
     "How likely is this patch to require follow-up PRs? error handling, unresolved TODOs/ FIXME, tight coupling, hardcoded constants, bandaid fixes"),
    ("scope_discipline",
     "Scope Discipline (higher is better, 1=very poor, 5=excellent): "
     "Is the patch minimal and precisely focused on the stated problem? "
     "Unnecessary file touches, unrelated refactors, API misuse semantics (Hyrum's law), changes outside the issue description."),
    ("overall_quality",
     "Overall Quality (higher is better, 1=very poor, 5=excellent): "
     "Holistic merge readiness — would you approve this PR as a senior engineer "
     "without requesting significant changes?"),
]

SYSTEM_PROMPT = """\
You are a senior software engineer reviewing two pull-request patches that both \
attempt to solve the same task. You will evaluate them on risk and quality dimensions \
that predict long-term stability: how likely each patch is to introduce bugs or require \
immediate follow-up work.

Important: a shorter patch is not automatically better — it depends on whether it fully \
solves the problem. Focus on risk signals, not patch size.

Respond ONLY in the exact JSON format requested. Do not include any text outside the JSON.\
"""

# Per-criterion comparison direction:
#   "which is better?" maps "A"→challenger wins, "B"→baseline wins, "tie"→draw
CRITERIA_QUESTIONS = {
    "bug_risk":         "Which patch has LOWER bug risk?",
    "followup_risk":    "Which patch is LESS LIKELY to need immediate follow-up PRs?",
    "scope_discipline": "Which patch is MORE focused and minimal in scope?",
    "overall_quality":  "Which patch has HIGHER overall merge readiness?",
}


def _build_prompt(problem: str, patch_a: str, patch_b: str) -> str:
    MAX_PATCH = 25_000
    if len(patch_a) > MAX_PATCH:
        patch_a = patch_a[:MAX_PATCH] + "\n... [truncated]"
    if len(patch_b) > MAX_PATCH:
        patch_b = patch_b[:MAX_PATCH] + "\n... [truncated]"

    criteria_desc = "\n".join(
        f"{i+1}. **{k}**: {desc}"
        for i, (k, desc) in enumerate(CRITERIA)
    )
    criteria_compare = "\n".join(
        f'  "{k}": "<A or B or tie>"  // {CRITERIA_QUESTIONS[k]}'
        for k, _ in CRITERIA
    )

    return f"""\
## Task / Problem Statement

{problem}

---

## Patch A

```diff
{patch_a}
```

---

## Patch B

```diff
{patch_b}
```

---

## Your Evaluation

For each criterion below, decide which patch is BETTER on that dimension, or if they are \
roughly equivalent write "tie". Then state your overall preferred patch (or "tie" if you \
would be equally comfortable merging either).

**Criteria definitions:**

{criteria_desc}

Respond in **exactly** this JSON format (no other text):

```json
{{
  "criterion_winner": {{
{criteria_compare}
  }},
  "preferred": "A",
  "justification": "2-3 sentence explanation focusing on the most important risk differences"
}}
```
Each criterion_winner value must be exactly "A", "B", or "tie". \
"preferred" must be exactly "A", "B", or "tie".\
"""


# ── LLM helpers (reuse from score_patch_judge_4_7) ─────────────────────────

def _load_base_scorer():
    spec = importlib.util.spec_from_file_location(
        "score_patch_judge_4_7",
        os.path.join(ROOT, "score_patch_judge_4_7.py"),
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _score_pair(pair: dict, model_cfg: dict, seed: int = 42) -> dict:
    """Score a (challenger vs baseline) pair with the Conway yes/no/tie rubric."""
    base = _load_base_scorer()
    rng  = random.Random(pair["iid_pr"] + str(seed))
    flip = rng.random() < 0.5

    if flip:
        patch_a, patch_b = pair["challenger_patch"], pair["baseline_patch"]
        a_is = "challenger"
    else:
        patch_a, patch_b = pair["baseline_patch"], pair["challenger_patch"]
        a_is = "baseline"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": _build_prompt(pair.get("problem", ""), patch_a, patch_b)},
    ]

    try:
        raw = base._chat(messages, model_cfg, max_tokens=1024)
    except Exception as e:
        return {"iid_pr": pair["iid_pr"], "error": str(e), "a_is": a_is}

    text = base._extract_json_from_response(raw)
    try:
        scored = json.loads(text)
    except json.JSONDecodeError:
        return {"iid_pr": pair["iid_pr"], "error": f"json_parse_failed: {raw[:300]}", "a_is": a_is}

    # Validate overall preferred
    preferred_raw = scored.get("preferred", "").strip().upper()
    if preferred_raw not in ("A", "B", "TIE"):
        return {"iid_pr": pair["iid_pr"], "error": f"invalid_preferred: {preferred_raw!r}", "a_is": a_is}

    # Translate A/B/TIE → challenger/baseline/tie
    if preferred_raw == "TIE":
        preferred_label = "tie"
    elif preferred_raw == "A":
        preferred_label = a_is                                          # "challenger" or "baseline"
    else:
        preferred_label = "baseline" if a_is == "challenger" else "challenger"

    # Parse per-criterion winners (A/B/tie in absolute terms → challenger/baseline/tie)
    crit_keys = [k for k, _ in CRITERIA]
    crit_winners_raw  = scored.get("criterion_winner", {})
    crit_winners: dict[str, str] = {}   # "challenger" | "baseline" | "tie"
    for k in crit_keys:
        raw_w = crit_winners_raw.get(k, "").strip().upper()
        if raw_w == "TIE":
            crit_winners[k] = "tie"
        elif raw_w == "A":
            crit_winners[k] = a_is
        elif raw_w == "B":
            crit_winners[k] = "baseline" if a_is == "challenger" else "challenger"
        else:
            crit_winners[k] = "unknown"

    # Advantage vector: +1 = challenger better, -1 = baseline better, 0 = tie/unknown
    advantage = {
        k: (1 if v == "challenger" else (-1 if v == "baseline" else 0))
        for k, v in crit_winners.items()
    }

    return {
        "iid_pr":          pair["iid_pr"],
        "a_is":            a_is,
        "preferred_label": preferred_label,          # "challenger" | "baseline" | "tie"
        "challenger_wins": preferred_label == "challenger",
        "criterion_winners": crit_winners,           # per-criterion challenger/baseline/tie
        "advantage":       advantage,                # +1/0/-1 per criterion
        "justification":   scored.get("justification", ""),
    }


# ── Data builders ──────────────────────────────────────────────────────────

def _pg_connect(pg_yaml: str):
    import psycopg2
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


def _build_oracle_pairs(
    iid_prs: list[str],
    baseline_patch_dir: str,
    baseline_traj_dir: str,
    pg_yaml: str,
    base_scorer,
) -> list[dict]:
    conn = _pg_connect(pg_yaml)
    cur  = conn.cursor()
    pairs = []
    for iid in iid_prs:
        repo, pr = _parse_iid(iid)
        cur.execute("SELECT non_test_patch FROM go_prs WHERE repo=%s AND pull_number=%s", (repo, pr))
        row = cur.fetchone()
        oracle_patch = row[0] if (row and row[0] and len(row[0].strip()) >= 50) else None
        if not oracle_patch:
            continue
        baseline_path = os.path.join(baseline_patch_dir, f"{iid}.patch")
        if not os.path.exists(baseline_path):
            continue
        baseline_patch = open(baseline_path).read()
        if len(baseline_patch.strip()) < 50:
            continue
        problem = ""
        traj_path = os.path.join(baseline_traj_dir, f"{iid}__a0.traj.json")
        if os.path.exists(traj_path):
            problem = base_scorer._extract_problem_from_traj(traj_path)
        pairs.append({
            "iid_pr":          iid,
            "challenger_patch": oracle_patch,
            "baseline_patch":  baseline_patch,
            "problem":         problem,
            "mode":            "oracle",
        })
    conn.close()
    print(f"  Built {len(pairs)} oracle-vs-baseline pairs")
    return pairs


def _build_steered_pairs(
    baseline_patch_dir: str,
    steered_traj_dir: str,
    iid_pr_filter: set[str] | None,
    base_scorer,
) -> list[dict]:
    all_pairs = base_scorer.build_pairs(
        limit=None, seed=42,
        patch_dir=baseline_patch_dir,
        steered_traj_dir=steered_traj_dir,
    )
    pairs = []
    for p in all_pairs:
        if iid_pr_filter and p["iid_pr"] not in iid_pr_filter:
            continue
        pairs.append({
            "iid_pr":          p["iid_pr"],
            "challenger_patch": p["steered_patch"],
            "baseline_patch":  p["baseline_patch"],
            "problem":         p.get("problem", ""),
            "mode":            "steered",
        })
    print(f"  Built {len(pairs)} steered-vs-baseline pairs")
    return pairs


# ── Per-judge runner ───────────────────────────────────────────────────────

def _run_judge(
    model_name: str,
    mode: str,
    pairs: list[dict],
    model_cfg: dict,
    out_path: str,
    concurrency: int,
    seed: int,
) -> list[dict]:
    if os.path.exists(out_path):
        done = [json.loads(l) for l in open(out_path) if l.strip()]
        if len(done) >= len(pairs):
            print(f"  [{model_name}] Already done ({len(done)} rows) — loading")
            return done

    results: list[dict] = []
    with open(out_path, "w") as fout:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_score_pair, p, model_cfg, seed): p["iid_pr"]
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
                    wins = sum(1 for r in results if r.get("challenger_wins"))
                    errs = sum(1 for r in results if "error" in r)
                    vn   = done_n - errs
                    rate = wins / vn if vn else 0.0
                    label = "oracle" if mode == "oracle" else "steered"
                    print(f"    [{done_n}/{len(pairs)}] {label}_win_rate={rate:.1%}  errors={errs}")
    return results


def _compute_summary(results: list[dict], mode: str) -> dict:
    valid = [r for r in results if "error" not in r]
    errs  = [r for r in results if "error" in r]
    n     = len(valid)
    if n == 0:
        return {"error": "no valid results", "n_errors": len(errs)}

    label = "oracle" if mode == "oracle" else "steered"
    wins  = sum(1 for r in valid if r.get("challenger_wins"))
    ties  = sum(1 for r in valid if r.get("preferred_label") == "tie")

    crit_keys = [k for k, _ in CRITERIA]
    per_criterion: dict[str, dict] = {}
    for k in crit_keys:
        c_wins  = sum(1 for r in valid if r.get("criterion_winners", {}).get(k) == "challenger")
        b_wins  = sum(1 for r in valid if r.get("criterion_winners", {}).get(k) == "baseline")
        c_ties  = sum(1 for r in valid if r.get("criterion_winners", {}).get(k) == "tie")
        adv_sum = sum(r.get("advantage", {}).get(k, 0) for r in valid)
        per_criterion[k] = {
            "challenger_wins": c_wins,
            "baseline_wins":   b_wins,
            "ties":            c_ties,
            "mean_advantage":  round(adv_sum / n, 3),   # avg of +1/0/-1
        }

    return {
        "mode":               mode,
        "n_evaluated":        n,
        "n_errors":           len(errs),
        f"{label}_wins":      wins,
        "baseline_wins":      n - wins - ties,
        "ties":               ties,
        f"{label}_win_rate":  round(wins / n, 4),
        "tie_rate":           round(ties / n, 4),
        "per_criterion":      per_criterion,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",                  choices=["oracle", "steered"], required=True)
    ap.add_argument("--baseline-patch-dir",    required=True)
    ap.add_argument("--baseline-traj-dir",     default=None,
                    help="Required for oracle mode (problem statement extraction)")
    ap.add_argument("--steered-traj-dir",      default=None,
                    help="Required for steered mode")
    ap.add_argument("--oracle-iid-prs-from",   default=None,
                    help="Oracle mode: .jsonl with iid_pr field to filter pairs")
    ap.add_argument("--out-dir",               required=True)
    ap.add_argument("--judge-models",          nargs="+", default=DEFAULT_JUDGES)
    ap.add_argument("--models-yaml",           default=os.path.join(ROOT, "models.yaml"))
    ap.add_argument("--pg-yaml",               default=os.path.join(ROOT, "postgres_connection.yaml"))
    ap.add_argument("--concurrency",           type=int, default=8)
    ap.add_argument("--seed",                  type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base_scorer = _load_base_scorer()

    # ── Build pairs ───────────────────────────────────────────────────────
    if args.mode == "oracle":
        if not args.baseline_traj_dir or not args.oracle_iid_prs_from:
            ap.error("oracle mode requires --baseline-traj-dir and --oracle-iid-prs-from")
        ref_rows = [json.loads(l) for l in open(args.oracle_iid_prs_from) if l.strip()]
        iid_prs  = [r["iid_pr"] for r in ref_rows if "error" not in r]
        print(f"Oracle mode: {len(iid_prs)} iid_prs from {args.oracle_iid_prs_from}")
        pairs = _build_oracle_pairs(
            iid_prs, args.baseline_patch_dir, args.baseline_traj_dir,
            args.pg_yaml, base_scorer,
        )
    else:  # steered
        if not args.steered_traj_dir:
            ap.error("steered mode requires --steered-traj-dir")
        iid_filter: set[str] | None = None
        if args.oracle_iid_prs_from:
            ref_rows  = [json.loads(l) for l in open(args.oracle_iid_prs_from) if l.strip()]
            iid_filter = {r["iid_pr"] for r in ref_rows if "error" not in r}
        pairs = _build_steered_pairs(
            args.baseline_patch_dir, args.steered_traj_dir, iid_filter, base_scorer,
        )

    print(f"Total pairs: {len(pairs)}")
    cfg_by_name = {
        m["model_name"]: m
        for m in yaml.safe_load(open(args.models_yaml))["model_list"]
    }

    # ── Score each judge ──────────────────────────────────────────────────
    all_summaries: dict[str, dict] = {}
    for model_name in args.judge_models:
        model_cfg = cfg_by_name.get(model_name)
        if not model_cfg:
            print(f"  WARN: {model_name} not found — skipping")
            continue

        suffix = f"{args.mode}_{model_name}"
        out_path = os.path.join(args.out_dir, f"{suffix}.jsonl")
        print(f"\n[{model_name}] Scoring {len(pairs)} pairs (mode={args.mode}, concurrency={args.concurrency})...")
        rows = _run_judge(model_name, args.mode, pairs, model_cfg, out_path,
                          args.concurrency, args.seed)
        summary = _compute_summary(rows, args.mode)
        all_summaries[model_name] = summary
        summ_path = os.path.join(args.out_dir, f"{suffix}_summary.json")
        with open(summ_path, "w") as f:
            json.dump(summary, f, indent=2)

    # ── Combined report ───────────────────────────────────────────────────
    label = "oracle" if args.mode == "oracle" else "steered"
    combined_path = os.path.join(args.out_dir, f"{args.mode}_combined_summary.json")
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'='*65}")
    title = "Oracle (GT) vs Baseline" if args.mode == "oracle" else "Steered vs Baseline"
    print(f"{title} — Conway Rubric  ({len(pairs)} pairs)")
    print(f"{'='*65}")
    print(f"  {'Judge':<32}  {label.capitalize()+' win%':>12}  {'n_valid':>7}  {'errors':>6}")
    print("  " + "-" * 62)
    rates = []
    for model, s in sorted(all_summaries.items(), key=lambda x: -x[1].get(f"{label}_win_rate", 0)):
        wr = s.get(f"{label}_win_rate", 0)
        rates.append(wr)
        print(f"  {model:<32}  {wr:>11.1%}  {s.get('n_evaluated',0):>7}  {s.get('n_errors',0):>6}")
    if rates:
        avg = sum(rates) / len(rates)
        valid_judges = [s for s in all_summaries.values() if s.get("n_evaluated", 0) >= 10]
        avg = sum(s.get(f"{label}_win_rate", 0) for s in valid_judges) / len(valid_judges) if valid_judges else 0
        print()
        print(f"  Avg {label} win rate: {avg:.1%}")
        if args.mode == "oracle":
            if avg >= 0.70:
                verdict = "GOOD — rubric reliably identifies higher-quality patches"
            elif avg >= 0.55:
                verdict = "MARGINAL"
            else:
                verdict = "POOR — rubric cannot distinguish oracle from agent output"
            print(f"  Rubric verdict: {verdict}")

    print()
    crit_keys = [k for k, _ in CRITERIA]
    if all_summaries:
        print(f"  Per-criterion {label} win rate (across all judges):")
        print(f"  {'Criterion':<20}  {'Challenger%':>11}  {'Baseline%':>9}  {'Tie%':>6}  {'AvgAdv':>7}")
        print("  " + "-" * 60)
        for k in crit_keys:
            cw_rates, bw_rates, tie_rates, advs = [], [], [], []
            for s in all_summaries.values():
                pc = s.get("per_criterion", {}).get(k)
                if pc and s.get("n_evaluated", 0) >= 5:
                    nn = s["n_evaluated"]
                    cw_rates.append(pc["challenger_wins"] / nn)
                    bw_rates.append(pc["baseline_wins"] / nn)
                    tie_rates.append(pc["ties"] / nn)
                    advs.append(pc["mean_advantage"])
            if cw_rates:
                avg_cw  = sum(cw_rates)  / len(cw_rates)
                avg_bw  = sum(bw_rates)  / len(bw_rates)
                avg_tie = sum(tie_rates) / len(tie_rates)
                avg_adv = sum(advs)      / len(advs)
                print(f"  {k:<20}  {avg_cw:>10.1%}  {avg_bw:>9.1%}  {avg_tie:>5.1%}  {avg_adv:>+7.3f}")

    print()
    print(f"  Combined summary → {combined_path}")


if __name__ == "__main__":
    main()
