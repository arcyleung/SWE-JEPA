#!/usr/bin/env python3
"""
Multi-judge panel evaluation for Experiment 4.7 steerer eval.

Loads existing gpt-5-codex scores for the `specific` hint-mode condition, runs
the same 52 pairs through 5 additional judges (all via the unified gateway at
http://1.95.77.23:3000), then computes inter-judge agreement statistics:

  • Pairwise Cohen's kappa (binary steered-wins preference)
  • Fleiss' kappa (all N judges simultaneously)
  • Per-criterion disagreement score: mean variance of δ=(steered−baseline) across judges
  • Recommendation: most disagreed criterion → reword or drop from weighted scheme

Usage:
    python score_judge_panel_4_7.py \\
        --baseline-patch-dir data/go_ablation_v1/patches_baseline \\
        --steered-traj-dir   data/go_ablation_v1/trajs_steered_specific \\
        --existing-scores    data/go_ablation_v1/judge_scores_specific.jsonl \\
        --existing-model     gpt-5-codex \\
        --out-dir            data/go_ablation_v1/judge_panel \\
        --concurrency        16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))

# Default panel (excluding gpt-5-codex which is already run)
DEFAULT_PANEL = [
    "kimi_k2.5",
    "qwen3.5_397b_a17b_judge",
    "claude_opus_4_6",
    "gemini_31_pro_thinking",
    "glm_5",
]


# ── Re-use scoring infrastructure from score_patch_judge_4_7 ──────────────

def _import_scorer():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "score_patch_judge_4_7",
        os.path.join(ROOT, "score_patch_judge_4_7.py"),
    )
    mod = importlib.util.load_from_spec(spec)  # type: ignore[attr-defined]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_scorer_module():
    import importlib.util
    path = os.path.join(ROOT, "score_patch_judge_4_7.py")
    spec = importlib.util.spec_from_file_location("score_patch_judge_4_7", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _quorum_iid_prs(
    judge_scores: dict[str, list[dict]],
    all_iid_prs: list[str],
    min_judges: int,
) -> list[str]:
    """Return iid_prs evaluated without error by at least min_judges judges."""
    from collections import Counter
    counts: Counter = Counter()
    for model, rows in judge_scores.items():
        for row in rows:
            if "error" not in row and row.get("iid_pr") in set(all_iid_prs):
                counts[row["iid_pr"]] += 1
    return [ip for ip in all_iid_prs if counts[ip] >= min_judges]


# ── Agreement statistics ───────────────────────────────────────────────────

def _cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Binary Cohen's kappa between two raters (1=steered wins, 0=baseline wins)."""
    n = len(labels_a)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    pa1 = sum(labels_a) / n
    pb1 = sum(labels_b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def _fleiss_kappa(ratings_matrix: list[list[int]], n_categories: int = 2) -> float:
    """Fleiss' kappa for N raters × M subjects, binary categories.

    ratings_matrix[i][j] = 1 if rater j says steered wins for pair i, else 0.
    """
    M = len(ratings_matrix)           # subjects (pairs)
    N = len(ratings_matrix[0]) if M else 0   # raters
    if M == 0 or N < 2:
        return float("nan")

    # Count how many raters assigned each category per subject
    # n_ij[i][c] = count of raters assigning category c to subject i
    n_ij: list[list[int]] = []
    for row in ratings_matrix:
        counts = [sum(1 for v in row if v == c) for c in range(n_categories)]
        n_ij.append(counts)

    # P_i = proportion of agreeing pairs for subject i
    P_i = []
    for counts in n_ij:
        s = sum(c * (c - 1) for c in counts)
        P_i.append(s / (N * (N - 1)))

    P_bar = statistics.mean(P_i)

    # p_j = proportion of all assignments in category j
    total = M * N
    p_j = [sum(n_ij[i][c] for i in range(M)) / total for c in range(n_categories)]
    P_e = sum(pj ** 2 for pj in p_j)

    if P_e >= 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def _criterion_disagreement(
    judge_scores: dict[str, list[dict]],
    criteria: list[str],
    iid_prs: list[str],
) -> dict[str, dict]:
    """
    For each criterion: compute mean variance of δ=(steered−baseline) across judges.

    Returns {criterion: {"mean_delta": float, "disagreement": float, "judge_deltas": {judge: float}}}
    Sorted by disagreement descending.
    """
    # judge_scores[model_name] = list of valid score rows (keyed by iid_pr)
    iid_set = set(iid_prs)
    # Build per-(iid_pr, criterion) deltas for each judge
    # deltas[iid_pr][criterion][judge] = steered_score - baseline_score
    deltas: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for model, rows in judge_scores.items():
        for row in rows:
            if row.get("iid_pr") not in iid_set or "error" in row:
                continue
            ss = row.get("scores_steered", {})
            bs = row.get("scores_baseline", {})
            for c in criteria:
                sv = ss.get(c)
                bv = bs.get(c)
                if isinstance(sv, (int, float)) and isinstance(bv, (int, float)):
                    deltas[row["iid_pr"]][c][model] = float(sv) - float(bv)

    results: dict[str, dict] = {}
    for c in criteria:
        pair_variances = []
        all_deltas: list[float] = []
        judge_totals: dict[str, list[float]] = defaultdict(list)
        for iid_pr in iid_prs:
            d = deltas.get(iid_pr, {}).get(c, {})
            if len(d) >= 2:
                vals = list(d.values())
                if len(vals) >= 2:
                    pair_variances.append(statistics.variance(vals))
                all_deltas.extend(vals)
                for m, v in d.items():
                    judge_totals[m].append(v)

        mean_delta = statistics.mean(all_deltas) if all_deltas else 0.0
        disagreement = statistics.mean(pair_variances) if pair_variances else 0.0
        judge_mean_deltas = {m: round(statistics.mean(vs), 3) for m, vs in judge_totals.items()}
        results[c] = {
            "mean_delta":    round(mean_delta, 3),
            "disagreement":  round(disagreement, 3),
            "judge_deltas":  judge_mean_deltas,
        }

    return dict(sorted(results.items(), key=lambda x: -x[1]["disagreement"]))


def _compute_panel_summary(
    judge_scores: dict[str, list[dict]],
    criteria: list[str],
    iid_prs: list[str],
    min_judges: int = 2,
) -> dict:
    """Full inter-judge agreement report using quorum-based common pairs."""
    models = list(judge_scores.keys())
    iid_set = set(iid_prs)

    # Build preference matrix: pref[model][iid_pr] = 1 if steered wins else 0
    pref: dict[str, dict[str, int]] = {}
    win_rates: dict[str, float] = {}
    for model, rows in judge_scores.items():
        d: dict[str, int] = {}
        for row in rows:
            if row.get("iid_pr") in iid_set and "error" not in row:
                d[row["iid_pr"]] = int(row.get("steered_wins", False))
        pref[model] = d
        n = len(d)
        win_rates[model] = round(sum(d.values()) / n, 4) if n else 0.0

    # Quorum pairs: evaluated by at least min_judges judges
    common = _quorum_iid_prs(judge_scores, iid_prs, min_judges)
    # For Fleiss' kappa use only pairs evaluated by ALL judges
    full_common = [ip for ip in iid_prs if all(ip in pref[m] for m in models)]

    # Fleiss' kappa over pairs evaluated by ALL judges
    if len(full_common) >= 2 and len(models) >= 2:
        ratings_matrix = [[pref[m][ip] for m in models] for ip in full_common]
        fk = round(_fleiss_kappa(ratings_matrix), 4)
    else:
        fk = float("nan")

    # Pairwise Cohen's kappa (uses pairs evaluated by both judges in each pair)
    pairwise: dict[str, dict[str, float]] = defaultdict(dict)
    for i, ma in enumerate(models):
        for mb in models[i + 1:]:
            shared = [ip for ip in iid_prs if ip in pref[ma] and ip in pref[mb]]
            if len(shared) < 5:
                kappa = float("nan")
            else:
                kappa = round(_cohen_kappa(
                    [pref[ma][ip] for ip in shared],
                    [pref[mb][ip] for ip in shared],
                ), 4)
            pairwise[ma][mb] = kappa
            pairwise[mb][ma] = kappa

    # Majority vote win rate
    majority_wins = 0
    for ip in common:
        votes = sum(pref[m].get(ip, 0) for m in models)
        if votes > len(models) / 2:
            majority_wins += 1
    majority_win_rate = round(majority_wins / len(common), 4) if common else 0.0

    # Per-criterion disagreement
    crit_disagree = _criterion_disagreement(judge_scores, criteria, common)

    most_disagreed = max(crit_disagree, key=lambda c: crit_disagree[c]["disagreement"]) if crit_disagree else None
    least_disagreed = min(crit_disagree, key=lambda c: crit_disagree[c]["disagreement"]) if crit_disagree else None

    return {
        "n_common_pairs":        len(common),
        "n_judges":              len(models),
        "judges":                models,
        "win_rates_per_judge":   win_rates,
        "majority_win_rate":     majority_win_rate,
        "fleiss_kappa":          fk,
        "pairwise_kappa":        dict(pairwise),
        "per_criterion":         crit_disagree,
        "most_disagreed_criterion":  most_disagreed,
        "least_disagreed_criterion": least_disagreed,
        "recommendation": _make_recommendation(crit_disagree, majority_win_rate, most_disagreed),
    }


def _make_recommendation(
    crit_disagree: dict[str, dict],
    majority_win_rate: float,
    most_disagreed: str | None,
) -> dict:
    rec: dict = {}

    if most_disagreed:
        d = crit_disagree[most_disagreed]
        rec["most_disagreed_criterion"] = most_disagreed
        rec["disagreement_score"] = d["disagreement"]
        rec["mean_delta_across_judges"] = d["mean_delta"]
        if d["disagreement"] > 1.5:
            rec["action"] = "drop"
            rec["rationale"] = (
                f"'{most_disagreed}' has high inter-judge variance (σ²={d['disagreement']:.2f}) "
                "and judges fundamentally disagree on which patch is better — "
                "this criterion adds noise rather than signal. Drop it from the weighted scheme."
            )
        elif d["mean_delta"] < -0.3:
            rec["action"] = "reword"
            rec["rationale"] = (
                f"'{most_disagreed}' consistently hurts the steered score (Δ={d['mean_delta']:+.3f}) "
                "and has high inter-judge variance. Consider rewording to focus on "
                "'does the patch add appropriate coverage given its scope' rather than absolute coverage."
            )
        else:
            rec["action"] = "monitor"
            rec["rationale"] = f"'{most_disagreed}' has moderate disagreement. Monitor across more pairs before rewording."

    if majority_win_rate < 0.50:
        rec["steerer_signal_weak"] = True
        rec["alternatives"] = [
            {
                "name": "SWE-bench test execution",
                "description": (
                    "Run the generated patch through the repo's own test suite. "
                    "Pass/fail is ground-truth, unaffected by judge bias. "
                    "Requires spinning up per-repo Docker environments (SWE-bench infra)."
                ),
            },
            {
                "name": "Ground-truth patch similarity",
                "description": (
                    "Embed the generated patch and the merged PR patch with a code embedding model "
                    "(e.g. UniXcoder, CodeBERT). High cosine similarity = closer to what the human chose. "
                    "Captures intent alignment without LLM judge bias."
                ),
            },
            {
                "name": "Conway-metric oracle",
                "description": (
                    "Directly measure the steerer's own features (scope_drift, trust_boundary_crossings, "
                    "conway_risk_proxy) on the generated patch vs baseline. If the steerer signal improves "
                    "these features, it's doing its job — validates the architecture independently of judge quality."
                ),
            },
            {
                "name": "Reviewer acceptance logistic model",
                "description": (
                    "Train a logistic model on (Conway features → accepted) using the Go PR label data. "
                    "Score generated patches with it. This is the steerer's own proxy — if steered patches "
                    "score higher on it, the steerer is self-consistent."
                ),
            },
            {
                "name": "Larger / more focused test set",
                "description": (
                    "52 pairs is marginal for detecting a ~50% signal. Use all 96 baseline-ok pairs "
                    "and restrict to 'hard' PRs (refactor_requested=1) where the steerer should have "
                    "the most impact. Run judge panel on 150+ pairs."
                ),
            },
        ]
    else:
        rec["steerer_signal_weak"] = False

    return rec


def _format_report(summary: dict) -> str:
    lines = ["# Judge Panel Agreement Report", ""]
    lines += [f"- Common pairs evaluated by all {summary['n_judges']} judges: **{summary['n_common_pairs']}**"]
    lines += [f"- Fleiss' κ (all judges, binary preference): **{summary['fleiss_kappa']:.4f}**"]
    lines += [f"- Majority-vote steered win rate: **{summary['majority_win_rate']:.1%}**", ""]

    lines += ["## Win Rate per Judge", ""]
    lines += ["| Judge | Steered win rate |", "|-------|-----------------|"]
    for m, wr in sorted(summary["win_rates_per_judge"].items(), key=lambda x: -x[1]):
        lines.append(f"| {m} | {wr:.1%} |")

    lines += ["", "## Pairwise Cohen's κ (preference agreement)", ""]
    judges = summary["judges"]
    header = "| |" + "|".join(f" {j} " for j in judges) + "|"
    sep    = "|---|" + "|".join("---" for _ in judges) + "|"
    lines += [header, sep]
    pk = summary["pairwise_kappa"]
    for ja in judges:
        row = f"| **{ja}** |"
        for jb in judges:
            if ja == jb:
                row += " — |"
            else:
                k = pk.get(ja, {}).get(jb, float("nan"))
                row += f" {k:.3f} |" if not math.isnan(k) else " — |"
        lines.append(row)

    lines += ["", "## Per-Criterion Disagreement (sorted by inter-judge variance)", ""]
    lines += ["| Criterion | Mean Δ (S−B) | Inter-judge variance | Action |", "|-----------|-------------|---------------------|--------|"]
    most = summary.get("most_disagreed_criterion")
    for c, v in summary["per_criterion"].items():
        flag = " ← most disagreed" if c == most else ""
        lines.append(f"| {c} | {v['mean_delta']:+.3f} | {v['disagreement']:.3f} | {flag} |")

    rec = summary.get("recommendation", {})
    if rec:
        lines += ["", "## Recommendation", ""]
        lines.append(f"**Most disagreed criterion:** `{rec.get('most_disagreed_criterion', '—')}`")
        lines.append(f"**Action:** {rec.get('action', '—')} — {rec.get('rationale', '')}")
        if rec.get("steerer_signal_weak"):
            lines += ["", "### Steered win rate < 50% — alternative evaluation methods", ""]
            for alt in rec.get("alternatives", []):
                lines.append(f"**{alt['name']}**: {alt['description']}")
                lines.append("")

    return "\n".join(lines) + "\n"


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-patch-dir", required=True)
    ap.add_argument("--steered-traj-dir",   required=True)
    ap.add_argument("--existing-scores",    required=True,
                    help="Path to existing gpt-5-codex judge_scores .jsonl")
    ap.add_argument("--existing-model",     default="gpt-5-codex")
    ap.add_argument("--judge-models",       nargs="+", default=DEFAULT_PANEL)
    ap.add_argument("--models-yaml",        default=os.path.join(ROOT, "models.yaml"))
    ap.add_argument("--out-dir",            required=True)
    ap.add_argument("--concurrency",        type=int, default=16)
    ap.add_argument("--seed",               type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scorer = _load_scorer_module()

    # ── Load existing gpt-5-codex scores ──────────────────────────────────
    existing_rows = [json.loads(l) for l in open(args.existing_scores) if l.strip()]
    valid_existing = [r for r in existing_rows if "error" not in r]
    evaluated_iid_prs = [r["iid_pr"] for r in valid_existing]
    print(f"Loaded {len(valid_existing)} existing {args.existing_model} scores "
          f"({len(existing_rows) - len(valid_existing)} errors)")

    # Copy existing scores into panel dir
    existing_out = os.path.join(args.out_dir, f"{args.existing_model}.jsonl")
    with open(existing_out, "w") as f:
        for row in existing_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved existing scores → {existing_out}")

    # ── Rebuild pairs filtered to the existing evaluated set ──────────────
    all_pairs = scorer.build_pairs(
        limit=None,
        seed=args.seed,
        patch_dir=args.baseline_patch_dir,
        steered_traj_dir=args.steered_traj_dir,
    )
    target_set = set(evaluated_iid_prs)
    pairs = [p for p in all_pairs if p["iid_pr"] in target_set]
    print(f"Rebuilt {len(pairs)} pairs matching existing evaluated set")

    # ── Load model configs ─────────────────────────────────────────────────
    models_cfg = yaml.safe_load(open(args.models_yaml))
    cfg_by_name = {m["model_name"]: m for m in models_cfg["model_list"]}

    # ── Score each new judge ───────────────────────────────────────────────
    all_judge_scores: dict[str, list[dict]] = {args.existing_model: valid_existing}

    for model_name in args.judge_models:
        model_cfg = cfg_by_name.get(model_name)
        if not model_cfg:
            print(f"  WARN: {model_name} not found in models.yaml — skipping")
            continue

        out_path = os.path.join(args.out_dir, f"{model_name}.jsonl")
        # Idempotent: skip if already fully scored
        if os.path.exists(out_path):
            done = [json.loads(l) for l in open(out_path) if l.strip()]
            if len(done) >= len(pairs):
                print(f"  [{model_name}] Already done ({len(done)} rows) — skipping")
                all_judge_scores[model_name] = [r for r in done if "error" not in r]
                continue

        print(f"  [{model_name}] Scoring {len(pairs)} pairs (concurrency={args.concurrency})...")
        results: list[dict] = []
        with open(out_path, "w") as fout:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {
                    pool.submit(scorer._score_pair, p, model_cfg, args.seed): p["iid_pr"]
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
                        wins = sum(1 for r in results if r.get("steered_wins"))
                        errs = sum(1 for r in results if "error" in r)
                        valid_n = done_n - errs
                        rate = wins / valid_n if valid_n else 0.0
                        print(f"    [{done_n}/{len(pairs)}] steered_win_rate={rate:.1%}  errors={errs}")

        valid_results = [r for r in results if "error" not in r]
        all_judge_scores[model_name] = valid_results
        print(f"  [{model_name}] Done → {out_path}")

    # ── Inter-judge agreement analysis ────────────────────────────────────
    print("\nComputing inter-judge agreement...")
    criteria = [k for k, _ in scorer.CRITERIA]
    summary = _compute_panel_summary(all_judge_scores, criteria, evaluated_iid_prs)

    summary_path = os.path.join(args.out_dir, "panel_agreement.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    report_path = os.path.join(args.out_dir, "panel_agreement.md")
    with open(report_path, "w") as f:
        f.write(_format_report(summary))

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Pairs evaluated: {summary['n_common_pairs']}  Judges: {summary['n_judges']}")
    print(f"Fleiss' κ: {summary['fleiss_kappa']:.4f}   "
          f"Majority win rate: {summary['majority_win_rate']:.1%}")
    print()
    print(f"{'Judge':<30}  {'Win rate':>9}")
    print("-" * 42)
    for m, wr in sorted(summary["win_rates_per_judge"].items(), key=lambda x: -x[1]):
        print(f"  {m:<28}  {wr:.1%}")
    print()
    print("Per-criterion disagreement (highest variance first):")
    print(f"  {'Criterion':<22}  {'Mean Δ':>8}  {'Variance':>9}")
    print("  " + "-" * 42)
    for c, v in summary["per_criterion"].items():
        flag = " ◄ most disagreed" if c == summary.get("most_disagreed_criterion") else ""
        print(f"  {c:<22}  {v['mean_delta']:>+8.3f}  {v['disagreement']:>9.3f}{flag}")
    print()
    rec = summary.get("recommendation", {})
    if rec.get("action"):
        print(f"Recommendation: {rec['action'].upper()} '{rec['most_disagreed_criterion']}'")
        print(f"  {rec['rationale']}")
    print()
    print(f"Agreement report → {report_path}")
    print(f"JSON summary     → {summary_path}")


if __name__ == "__main__":
    main()
