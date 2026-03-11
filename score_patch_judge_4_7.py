#!/usr/bin/env python3
"""
Judge scoring for Experiment 4.7 steerer eval.

Compares steered vs non-steered agent patches using gpt-5-codex as a blind
A/B reviewer. The judge does NOT know which patch came from the steered agent.

Rubric: 10 high-level SE quality criteria (1-5 scale) + preference choice.
Criteria are NOT derived from the steerer's feature set — they are independent
senior-engineer quality proxies.

Usage:
    python score_patch_judge_4_7.py [--limit 100] [--concurrency 8] \
        [--out data/phase4_7_judge_scores.jsonl] \
        [--summary data/phase4_7_judge_summary.json]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Data paths ─────────────────────────────────────────────────────────────

TRAJ_DIR_STEERED  = os.path.join(ROOT, "data", "phase4_7_trajectories_steered_7k")
PATCH_DIR_BASELINE = os.path.join(ROOT, "data", "phase4_7_patches_feature_sl80")

# ── LiteLLM setup ──────────────────────────────────────────────────────────

def _load_model_cfg(models_yaml: str, model_name: str) -> dict:
    cfg = yaml.safe_load(open(models_yaml))
    for m in cfg["model_list"]:
        if m["model_name"] == model_name:
            return m
    raise ValueError(f"Model {model_name!r} not found in {models_yaml}")


def _chat(messages: list[dict], model_cfg: dict, max_tokens: int = 4096,
          retries: int = 3) -> str:
    """Call litellm endpoint; return assistant content string."""
    import litellm
    params = model_cfg["litellm_params"]
    api_base = params["api_base"]
    # Normalize: remove trailing /chat/completions if present
    if api_base.endswith("/chat/completions"):
        api_base = api_base[: -len("/chat/completions")]
    for attempt in range(retries):
        try:
            resp = litellm.completion(
                model=params["model"],
                messages=messages,
                api_base=api_base,
                api_key=params["api_key"],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


# ── Patch extraction from trajectories ────────────────────────────────────

def _extract_patch_from_traj(traj_path: str) -> str:
    """Extract the largest git diff block found in any tool output message."""
    try:
        traj = json.load(open(traj_path))
        best = ""
        for m in traj.get("messages", []):
            content = m.get("content", "")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict):
                        text = c.get("output", "") or c.get("text", "") or ""
                        if "diff --git" in text and len(text) > len(best):
                            best = text
            elif isinstance(content, str) and "diff --git" in content:
                if len(content) > len(best):
                    best = content
        return best
    except Exception:
        return ""


def _extract_problem_from_traj(traj_path: str) -> str:
    """Get the problem statement from the first user message in the traj."""
    try:
        traj = json.load(open(traj_path))
        for m in traj.get("messages", []):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            text = c["text"]
                            # Strip steering constraints (added by steered runner)
                            if "Steering constraints:" in text:
                                text = text[:text.index("Steering constraints:")].strip()
                            return text[:3000]
                elif isinstance(content, str) and len(content) > 20:
                    if "Steering constraints:" in content:
                        content = content[:content.index("Steering constraints:")].strip()
                    return content[:3000]
    except Exception:
        pass
    return ""


# ── Pair building ──────────────────────────────────────────────────────────

def build_pairs(limit: int | None = None, seed: int = 42) -> list[dict]:
    """Find instances with both a baseline patch and an extractable steered patch."""
    rng = random.Random(seed)

    # Index baseline patches by iid_pr key
    baseline_index: dict[str, str] = {}
    for fname in os.listdir(PATCH_DIR_BASELINE):
        if not fname.endswith(".patch"):
            continue
        if os.path.getsize(os.path.join(PATCH_DIR_BASELINE, fname)) < 50:
            continue
        key = fname[:-len(".patch")]
        baseline_index[key] = os.path.join(PATCH_DIR_BASELINE, fname)

    # Index steered trajs by iid_pr
    from collections import defaultdict as dd
    steered_index: dict[str, list[tuple[int, str]]] = dd(list)
    for fname in os.listdir(TRAJ_DIR_STEERED):
        if "__a" not in fname or not fname.endswith(".traj.json"):
            continue
        base = fname[:-len(".traj.json")]
        iid_pr, att_str = base.rsplit("__a", 1)
        steered_index[iid_pr].append((int(att_str), os.path.join(TRAJ_DIR_STEERED, fname)))

    pairs: list[dict] = []
    for iid_pr, atts in steered_index.items():
        if iid_pr not in baseline_index:
            continue
        # Pick the first (lowest index) attempt as "best" steered
        _, best_traj = sorted(atts)[0]
        steered_patch = _extract_patch_from_traj(best_traj)
        if not steered_patch or len(steered_patch.strip()) < 50:
            continue
        baseline_patch = open(baseline_index[iid_pr]).read()
        if len(baseline_patch.strip()) < 50:
            continue
        problem = _extract_problem_from_traj(best_traj)
        pairs.append({
            "iid_pr": iid_pr,
            "baseline_patch": baseline_patch,
            "steered_patch": steered_patch,
            "problem": problem,
            "baseline_patch_path": baseline_index[iid_pr],
            "steered_traj_path": best_traj,
        })

    rng.shuffle(pairs)
    if limit:
        pairs = pairs[:limit]
    return pairs


# ── Rubric and judge prompt ────────────────────────────────────────────────

CRITERIA = [
    ("correctness",       "Correctness: Does the patch actually address the stated problem? Are the changes logically sound and complete?"),
    ("code_quality",      "Code Quality: Readability, naming conventions, idiomatic use of the language. Is the code clean and maintainable?"),
    ("scope_discipline",  "Scope Discipline: Is the change focused and minimal? Does it avoid unnecessary file touches, refactors, or unrelated edits?"),
    ("error_handling",    "Error Handling: Are failure modes, edge cases, and exceptions handled appropriately? No silent swallowing of errors?"),
    ("security",          "Security: Does the change avoid introducing security issues (injection, secrets, unsafe deserialization, shell=True, etc.)?"),
    ("observability",     "Observability: Where warranted by the change, are logs, metrics, or health signals included or improved?"),
    ("test_coverage",     "Test Coverage: Does the patch add or update tests in proportion to the complexity of the change?"),
    ("interface_design",  "Interface Design: Are new functions, APIs, or public interfaces clean, well-typed, and consistent with the existing codebase?"),
    ("documentation",     "Documentation: Are complex or non-obvious parts of the change appropriately commented or documented?"),
    ("review_readiness",  "Review Readiness: Overall merge readiness — would you approve this PR as a senior engineer without requesting significant changes?"),
]

SYSTEM_PROMPT = """\
You are a senior software engineer with deep expertise in software architecture, \
security, observability, and code review. You have reviewed thousands of pull requests \
across many languages and frameworks.

You will be shown two PR patches (A and B) that both attempt to solve the same \
programming task. They were generated independently. Your goal is to evaluate \
them objectively and indicate which you would prefer to merge.

Important: evaluate each patch on its own merits. A shorter patch is not \
automatically better — it depends on whether it fully solves the problem. \
A larger patch is not automatically worse — it depends on whether the additional \
changes are necessary.

Respond ONLY in the exact JSON format requested. Do not include any text outside \
the JSON block.\
"""

def _build_prompt(problem: str, patch_a: str, patch_b: str) -> str:
    # Truncate patches to avoid token overflow (400k ctx but be conservative)
    MAX_PATCH = 30_000
    if len(patch_a) > MAX_PATCH:
        patch_a = patch_a[:MAX_PATCH] + "\n... [truncated]"
    if len(patch_b) > MAX_PATCH:
        patch_b = patch_b[:MAX_PATCH] + "\n... [truncated]"

    criteria_list = "\n".join(
        f"{i+1}. {desc}"
        for i, (_, desc) in enumerate(CRITERIA)
    )
    criteria_keys = ", ".join(f'"{k}"' for k, _ in CRITERIA)

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

Score each patch on the following criteria using a scale of **1–5**:
- 1 = Very poor / completely missing
- 2 = Below average / significant issues
- 3 = Adequate / meets minimum expectations
- 4 = Good / above average
- 5 = Excellent / exemplary

**Criteria:**

{criteria_list}

After scoring, state your overall preference: which patch would you merge?

Respond in **exactly** this JSON format (no other text):

```json
{{
  "scores_A": {{{criteria_keys}}},
  "scores_B": {{{criteria_keys}}},
  "preferred": "A",
  "justification": "2-3 sentence explanation of your preference focusing on the most important differences"
}}
```
Replace each value with an integer 1–5. "preferred" must be exactly "A" or "B".\
"""


# ── Scoring worker ─────────────────────────────────────────────────────────

def _score_pair(pair: dict, model_cfg: dict, seed: int = 42) -> dict:
    rng = random.Random(pair["iid_pr"] + str(seed))
    flip = rng.random() < 0.5  # randomize A/B assignment

    if flip:
        patch_a, patch_b = pair["steered_patch"], pair["baseline_patch"]
        a_is = "steered"
    else:
        patch_a, patch_b = pair["baseline_patch"], pair["steered_patch"]
        a_is = "baseline"

    prompt = _build_prompt(pair["problem"], patch_a, patch_b)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    try:
        raw = _chat(messages, model_cfg, max_tokens=2048)
    except Exception as e:
        return {
            "iid_pr": pair["iid_pr"],
            "error": str(e),
            "a_is": a_is,
        }

    # Parse JSON from response (handle markdown fences)
    text = raw.strip()
    if "```json" in text:
        text = text[text.index("```json") + 7:]
        text = text[:text.index("```")].strip()
    elif "```" in text:
        text = text[text.index("```") + 3:]
        text = text[:text.index("```")].strip()

    try:
        scored = json.loads(text)
    except json.JSONDecodeError:
        return {
            "iid_pr": pair["iid_pr"],
            "error": f"json_parse_failed: {raw[:300]}",
            "a_is": a_is,
            "raw_response": raw[:1000],
        }

    preferred_raw = scored.get("preferred", "").strip().upper()
    if preferred_raw not in ("A", "B"):
        return {
            "iid_pr": pair["iid_pr"],
            "error": f"invalid_preferred: {preferred_raw!r}",
            "a_is": a_is,
            "raw_response": raw[:500],
        }

    # Translate A/B back to steered/baseline
    if preferred_raw == "A":
        preferred_label = a_is  # "steered" or "baseline"
    else:
        preferred_label = "steered" if a_is == "baseline" else "baseline"

    scores_a = scored.get("scores_A", {})
    scores_b = scored.get("scores_B", {})

    # Map back to steered/baseline scores
    if a_is == "steered":
        scores_steered = scores_a
        scores_baseline = scores_b
    else:
        scores_steered = scores_b
        scores_baseline = scores_a

    return {
        "iid_pr":           pair["iid_pr"],
        "a_is":             a_is,
        "preferred_ab":     preferred_raw,
        "preferred_label":  preferred_label,       # "steered" or "baseline"
        "steered_wins":     preferred_label == "steered",
        "scores_steered":   scores_steered,
        "scores_baseline":  scores_baseline,
        "justification":    scored.get("justification", ""),
        "baseline_patch_chars": len(pair["baseline_patch"]),
        "steered_patch_chars":  len(pair["steered_patch"]),
    }


# ── Aggregate statistics ───────────────────────────────────────────────────

def _compute_summary(results: list[dict]) -> dict:
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if not valid:
        return {"error": "no valid results", "n_errors": len(errors)}

    n = len(valid)
    steered_wins = sum(1 for r in valid if r["steered_wins"])
    baseline_wins = n - steered_wins

    # Per-criterion mean scores
    criteria_keys = [k for k, _ in CRITERIA]
    mean_steered: dict[str, float] = {}
    mean_baseline: dict[str, float] = {}
    for k in criteria_keys:
        s_scores = [r["scores_steered"].get(k) for r in valid if isinstance(r["scores_steered"].get(k), (int, float))]
        b_scores = [r["scores_baseline"].get(k) for r in valid if isinstance(r["scores_baseline"].get(k), (int, float))]
        mean_steered[k]  = round(sum(s_scores) / len(s_scores), 3) if s_scores else 0.0
        mean_baseline[k] = round(sum(b_scores) / len(b_scores), 3) if b_scores else 0.0

    overall_steered  = round(sum(mean_steered.values())  / len(criteria_keys), 3)
    overall_baseline = round(sum(mean_baseline.values()) / len(criteria_keys), 3)

    return {
        "n_evaluated":       n,
        "n_errors":          len(errors),
        "steered_wins":      steered_wins,
        "baseline_wins":     baseline_wins,
        "steered_win_rate":  round(steered_wins / n, 4),
        "overall_mean_steered":  overall_steered,
        "overall_mean_baseline": overall_baseline,
        "delta_overall":    round(overall_steered - overall_baseline, 3),
        "per_criterion": {
            k: {
                "steered":  mean_steered[k],
                "baseline": mean_baseline[k],
                "delta":    round(mean_steered[k] - mean_baseline[k], 3),
            }
            for k in criteria_keys
        },
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",       type=int, default=100,
                    help="Max pairs to evaluate (default 100)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--models-yaml", default=os.path.join(ROOT, "models.yaml"))
    ap.add_argument("--judge-model", default="qwen3.5_397b_a17b")
    ap.add_argument("--out",         default=os.path.join(ROOT, "data", "phase4_7_judge_scores.jsonl"))
    ap.add_argument("--summary",     default=os.path.join(ROOT, "data", "phase4_7_judge_summary.json"))
    args = ap.parse_args()

    model_cfg = _load_model_cfg(args.models_yaml, args.judge_model)
    print(f"Judge model: {args.judge_model}  ({model_cfg['litellm_params']['model']})")

    print("Building comparable pairs...")
    pairs = build_pairs(limit=args.limit, seed=args.seed)
    print(f"  Found {len(pairs)} valid pairs (baseline + steered patch both non-empty)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results: list[dict] = []
    done = 0

    with open(args.out, "w") as fout:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(_score_pair, p, model_cfg, args.seed): p["iid_pr"]
                for p in pairs
            }
            for fut in as_completed(futures):
                iid_pr = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {"iid_pr": iid_pr, "error": str(exc)}
                results.append(result)
                fout.write(json.dumps(result) + "\n")
                fout.flush()
                done += 1
                if done % 10 == 0 or done == len(pairs):
                    wins = sum(1 for r in results if r.get("steered_wins"))
                    errs = sum(1 for r in results if "error" in r)
                    valid = done - errs
                    rate = wins / valid if valid else 0
                    print(f"  [{done}/{len(pairs)}]  steered_win_rate={rate:.2%}  errors={errs}")

    summary = _compute_summary(results)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    if "error" in summary:
        print(f"ERROR: {summary['error']}  ({summary.get('n_errors', 0)} errors)")
    else:
        print(f"RESULTS: {summary['n_evaluated']} evaluated, {summary['n_errors']} errors")
        print(f"Steered win rate:   {summary['steered_win_rate']:.1%}  ({summary['steered_wins']}/{summary['n_evaluated']})")
        print(f"Overall mean score: steered={summary['overall_mean_steered']}  baseline={summary['overall_mean_baseline']}  Δ={summary['delta_overall']:+.3f}")
        print()
        print(f"{'Criterion':<22}  {'Steered':>8}  {'Baseline':>8}  {'Delta':>8}")
        print("-" * 52)
        for k, v in summary["per_criterion"].items():
            bar = "▲" if v["delta"] > 0 else ("▼" if v["delta"] < 0 else " ")
            print(f"  {k:<20}  {v['steered']:>8.3f}  {v['baseline']:>8.3f}  {bar}{abs(v['delta']):>6.3f}")
    print()
    print(f"Scores  → {args.out}")
    print(f"Summary → {args.summary}")


if __name__ == "__main__":
    main()
