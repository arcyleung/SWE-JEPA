#!/usr/bin/env python3
"""Build extended MDP dataset for Experiment 5.1 (Conway-Aware RL Steerer).

Extends build_pr_mdp_dataset.py with:
- cross_module_spread: log1p(# distinct top-level dirs), Conway ownership friction proxy
- has_tests: binary, test files present in PR → test coverage proxy
- churn_asymmetry: deletions / (additions + deletions + 1), refactoring vs pure addition
- followup_risk: pre-computed per-instance score from Exp 4.3 probe cache (0.0 if unavailable)
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

import pg8000.native
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
PG_CONFIG_FILE = os.path.join(ROOT, "postgres_connection.yaml")
OUT_JSONL = os.path.join(ROOT, "data", "phase5_1_pr_mdp_dataset_v51.jsonl")
OUT_SUMMARY = os.path.join(ROOT, "data", "phase5_1_pr_mdp_summary_v51.json")

# Optional cache from Exp 4.3 region localization probe
FOLLOWUP_RISK_CACHE = os.path.join(ROOT, "data", "phase4_3_topk_predictions.jsonl")

REFACTOR_RE = re.compile(
    r"\b(refactor|restructure|split\s+this|extract\s+\w+|too\s+large|too\s+big|naming|cleanup|architecture|design)\b",
    re.IGNORECASE,
)
TEST_FILE_RE = re.compile(
    r"(test[_/]|[_/]test|\.test\.|_spec\.|spec[_/]|[_/]spec|__tests?__)",
    re.IGNORECASE,
)


def _load_db():
    cfg = yaml.safe_load(open(PG_CONFIG_FILE))
    return pg8000.native.Connection(
        host=cfg["ip"],
        port=cfg.get("port", 9999),
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
    )


def _j(v):
    if isinstance(v, (list, dict)):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return []
    return []


def _count_refactor_mentions(review_threads, comments) -> tuple[int, int]:
    ref_comments = 0
    ref_threads = 0
    for th in _j(review_threads):
        if not isinstance(th, dict):
            continue
        hit_thread = False
        for c in _j(th.get("comments", [])):
            if isinstance(c, dict) and REFACTOR_RE.search(str(c.get("body", ""))):
                ref_comments += 1
                hit_thread = True
        if hit_thread:
            ref_threads += 1
    for c in _j(comments):
        if isinstance(c, dict) and REFACTOR_RE.search(str(c.get("body", ""))):
            ref_comments += 1
    return ref_comments, ref_threads


def _requested_reviewer_count(v) -> int:
    rv = _j(v)
    return len(rv) if isinstance(rv, list) else 0


def _parse_filenames(fp_raw) -> list[str]:
    """Extract list of filenames from file_patches JSON (GitHub API list-of-dicts or plain list)."""
    try:
        fp = _j(fp_raw)
        if isinstance(fp, list):
            names = []
            for item in fp:
                if isinstance(item, dict):
                    fn = item.get("filename") or item.get("file") or item.get("path") or ""
                    if fn:
                        names.append(str(fn))
                elif isinstance(item, str) and item:
                    names.append(item)
            return names
    except Exception:
        pass
    return []


def _cross_module_spread(filenames: list[str]) -> int:
    """Count distinct top-level directory prefixes — Conway ownership friction proxy."""
    dirs: set[str] = set()
    for f in filenames:
        parts = f.split("/")
        dirs.add(parts[0] if len(parts) > 1 else ".")
    return len(dirs)


def _has_tests(filenames: list[str]) -> int:
    return int(any(TEST_FILE_RE.search(f) for f in filenames))


def _load_followup_risk_cache(path: str) -> dict[str, float]:
    """Load per-instance_id mean followup risk from Exp 4.3 predictions if available."""
    cache: dict[str, list[float]] = {}
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                iid = rec.get("instance_id") or rec.get("feature_instance_id", "")
                score = float(rec.get("score", 0.0))
                if iid:
                    cache.setdefault(iid, []).append(score)
        return {k: float(sum(v) / len(v)) for k, v in cache.items()}
    except Exception:
        return {}


def _load_llm_labels(path: str) -> dict[tuple[str, int], dict]:
    """Load pre-computed LLM refactor labels from label_refactor_llm.py output."""
    labels: dict[tuple[str, int], dict] = {}
    with open(path) as f:
        for ln in f:
            if not ln.strip():
                continue
            row = json.loads(ln)
            key = (str(row["repo"]), int(row["pull_number"]))
            labels[key] = row
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30000)
    ap.add_argument("--out", default=OUT_JSONL)
    ap.add_argument("--summary-out", default=OUT_SUMMARY)
    ap.add_argument("--followup-risk-cache", default=FOLLOWUP_RISK_CACHE)
    ap.add_argument("--llm-labels", default=None,
                    help="Path to LLM-judged refactor labels JSONL. When provided, replaces regex labeling.")
    args = ap.parse_args()

    risk_cache = _load_followup_risk_cache(args.followup_risk_cache)
    print(f"Loaded followup_risk cache: {len(risk_cache)} instance_ids")

    llm_label_map: dict[tuple[str, int], dict] = {}
    if args.llm_labels:
        llm_label_map = _load_llm_labels(args.llm_labels)
        print(f"Loaded {len(llm_label_map)} LLM refactor labels from {args.llm_labels}")

    conn = _load_db()
    rows = conn.run(
        """
        SELECT
            repo, instance_id, pull_number, pr_merged, pr_state, pr_is_draft,
            pr_title, pr_body, changed_files, additions, deletions,
            requested_reviewers, total_review_threads, review_threads,
            total_comments, comments, closing_issue_id, created_at::text, merged_at::text,
            file_patches
        FROM prs_copy
        WHERE patch IS NOT NULL
          AND file_patches IS NOT NULL
          AND changed_files BETWEEN 1 AND 120
          AND (COALESCE(additions,0) + COALESCE(deletions,0)) BETWEEN 5 AND 8000
        ORDER BY created_at DESC
        LIMIT :lim
        """,
        lim=args.limit,
    )
    conn.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    stats = Counter()

    with open(args.out, "w") as f:
        for r in rows:
            (
                repo, iid, pr_num, pr_merged, pr_state, pr_is_draft,
                pr_title, pr_body, changed_files, additions, deletions,
                requested_reviewers, total_review_threads, review_threads,
                total_comments, comments, closing_issue_id, created_at, merged_at,
                file_patches,
            ) = r

            llm_key = (str(repo), int(pr_num or 0))
            if llm_label_map and llm_key in llm_label_map:
                llm_row = llm_label_map[llm_key]
                refactor_requested = int(bool(llm_row.get("refactor_requested", False)))
                ref_comment_count = int(llm_row.get("refactor_thread_count", 0))
                ref_thread_count = ref_comment_count
            else:
                ref_comment_count, ref_thread_count = _count_refactor_mentions(review_threads, comments)
                refactor_requested = int(ref_comment_count > 0 or ref_thread_count > 0)
            reviewer_n = _requested_reviewer_count(requested_reviewers)
            churn = int((additions or 0) + (deletions or 0))
            review_friction = int(
                (total_review_threads or 0) >= 3 or (total_comments or 0) >= 8 or refactor_requested
            )
            accepted = int(bool(pr_merged))

            # Conway proxy features
            filenames = _parse_filenames(file_patches)
            spread = _cross_module_spread(filenames)
            has_tests = _has_tests(filenames)
            add = int(additions or 0)
            dele = int(deletions or 0)
            churn_asym = dele / (add + dele + 1)

            # Longitudinal followup risk (from Exp 4.3 probe cache, 0.0 if unavailable)
            followup_risk = risk_cache.get(str(iid), 0.0)

            item = {
                "repo": repo,
                "instance_id": iid,
                "pull_number": int(pr_num or 0),
                "s_t": {
                    # Original Exp 4.7 features
                    "is_draft": int(bool(pr_is_draft)),
                    "changed_files": int(changed_files or 0),
                    "additions": add,
                    "deletions": dele,
                    "requested_reviewers_count": reviewer_n,
                    "has_closing_issue": int(bool(closing_issue_id)),
                    # New v51 Conway proxy features
                    "cross_module_spread": spread,
                    "has_tests": has_tests,
                    "churn_asymmetry": round(churn_asym, 4),
                    # Longitudinal signal from Exp 4.3
                    "followup_risk": round(followup_risk, 4),
                },
                "a_t": {
                    "action": "submit_or_update_pr",
                },
                "s_t1": {
                    "total_review_threads": int(total_review_threads or 0),
                    "total_comments": int(total_comments or 0),
                    "refactor_comment_count": int(ref_comment_count),
                    "refactor_thread_count": int(ref_thread_count),
                    "refactor_requested": refactor_requested,
                    "review_friction": review_friction,
                    "accepted": accepted,
                    "pr_state": pr_state or "",
                },
                "reward": float(
                    (1.0 if accepted else -1.0)
                    - 0.2 * review_friction
                    - 0.1 * refactor_requested
                ),
                "meta": {
                    "title": pr_title or "",
                    "body": (pr_body or "")[:1500],
                    "created_at": created_at,
                    "merged_at": merged_at,
                    "n_files": len(filenames),
                    "followup_risk_from_cache": followup_risk > 0.0,
                },
            }
            f.write(json.dumps(item) + "\n")
            n += 1
            stats["accepted"] += accepted
            stats["refactor_requested"] += refactor_requested
            stats["review_friction"] += review_friction
            stats["has_tests"] += has_tests
            stats["multi_module"] += int(spread > 2)
            stats["followup_risk_cached"] += int(followup_risk > 0.0)

    summary = {
        "rows": n,
        "accepted_rate": (stats["accepted"] / n) if n else 0.0,
        "refactor_requested_rate": (stats["refactor_requested"] / n) if n else 0.0,
        "review_friction_rate": (stats["review_friction"] / n) if n else 0.0,
        "has_tests_rate": (stats["has_tests"] / n) if n else 0.0,
        "multi_module_rate": (stats["multi_module"] / n) if n else 0.0,
        "followup_risk_cached_rate": (stats["followup_risk_cached"] / n) if n else 0.0,
        "out_jsonl": args.out,
    }
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
