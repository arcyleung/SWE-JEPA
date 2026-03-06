#!/usr/bin/env python3
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
OUT_JSONL = os.path.join(ROOT, "docs", "phase4_7_pr_mdp_dataset.jsonl")
OUT_SUMMARY = os.path.join(ROOT, "docs", "phase4_7_pr_mdp_summary.json")

REFACTOR_RE = re.compile(
    r"\b(refactor|restructure|split\s+this|extract\s+\w+|too\s+large|too\s+big|naming|cleanup|architecture|design)\b",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30000)
    ap.add_argument("--out", default=OUT_JSONL)
    ap.add_argument("--summary-out", default=OUT_SUMMARY)
    args = ap.parse_args()

    conn = _load_db()
    rows = conn.run(
        """
        SELECT
            repo, instance_id, pull_number, pr_merged, pr_state, pr_is_draft,
            pr_title, pr_body, changed_files, additions, deletions,
            requested_reviewers, total_review_threads, review_threads,
            total_comments, comments, closing_issue_id, created_at::text, merged_at::text
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
                repo,
                iid,
                pr_num,
                pr_merged,
                pr_state,
                pr_is_draft,
                pr_title,
                pr_body,
                changed_files,
                additions,
                deletions,
                requested_reviewers,
                total_review_threads,
                review_threads,
                total_comments,
                comments,
                closing_issue_id,
                created_at,
                merged_at,
            ) = r

            ref_comment_count, ref_thread_count = _count_refactor_mentions(review_threads, comments)
            refactor_requested = int(ref_comment_count > 0 or ref_thread_count > 0)
            reviewer_n = _requested_reviewer_count(requested_reviewers)
            churn = int((additions or 0) + (deletions or 0))
            review_friction = int((total_review_threads or 0) >= 3 or (total_comments or 0) >= 8 or refactor_requested)
            accepted = int(bool(pr_merged))

            # PR-evolution transition proxy: pre-review state -> observed review outcome state.
            item = {
                "repo": repo,
                "instance_id": iid,
                "pull_number": int(pr_num or 0),
                "s_t": {
                    "is_draft": int(bool(pr_is_draft)),
                    "changed_files": int(changed_files or 0),
                    "additions": int(additions or 0),
                    "deletions": int(deletions or 0),
                    "requested_reviewers_count": reviewer_n,
                    "has_closing_issue": int(bool(closing_issue_id)),
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
                "reward": float((1.0 if accepted else -1.0) - 0.2 * review_friction - 0.1 * refactor_requested),
                "meta": {
                    "title": pr_title or "",
                    "body": (pr_body or "")[:1500],
                    "created_at": created_at,
                    "merged_at": merged_at,
                },
            }
            f.write(json.dumps(item) + "\n")
            n += 1
            stats["accepted"] += accepted
            stats["refactor_requested"] += refactor_requested
            stats["review_friction"] += review_friction

    summary = {
        "rows": n,
        "accepted_rate": (stats["accepted"] / n) if n else 0.0,
        "refactor_requested_rate": (stats["refactor_requested"] / n) if n else 0.0,
        "review_friction_rate": (stats["review_friction"] / n) if n else 0.0,
        "out_jsonl": args.out,
    }
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
