#!/usr/bin/env python3
"""Merge JSONL shards and write refinement-history summary/report."""
from __future__ import annotations

import argparse
import glob
import json
import os

from extract_pr_refinement_history import _build_summary, _write_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", dest="glob_pattern", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--report-out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob_pattern))
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    rows.sort(key=lambda r: (r["repo"], r["instance_id"], r["commit_idx"]))
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = _build_summary(rows)
    summary["shard_files"] = paths
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    _write_report(summary, args.report_out, args.limit or None, args.out_jsonl, args.summary_out)
    print(f"Merged {len(rows)} rows from {len(paths)} shard files")
    print(f"Summary -> {args.summary_out}")
    print(f"Report -> {args.report_out}")


if __name__ == "__main__":
    main()
