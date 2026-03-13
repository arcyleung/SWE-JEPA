#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from extract_pr_refinement_history import _build_summary

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(ROOT, "data", "phase4_7_2_slurm_ramcache_v1", "merged.jsonl")
DEFAULT_JSON = os.path.join(ROOT, "data", "phase4_7_2_merge_sensitivity.json")
DEFAULT_MD = os.path.join(ROOT, "docs", "phase4_7_2_merge_sensitivity.md")

KEY_METRICS = [
    "conway_risk_proxy",
    "conway_risk_flags",
    "api_change_without_tests",
    "public_api_without_docs",
    "shared_change_isolated",
    "ownership_diffusion",
    "boundary_density",
    "operability_score",
]


def _load_rows(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _slice_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "all":
        return rows
    if mode == "exclude_merge_rows":
        return [r for r in rows if int(r.get("is_merge_commit_sampled", 0) or 0) == 0]
    if mode == "exclude_merge_prs":
        merge_prs = {r["instance_id"] for r in rows if int(r.get("is_merge_commit_sampled", 0) or 0) == 1}
        return [r for r in rows if r["instance_id"] not in merge_prs]
    raise ValueError(f"unsupported mode: {mode}")


def _median_pair(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "first": float(summary.get("median_risk_first", 0.0) or 0.0),
        "final": float(summary.get("median_risk_final", 0.0) or 0.0),
        "pre_review": float(summary.get("median_risk_response_pre", 0.0) or 0.0),
        "post_review": float(summary.get("median_risk_response_post", 0.0) or 0.0),
    }


def _summary_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _build_summary(rows)
    merge_rows = sum(int(r.get("is_merge_commit_sampled", 0) or 0) for r in rows)
    merge_prs = len({r["instance_id"] for r in rows if int(r.get("is_merge_commit_sampled", 0) or 0) == 1})
    return {
        "n_rows": int(summary["n_rows"]),
        "n_prs": int(summary["n_prs"]),
        "n_prs_multi_commit": int(summary["n_prs_multi_commit"]),
        "n_prs_with_review_response": int(summary["n_prs_with_review_response"]),
        "n_review_response_transitions": int(summary["n_review_response_transitions"]),
        "review_comment_events": int(summary["review_comment_events"]),
        "submitted_review_events": int(summary["submitted_review_events"]),
        "median_risk": _median_pair(summary),
        "first_to_final": summary["first_to_final"],
        "post_review_transition": summary["post_review_transition"],
        "merge_rows_present": merge_rows,
        "merge_prs_present": merge_prs,
    }


def _metric_diff(ref: dict[str, Any], cur: dict[str, Any], metric: str) -> dict[str, float]:
    ref_ff = ref["first_to_final"][metric]
    ref_pr = ref["post_review_transition"][metric]
    cur_ff = cur["first_to_final"][metric]
    cur_pr = cur["post_review_transition"][metric]
    return {
        "first_to_final_median_delta_change": float(cur_ff["median_delta"] - ref_ff["median_delta"]),
        "first_to_final_improved_fraction_change": float(cur_ff["improved_fraction"] - ref_ff["improved_fraction"]),
        "post_review_median_delta_change": float(cur_pr["median_delta"] - ref_pr["median_delta"]),
        "post_review_improved_fraction_change": float(cur_pr["improved_fraction"] - ref_pr["improved_fraction"]),
    }


def _render_report(payload: dict[str, Any], out_path: str) -> None:
    def pct(v: float) -> str:
        return f"{100.0 * v:.1f}%"

    def fmt(v: float) -> str:
        return f"{v:.3f}"

    all_summary = payload["slices"]["all"]
    lines = [
        "# Experiment 4.7.2 Merge-Commit Sensitivity",
        "",
        "This report re-runs the 4.7.2 refinement-history summary on the finished `merged.jsonl`",
        "under three slices:",
        "",
        "- `all`: original dataset",
        "- `exclude_merge_rows`: keep PRs but drop sampled merge-commit snapshots",
        "- `exclude_merge_prs`: drop any PR that sampled a merge commit",
        "",
        "## Dataset slices",
        "",
        "| Slice | PRs | Rows | Review-response transitions | Merge rows present | Merge PRs present |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in payload["slices"].items():
        lines.append(
            f"| `{name}` | {summary['n_prs']} | {summary['n_rows']} | "
            f"{summary['n_review_response_transitions']} | {summary['merge_rows_present']} | {summary['merge_prs_present']} |"
        )

    lines.extend(
        [
            "",
            "## Risk proxy medians",
            "",
            "| Slice | First | Final | Pre-review | Post-review |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, summary in payload["slices"].items():
        mr = summary["median_risk"]
        lines.append(
            f"| `{name}` | {fmt(mr['first'])} | {fmt(mr['final'])} | "
            f"{fmt(mr['pre_review'])} | {fmt(mr['post_review'])} |"
        )

    for name in ("exclude_merge_rows", "exclude_merge_prs"):
        summary = payload["slices"][name]
        diff = payload["vs_all"][name]
        lines.extend(
            [
                "",
                f"## Delta Comparison vs `all`: `{name}`",
                "",
                "| Metric | First->final improved | Change vs all | Post-review improved | Change vs all |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric in KEY_METRICS:
            ff = summary["first_to_final"][metric]
            pr = summary["post_review_transition"][metric]
            dd = diff[metric]
            lines.append(
                f"| `{metric}` | {pct(ff['improved_fraction'])} | {pct(dd['first_to_final_improved_fraction_change'])} | "
                f"{pct(pr['improved_fraction'])} | {pct(dd['post_review_improved_fraction_change'])} |"
            )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- `exclude_merge_rows` does not improve the primary `conway_risk_proxy` drift signal versus `all` if its improved fraction or post-review improved fraction falls below the original.",
            "- `exclude_merge_prs` is a stricter robustness check, but it is not the intended downstream training slice because it discards full PRs rather than only merge snapshots.",
            "",
            "The selection rule for downstream training is simple:",
            "",
            "- keep merge rows if the primary risk-proxy trend weakens when they are removed",
            "- otherwise train on the merge-excluded row slice",
        ]
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--json-out", default=DEFAULT_JSON)
    ap.add_argument("--report-out", default=DEFAULT_MD)
    args = ap.parse_args()

    rows = _load_rows(args.input)
    slices = {
        "all": _summary_payload(_slice_rows(rows, "all")),
        "exclude_merge_rows": _summary_payload(_slice_rows(rows, "exclude_merge_rows")),
        "exclude_merge_prs": _summary_payload(_slice_rows(rows, "exclude_merge_prs")),
    }
    payload = {
        "input": args.input,
        "rows_total": len(rows),
        "slices": slices,
        "vs_all": {
            name: {metric: _metric_diff(slices["all"], summary, metric) for metric in KEY_METRICS}
            for name, summary in slices.items()
            if name != "all"
        },
    }

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(payload, f, indent=2)
    _render_report(payload, args.report_out)
    print(json.dumps(
        {
            "json_out": args.json_out,
            "report_out": args.report_out,
            "all_prs": slices["all"]["n_prs"],
            "exclude_merge_rows_prs": slices["exclude_merge_rows"]["n_prs"],
            "exclude_merge_prs_prs": slices["exclude_merge_prs"]["n_prs"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
