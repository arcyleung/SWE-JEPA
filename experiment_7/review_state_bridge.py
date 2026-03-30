#!/usr/bin/env python3
"""Shared review-state bridge for Experiment 7.1.

This module does two jobs:

1. Deterministic teacher-side tag extraction from a concrete patch diff.
2. Symbolic state -> steering text rendering from predicted tags / clusters.

The intent is to keep the "decoder" simple and auditable. The student predicts
latent state and tags; this bridge turns that state into the same kind of
targeted review feedback used by the v3 steered agent.
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

TEST_PAT = re.compile(r"test[_s]|_test\.py|tests/|spec/", re.IGNORECASE)
API_PAT = re.compile(r"api|route|endpoint|handler|view|schema|serializer", re.IGNORECASE)
EVAL_EXEC_PAT = re.compile(r"eval\s*\(|exec\s*\(")
SQL_KEYWORD_PAT = re.compile(r"\b(?:SELECT|INSERT|UPDATE|DELETE)\b", re.IGNORECASE)
FSTRING_PREFIX_PAT = re.compile(r"""(?:^|[^A-Za-z0-9_])(?:[rRuUbB]{0,2}[fF]|[fF][rRuUbB]{0,2})["']""")
HARDCODED_CREDENTIAL_PAT = re.compile(
    r"""(?:password|secret|token|api_key)\s*=\s*["'][^"']{4,}""",
    re.IGNORECASE,
)
BARE_EXCEPT_PAT = re.compile(r"except\s*:")
EXCEPT_PASS_PAT = re.compile(r"except[^\n]*:\s*\n\s*pass")
HTTP_CALL_PAT = re.compile(r"requests\.(get|post|put|delete|patch)\(|httpx\.|aiohttp\.")
TIMEOUT_PAT = re.compile(r"timeout\s*=")
SHARED_DIRS = {"utils", "common", "lib", "shared", "helpers", "core", "base", "compat"}

TAG_MESSAGES: dict[str, str] = {
    "patch_too_large": (
        "Patch scope looks broad. Remove unrelated edits and keep the change "
        "focused on the task requirements."
    ),
    "shared_without_tests": (
        "Shared or utility code changed without matching tests. Add regression "
        "coverage or localize the change to the implementation site."
    ),
    "api_without_tests": (
        "API or interface code changed without tests. Add integration or API-level "
        "coverage for the changed surface."
    ),
    "eval_exec": (
        "Avoid eval()/exec() in the patch. Replace with explicit dispatch or a "
        "safer parsing mechanism."
    ),
    "sql_fstring": (
        "The patch may be constructing SQL unsafely. Use parameterized queries "
        "instead of string interpolation."
    ),
    "hardcoded_credential": (
        "The patch appears to introduce a hardcoded secret or credential. Move "
        "that value to configuration or environment."
    ),
    "bare_except": (
        "Avoid bare except clauses. Catch specific exceptions so failures stay visible."
    ),
    "except_pass": (
        "Avoid silently swallowing exceptions with except/pass. Log the error or re-raise."
    ),
    "http_without_timeout": (
        "Network calls appear to be added without explicit timeouts. Add timeouts "
        "to prevent hangs in production."
    ),
}

TAG_NAMES = list(TAG_MESSAGES.keys())


def _has_sql_fstring(patch_text: str) -> bool:
    """Cheap SQL f-string detector without catastrophic backtracking on large diffs."""
    if "{" not in patch_text or "}" not in patch_text:
        return False
    if SQL_KEYWORD_PAT.search(patch_text) is None:
        return False
    for line in patch_text.splitlines():
        if FSTRING_PREFIX_PAT.search(line) is None:
            continue
        if "{" not in line or "}" not in line:
            continue
        if SQL_KEYWORD_PAT.search(line):
            return True
    return False


def count_patch_stats(patch_text: str) -> dict[str, Any]:
    """Extract basic stats from a unified diff."""
    files: list[str] = []
    additions = 0
    deletions = 0
    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            m = re.match(r"diff --git a/(.*?) b/(.*)", line)
            if m:
                files.append(m.group(2))
        elif line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return {
        "files": files,
        "n_files": len(files),
        "additions": additions,
        "deletions": deletions,
    }


def classify_files(files: list[str]) -> dict[str, list[str]]:
    """Classify patch files into rough semantic buckets."""
    result: dict[str, list[str]] = {"test": [], "shared": [], "api": [], "impl": []}
    for f in files:
        parts = PurePosixPath(f).parts
        if TEST_PAT.search(f):
            result["test"].append(f)
        elif any(p.lower() in SHARED_DIRS for p in parts):
            result["shared"].append(f)
        elif API_PAT.search(f):
            result["api"].append(f)
        else:
            result["impl"].append(f)
    return result


def detect_review_issue_flags(patch_text: str) -> dict[str, int]:
    """Return deterministic binary issue tags for a patch diff."""
    stats = count_patch_stats(patch_text)
    classified = classify_files(stats["files"])

    flags = {name: 0 for name in TAG_NAMES}
    flags["patch_too_large"] = int(stats["n_files"] > 8)
    flags["shared_without_tests"] = int(bool(classified["shared"]) and not classified["test"])
    flags["api_without_tests"] = int(bool(classified["api"]) and not classified["test"])
    flags["eval_exec"] = int(bool(EVAL_EXEC_PAT.search(patch_text)))
    flags["sql_fstring"] = int(_has_sql_fstring(patch_text))
    flags["hardcoded_credential"] = int(bool(HARDCODED_CREDENTIAL_PAT.search(patch_text)))
    flags["bare_except"] = int(bool(BARE_EXCEPT_PAT.search(patch_text)))
    flags["except_pass"] = int(bool(EXCEPT_PASS_PAT.search(patch_text)))
    has_http = bool(HTTP_CALL_PAT.search(patch_text))
    has_timeout = bool(TIMEOUT_PAT.search(patch_text))
    flags["http_without_timeout"] = int(has_http and not has_timeout)
    return flags


def render_review_messages(
    tag_scores: dict[str, float],
    cluster_id: int | None = None,
    cluster_hints: dict[str, Any] | None = None,
    threshold: float = 0.5,
    max_issues: int = 4,
) -> list[str]:
    """Turn predicted tags / cluster state into deterministic review guidance."""
    issues: list[str] = []
    for tag_name in TAG_NAMES:
        if float(tag_scores.get(tag_name, 0.0)) >= threshold:
            issues.append(TAG_MESSAGES[tag_name])

    if cluster_hints is not None and cluster_id is not None:
        info = cluster_hints.get(str(int(cluster_id)))
        if info:
            risk_tier = str(info.get("risk_tier", "MED")).upper()
            label = str(info.get("label", "")).strip()
            acc_rate = float(info.get("acceptance_rate", 0.0) or 0.0)
            bugfix_rate = float(info.get("bugfix_rate", 0.0) or 0.0)
            if risk_tier == "HIGH":
                issues.append(
                    "Patch matches a historically high-risk change pattern"
                    + (f" ({label})" if label else "")
                    + f". These changes had low acceptance ({acc_rate:.0%}) and high "
                      f"bugfix followup ({bugfix_rate:.0%}); tighten scope and add tests."
                )
            elif bugfix_rate >= 0.6:
                issues.append(
                    "Patch resembles a historically bug-prone pattern"
                    + (f" ({label})" if label else "")
                    + f". Add targeted tests and review boundary conditions carefully."
                )

    # Preserve order and trim to a focused checklist
    deduped: list[str] = []
    seen: set[str] = set()
    for issue in issues:
        if issue not in seen:
            deduped.append(issue)
            seen.add(issue)
        if len(deduped) >= max_issues:
            break
    return deduped


def render_student_review_messages(
    tag_scores: dict[str, float],
    accept_prob: float,
    cluster_id: int | None = None,
    cluster_hints: dict[str, Any] | None = None,
    threshold: float = 0.5,
    fallback_threshold: float = 0.2,
    accept_threshold: float = 0.65,
    max_issues: int = 4,
) -> list[str]:
    """Render review guidance from student outputs.

    The first pass uses the standard thresholded tag rendering. If the
    acceptance head predicts a risky patch and nothing concrete fires at the
    primary threshold, lower the tag threshold. As a last resort, use the
    single highest-scoring tag so the review prompt stays concrete.
    """
    issues = render_review_messages(
        tag_scores=tag_scores,
        cluster_id=cluster_id,
        cluster_hints=cluster_hints,
        threshold=threshold,
        max_issues=max_issues,
    )
    if issues or accept_prob > accept_threshold:
        return issues

    issues = render_review_messages(
        tag_scores=tag_scores,
        cluster_id=cluster_id,
        cluster_hints=cluster_hints,
        threshold=fallback_threshold,
        max_issues=max_issues,
    )
    if issues:
        return issues

    if not tag_scores:
        return []
    top_tag, top_score = max(tag_scores.items(), key=lambda item: float(item[1]))
    if float(top_score) < fallback_threshold:
        return []

    fallback_issue = TAG_MESSAGES.get(top_tag)
    if fallback_issue is None:
        return []
    return [fallback_issue]
