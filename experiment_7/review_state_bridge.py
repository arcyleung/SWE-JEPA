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
from dataclasses import dataclass
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
DEF_OR_CLASS_PAT = re.compile(r"^[+-]\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
HUNK_CONTEXT_PAT = re.compile(r"@@.*?@@\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
ASYNC_RUNTIME_PAT = re.compile(
    r"\b(?:asyncio|await|IOLoop|run_until_complete|get_event_loop|new_event_loop|pytest\.mark\.asyncio)\b"
)
SCHEMA_RISK_PAT = re.compile(
    r"\b(?:kwargs|VAR_KEYWORD|inspect\.Parameter\.empty|Signature|schema|properties|serializer|field_info)\b"
)
TYPE_HINT_RISK_PAT = re.compile(
    r"\b(?:Annotated|Literal|Union|Optional|ForwardRef|get_type_hints|typing\.|__future__\s+import\s+annotations)\b"
)
SENTINEL_COMPARE_PAT = re.compile(
    r"(?:==|!=)\s*(?:None|inspect\.Parameter\.empty)\b|"
    r"\b(?:is|is\s+not)\s*(?:True|False)\b"
)
PRIVATE_ATTR_PAT = re.compile(r"\._[A-Za-z][A-Za-z0-9_]*")

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
    "contract_without_targeted_tests": (
        "Public interface or shared behavior changed without clearly targeted regression "
        "coverage. Keep the contract stable and add tests for the exact changed path "
        "and edge cases."
    ),
    "schema_contract_mismatch": (
        "This patch may be changing the runtime/schema contract. Make sure kwargs, "
        "parameter metadata, and serialized fields stay consistent with the public API."
    ),
    "sentinel_identity_risk": (
        "Sentinel or identity handling looks risky. Prefer identity checks for sentinels "
        "and verify default or empty-value behavior with focused tests."
    ),
    "private_attr_test_reliance": (
        "New tests appear to rely on private attributes. Prefer assertions through the "
        "public API or externally visible behavior."
    ),
    "async_boundary_change": (
        "The patch changes async or event-loop boundaries. Keep sync/async contracts "
        "stable and add coverage for callsites, return types, and runtime lifecycle."
    ),
    "type_annotation_contract_change": (
        "Type or annotation handling changed on a public/shared path. Preserve existing "
        "helper flow and add a regression test for the exact annotation shape being fixed."
    ),
    "overbroad_fix_scope": (
        "The fix may be broader than necessary. Keep the change local and avoid "
        "duplicating or rewriting surrounding helper logic unless the task requires it."
    ),
}

TAG_NAMES = list(TAG_MESSAGES.keys())


@dataclass(frozen=True)
class PatchFileSection:
    path: str
    added_lines: tuple[str, ...]
    removed_lines: tuple[str, ...]
    context_symbols: tuple[str, ...]


def split_patch_sections(patch_text: str) -> list[PatchFileSection]:
    """Split a unified diff into per-file added/removed line groups."""
    sections: list[PatchFileSection] = []
    current_path: str | None = None
    added_lines: list[str] = []
    removed_lines: list[str] = []
    context_symbols: list[str] = []

    def flush() -> None:
        nonlocal current_path, added_lines, removed_lines, context_symbols
        if current_path is None:
            return
        sections.append(
            PatchFileSection(
                path=current_path,
                added_lines=tuple(added_lines),
                removed_lines=tuple(removed_lines),
                context_symbols=tuple(context_symbols),
            )
        )
        current_path = None
        added_lines = []
        removed_lines = []
        context_symbols = []

    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            flush()
            m = re.match(r"diff --git a/(.*?) b/(.*)", line)
            if m:
                current_path = m.group(2)
            continue
        if current_path is None:
            continue
        if line.startswith("@@"):
            for symbol in HUNK_CONTEXT_PAT.findall(line):
                if not symbol.startswith("_"):
                    context_symbols.append(symbol)
            continue
        if line.startswith("+++ ") or line.startswith("--- "):
            continue
        if line.startswith("+"):
            added_lines.append(line[1:])
        elif line.startswith("-"):
            removed_lines.append(line[1:])

    flush()
    return sections


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


def _extract_public_symbol_changes(sections: list[PatchFileSection]) -> set[str]:
    symbols: set[str] = set()
    for section in sections:
        if TEST_PAT.search(section.path):
            continue
        symbols.update(symbol for symbol in section.context_symbols if not symbol.startswith("_"))
        for line in (*section.added_lines, *section.removed_lines):
            match = DEF_OR_CLASS_PAT.match(line)
            if not match:
                continue
            symbol = match.group(1)
            if symbol.startswith("_"):
                continue
            symbols.add(symbol)
    return symbols


def _collect_added_test_text(sections: list[PatchFileSection]) -> str:
    test_lines: list[str] = []
    for section in sections:
        if TEST_PAT.search(section.path):
            test_lines.extend(section.added_lines)
    return "\n".join(test_lines)


def _has_targeted_test_mentions(test_text: str, changed_symbols: set[str]) -> bool:
    if not test_text or not changed_symbols:
        return False
    for symbol in changed_symbols:
        if re.search(rf"\b{re.escape(symbol)}\b", test_text):
            return True
    return False


def detect_runtime_review_messages(
    patch_text: str,
    max_issues: int = 4,
) -> list[str]:
    """Extract runtime-only review hints from the concrete diff.

    These heuristics expand the current 7.1 bridge without changing the
    checkpoint tag schema. The student still predicts the original 9 tags,
    while this layer adds concrete contract/test hints that frontier judges
    repeatedly called out in failed comparisons.
    """
    issue_flags = detect_review_issue_flags(patch_text)
    runtime_tag_scores = {
        tag_name: float(issue_flags.get(tag_name, 0))
        for tag_name in TAG_NAMES
    }
    return render_review_messages(
        tag_scores=runtime_tag_scores,
        cluster_id=None,
        cluster_hints=None,
        threshold=0.5,
        max_issues=max_issues,
    )


def detect_review_issue_flags(patch_text: str) -> dict[str, int]:
    """Return deterministic binary issue tags for a patch diff."""
    stats = count_patch_stats(patch_text)
    classified = classify_files(stats["files"])
    sections = split_patch_sections(patch_text)
    changed_symbols = _extract_public_symbol_changes(sections)
    added_test_text = _collect_added_test_text(sections)
    added_non_test_text = "\n".join(
        line
        for section in sections
        if not TEST_PAT.search(section.path)
        for line in section.added_lines
    )
    touches_contract_surface = bool(changed_symbols or classified["shared"] or classified["api"])
    has_targeted_test_mentions = _has_targeted_test_mentions(added_test_text, changed_symbols)

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
    flags["contract_without_targeted_tests"] = int(
        touches_contract_surface and not has_targeted_test_mentions
    )
    flags["schema_contract_mismatch"] = int(
        bool(SCHEMA_RISK_PAT.search(added_non_test_text))
        and (
            touches_contract_surface
            or "kwargs" in added_non_test_text
            or "schema" in added_non_test_text
            or "validate(" in added_non_test_text
        )
    )
    flags["sentinel_identity_risk"] = int(bool(SENTINEL_COMPARE_PAT.search(added_non_test_text)))
    flags["private_attr_test_reliance"] = int(bool(PRIVATE_ATTR_PAT.search(added_test_text)))
    flags["async_boundary_change"] = int(bool(ASYNC_RUNTIME_PAT.search(added_non_test_text)))
    flags["type_annotation_contract_change"] = int(
        touches_contract_surface and bool(TYPE_HINT_RISK_PAT.search(added_non_test_text))
    )
    flags["overbroad_fix_scope"] = int(
        stats["n_files"] > 4 or stats["additions"] > 120 or len(changed_symbols) > 2
    )
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
    patch_text: str | None = None,
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
    heuristic_issues = detect_runtime_review_messages(
        patch_text=patch_text or "",
        max_issues=max_issues,
    ) if patch_text else []
    issues = render_review_messages(
        tag_scores=tag_scores,
        cluster_id=cluster_id,
        cluster_hints=cluster_hints,
        threshold=threshold,
        max_issues=max_issues,
    )
    if heuristic_issues:
        issues = heuristic_issues + issues
        deduped: list[str] = []
        seen: set[str] = set()
        for issue in issues:
            if issue in seen:
                continue
            deduped.append(issue)
            seen.add(issue)
            if len(deduped) >= max_issues:
                break
        issues = deduped
    if issues or accept_prob > accept_threshold:
        return issues

    issues = render_review_messages(
        tag_scores=tag_scores,
        cluster_id=cluster_id,
        cluster_hints=cluster_hints,
        threshold=fallback_threshold,
        max_issues=max_issues,
    )
    if heuristic_issues:
        issues = heuristic_issues + issues
        deduped = []
        seen = set()
        for issue in issues:
            if issue in seen:
                continue
            deduped.append(issue)
            seen.add(issue)
            if len(deduped) >= max_issues:
                break
        issues = deduped
    if issues:
        return issues

    if heuristic_issues:
        return heuristic_issues[:max_issues]

    if not tag_scores:
        return []
    top_tag, top_score = max(tag_scores.items(), key=lambda item: float(item[1]))
    if float(top_score) < fallback_threshold:
        return []

    fallback_issue = TAG_MESSAGES.get(top_tag)
    if fallback_issue is None:
        return []
    return [fallback_issue]
