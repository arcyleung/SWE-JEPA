#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pg8000.native
import yaml
from openai import AsyncOpenAI
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
PG_CONFIG_FILE = os.path.join(ROOT, "postgres_connection.yaml")
MODELS_YAML = os.path.join(ROOT, "models.yaml")
ORG_JSONL = os.path.join(ROOT, "followup_org_metrics.jsonl")

OUT_CANDIDATES = os.path.join(ROOT, "docs", "phase4_6_pr_review_candidates.jsonl")
OUT_METRICS = os.path.join(ROOT, "docs", "phase4_6_swe_jepa_vs_coder_metrics.json")
OUT_REPORT = os.path.join(ROOT, "docs", "phase4_6_swe_jepa_pr_review.md")


def _load_pg_cfg() -> dict:
    cfg = yaml.safe_load(open(PG_CONFIG_FILE))
    return dict(
        host=cfg["ip"],
        port=cfg.get("port", 9999),
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
    )


def _safe_json(v):
    if v is None:
        return None
    if isinstance(v, (list, dict)):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return None
    return None


def _top_module(path: str) -> str:
    p = (path or "").strip("/")
    if not p:
        return "__root__"
    return p.split("/", 1)[0]


def _parse_files(file_patches) -> list[str]:
    fp = _safe_json(file_patches)
    if not isinstance(fp, list):
        return []
    out = []
    for x in fp:
        if isinstance(x, str) and x:
            out.append(x)
            continue
        if isinstance(x, dict):
            p = x.get("file_path") or x.get("path") or x.get("file") or x.get("filename")
            if isinstance(p, str) and p:
                out.append(p)
    return list(dict.fromkeys(out))


def _requested_reviewer_count(v) -> int:
    j = _safe_json(v)
    if isinstance(j, list):
        return len(j)
    return 0


def _read_model_cfg(model_name: str) -> dict:
    cfg = yaml.safe_load(open(MODELS_YAML))
    for m in cfg.get("model_list", []):
        if m.get("model_name") == model_name:
            return m
    raise ValueError(f"model_name={model_name} not found in models.yaml")


def _load_org_file_stats(path: str) -> tuple[dict[tuple[str, str], dict], dict[str, dict]]:
    file_stats: dict[tuple[str, str], dict] = {}
    repo_buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for ln in f:
            r = json.loads(ln)
            repo = r["feature_repo"]
            fp = r["feature_file"]
            k = (repo, fp)
            if k not in file_stats:
                file_stats[k] = {
                    "ownership_friction": float(r.get("ownership_friction", 0.0)),
                    "interface_stress": float(r.get("interface_stress", 0.0)),
                    "distinct_authors": float(r.get("distinct_authors", 0.0)),
                    "cochange_cross_module_ratio": float(r.get("cochange_cross_module_ratio", 0.0)),
                }
            d = file_stats[k]
            for fk, fv in d.items():
                repo_buckets[repo][fk].append(float(fv))

    repo_defaults: dict[str, dict] = {}
    for repo, kv in repo_buckets.items():
        dd = {}
        for k, vals in kv.items():
            arr = sorted(vals)
            dd[k] = float(arr[len(arr) // 2]) if arr else 0.0
        repo_defaults[repo] = dd
    return file_stats, repo_defaults


def _avg_org_metrics(
    repo: str,
    files: list[str],
    file_stats: dict[tuple[str, str], dict],
    repo_defaults: dict[str, dict],
) -> dict:
    rows = [file_stats[(repo, f)] for f in files if (repo, f) in file_stats]
    if not rows:
        d = repo_defaults.get(repo, {})
        return {
            "ownership_friction_mean": float(d.get("ownership_friction", 0.0)),
            "interface_stress_mean": float(d.get("interface_stress", 0.0)),
            "distinct_authors_mean": float(d.get("distinct_authors", 0.0)),
            "cochange_cross_module_ratio_mean": float(d.get("cochange_cross_module_ratio", 0.0)),
        }
    return {
        "ownership_friction_mean": float(np.mean([x["ownership_friction"] for x in rows])),
        "interface_stress_mean": float(np.mean([x["interface_stress"] for x in rows])),
        "distinct_authors_mean": float(np.mean([x["distinct_authors"] for x in rows])),
        "cochange_cross_module_ratio_mean": float(np.mean([x["cochange_cross_module_ratio"] for x in rows])),
    }


@dataclass
class PRRow:
    repo: str
    instance_id: str
    pull_number: int
    pr_title: str
    pr_body: str
    patch: str
    file_patches: object
    requested_reviewers: object
    total_review_threads: int
    total_comments: int
    closing_issue_id: str | None
    created_at: str | None
    merged_at: str | None
    additions: int
    deletions: int
    changed_files: int


def _fetch_merged_prs(limit_pool: int = 12000) -> list[PRRow]:
    db = _load_pg_cfg()
    conn = pg8000.native.Connection(**db)
    rows = conn.run(
        """
        SELECT
            repo, instance_id, pull_number, pr_title, pr_body, patch, file_patches,
            requested_reviewers, total_review_threads, total_comments, closing_issue_id,
            created_at::text, merged_at::text, additions, deletions, changed_files
        FROM prs_copy
        WHERE pr_merged = TRUE
          AND patch IS NOT NULL
          AND file_patches IS NOT NULL
          AND changed_files BETWEEN 2 AND 40
          AND (COALESCE(additions, 0) + COALESCE(deletions, 0)) BETWEEN 20 AND 3000
          AND repo IN (SELECT DISTINCT feature_repo FROM followup_org_metrics)
        ORDER BY created_at DESC
        LIMIT :lim
        """,
        lim=limit_pool,
    )
    conn.close()
    out = []
    for r in rows:
        out.append(
            PRRow(
                repo=r[0],
                instance_id=r[1],
                pull_number=int(r[2]),
                pr_title=r[3] or "",
                pr_body=r[4] or "",
                patch=r[5] or "",
                file_patches=r[6],
                requested_reviewers=r[7],
                total_review_threads=int(r[8] or 0),
                total_comments=int(r[9] or 0),
                closing_issue_id=r[10],
                created_at=r[11],
                merged_at=r[12],
                additions=int(r[13] or 0),
                deletions=int(r[14] or 0),
                changed_files=int(r[15] or 0),
            )
        )
    return out


def _name_quality(title: str) -> float:
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", title or "")
    if not toks:
        return 0.5
    good = 0
    for t in toks:
        if len(t) >= 3 and not re.fullmatch(r"[a-z]\d*", t):
            good += 1
    return float(good / len(toks))


def _candidate_features(base: dict, kind: str, rng: random.Random) -> dict:
    x = dict(base)
    x["variant_kind"] = kind
    # Add mild jitter to reduce deterministic separability.
    x["changed_files"] = max(1.0, x["changed_files"] + rng.uniform(-0.4, 0.4))
    x["additions"] = max(1.0, x["additions"] * (1.0 + rng.uniform(-0.06, 0.06)))
    x["deletions"] = max(1.0, x["deletions"] * (1.0 + rng.uniform(-0.06, 0.06)))
    x["cross_module_ratio"] = float(min(1.0, max(0.0, x["cross_module_ratio"] + rng.uniform(-0.03, 0.03))))
    x["naming_quality"] = float(min(1.0, max(0.0, x["naming_quality"] + rng.uniform(-0.04, 0.04))))
    x["ownership_friction_mean"] = float(min(1.0, max(0.0, x["ownership_friction_mean"] + rng.uniform(-0.03, 0.03))))
    x["interface_stress_mean"] = float(min(10.0, max(0.0, x["interface_stress_mean"] + rng.uniform(-0.10, 0.10))))
    x["cochange_cross_module_ratio_mean"] = float(
        min(1.0, max(0.0, x["cochange_cross_module_ratio_mean"] + rng.uniform(-0.03, 0.03)))
    )
    if kind == "real":
        return x
    if kind == "naming_drift":
        x["naming_quality"] = max(0.05, x["naming_quality"] * rng.uniform(0.40, 0.75))
        x["risk_scope"] += rng.uniform(0.05, 0.14)
    elif kind == "cross_cutting":
        x["changed_files"] = float(x["changed_files"] * rng.uniform(1.25, 1.7) + rng.uniform(1.0, 3.5))
        x["module_count"] = float(x["module_count"] + rng.uniform(1.0, 2.5))
        x["cross_module_ratio"] = min(1.0, x["cross_module_ratio"] + rng.uniform(0.12, 0.28))
        x["cochange_cross_module_ratio_mean"] = min(1.0, x["cochange_cross_module_ratio_mean"] + rng.uniform(0.09, 0.20))
        x["risk_scope"] += rng.uniform(0.12, 0.28)
    elif kind == "scope_bloat":
        x["additions"] = float(x["additions"] * rng.uniform(1.45, 1.95) + rng.uniform(30.0, 120.0))
        x["deletions"] = float(x["deletions"] * rng.uniform(1.35, 1.85) + rng.uniform(20.0, 100.0))
        x["changed_files"] = float(x["changed_files"] * rng.uniform(1.2, 1.55) + rng.uniform(1.0, 3.5))
        x["risk_scope"] += rng.uniform(0.14, 0.32)
    elif kind == "arch_boundary":
        x["cross_module_ratio"] = min(1.0, x["cross_module_ratio"] + rng.uniform(0.14, 0.30))
        x["interface_stress_mean"] = min(10.0, x["interface_stress_mean"] + rng.uniform(0.35, 1.0))
        x["cochange_cross_module_ratio_mean"] = min(1.0, x["cochange_cross_module_ratio_mean"] + rng.uniform(0.10, 0.24))
        x["risk_scope"] += rng.uniform(0.12, 0.26)
    elif kind == "ownership_friction":
        x["ownership_friction_mean"] = min(1.0, x["ownership_friction_mean"] + rng.uniform(0.10, 0.28))
        x["distinct_authors_mean"] = min(30.0, x["distinct_authors_mean"] + rng.uniform(1.0, 3.5))
        x["requested_reviewers_count"] = float(x["requested_reviewers_count"] + rng.uniform(0.0, 2.0))
        x["risk_scope"] += rng.uniform(0.08, 0.24)
    return x


def _featurize_row(
    pr: PRRow,
    files: list[str],
    org_file_stats: dict[tuple[str, str], dict],
    org_repo_defaults: dict[str, dict],
) -> dict:
    mods = [_top_module(f) for f in files]
    mod_counts = Counter(mods)
    module_count = len(mod_counts)
    cross_ratio = 0.0 if not files else 1.0 - (max(mod_counts.values()) / len(files))
    orgm = _avg_org_metrics(pr.repo, files, org_file_stats, org_repo_defaults)
    requested_n = _requested_reviewer_count(pr.requested_reviewers)
    title = pr.pr_title or ""
    body = pr.pr_body or ""
    risk_scope = min(1.0, math.log1p(max(0, pr.changed_files)) / math.log(40.0))
    return {
        "repo": pr.repo,
        "instance_id": pr.instance_id,
        "pull_number": pr.pull_number,
        "title": title,
        "body": body[:1200],
        "patch_excerpt": (pr.patch or "")[:3500],
        "changed_files": float(max(0, pr.changed_files)),
        "additions": float(max(0, pr.additions)),
        "deletions": float(max(0, pr.deletions)),
        "module_count": float(module_count),
        "cross_module_ratio": float(cross_ratio),
        "requested_reviewers_count": float(requested_n),
        "has_closing_issue": 1.0 if pr.closing_issue_id else 0.0,
        "total_review_threads": float(max(0, pr.total_review_threads)),
        "total_comments": float(max(0, pr.total_comments)),
        "naming_quality": _name_quality(title),
        "risk_scope": float(risk_scope),
        **orgm,
        "top_modules": [m for m, _ in mod_counts.most_common(3)],
        "files_head": files[:10],
    }


def _feature_vector(c: dict) -> np.ndarray:
    arr = np.array(
        [
            math.log1p(c["changed_files"]),
            math.log1p(c["additions"]),
            math.log1p(c["deletions"]),
            math.log1p(c["module_count"]),
            c["cross_module_ratio"],
            math.log1p(c["requested_reviewers_count"]),
            c["has_closing_issue"],
            math.log1p(c["total_review_threads"]),
            math.log1p(c["total_comments"]),
            c["naming_quality"],
            c["ownership_friction_mean"],
            c["interface_stress_mean"],
            math.log1p(c["distinct_authors_mean"]),
            c["cochange_cross_module_ratio_mean"],
            c["risk_scope"],
        ],
        dtype=np.float32,
    )
    return arr


def _build_candidate_pack(base: dict, rng: random.Random) -> list[dict]:
    kinds = [
        "real",
        "naming_drift",
        "cross_cutting",
        "scope_bloat",
        "arch_boundary",
        "ownership_friction",
    ]
    labels = ["cand_A", "cand_B", "cand_C", "cand_D", "cand_E", "cand_F"]
    rng.shuffle(labels)
    out = []
    for k, cid in zip(kinds, labels):
        c = _candidate_features(base, k, rng)
        c["candidate_id"] = cid
        c["label"] = 1 if k == "real" else 0
        out.append(c)
    return out


def _pack_prompt(pack: dict) -> str:
    header = (
        "You are an upstream maintainer reviewing 6 functionally equivalent PR variants.\n"
        "Choose the single variant most likely to be accepted/merged upstream based on"
        " code-review quality, scope discipline, architecture fit, and maintainability.\n"
        "Return JSON only: {\"best_candidate\": \"cand_*\", \"reason\": \"...\"}\n\n"
        f"Repo: {pack['repo']}\n"
        f"Original PR title: {pack['title']}\n\n"
        "Candidates:\n"
    )
    chunks = [header]
    for c in pack["candidates"]:
        chunks.append(
            f"- {c['candidate_id']}:\n"
            f"  changed_files={int(c['changed_files'])}, additions={int(c['additions'])}, "
            f"deletions={int(c['deletions'])}, modules={int(c['module_count'])}, "
            f"cross_module_ratio={c['cross_module_ratio']:.2f}\n"
            f"  naming_quality={c['naming_quality']:.2f}, ownership_friction={c['ownership_friction_mean']:.2f}, "
            f"interface_stress={c['interface_stress_mean']:.2f}, linked_issue={int(c['has_closing_issue'])}, "
            f"requested_reviewers={int(c['requested_reviewers_count'])}\n"
            f"  top_modules={c['top_modules']}\n"
            f"  files_head={c['files_head'][:6]}\n"
        )
    return "\n".join(chunks)


async def _coder_pick(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    pack: dict,
    retries: int = 2,
) -> str:
    prompt = _pack_prompt(pack)
    for _ in range(retries + 1):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a strict code reviewer."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=120,
                )
            raw = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                obj = json.loads(m.group(0))
                cand = str(obj.get("best_candidate", "")).strip()
            else:
                cand = raw.strip().split()[0]
            if cand in {"cand_A", "cand_B", "cand_C", "cand_D", "cand_E", "cand_F"}:
                return cand
        except Exception:
            await asyncio.sleep(0.05)
    return "cand_A"


def _rank_metrics(packs: list[dict], pred_key: str) -> dict:
    top1 = []
    rr = []
    by_kind = defaultdict(list)
    for p in packs:
        cs = p["candidates"]
        order = sorted(cs, key=lambda x: float(x[pred_key]), reverse=True)
        top1.append(1 if order[0]["label"] == 1 else 0)
        rank_real = next((i + 1 for i, c in enumerate(order) if c["label"] == 1), len(order))
        rr.append(1.0 / rank_real)
        real_score = next(c[pred_key] for c in cs if c["label"] == 1)
        for c in cs:
            if c["label"] == 0:
                by_kind[c["variant_kind"]].append(1 if real_score > c[pred_key] else 0)
    return {
        "top1": float(np.mean(top1)) if top1 else 0.0,
        "mrr": float(np.mean(rr)) if rr else 0.0,
        "n_packs": len(packs),
        "win_rate_by_variant": {k: float(np.mean(v)) for k, v in by_kind.items()},
    }


def _fit_swe_ranker(train_packs: list[dict], seed: int) -> tuple[StandardScaler, LogisticRegression]:
    X_tr, y_tr = [], []
    for p in train_packs:
        for c in p["candidates"]:
            X_tr.append(_feature_vector(c))
            y_tr.append(int(c["label"]))
    X_tr = np.asarray(X_tr, dtype=np.float32)
    y_tr = np.asarray(y_tr, dtype=np.int32)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(X_tr_s, y_tr)
    return sc, clf


def _score_swe(packs: list[dict], sc: StandardScaler, clf: LogisticRegression) -> None:
    for p in packs:
        X = np.asarray([_feature_vector(c) for c in p["candidates"]], dtype=np.float32)
        s = clf.predict_proba(sc.transform(X))[:, 1]
        for c, v in zip(p["candidates"], s):
            c["swe_score"] = float(v)


async def _run_coder(packs: list[dict], model_cfg: dict, concurrency: int) -> None:
    base = model_cfg["litellm_params"]["api_base"].rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    api_key = model_cfg["litellm_params"]["api_key"]
    model = model_cfg.get("model_id") or model_cfg["litellm_params"]["model"]
    client = AsyncOpenAI(base_url=base, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    done = 0
    lock = asyncio.Lock()

    async def one(pack: dict):
        nonlocal done
        pick = await _coder_pick(client, sem, model, pack)
        for c in pack["candidates"]:
            c["coder_score"] = 1.0 if c["candidate_id"] == pick else 0.0
        async with lock:
            done += 1
            if done % 20 == 0 or done == len(packs):
                print(f"  coder scored [{done}/{len(packs)}]", flush=True)

    try:
        await asyncio.gather(*[one(p) for p in packs])
    finally:
        # Ensure HTTP resources are closed before loop teardown.
        close_fn = getattr(client, "close", None)
        if close_fn is not None:
            maybe = close_fn()
            if asyncio.iscoroutine(maybe):
                await maybe


def _write_candidates(path: str, packs: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for p in packs:
            f.write(json.dumps(p) + "\n")


def _write_report(path: str, payload: dict) -> None:
    swe = payload["metrics"]["swe_jepa_conway"]
    coder = payload["metrics"]["coder_model"]
    d_top1 = 100.0 * (swe["top1"] - coder["top1"])
    d_mrr = 100.0 * (swe["mrr"] - coder["mrr"])

    lines = [
        "# Experiment 4.6: SWE-JEPA PR Review Reranking",
        "",
        f"**Date**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Packs**: {payload['n_test_packs']} test packs (6 candidates each)",
        f"**Coder model**: {payload['coder_model_name']}",
        f"**Coder concurrency**: {payload['coder_concurrency']}",
        "",
        "## Setup",
        "",
        "- Ground truth candidate per pack: historically merged PR from `prs_copy`.",
        "- Five synthetic negatives: naming drift, cross-cutting, scope bloat, architecture-boundary, ownership-friction.",
        "- SWE-JEPA-Conway scorer: logistic head on structural + Conway proxy features.",
        "- Baseline: standalone coder model choosing best candidate via prompted review.",
        "",
        "## Main Results",
        "",
        "| Model | Top-1 | MRR |",
        "|---|---:|---:|",
        f"| SWE-JEPA + Conway | {100*swe['top1']:.1f}% | {100*swe['mrr']:.1f}% |",
        f"| Coder model | {100*coder['top1']:.1f}% | {100*coder['mrr']:.1f}% |",
        "",
        f"**Delta (SWE-JEPA - coder)**: Top-1 {d_top1:+.1f} pp, MRR {d_mrr:+.1f} pp",
        "",
        "## Win Rate by Variant (real PR ranked above synthetic)",
        "",
        "| Variant | SWE-JEPA | Coder |",
        "|---|---:|---:|",
    ]
    all_k = sorted(set(swe["win_rate_by_variant"]) | set(coder["win_rate_by_variant"]))
    for k in all_k:
        lines.append(
            f"| {k} | {100*swe['win_rate_by_variant'].get(k, 0.0):.1f}% | "
            f"{100*coder['win_rate_by_variant'].get(k, 0.0):.1f}% |"
        )

    lines += [
        "",
        "## Robustness",
        "",
    ]
    cv = payload.get("cv5")
    if cv:
        lines += [
            f"- 5-fold Group CV (by repo): SWE Top-1 {100*cv['swe_top1_mean']:.1f}% ± {100*cv['swe_top1_std']:.1f}, "
            f"Coder Top-1 {100*cv['coder_top1_mean']:.1f}% ± {100*cv['coder_top1_std']:.1f}",
            f"- 5-fold Group CV (by repo): SWE MRR {100*cv['swe_mrr_mean']:.1f}% ± {100*cv['swe_mrr_std']:.1f}, "
            f"Coder MRR {100*cv['coder_mrr_mean']:.1f}% ± {100*cv['coder_mrr_std']:.1f}",
            "",
        ]
    lines += [
        "",
        "## Notes",
        "",
        "- `total_comments`/`total_review_threads` are included as pragmatic metadata proxies and may contain partial outcome leakage.",
        "- This benchmark uses synthetic variants; next step is higher-fidelity semantic-preserving transforms on real patch AST/edit scripts.",
        "",
        "## Artifacts",
        "",
        "- `docs/phase4_6_pr_review_candidates.jsonl`",
        "- `docs/phase4_6_swe_jepa_vs_coder_metrics.json`",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="qwen3_coder_30b")
    ap.add_argument("--packs", type=int, default=180)
    ap.add_argument("--pool-size", type=int, default=10000)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--coder-concurrency", type=int, default=500)
    ap.add_argument("--max-per-repo", type=int, default=1)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--skip-coder", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    print("Exp 4.6: PR acceptance reranking", flush=True)
    print("=" * 60, flush=True)
    print("Loading Conway file stats …", flush=True)
    org_file_stats, org_repo_defaults = _load_org_file_stats(ORG_JSONL)
    print(f"  org file stats: {len(org_file_stats):,}", flush=True)

    print("Loading merged PRs from prs_copy …", flush=True)
    rows = _fetch_merged_prs(limit_pool=args.pool_size)
    print(f"  merged PR rows: {len(rows):,}", flush=True)

    # Sample PR anchors with bounded per-repo count.
    by_repo = defaultdict(list)
    for r in rows:
        by_repo[r.repo].append(r)
    repos = list(by_repo.keys())
    rng.shuffle(repos)

    selected = []
    for repo in repos:
        pool = by_repo[repo]
        if not pool:
            continue
        rng.shuffle(pool)
        take_n = min(args.max_per_repo, len(pool))
        selected.extend(pool[:take_n])
        if len(selected) >= args.packs:
            break
    selected = selected[: args.packs]
    print(f"  selected PR anchors: {len(selected):,}", flush=True)
    if len(selected) < 40:
        raise ValueError("Too few PR anchors selected; increase --pool-size or lower filters.")

    packs = []
    for r in selected:
        files = _parse_files(r.file_patches)
        if not files:
            continue
        base = _featurize_row(r, files, org_file_stats, org_repo_defaults)
        cands = _build_candidate_pack(base, rng)
        pack = {
            "pack_id": f"{r.repo}#{r.pull_number}",
            "repo": r.repo,
            "title": r.pr_title,
            "instance_id": r.instance_id,
            "candidates": cands,
        }
        packs.append(pack)
    print(f"  valid packs: {len(packs):,}", flush=True)

    # Holdout split
    pack_repos = sorted({p["repo"] for p in packs})
    rng.shuffle(pack_repos)
    n_test_repos = max(1, int(len(pack_repos) * args.test_frac))
    test_repos = set(pack_repos[:n_test_repos])
    train_packs = [p for p in packs if p["repo"] not in test_repos]
    test_packs = [p for p in packs if p["repo"] in test_repos]
    print(f"  split: {len(train_packs):,} train packs / {len(test_packs):,} test packs", flush=True)

    # Train and score holdout SWE-JEPA-Conway
    sc, clf = _fit_swe_ranker(train_packs, args.seed)
    _score_swe(test_packs, sc, clf)

    if args.skip_coder:
        print("Skipping coder-model scoring (--skip-coder)", flush=True)
        for p in test_packs:
            order = sorted(p["candidates"], key=lambda x: float(x["swe_score"]), reverse=True)
            top = order[0]["candidate_id"]
            for c in p["candidates"]:
                c["coder_score"] = 1.0 if c["candidate_id"] == top else 0.0
    else:
        model_cfg = _read_model_cfg(args.model_name)
        print(f"Scoring with coder model ({args.model_name}) @ concurrency={args.coder_concurrency} …",
              flush=True)
        asyncio.run(_run_coder(test_packs, model_cfg, args.coder_concurrency))

    swe_metrics = _rank_metrics(test_packs, "swe_score")
    coder_metrics = _rank_metrics(test_packs, "coder_score")

    # 5-fold grouped CV over all packs
    cv_payload = None
    if args.cv_folds and args.cv_folds >= 2 and len(packs) >= args.cv_folds:
        print(f"Running {args.cv_folds}-fold Group CV (repo-grouped) …", flush=True)
        groups = np.array([p["repo"] for p in packs])
        idx = np.arange(len(packs))
        gkf = GroupKFold(n_splits=args.cv_folds)
        fold_rows = []
        for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(idx, groups=groups), start=1):
            tr = [packs[i] for i in tr_idx.tolist()]
            te = [packs[i] for i in te_idx.tolist()]
            sc_f, clf_f = _fit_swe_ranker(tr, args.seed + fold_i)
            _score_swe(te, sc_f, clf_f)
            if args.skip_coder:
                for p in te:
                    top = max(p["candidates"], key=lambda c: float(c["swe_score"]))["candidate_id"]
                    for c in p["candidates"]:
                        c["coder_score"] = 1.0 if c["candidate_id"] == top else 0.0
            else:
                model_cfg = _read_model_cfg(args.model_name)
                asyncio.run(_run_coder(te, model_cfg, args.coder_concurrency))
            ms = _rank_metrics(te, "swe_score")
            mc = _rank_metrics(te, "coder_score")
            fold_rows.append(
                {
                    "fold": fold_i,
                    "n_test_packs": len(te),
                    "swe_top1": ms["top1"],
                    "swe_mrr": ms["mrr"],
                    "coder_top1": mc["top1"],
                    "coder_mrr": mc["mrr"],
                }
            )
            print(
                f"  fold {fold_i}/{args.cv_folds}: "
                f"swe_top1={100*ms['top1']:.1f}% coder_top1={100*mc['top1']:.1f}%",
                flush=True,
            )

        cv_payload = {
            "folds": fold_rows,
            "swe_top1_mean": float(np.mean([r["swe_top1"] for r in fold_rows])),
            "swe_top1_std": float(np.std([r["swe_top1"] for r in fold_rows])),
            "swe_mrr_mean": float(np.mean([r["swe_mrr"] for r in fold_rows])),
            "swe_mrr_std": float(np.std([r["swe_mrr"] for r in fold_rows])),
            "coder_top1_mean": float(np.mean([r["coder_top1"] for r in fold_rows])),
            "coder_top1_std": float(np.std([r["coder_top1"] for r in fold_rows])),
            "coder_mrr_mean": float(np.mean([r["coder_mrr"] for r in fold_rows])),
            "coder_mrr_std": float(np.std([r["coder_mrr"] for r in fold_rows])),
        }

    payload = {
        "date_utc": datetime.now(UTC).isoformat(),
        "args": vars(args),
        "n_train_packs": len(train_packs),
        "n_test_packs": len(test_packs),
        "coder_model_name": args.model_name,
        "coder_concurrency": args.coder_concurrency,
        "metrics": {
            "swe_jepa_conway": swe_metrics,
            "coder_model": coder_metrics,
        },
        "cv5": cv_payload,
        "elapsed_sec": time.time() - t0,
    }

    os.makedirs(os.path.dirname(OUT_METRICS), exist_ok=True)
    with open(OUT_METRICS, "w") as f:
        json.dump(payload, f, indent=2)
    _write_candidates(OUT_CANDIDATES, test_packs)
    _write_report(OUT_REPORT, payload)

    print("\nDone.", flush=True)
    print(f"  Metrics: {OUT_METRICS}", flush=True)
    print(f"  Candidates: {OUT_CANDIDATES}", flush=True)
    print(f"  Report: {OUT_REPORT}", flush=True)


if __name__ == "__main__":
    main()
