"""Retrieval evaluation over the Conditions of Carriage: does the right clause come back, and how high?

Gold set: eval/retrieval/gold.yaml (20 topics x EN/AR, graded labels 2 = answers it, 1 = also relevant).
Each retrieval variant is run on every answerable question, raw (exactly as the customer typed it), and scored with:

- recall@k   share of the labelled clauses (grade >= 1) found in the top k
- hit@k      the question counts if at least one grade-2 clause is in the top k
- MRR        1 / rank of the first grade-2 clause (0 if not in the top 10)
- nDCG@10    graded ranking quality, gain 2^grade - 1

Run (needs the stack, costs a fraction of a cent in query embeddings and rewrites):
    docker compose run --rm --build eval python -m falcon.eval.retrieval
    ... --variants dense,dense+rw
Variants: dense (production) and dense+rw (LLM query rewriting, falcon.policy.rewrite). The lexical and hybrid variants
measured on 17 Oct are in docs/EVALUATION.md; the lexical layer was removed afterwards (decision #24).
Writes eval/results/retrieval-<timestamp>/{report.md, per_query.jsonl}.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from datetime import datetime
from pathlib import Path

import yaml

from falcon.core.config import REPO_ROOT, settings
from falcon.core.db import Database
from falcon.policy.embed import Embedder
from falcon.policy.rewrite import QueryRewriter
from falcon.policy.search import PolicySearch

GOLD = REPO_ROOT / "eval" / "retrieval" / "gold.yaml"
TOP = 10
KS = (1, 3, 5, 10)


def load_gold(path: Path = GOLD) -> list[dict]:
    items = []
    for t in yaml.safe_load(path.read_text(encoding="utf-8")):
        for lang in ("en", "ar"):
            if t.get(lang):
                items.append({"id": f"{t['topic']}_{lang}", "topic": t["topic"], "lang": lang, "question": t[lang],
                              "relevant": {str(k): int(v) for k, v in (t.get("relevant") or {}).items()},
                              "answerable": t.get("answerable", True)})
    return items


def score_ranking(ranked: list[str], relevant: dict[str, int]) -> dict:
    primary = {c for c, g in relevant.items() if g >= 2}
    out = {f"recall@{k}": len(set(ranked[:k]) & set(relevant)) / len(relevant) for k in KS}
    out.update({f"hit@{k}": float(bool(set(ranked[:k]) & primary)) for k in KS})
    first = next((i for i, c in enumerate(ranked[:TOP], start=1) if c in primary), None)
    out["mrr"] = 1.0 / first if first else 0.0
    out["first_primary_rank"] = first
    dcg = sum((2 ** relevant.get(c, 0) - 1) / math.log2(i + 1) for i, c in enumerate(ranked[:TOP], start=1))
    ideal = sorted(relevant.values(), reverse=True)[:TOP]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out


METRICS = [f"recall@{k}" for k in KS] + ["hit@1", "hit@3", "hit@5", "mrr", "ndcg@10"]


def _mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else float("nan")


def parse_variant(variant: str) -> dict:
    """dense | dense+rw (always rewrite) | dense+auto@<threshold> (rewrite below that confidence) | +alias (the alias layer; off otherwise)."""
    name, _, thr = variant.partition("@")
    tokens = name.split("+")
    if tokens[0] != "dense" or any(t not in ("rw", "auto", "alias") for t in tokens[1:]):
        raise SystemExit(f"unknown variant '{variant}': dense[+rw|+auto@<threshold>][+alias] (the lexical layer was removed, decision #24)")
    rw = "always" if "rw" in tokens else "auto" if "auto" in tokens else "off"
    return {"rewrite": rw, "aliases": "alias" in tokens, **({"threshold": float(thr)} if thr else {})}


async def run_variant(search: PolicySearch, variant: str, items: list[dict]) -> list[dict]:
    rows = []
    opts = parse_variant(variant)
    for it in items:
        started = time.perf_counter()
        res = await search.search(it["question"], top_k=TOP, **opts)
        ms = int((time.perf_counter() - started) * 1000)
        ranked = [r["clause_id"] for r in res["results"]]
        print(f"  {variant} {it['id']}: {ms} ms, confidence {res.get('confidence')}, rewritten {res.get('rewritten', False)}", flush=True)
        rows.append({"variant": variant, "id": it["id"], "lang": it["lang"], "question": it["question"], "ranked": ranked,
                     "relevant": it["relevant"], "rewrites": res.get("rewrites", []), "rewritten": res.get("rewritten", False),
                     "confidence": res.get("confidence"), "latency_ms": ms,
                     **score_ranking(ranked, it["relevant"])})
    return rows


def report(all_rows: list[dict], variants: list[str], n_items: int) -> str:
    lines = [f"# Retrieval evaluation", "",
             f"{datetime.now():%Y-%m-%d %H:%M} · gold: {n_items} answerable questions (EN + AR, same topics) · top {TOP} · "
             f"embeddings: {settings.embedding_model} · rewriter: {QueryRewriter().model}", "",
             "| variant | lang | " + " | ".join(METRICS) + " | latency ms |", "|---|---|" + "---|" * (len(METRICS) + 1)]
    for v in variants:
        for lang in ("all", "en", "ar"):
            rows = [r for r in all_rows if r["variant"] == v and (lang == "all" or r["lang"] == lang)]
            lines.append(f"| {v} | {lang} | " + " | ".join(f"{_mean(rows, m):.2f}" for m in METRICS) + f" | {_mean(rows, 'latency_ms'):.0f} |")
    lines += ["", "Rewrites triggered: " + " · ".join(f"{v} {sum(1 for r in all_rows if r['variant'] == v and r['rewritten'])}/{sum(1 for r in all_rows if r['variant'] == v)}" for v in variants)]
    dense = [r for r in all_rows if r["variant"] == "dense" and r.get("confidence") is not None]
    if dense:
        hits = sorted(r["confidence"] for r in dense if r["hit@5"])
        miss = sorted(r["confidence"] for r in dense if not r["hit@5"])
        lines += ["", f"First-pass confidence (best cosine similarity), dense: hits@5 median {hits[len(hits) // 2]:.3f} (min {hits[0]:.3f}) · "
                      f"misses {', '.join(f'{c:.3f}' for c in miss) or 'none'}"]
    lines += ["", "## Misses (no grade-2 clause in the top 5)", ""]
    for v in variants:
        misses = [r for r in all_rows if r["variant"] == v and not r["hit@5"]]
        lines.append(f"**{v}** ({len(misses)}):")
        for r in misses:
            want = [c for c, g in r["relevant"].items() if g >= 2]
            lines.append(f"- `{r['id']}` wanted {want}, first at {r['first_primary_rank'] or '>10'}; got {r['ranked'][:5]}")
        lines.append("")
    return "\n".join(lines)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="dense,dense+alias,dense+auto@0.6,dense+alias+auto@0.6")
    args = ap.parse_args()
    variants = args.variants.split(",")
    items = [i for i in load_gold() if i["answerable"]]
    db = Database(settings.dsn_policy, max_size=2)
    await db.open()
    search = PolicySearch(db, Embedder(), QueryRewriter())
    all_rows = []
    for v in variants:
        all_rows += await run_variant(search, v, items)
    out = REPO_ROOT / "eval" / "results" / f"retrieval-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_query.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows), encoding="utf-8")
    text = report(all_rows, variants, len(items))
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    asyncio.run(main())
