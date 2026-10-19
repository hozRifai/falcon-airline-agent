"""Policy retrieval: dense (semantic) search over the Conditions of Carriage, grouped by clause id so one hit
returns the clause in both languages.

Every clause is embedded once at ingest (bge-m3 through OpenRouter: one multilingual space, so an Arabic question
finds the English clause and vice versa) and stored in pgvector; a query is embedded the same way and ranked by
cosine similarity. Query rewriting and expansion (rewrite.py: three LLM restatements of the question) runs adaptively: only
when the first pass is low-confidence (decision #31); always-on rewriting was measured and found to lower the first result
(decision #23).

There is no lexical layer (decision #24): when no vectors are available (no embedding provider configured, index
not embedded yet, or the embedding call failed) the search reports itself unavailable instead of guessing, and the
agent tells the customer it cannot check the Conditions right now and offers a human agent.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from falcon.core.config import settings
from falcon.core.db import Database

from .aliases import expand

log = logging.getLogger("falcon.policy.search")

RRF_K = 60
UNAVAILABLE = ("Policy search is unavailable right now (the semantic index could not be queried). Do not answer the "
               "policy question from memory: tell the customer you cannot check the Conditions of Carriage at the moment "
               "and offer a human agent. get_clause and get_article still work if the customer names a clause.")


@dataclass
class Hit:
    clause_id: str
    article_no: int
    article_title: str
    texts: dict[str, str] = field(default_factory=dict)  # lang -> text
    score: float = 0.0
    similarity: float = 0.0  # best cosine similarity across the lists that returned this clause
    matched_by: set[str] = field(default_factory=set)


class PolicySearch:
    def __init__(self, db: Database, embedder=None, rewriter=None):
        self.db = db
        self.embedder = embedder
        self.rewriter = rewriter
        self._has_vectors = False  # becomes True the first time vectors are seen in the index

    async def dense_many(self, queries: list[str], limit: int = 12) -> list[list[dict]]:
        """One embedding call for all queries (the embedding API dominates latency), then one vector search each.
        Returns [] when dense search is not available: no provider, no vectors in the index, or the embedding call failed."""
        if not queries or not self.embedder or not self.embedder.enabled:
            return []
        if not self._has_vectors:  # a positive answer is cached; a negative one is re-checked (the index may be embedded later)
            has = await self.db.fetch_one("SELECT count(*) AS n FROM policy.clauses WHERE embedding IS NOT NULL")
            self._has_vectors = bool(has and has["n"])
            if not self._has_vectors:
                return []
        try:
            vectors = await asyncio.to_thread(self.embedder.embed, queries)  # the client is synchronous; keep the server's event loop free
        except Exception as exc:  # noqa: BLE001 - the embedding API is a remote dependency
            log.warning("embedding call failed: %s", exc)
            return []
        out = []
        for vec in vectors:
            out.append(await self.db.fetch_all(
                """SELECT clause_id, article_no, article_title, lang, text, 1 - (embedding <=> %s::vector) AS rank
                   FROM policy.clauses WHERE embedding IS NOT NULL ORDER BY embedding <=> %s::vector LIMIT %s""",
                (str(vec), str(vec), limit)))
        return out

    async def search(self, query: str, top_k: int = 6, rewrite: str | bool = "off", threshold: float | None = None, original_weight: float = 2.0,
                     aliases: bool | None = None) -> dict:
        """Semantic search with adaptive query rewriting (decision #31).

        rewrite: "off" | "auto" | "always". The first pass embeds the question as typed; its confidence is the best cosine
        similarity in the index. In "auto" mode a confident first pass is the answer (one embedding call, ~1 s); below
        `threshold` (settings.policy_rewrite_threshold) the LLM restatements from rewrite.py (formal English, formal Arabic, a
        sentence the answering clause would contain) are searched too and the lists are fused by weighted Reciprocal Rank
        Fusion, the question as typed counting `original_weight` times a restatement, so a good first pass is not pushed down by
        a loose rewrite. Every result carries `confidence`, `rewritten` and the rewrites, so a trace shows why a search took
        the second pass."""
        mode = {True: "always", False: "off"}.get(rewrite, str(rewrite).strip().lower())
        mode = {"true": "always", "1": "always", "false": "off", "0": "off", "": "off"}.get(mode, mode)
        if mode not in ("off", "auto", "always"):
            raise ValueError(f"rewrite must be off, auto or always, not {mode!r}")
        threshold = settings.policy_rewrite_threshold if threshold is None else threshold
        aliases = settings.policy_query_aliases if aliases is None else aliases
        probe, added = expand(query) if aliases else (query, [])
        first = await self.dense_many([probe])
        if not first or not first[0]:
            return {"query": query, "available": False, "method": "semantic", "rewrites": [], "results": [], "note": UNAVAILABLE}
        confidence = round(max(float(r["rank"]) for r in first[0]), 3)
        rewrites: list[str] = []
        if self.rewriter is not None and (mode == "always" or (mode == "auto" and confidence < threshold)):
            rewrites = await self.rewriter.rewrite(query)
        lists = first + (await self.dense_many(rewrites) if rewrites else [])
        # Weighted Reciprocal Rank Fusion over base clause ids (part suffix stripped); with one list this is the dense ranking
        fused: dict[str, Hit] = {}
        for i, rows in enumerate(lists):
            weight = original_weight if i == 0 else 1.0
            for rank, r in enumerate(rows, start=1):
                base = r["clause_id"].split("#")[0]
                hit = fused.setdefault(base, Hit(base, r["article_no"], r["article_title"]))
                hit.score += weight / (RRF_K + rank)
                hit.similarity = max(hit.similarity, float(r["rank"]))
                hit.matched_by.add("semantic" if i == 0 else f"semantic:rewrite{i}")
        top = sorted(fused.values(), key=lambda h: -h.score)[:top_k]
        texts = await self.db.fetch_all(
            "SELECT clause_id, lang, text FROM policy.clauses WHERE split_part(clause_id, '#', 1) = ANY(%s) ORDER BY clause_id",
            ([h.clause_id for h in top],))
        for t in texts:
            base = t["clause_id"].split("#")[0]
            for h in top:
                if h.clause_id == base:
                    h.texts[t["lang"]] = (h.texts.get(t["lang"], "") + " " + t["text"]).strip()
        return {
            "query": query, "available": True, "method": "semantic", "confidence": confidence, "rewrite_mode": mode, "expanded_with": added,
            "rewritten": bool(rewrites), "rewrites": rewrites,
            **({"note": f"First pass was low-confidence ({confidence} < {threshold}); the question was restated and expanded and the lists fused."} if rewrites and mode == "auto" else {}),
            "results": [{"clause_id": h.clause_id, "article": f"{h.article_no} – {h.article_title}", "score": round(h.score, 4), "similarity": round(h.similarity, 3),
                         "matched_by": sorted(h.matched_by), "text_en": h.texts.get("en"), "text_ar": h.texts.get("ar")} for h in top],
        }

    async def clause(self, clause_id: str) -> dict | None:
        rows = await self.db.fetch_all("SELECT clause_id, article_no, article_title, lang, text FROM policy.clauses WHERE split_part(clause_id, '#', 1) = %s ORDER BY clause_id", (clause_id,))
        if not rows:
            return None
        texts: dict[str, str] = {}
        for r in rows:
            texts[r["lang"]] = (texts.get(r["lang"], "") + " " + r["text"]).strip()
        return {"clause_id": clause_id, "article": f"{rows[0]['article_no']} – {rows[0]['article_title']}", "text_en": texts.get("en"), "text_ar": texts.get("ar")}

    async def articles(self) -> list[dict]:
        rows = await self.db.fetch_all("SELECT article_no, article_title, count(*) AS clauses FROM policy.clauses WHERE lang='en' GROUP BY article_no, article_title ORDER BY article_no")
        return [dict(r) for r in rows]

    async def article(self, article_no: int) -> list[dict]:
        rows = await self.db.fetch_all("SELECT clause_id, lang, left(text, 160) AS preview FROM policy.clauses WHERE article_no = %s AND clause_id NOT LIKE '%%#%%' ORDER BY lang, clause_id", (article_no,))
        return [dict(r) for r in rows]
