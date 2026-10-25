"""Query rewriting for policy retrieval: customer words -> the contract's words, in both languages.

Customers ask "my suitcase never showed up"; the Conditions of Carriage say "total or partial loss of Checked
Baggage". Measured on the retrieval gold set, plain lexical search found the answering clause in the top 5 for 35% of
such questions. One cheap LLM call turns the question into three search queries:

1. the question restated in the contract's formal English vocabulary,
2. the same in formal Arabic (the index holds both languages),
3. one sentence the answering clause would plausibly contain (hypothetical-document style, "HyDE").

The original question is always searched too, so a bad rewrite can add noise but cannot remove the plain search.
If the model is unavailable the rewriter returns nothing and search carries on without it.
"""

from __future__ import annotations

import json
import logging

from falcon.core.config import settings

log = logging.getLogger("falcon.policy.rewrite")

PROMPT = """You write search queries for an airline's Conditions of Carriage, a formal contract with articles on \
tickets, fares and taxes, reservations, check-in deadlines, refusal of carriage, medical clearance, pregnancy, \
unaccompanied children, baggage (free allowance, prohibited items, checked and unchecked baggage, animals, firearms), \
schedules, cancellations and denied boarding, refunds (involuntary and voluntary), conduct aboard, travel documents, \
liability and time limits for claims.

Given a customer's question (English or Arabic), return JSON {"queries": [q1, q2, q3]}:
q1: the question restated in the contract's formal English vocabulary (for example "loss of Checked Baggage",
    "refused carriage", "Check-in Deadline", "involuntary refund", "Unchecked Baggage", "Flight Coupons").
q2: the same restatement in formal Modern Standard Arabic.
q3: one English sentence, in the contract's style, that the clause answering the question would plausibly contain.
Do not answer the question. Return only the JSON."""


class QueryRewriter:
    def __init__(self, model: str | None = None):
        self.model = model or settings.llm_model
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(settings.openrouter_api_key)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(base_url=settings.openrouter_base_url, api_key=settings.openrouter_api_key, timeout=20, max_retries=1)
        return self._client

    async def rewrite(self, question: str) -> list[str]:
        if not self.enabled:
            return []
        try:
            resp = await self._get_client().chat.completions.create(
                model=self.model, temperature=0, max_tokens=300, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": question}])
            queries = json.loads(resp.choices[0].message.content or "{}").get("queries", [])
            return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:3]
        except Exception as exc:  # noqa: BLE001 - retrieval must not fail because the rewriter did
            log.warning("query rewrite failed, searching the original question only: %s", exc)
            return []
