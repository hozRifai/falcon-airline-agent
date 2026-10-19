"""Policy MCP server: the knowledge base (Conditions of Carriage, English and Arabic, linked by clause id).

Tools let the agent search, open a clause, and navigate by article, which is what an "agentic"
retrieval loop needs: search → read the cited clause in full → check the neighbouring clauses. `check_travel_eligibility`
applies Article 7 (pregnancy, newborns, children alone) as fixed thresholds and returns the clauses it rests on (decision #34).
Role: Policy.Read. Read-only database role: policy_ro.

Run:  python -m falcon.services.mcp_policy.server   (port 8003)
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import uvicorn
from mcp.server.fastmcp.exceptions import ToolError

from falcon.core import auth as roles
from falcon.core import rules
from falcon.core.config import settings
from falcon.core.db import Database
from falcon.core.mcp_server import Audit, audited, build_server, require_role, wrap_app
from falcon.policy.embed import Embedder
from falcon.policy.rewrite import QueryRewriter
from falcon.policy.search import PolicySearch

log = logging.getLogger("falcon.policy")
PORT = 8003

db = Database(settings.dsn_policy, max_size=4)
search = PolicySearch(db, Embedder(), QueryRewriter())
audit = Audit("policy", Database(settings.dsn_audit, max_size=2))

mcp = build_server(
    "falcon-policy",
    audience=settings.entra_aud_policy,
    public_url=settings.mcp_policy_url,
    port=PORT,
    instructions="Falcon's Conditions of Carriage in English and Arabic. Semantic search returns clause ids; cite them.",
)


@mcp.tool()
@audited(audit, "search_policy")
async def search_policy(query: str, top_k: int = 6) -> dict:
    """Search the General Conditions of Carriage. Pass the customer's question as a short natural sentence, in English
    or Arabic (for example "can my 8 year old fly alone"); the search is semantic and cross-lingual, so one language is
    enough. Leave top_k at its default: one clause rarely answers a question on its own. Returns clauses with their ids and the
    text in both languages when available, with a confidence score; a low-confidence first pass is restated and expanded automatically.
    Always cite the clause id you rely on, and only for what its text says.
    Fees, deadlines and refund amounts come from get_fare_rules in the reservations system, not from this document."""
    require_role(roles.POLICY_READ)
    if not query or len(query.strip()) < 3:
        raise ToolError("Give a few key terms to search for.")
    top_k = max(1, min(int(top_k), 10))
    result = await search.search(query.strip(), top_k, rewrite=settings.policy_query_rewrite)
    if result["available"] and not result["results"]:
        result["note"] = "No clause matched. Try other words, or browse list_articles."
    return result


@mcp.tool()
@audited(audit, "get_clause")
async def get_clause(clause_id: str) -> dict:
    """Read one clause in full, in English and Arabic, by its id (for example 12.2.1)."""
    require_role(roles.POLICY_READ)
    row = await search.clause(clause_id.strip())
    if not row:
        raise ToolError(f"No clause {clause_id}.")
    return row


@mcp.tool()
@audited(audit, "list_articles")
async def list_articles() -> dict:
    """The 21 articles of the Conditions of Carriage with their titles (a table of contents for navigation)."""
    require_role(roles.POLICY_READ)
    return {"articles": await search.articles()}


@mcp.tool()
@audited(audit, "get_article")
async def get_article(article_no: int) -> dict:
    """All clause ids of one article with a short preview of each, in both languages. Use get_clause to read one in full."""
    require_role(roles.POLICY_READ)
    rows = await search.article(int(article_no))
    if not rows:
        raise ToolError(f"No article {article_no}.")
    return {"article_no": article_no, "clauses": rows}


@mcp.tool()
@audited(audit, "get_glossary")
async def get_glossary() -> dict:
    """English/Arabic glossary of the policy's own terms (ticket, fare, checked baggage, denied boarding...). Use it to build bilingual search queries."""
    require_role(roles.POLICY_READ)
    rows = await db.fetch_all("SELECT term_en, term_ar, note FROM policy.glossary ORDER BY term_en")
    return {"terms": [dict(r) for r in rows]}


@mcp.tool()
@audited(audit, "check_travel_eligibility")
async def check_travel_eligibility(pregnancy_weeks: int | None = None, pregnancy_months: int | None = None, multiple_pregnancy: bool = False,
                                   complicated_pregnancy: bool = False, infant_age_days: int | None = None, child_age_years: int | None = None,
                                   travelling_alone: bool = False) -> dict:
    """Deterministic check of what the Conditions of Carriage (Article 7) say for a pregnancy, a newborn or infant, or a child travelling
    alone. Call it as soon as a customer mentions one of these, even in passing and without a question, before greeting them or confirming
    anything. Pass the facts as stated: pregnancy_weeks, or pregnancy_months when the customer spoke in months (the rule counts weeks; the
    result then tells you to ask for the week); multiple_pregnancy for twins; complicated_pregnancy when they say so; infant_age_days for a
    newborn (weeks x 7); child_age_years with travelling_alone. Returns a status (ok, needs_weeks, conditions_apply, document_required,
    escort_required, clearance_required, not_carried), one finding per fact with the rule in English and Arabic and the figures it used, and
    the full text of every clause it relied on, so you can cite the clause id. State the finding first, then the options when travel is
    restricted: the document to obtain, cancellation under the fare rules (quote_cancellation), or an exception case with the document
    attached (request_exception). Other medical conditions, animals, items and documents: use search_policy."""
    require_role(roles.POLICY_READ)
    try:
        findings = rules.travel_eligibility(pregnancy_weeks=pregnancy_weeks, pregnancy_months=pregnancy_months, multiple_pregnancy=bool(multiple_pregnancy),
                                            complicated_pregnancy=bool(complicated_pregnancy), infant_age_days=infant_age_days, child_age_years=child_age_years,
                                            travelling_alone=bool(travelling_alone))
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    if not findings:
        raise ToolError("Give at least one fact: pregnancy_weeks or pregnancy_months, infant_age_days, or child_age_years (with travelling_alone).")
    clauses = []
    for cid in dict.fromkeys(c for f in findings for c in f.clause_ids):
        row = await search.clause(cid)
        if row:
            clauses.append(row)
    status = max(findings, key=lambda f: rules.ELIGIBILITY_SEVERITY[f.status]).status
    return {"status": status, "findings": [asdict(f) for f in findings], "clauses": clauses,
            "source": "Article 7 of the Conditions of Carriage, applied as fixed thresholds; cite the clause ids above. Fees, refunds and deadlines come from the quote tools."}


app = wrap_app(mcp)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
