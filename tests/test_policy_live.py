"""The policy server's deterministic eligibility tool, live: the clauses it cites exist in the index, in both languages,
and the figures it reports are the ones in the clause text.

Requires the policy MCP server (scripts/up.sh). Skipped when it is down, or when its image predates the tool.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from falcon.core import auth as roles
from falcon.core.auth import UserContext
from falcon.core.config import settings
from falcon.core.mcp_client import client_for


def _up(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2).status_code < 500
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(not _up(settings.mcp_policy_url), reason="policy server not running")


def _parse(result):
    text = "".join(b.get("text", "") for b in result if isinstance(b, dict)) if isinstance(result, list) else str(result)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_text": text}


async def _tools() -> dict:
    user = UserContext(user_id="u-layla", display_name="Layla Nasser", customer_id="C-1004", lang="ar", session_id="test-policy")
    return {t.name: t for t in await client_for(user, [roles.POLICY_READ], ["policy"]).get_tools()}


@pytest.fixture(scope="module")
def tools() -> dict:
    found = asyncio.run(_tools())
    if "check_travel_eligibility" not in found:
        pytest.skip("the policy server image predates check_travel_eligibility: run scripts/up.sh")
    return found


def _call(tools: dict, **args) -> dict:
    return _parse(asyncio.run(tools["check_travel_eligibility"].ainvoke(args)))


def test_the_38th_month_returns_the_pregnancy_clauses_with_their_text(tools):
    d = _call(tools, pregnancy_months=38)
    assert d["status"] == "needs_weeks" and [c["clause_id"] for c in d["clauses"]] == ["7.4.1", "7.4.3"]
    assert "36th week" in d["clauses"][1]["text_en"] and "السادس والثلاثين" in d["clauses"][1]["text_ar"]
    assert "28th week" in d["clauses"][0]["text_en"]
    assert d["findings"][0]["figures"]["last_week"] == 36 and "38th week" in d["findings"][0]["rule_en"]


def test_38_weeks_is_not_carried_and_30_weeks_needs_the_letter(tools):
    assert _call(tools, pregnancy_weeks=38)["status"] == "not_carried"
    d = _call(tools, pregnancy_weeks=30)
    assert d["status"] == "document_required" and "midwife" in d["clauses"][0]["text_en"]


def test_a_newborn_and_a_child_alone(tools):
    d = _call(tools, infant_age_days=3)
    assert d["status"] == "not_carried" and d["clauses"][0]["clause_id"] == "7.4.5" and "7 Days" in d["clauses"][0]["text_en"]
    d = _call(tools, child_age_years=8, travelling_alone=True)
    assert d["status"] == "conditions_apply" and [c["clause_id"] for c in d["clauses"]] == ["7.7.3", "7.7.2"]
    assert "12 hours" in d["clauses"][1]["text_en"] and "8 hours" in d["clauses"][0]["text_en"]  # 7.7.3 includes its second part


def test_no_fact_and_a_bad_figure_are_errors_not_answers(tools):
    text = _parse(asyncio.run(tools["check_travel_eligibility"].ainvoke({}))).get("_text", "")
    assert "at least one fact" in text
    text = _parse(asyncio.run(tools["check_travel_eligibility"].ainvoke({"child_age_years": -3}))).get("_text", "")
    assert "between 0 and 120" in text
