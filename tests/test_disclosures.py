"""Disclosed facts end to end, without services or a key.

The guard notices "38th month" / "30 weeks" / "3 days" / "8 years alone", the scripted model calls check_travel_eligibility (a stub here,
backed by the same rules function the policy server uses, with the clause text the server would attach), the reply states the rule
with its clause id and figures, and the reviewer verifies it against the tool result: cited ids retrieved, figures present, language right.
The 18 Oct reply ("have a safe trip") cannot come out of this path.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from falcon.core import rules
from falcon.services.runtime.fake_llm import RuleBasedFakeModel
from falcon.services.runtime.graph import build_graph

# the clause text the policy server attaches (data/policy/coc_en.pdf, Article 7), abridged
CLAUSES = {
    "7.4.1": "If you are pregnant we will carry you after your 28th week of gestation only if you produce to us a medical report or letter, signed by an appropriately qualified doctor or midwife.",
    "7.4.2": "You must submit a Medical Information Form (MEDIF) and obtain prior clearance from our medical department for travel in case of a complicated pregnancy.",
    "7.4.3": "We will not carry you: (a) in case of a single pregnancy, after the 36th week of gestation and (b) in case of a multiple pregnancy, after the 32nd week of gestation.",
    "7.4.5": "Carriage of newly born infants will generally not be permitted until 7 Days after delivery.",
    "7.7.1": "You will not be accepted for carriage if you are under 5 years of age unless you are accompanied by an escort aged 16 years or more.",
    "7.7.2": "If you are between 5 and 16 years of age, we can arrange for an escort, provided we receive notice at least 12 hours prior to travel.",
    "7.7.3": "If you are more than 5 years of age, but less than 12 years old, you will not be accepted for carriage unless the following conditions are met in full.",
}


def check_travel_eligibility(pregnancy_weeks: int | None = None, pregnancy_months: int | None = None, multiple_pregnancy: bool = False, complicated_pregnancy: bool = False,
                             infant_age_days: int | None = None, child_age_years: int | None = None, travelling_alone: bool = False) -> str:
    """What the policy server's tool returns, built from the same pure function (falcon.core.rules.travel_eligibility)."""
    findings = rules.travel_eligibility(pregnancy_weeks=pregnancy_weeks, pregnancy_months=pregnancy_months, multiple_pregnancy=multiple_pregnancy, complicated_pregnancy=complicated_pregnancy,
                                        infant_age_days=infant_age_days, child_age_years=child_age_years, travelling_alone=travelling_alone)
    if not findings:
        return "Error: give at least one fact."
    ids = list(dict.fromkeys(c for f in findings for c in f.clause_ids))
    return json.dumps({"status": max(findings, key=lambda f: rules.ELIGIBILITY_SEVERITY[f.status]).status, "findings": [asdict(f) for f in findings],
                       "clauses": [{"clause_id": c, "article": "7 – Refusal and limitation of carriage", "text_en": CLAUSES.get(c, ""), "text_ar": None} for c in ids]}, ensure_ascii=False)


TOOL = StructuredTool.from_function(check_travel_eligibility, name="check_travel_eligibility", description="Article 7 as thresholds (stub for the tests).")


def _talk(text: str, lang: str = "en", name: str = "Layla Nasser") -> dict:
    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "t1", "tools": [TOOL], "llm": RuleBasedFakeModel(), "default_lang": lang}}
    initial = {"messages": [HumanMessage(content=text)], "user_name": name, "preferred_lang": lang}
    asyncio.run(graph.ainvoke(initial, config))
    return asyncio.run(graph.aget_state(config)).values


def _reply(values: dict) -> str:
    last = values["messages"][-1]
    assert isinstance(last, AIMessage) and not last.tool_calls
    return last.content


def test_layla_in_her_38th_month_gets_the_rule_and_a_question_not_a_wish():
    v = _talk("جميل. أنا حامل في شهري الثامن والثلاثين, سأسافر في الوقت المحدد ان شاء الله", lang="ar")
    reply = _reply(v)
    assert [e["tool"] for e in v["turn_log"]] == ["check_travel_eligibility"] and v["turn_log"][0]["args"] == {"pregnancy_months": 38}
    assert "7.4.3" in reply and "36" in reply and "7.4.1" in reply and "28" in reply
    assert "رحلة سعيدة" not in reply and "آمنة" not in reply and "استثناء" in reply
    assert v["verification"]["status"] == "verified" and v["verification"]["rounds"] == []
    assert sorted(v["turn_log"][0]["clause_ids"]) == ["7.4.1", "7.4.3"]  # grounding: the ids the reply cites were retrieved in this turn
    assert [d["kind"] for d in v["guardrail"]["disclosures"]] == ["pregnancy"]


def test_30_weeks_pregnant_needs_the_letter_of_clause_7_4_1():
    v = _talk("I'm 30 weeks pregnant, I'll be flying to Cairo tomorrow as planned", name="Fatima Al Mansoori")
    reply = _reply(v)
    assert v["turn_log"][0]["args"] == {"pregnancy_weeks": 30}
    assert "7.4.1" in reply and "midwife" in reply and "28th week" in reply and "36th week" in reply
    assert v["verification"]["status"] == "verified"


def test_a_three_day_old_newborn_cannot_travel_yet():
    v = _talk("وُلد ابني قبل ثلاثة أيام وسنسافر به معنا على رحلتنا", lang="ar", name="Hassan Qasim")
    reply = _reply(v)
    assert v["turn_log"][0]["args"] == {"infant_age_days": 3}
    assert "7.4.5" in reply and "7 أيام" in reply and v["verification"]["status"] == "verified"


def test_an_eight_year_old_alone_gets_the_unaccompanied_minor_conditions():
    v = _talk("my 8 year old son will be travelling alone on this trip, we booked him on his own", name="Noura Abdullah")
    reply = _reply(v)
    assert v["turn_log"][0]["args"] == {"child_age_years": 8, "travelling_alone": True}
    assert "7.7.3" in reply and "7.7.2" in reply and "12 hours" in reply and v["verification"]["status"] == "verified"


def test_twins_are_passed_to_the_tool():
    v = _talk("just letting you know I am eight months pregnant with twins, see you on the flight")
    assert v["turn_log"][0]["args"] == {"pregnancy_months": 8, "multiple_pregnancy": True}
    assert "32nd week" in _reply(v)


def test_a_message_without_a_disclosure_takes_the_usual_path():
    v = _talk("hello, what can you do?")
    assert v["turn_log"] == [] and v["guardrail"]["disclosures"] == [] and "assistant" in _reply(v).lower()
