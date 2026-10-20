"""The reviewer loop (decision #26): deterministic checks shared with the scorer, the retry with a precise nudge, the hold.

No database and no key: the graph runs with an in-memory checkpointer and a stub model that returns scripted drafts.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from falcon.core import faithfulness as f
from falcon.core.config import settings
from falcon.services.runtime.graph import build_graph
from falcon.services.runtime.review import HELD, review_draft

QUOTE = json.dumps({"kind": "cancel", "pnr": "K3P7QZ", "allowed": True, "fee": "100.00", "refund": "1291.00", "currency": "AED", "quote_id": "Q-1"})
CANCELLED = json.dumps({"already_done": False, "pnr": "K3P7QZ", "status": "cancelled", "refund_amount": "1291.00", "refund_method": "credit"})
DECLINED = json.dumps({"declined": True, "message": "The person did not confirm this action. Nothing was changed."})


# ---------------------------------------------------------------- the checks
def test_amounts_must_come_from_tool_results_or_the_customer():
    ok = f.check_reply("The fee is 100 AED and you get 1,291 AED back as travel credit.", [("quote_cancellation", QUOTE)], [])
    assert ok.ok and ok.amounts == ["100.00", "1291.00"] and ok.evidence == 1
    bad = f.check_reply("You will get AED 1,391 back.", [("quote_cancellation", QUOTE)], [])
    assert not bad.ok and bad.unsupported_amounts == ["1391.00"] and "1391.00" in bad.nudge()
    said_by_customer = f.check_reply("You paid 1,391 AED, as you said.", [("quote_cancellation", QUOTE)], ["I paid 1391 for this"])
    assert said_by_customer.ok
    no_tools = f.check_reply("Your refund would be about 900.00 AED.", [], [])
    assert not no_tools.ok and no_tools.evidence == 0


def test_clause_ids_arabic_digits_and_plain_numbers_are_not_money():
    assert f.amounts_in("See clause 8.10 and Article 10.2.1; also [7.1] applies.") == set()
    assert f.amounts_in("الرسوم ١٠٠ درهم والمبلغ المسترد ١٬٢٩١ درهم") == {"100.00", "1291.00"}
    assert f.amounts_in("Flight FA929 departs at 19:40 on 2026-10-21, within 24 hours, 30 kg included.") == set()
    assert f.amounts_in("AED 60 per bag") == {"60.00"}


def test_done_claims_need_evidence_in_the_trace():
    claim = "Your booking has been cancelled and 1,291 AED will be refunded as travel credit."
    assert not f.check_reply(claim, [("quote_cancellation", QUOTE)], []).ok                       # quoted only, nothing executed
    assert f.check_reply(claim, [("quote_cancellation", QUOTE), ("cancel_booking", CANCELLED)], []).ok  # the write succeeded
    assert not f.check_reply(claim, [("quote_cancellation", QUOTE), ("cancel_booking", DECLINED)], []).ok  # the person declined
    assert not f.check_reply(claim, [("quote_cancellation", QUOTE), ("cancel_booking", "Error: the quote expired")], []).ok
    # a cancellation decided elsewhere shows up in the booking's state: that is evidence too
    state = json.dumps({"pnr": "K3P7QZ", "status": "cancelled", "segments": []})
    assert f.check_reply("Your booking was cancelled on 19 October.", [("get_booking", state)], []).ok
    # a cancelled *flight* is not a cancelled booking
    flight = json.dumps({"found": True, "flights": [{"flight_number": "FA927", "status": "cancelled"}]})
    assert not f.check_reply("Your booking has been cancelled.", [("get_flight_status", flight)], []).ok
    assert f.check_reply("Your flight has been cancelled by the airline; you are entitled to a full refund.", [("get_flight_status", flight)], []).ok
    # Arabic
    assert not f.check_reply("تم إلغاء حجزك.", [("quote_cancellation", QUOTE)], []).ok
    assert f.check_reply("سيتم استرداد المبلغ 1291.00 درهم.", [("quote_cancellation", QUOTE)], []).ok  # a promise, not a claim


def test_policy_statements_need_a_policy_tool():
    draft = AIMessage(content="According to clause 10.2.1 of the Conditions of Carriage, infants travel free.")
    r = review_draft(draft, [HumanMessage(content="do infants pay?")])
    assert "policy_without_source" in r.problems and "unretrieved_clause" in r.problems and "search_policy" in r.nudge
    with_source = review_draft(draft, [HumanMessage(content="do infants pay?"), ToolMessage(content=json.dumps({"results": [{"clause_id": "10.2.1"}]}), tool_call_id="1", name="search_policy")])
    assert with_source.ok


SEARCH_ONE_DAY = json.dumps({"dates": "2026-10-27 to 2026-10-27", "days_searched": 1, "options_cheapest_first": [{"flight_number": "FA951", "price_per_passenger_aed": {"SPECIAL": 965}}]})


def _search_turn(customer: str, calls: list[dict]) -> list:
    msgs = [HumanMessage(content=customer)]
    for i, args in enumerate(calls):
        msgs.append(AIMessage(content="", tool_calls=[{"name": "search_flights", "args": args, "id": f"c{i}", "type": "tool_call"}]))
        msgs.append(ToolMessage(content=SEARCH_ONE_DAY, tool_call_id=f"c{i}", name="search_flights"))
    return msgs


def test_a_period_searched_as_one_day_is_sent_back_with_the_day():
    # FAILURE_DIARY #39: "next week" → search_flights(date_from=2026-10-27) → "the cheapest is FA951 at 965" (the 26th was 900)
    draft = AIMessage(content="The cheapest flight from Dubai to Beirut on 27 October is FA951: Special 965 AED.")
    r = review_draft(draft, _search_turn("I want to book a flight the next week, find me the cheapest one from dubai to beirut", [{"origin": "DXB", "destination": "BEY", "date_from": "2026-10-27"}]))
    assert r.problems == ["period_searched_as_one_day"] and "2026-10-27" in r.nudge and "date_to" in r.nudge
    ar = review_draft(AIMessage(content="أرخص رحلة هي FA951 بسعر 965 درهم."), _search_turn("أريد أرخص رحلة من دبي إلى بيروت الأسبوع القادم", [{"origin": "دبي", "destination": "بيروت", "date_from": "2026-10-27"}]))
    assert ar.problems == ["period_searched_as_one_day"]


@pytest.mark.parametrize("customer, calls, draft", [
    # a range was searched: the period is covered (even if an earlier call was one day)
    ("cheapest from dubai to beirut next week", [{"date_from": "2026-10-27"}, {"date_from": "2026-10-26", "date_to": "2026-11-01"}], "FA951 on Monday 26 October: 965 AED."),
    # the customer named a day inside the period: one day is right
    ("Tuesday next week, dubai to beirut", [{"date_from": "2026-10-27"}], "FA951 on 27 October: 965 AED."),
    ("the 27th next week please", [{"date_from": "2026-10-27"}], "FA951 on 27 October: 965 AED."),
    ("أريد رحلة يوم الثلاثاء من الأسبوع القادم", [{"date_from": "2026-10-27"}], "FA951 بسعر 965 درهم."),
    # no flights presented (a question back), or no search at all: nothing to check
    ("next week, dubai to beirut", [{"date_from": "2026-10-27"}], "Which day next week suits you?"),
    ("what is the status of my flight next week?", [], "FA923 on 21 October is cancelled."),
])
def test_scope_check_stays_quiet_when_one_day_is_right_or_nothing_was_presented(customer, calls, draft):
    assert review_draft(AIMessage(content=draft), _search_turn(customer, calls)).ok


# ---------------------------------------------------------------- the loop in the graph
class ScriptedModel(GenericFakeChatModel):
    """Returns the scripted drafts in order; bind_tools is a no-op so the graph can use it like a real model."""

    def bind_tools(self, tools: Any, **kwargs: Any):  # noqa: D401
        return self


def _run(drafts: list[str], human: str = "what do I get back if I cancel?", lang_hint: str = "en") -> dict:
    model = ScriptedModel(messages=iter([AIMessage(content=d) for d in drafts]))
    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "t1", "tools": [], "llm": model, "default_lang": lang_hint}}
    initial = {"messages": [HumanMessage(content=human)], "role": "customer", "user_name": "Test", "preferred_lang": lang_hint}
    asyncio.run(graph.ainvoke(initial, config))
    return asyncio.run(graph.aget_state(config)).values


def test_a_good_first_draft_is_verified_without_a_retry():
    values = _run(["I can price the cancellation for you: shall I check the quote?"])
    assert values["messages"][-1].content.startswith("I can price")
    assert values["verification"]["status"] == "verified" and values["verification"]["rounds"] == []


def test_unsupported_amount_then_false_claim_then_a_clean_draft():
    values = _run(["You would get about 900.00 AED back.", "Your booking has been cancelled.", "I can quote the exact refund: shall I check?"])
    v = values["verification"]
    assert values["messages"][-1].content.startswith("I can quote")
    assert v["status"] == "retried" and [r["problems"] for r in v["rounds"]] == [["unsupported_amounts"], ["false_claim"]]


def test_every_round_failing_holds_the_reply_in_the_customer_language():
    bad = ["المبلغ المسترد 900.00 درهم."] * (settings.review_max_rounds + 1)
    values = _run(bad, human="كم سأسترد إذا ألغيت؟", lang_hint="ar")
    assert values["messages"][-1].content == HELD["ar"]
    v = values["verification"]
    assert v["status"] == "held" and len(v["rounds"]) == settings.review_max_rounds + 1
    assert all(r["problems"] == ["unsupported_amounts"] for r in v["rounds"])


@pytest.mark.parametrize("text", ["Flight FA929 leaves at 19:40 tomorrow.", "Your Saver fare includes 30 kg of checked baggage."])
def test_replies_without_money_or_claims_pass(text):
    values = _run([text])
    assert values["verification"]["status"] == "verified" and values["verification"]["amounts"] == []


# ---------------------------------------------------------------- language and policy answers (FAILURE_DIARY #43)
from falcon.services.runtime.review import cited_clause_ids  # noqa: E402

SEARCH = json.dumps({"results": [{"clause_id": "7.4.4", "text_en": "It is your responsibility to check if Article 7.4.1, 7.4.2 or 7.4.3 applies to you."},
                                 {"clause_id": "8.1", "text_en": "Your free Baggage allowance and restrictions on size are shown in your Ticket."}]})
POLICY_TRACE = [HumanMessage(content="ما هي سياسة الأمتعة للرضع؟"), ToolMessage(content=SEARCH, tool_call_id="1", name="search_policy")]


def test_an_arabic_question_gets_an_arabic_reply():
    bad = review_draft(AIMessage(content="According to clause 8.1, the allowance is shown on your ticket."), POLICY_TRACE, lang="ar")
    assert "wrong_language" in bad.problems and "Modern Standard Arabic" in bad.nudge
    ok = review_draft(AIMessage(content="وفقًا للبند 8.1، يظهر وزن الأمتعة المسموح به في تذكرتك (الحجز K3P7QZ، الرحلة FA951 من DXB)."), POLICY_TRACE, lang="ar")
    assert ok.ok
    assert review_draft(AIMessage(content="Flight FA951 departs 08:30."), [HumanMessage(content="when does it leave?")], lang="en").ok


def test_policy_figures_must_come_from_the_retrieved_text():
    # the laptop reply of 18 Oct: 7.4.4 is a pregnancy clause and the document gives no 10 kg / 115 cm anywhere
    bad = review_draft(AIMessage(content="According to clause 7.4.4, infants get one checked item not exceeding 10 kg and 115 cm."), POLICY_TRACE, lang="en")
    assert "unsupported_quantities" in bad.problems and "10, 115" in bad.nudge
    rules = ToolMessage(content=json.dumps({"fare_families": [{"code": "SAVER", "included_checked_bag_kg": 30}], "extra": {"10 kg": "60.00"}}), tool_call_id="2", name="get_fare_rules")
    ok = review_draft(AIMessage(content="Under the fare rules your Saver fare includes 30 kg; an extra 10 kg costs 60 AED (clause 8.1 says the allowance is on your ticket)."),
                      POLICY_TRACE + [rules], lang="en")
    assert ok.ok
    # a flight reply is not a policy answer: durations are not checked against the document
    assert review_draft(AIMessage(content="FA951 departs 08:30, duration 3 hours 50 minutes."), [HumanMessage(content="flights to Beirut?")], lang="en").ok


def test_cited_clauses_must_have_been_retrieved():
    assert cited_clause_ids("See clauses 7.4.1 and 7.4.3, Article 10.2.1 and [8.1]; البند 8.10.") == ["7.4.1", "7.4.3", "10.2.1", "8.1", "8.10"]
    # FAILURE_DIARY #50: the forms DeepSeek writes in Arabic (a preposition drops the article's alef) and with markdown emphasis
    assert cited_clause_ids("وفقًا للبند **6.1** من شروط النقل، يجب الحضور قبل **3 ساعات**.") == ["6.1"]
    assert cited_clause_ids("وفقًا للفقرة 3.2.1 والمادة رقم 10.2 وبالبند 7.4.1 و7.4.3") == ["3.2.1", "10.2", "7.4.1", "7.4.3"]
    assert cited_clause_ids("According to **clause 13.2.1**, and clause **13.2.2** states; see clauses **7.4.1** and **7.4.3**.") == ["13.2.1", "13.2.2", "7.4.1", "7.4.3"]
    assert cited_clause_ids("وفقًا لشروط النقل (المادة 17)، تحدد اللوائح بدل الأمتعة. See Article 8 and clause 8.1.") == ["17", "8", "8.1"]
    assert cited_clause_ids("You may bring 2 bags; paragraph 3 of your ticket; clause 8.1 and 2 extra bags") == ["8.1"]  # bare numbers only after article words
    fabricated = review_draft(AIMessage(content="See clause 8.3.2 for prohibited items."), POLICY_TRACE, lang="en")
    assert "unretrieved_clause" in fabricated.problems and "8.3.2" in fabricated.nudge
    parent = review_draft(AIMessage(content="See clause 7.4 for the pregnancy rules."), POLICY_TRACE, lang="en")
    assert parent.ok  # the parent of a retrieved clause is fine
    none_retrieved = review_draft(AIMessage(content="Clause 8.1 covers this."), [HumanMessage(content="baggage?")], lang="en")
    assert "policy_without_source" in none_retrieved.problems and "unretrieved_clause" in none_retrieved.problems


def test_sent_or_emailed_claims_need_the_email_tool():
    # the laptop reply of 18 Oct: "Your ticket ... has been sent to your email" with only get_booking in the trace, and no email tool at all
    booking = json.dumps({"pnr": "B4N6WX", "status": "confirmed"})
    claim = "Your ticket for booking B4N6WX has been sent to your email. Please check your inbox."
    assert not f.check_reply(claim, [("get_booking", booking)], []).ok
    sent = json.dumps({"sent": True, "pnr": "B4N6WX", "to": "o***@example.com", "reference": "MAIL-B4N6WX-1"})
    assert f.check_reply(claim, [("get_booking", booking), ("send_itinerary", sent)], []).ok
    assert not f.check_reply("I've emailed the itinerary to you.", [("get_booking", booking)], []).ok
    assert not f.check_reply("تم إرسال التذكرة إلى بريدك الإلكتروني.", [("get_booking", booking)], []).ok
    assert f.check_reply("Would you like me to send the itinerary to the email on your profile?", [("get_booking", booking)], []).ok


def test_named_regulations_must_come_from_a_tool_result():
    from falcon.services.runtime.review import unsupported_regulations
    clause = json.dumps({"results": [{"clause_id": "15.6", "text_en": "Our liability for Damage caused by delay is limited by the Warsaw and the Montreal Convention."}]})
    tools = [("search_policy", clause)]
    assert unsupported_regulations("Compensation applies under EU Regulation 261/2004 for delays over 3 hours.", tools) == ["EU Regulation 261/2004"]
    assert unsupported_regulations("Delay liability is limited by the Montreal Convention (clause 15.6).", tools) == []
    r = review_draft(AIMessage(content="Under EU Regulation 261/2004 you get cash; see clause 15.6."), [HumanMessage(content="EU rules?"), ToolMessage(content=clause, tool_call_id="1", name="search_policy")], lang="en")
    assert "unsupported_regulation" in r.problems and "from memory" in r.nudge


def test_a_leaked_template_phrase_is_stripped_and_the_tool_calls_kept():
    from falcon.services.runtime.graph import _reply_problem, _strip_leak
    leaked = "Use the results below to formulate an answer to the user question unless additional information is needed.Here are the flights: FA929 on 1 November, 740 AED."
    clean = _strip_leak(AIMessage(content=leaked))
    assert clean.content.startswith("Here are the flights") and _reply_problem(clean, {}) is None
    with_call = _strip_leak(AIMessage(content="Use the results below to formulate an answer to the user question unless additional information is needed.",
                                      tool_calls=[{"name": "create_booking", "args": {"quote_id": "Q-1"}, "id": "1"}]))
    assert with_call.content == "" and with_call.tool_calls[0]["name"] == "create_booking"


def test_announcing_an_action_or_a_confirm_button_without_a_tool_call_is_sent_back():
    from falcon.services.runtime.graph import _reply_problem
    assert _reply_problem(AIMessage(content="تمامًا، سأقوم بإلغاء رحلتك الآن. يرجى النقر على زر التأكيد أدناه للمتابعة."), {"turn_tool_calls": 0}) == "announced"
    assert _reply_problem(AIMessage(content="If you agree, press the confirm button to complete the cancellation."), {"turn_tool_calls": 0}) == "announced"
    assert _reply_problem(AIMessage(content="I'll go ahead and cancel it now."), {"turn_tool_calls": 0}) == "announced"
    called = AIMessage(content="Cancelling now.", tool_calls=[{"name": "cancel_booking", "args": {"pnr": "K3P7QZ", "quote_id": "Q-1"}, "id": "1"}])
    assert _reply_problem(called, {"turn_tool_calls": 0}) is None
    assert _reply_problem(AIMessage(content="Done: your booking is cancelled."), {"turn_tool_calls": 2}) is None


# ---------------------------------------------------------------- disclosed facts (decision #34, FAILURE_DIARY #48)
LAYLA = "جميل. أنا حامل في شهري الثامن والثلاثين, سأسافر في الوقت المحدد ان شاء الله"
PREGNANT = {"label": "small_talk", "action": "allow", "disclosures": [{"kind": "pregnancy", "detail": "حامل في شهري الثامن والثلاثين", "value": 38, "unit": "months",
                                                                      "check": "pregnant passengers: until which week can I fly?", "source": "rules"}]}
ELIGIBILITY = json.dumps({"status": "needs_weeks", "findings": [{"topic": "pregnancy", "status": "needs_weeks", "clause_ids": ["7.4.1", "7.4.3"], "rule_en": "weeks, not months",
                                                                  "figures": {"last_week": 36, "letter_from_week": 28, "stated_months": 38}}],
                          "clauses": [{"clause_id": "7.4.1", "text_en": "after your 28th week of gestation only with a medical report"}, {"clause_id": "7.4.3", "text_en": "after the 36th week"}]})
WISH = "أهلاً بك، ليلى! نتمنى لك رحلة مريحة وآمنة بإذن الله."  # the 18 Oct reply, verbatim


def test_a_disclosure_answered_without_a_check_is_sent_back_naming_the_fact_and_the_tools():
    r = review_draft(AIMessage(content=WISH), [HumanMessage(content=LAYLA)], lang="ar", guardrail=PREGNANT)
    assert r.problems == ["disclosure_unchecked"]
    assert "check_travel_eligibility" in r.nudge and "الثامن والثلاثين" in r.nudge and "until which week" in r.nudge and "good trip" in r.nudge
    assert review_draft(AIMessage(content=WISH), [HumanMessage(content=LAYLA)], lang="ar", guardrail={"label": "small_talk", "disclosures": []}).ok
    assert review_draft(AIMessage(content=WISH), [HumanMessage(content=LAYLA)], lang="ar").ok  # no guardrail record (the opening message)


def test_a_check_in_this_turn_satisfies_the_floor_and_an_earlier_or_failed_one_does_not():
    call = AIMessage(content="", tool_calls=[{"name": "check_travel_eligibility", "args": {"pregnancy_months": 38}, "id": "c1", "type": "tool_call"}])
    answer = AIMessage(content="وفقًا للبند 7.4.3 لا يمكن النقل بعد الأسبوع 36 من الحمل، ويلزم تقرير طبي بعد الأسبوع 28 (البند 7.4.1). في أي أسبوع أنتِ الآن؟")
    checked = [HumanMessage(content=LAYLA), call, ToolMessage(content=ELIGIBILITY, tool_call_id="c1", name="check_travel_eligibility")]
    assert review_draft(answer, checked, lang="ar", guardrail=PREGNANT).ok
    search = json.dumps({"results": [{"clause_id": "7.4.3", "text_en": "after the 36th week"}, {"clause_id": "7.4.1", "text_en": "after your 28th week"}]})
    searched = [HumanMessage(content=LAYLA), AIMessage(content="", tool_calls=[{"name": "search_policy", "args": {"query": "pregnancy"}, "id": "c2", "type": "tool_call"}]),
                ToolMessage(content=search, tool_call_id="c2", name="search_policy")]
    assert review_draft(answer, searched, lang="ar", guardrail=PREGNANT).ok
    failed = [HumanMessage(content=LAYLA), call, ToolMessage(content="Error: pregnancy_months must be between 0 and 99; got 380.", tool_call_id="c1", name="check_travel_eligibility")]
    assert "disclosure_unchecked" in review_draft(AIMessage(content=WISH), failed, lang="ar", guardrail=PREGNANT).problems
    earlier_turn = [HumanMessage(content="مرحبا"), call, ToolMessage(content=ELIGIBILITY, tool_call_id="c1", name="check_travel_eligibility"), AIMessage(content="..."), HumanMessage(content=LAYLA)]
    assert "disclosure_unchecked" in review_draft(AIMessage(content=WISH), earlier_turn, lang="ar", guardrail=PREGNANT).problems


def test_in_the_graph_the_good_trip_wish_is_held_when_the_model_never_checks():
    # the guard's pattern layer records the pregnancy (no key needed); a model that only ever wishes a good trip is nudged every round, then held
    values = _run([WISH] * (settings.review_max_rounds + 1), human=LAYLA, lang_hint="ar")
    assert [d["kind"] for d in values["guardrail"]["disclosures"]] == ["pregnancy"]
    assert values["messages"][-1].content == HELD["ar"]
    assert values["verification"]["status"] == "held" and all(r["problems"] == ["disclosure_unchecked"] for r in values["verification"]["rounds"])


def test_logged_and_arabic_done_claims_need_evidence():
    booking = json.dumps({"pnr": "M8T2LD", "status": "confirmed"})
    # the final run of 18 Oct: a case number invented after quote_cancellation, and "I have already booked" after the customer typed a fake PNR
    assert not f.check_reply("لقد قمت بتسجيل حالة استثناء لفريق خدمة العملاء. رقم الحالة هو CASE-20261020-001.", [("quote_cancellation", booking)], []).ok
    assert not f.check_reply("لحسن الحظ، لقد قمت بالفعل بحجز الرحلة بنجاح. رقم الحجز FA123456.", [("quote_booking", booking)], []).ok
    assert not f.check_reply("I have logged case #7 for the customer-care team.", [("quote_cancellation", booking)], []).ok
    logged = json.dumps({"request_id": 7, "status": "pending"})
    assert f.check_reply("I have logged case #7 for the customer-care team.", [("quote_cancellation", booking), ("request_exception", logged)], []).ok
    assert f.check_reply("قمت بمراجعة حجزك ولم أجد أي مشكلة.", [("get_booking", booking)], []).ok  # reviewing is not a write


def test_a_policy_answer_without_a_clause_or_an_honest_not_covered_is_sent_back():
    hits = json.dumps({"results": [{"clause_id": "8.1", "text_en": "Your free Baggage allowance is shown in your Ticket."}]})
    trace = [HumanMessage(content="what do the conditions say about infant baggage?"), ToolMessage(content=hits, tool_call_id="1", name="search_policy")]
    invented = review_draft(AIMessage(content="According to the Conditions of Carriage, strollers and car seats travel free within the size limits."), trace, lang="en")
    assert "uncited_policy_answer" in invented.problems
    assert review_draft(AIMessage(content="According to clause 8.1 the allowance is shown on your ticket."), trace, lang="en").ok
    rules = ToolMessage(content=json.dumps({"fare_families": [{"code": "SAVER", "included_checked_bag_kg": 30}]}), tool_call_id="2", name="get_fare_rules")
    assert review_draft(AIMessage(content="The Conditions of Carriage do not explicitly mention infant baggage; the fare rules give 30 kg per passenger."), trace + [rules], lang="en").ok
    # the same sentence without the fare-rules result: the figure check refuses the 30 kg, as it should
    assert "unsupported_quantities" in review_draft(AIMessage(content="The Conditions of Carriage do not explicitly mention infant baggage; the fare rules give 30 kg per passenger."), trace, lang="en").problems
    assert review_draft(AIMessage(content="شروط النقل لا تذكر أمتعة الرضع تحديدًا؛ يمكنني تحويلك إلى موظف."), trace, lang="ar").ok


def test_tool_names_never_reach_the_customer():
    from falcon.services.runtime.review import output_problems
    # the newborn reply of 18 Oct: "إلغاء الحجز ... (quote_cancellation)" and "طلب استثناء ... (request_exception)"
    problems = output_problems("يمكنكم إلغاء الحجز واسترداد المبلغ وفقًا لشروط السعر (quote_cancellation) أو تقديم طلب استثناء (request_exception).")
    assert [p for p, _ in problems] == ["tool_name_in_reply"] and "quote_cancellation, request_exception" in problems[0][1]
    assert output_problems("I can quote the cancellation under your fare rules, or log an exception case for the customer-care team with your document attached.") == []


def test_a_repeated_disclosure_needs_a_check_in_the_conversation_not_in_the_turn():
    # the first real-model run: turn 1 checked the pregnancy; in turn 2 "I'm in week 38, single" was flagged again and the reply was held
    call = AIMessage(content="", tool_calls=[{"name": "check_travel_eligibility", "args": {"pregnancy_months": 38}, "id": "c1", "type": "tool_call"}])
    thread = [HumanMessage(content=LAYLA), call, ToolMessage(content=ELIGIBILITY, tool_call_id="c1", name="check_travel_eligibility"),
              AIMessage(content="الرجاء توضيح الأسبوع."), HumanMessage(content="أنا حامل في الأسبوع الثامن والثلاثين بجنين واحد.")]
    answer = AIMessage(content="في الأسبوع 38 لا يمكن نقلك: لا يُسمح بالسفر بعد الأسبوع 36 (البند 7.4.3). يمكنني تسعير الإلغاء أو تسجيل حالة استثناء مع تقرير طبي.")
    repeat = {**PREGNANT, "repeat": ["pregnancy"]}
    assert review_draft(answer, thread, lang="ar", guardrail=repeat).ok                      # checked earlier in the conversation
    assert "disclosure_unchecked" in review_draft(answer, thread, lang="ar", guardrail=PREGNANT).problems   # a first disclosure: this turn
    no_check = [HumanMessage(content="مرحبا"), AIMessage(content="أهلاً"), HumanMessage(content=LAYLA)]
    assert "disclosure_unchecked" in review_draft(answer, no_check, lang="ar", guardrail=repeat).problems   # a repeat with no check anywhere
    mixed = {**repeat, "disclosures": PREGNANT["disclosures"] + [{"kind": "animal", "detail": "my cat", "check": "pets in the cabin", "source": "llm"}]}
    r = review_draft(answer, thread, lang="ar", guardrail=mixed)
    assert "disclosure_unchecked" in r.problems and 'animal ("my cat")' in r.nudge and "pets in the cabin" in r.nudge
    assert 'pregnancy ("' not in r.nudge and "until which week" not in r.nudge  # only the new kind and its question are demanded


def test_the_graph_remembers_disclosed_kinds_across_turns():
    model = ScriptedModel(messages=iter([AIMessage(content=WISH) for _ in range(10)]))  # distinct objects: a reused message keeps the id the state gave it
    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "t2", "tools": [], "llm": model, "default_lang": "ar"}}
    asyncio.run(graph.ainvoke({"messages": [HumanMessage(content=LAYLA)], "role": "customer", "user_name": "Layla", "preferred_lang": "ar"}, config))
    first = asyncio.run(graph.aget_state(config)).values
    assert first["disclosed"] == ["pregnancy"] and first["guardrail"]["repeat"] == []
    asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="أنا حامل في الأسبوع الثامن والثلاثين بجنين واحد")]}, config))
    second = asyncio.run(graph.aget_state(config)).values
    assert second["guardrail"]["repeat"] == ["pregnancy"] and second["disclosed"] == ["pregnancy"]
    assert second["messages"][-1].content == HELD["ar"]  # no policy call anywhere in this conversation: the floor still holds the wish


def test_an_arabic_draft_citing_with_a_preposition_and_emphasis_passes_the_policy_checks():
    # FAILURE_DIARY #50: "وفقًا للبند **6.1**" was sent back four times for "no citation" and the customer got the held message
    from falcon.services.runtime.review import policy_problems

    result = json.dumps({"query": "q", "available": True, "results": [{"clause_id": "6.1", "article": "6 – Check-in and boarding",
                         "text_en": "You must present yourself for check-in no later than 3 hours for Economy Class.",
                         "text_ar": "يجب أن تحضر لتسجيل الوصول قبل 3 ساعات على الأقل للدرجة السياحية."}]}, ensure_ascii=False)
    draft = "وفقًا للبند **6.1** من شروط النقل، يجب أن تكون موجودًا في مكتب تسجيل الوصول قبل **3 ساعات** من موعد إقلاع رحلتك."
    assert policy_problems(draft, [("search_policy", result)]) == []
    assert [p for p, _ in policy_problems("وفقًا للبند **6.2** من شروط النقل، قبل 3 ساعات.", [("search_policy", result)])] == ["unretrieved_clause"]
    assert [p for p, _ in policy_problems("وفقًا لشروط النقل، قبل 3 ساعات.", [("search_policy", result)])] == ["uncited_policy_answer"]


# ---------------------------------------------------------------- decision #37: no citation, no answer; a citation must be supported
def test_a_rules_question_answered_without_the_document_is_sent_back():
    from falcon.services.runtime.review import consulted_this_turn, policy_question_problem
    fare = ToolMessage(content=json.dumps({"fare_families": [{"code": "SAVER", "cancel_or_change_deadline_hours_before_departure": 24}]}), tool_call_id="1", name="get_fare_rules")
    msgs = [HumanMessage(content="Is there a deadline for asking for a refund on a ticket I never used?"), fare]
    assert not consulted_this_turn(msgs)
    problem = policy_question_problem(msgs, {"policy_question": True})
    assert problem and problem[0] == "policy_unconsulted" and "search_policy" in problem[1]
    assert policy_question_problem(msgs, {"policy_question": False}) is None
    searched = msgs + [ToolMessage(content=json.dumps({"results": [{"clause_id": "10.4.1", "text_en": "We will not give a refund if application is made more than 2 years from issue."}]}), tool_call_id="2", name="search_policy")]
    assert consulted_this_turn(searched) and policy_question_problem(searched, {"policy_question": True}) is None
    r = review_draft(AIMessage(content="Under your Saver fare the deadline is 24 hours before departure."), msgs, lang="en", guardrail={"policy_question": True})
    assert "policy_unconsulted" in r.problems
    ok = review_draft(AIMessage(content="Under clause 10.4.1, a refund must be applied for within 2 years of the ticket's issue."), searched, lang="en", guardrail={"policy_question": True})
    assert ok.ok
    # a policy question answered after a search but with no citation and no "not covered" is sent back even without the word "Conditions"
    uncited = review_draft(AIMessage(content="You have two years to ask for it."), searched, lang="en", guardrail={"policy_question": True})
    assert "uncited_policy_answer" in uncited.problems


def test_citation_support_uses_the_clause_text(monkeypatch):
    import asyncio as _asyncio
    from falcon.services.runtime import review as rv
    clause = json.dumps({"results": [{"clause_id": "16.2.1", "text_en": "All claims for compensation for Damage to Baggage must be accompanied by an itemised list identifying each affected item by description, manufacturer and age, together with proof of purchase."}]})
    msgs = [HumanMessage(content="what do I need to send for a damaged bag?"), ToolMessage(content=clause, tool_call_id="1", name="search_policy")]
    assert rv.clause_texts([("search_policy", clause)])["16.2.1"].startswith("All claims")
    assert "photos" in rv.citing_sentences("Under clause 16.2.1 you must send photos of the bag and the tag number. Please file within 7 days.", "16.2.1")

    class Stub:
        def __init__(self, reply):
            self.reply = reply
        async def ainvoke(self, _msgs):
            return AIMessage(content=self.reply)
    monkeypatch.setattr(rv, "_reviewer", lambda: Stub(json.dumps({"items": [{"clause_id": "16.2.1", "supported": False, "reason": "the clause requires an itemised list and proof of purchase, not photos"}]})))
    bad = _asyncio.run(rv.citation_support(AIMessage(content="Under clause 16.2.1 you must send photos of the bag and the tag number."), msgs))
    assert bad.problems == ["citation_unsupported"] and "itemised list" in bad.nudge
    monkeypatch.setattr(rv, "_reviewer", lambda: Stub(json.dumps({"items": [{"clause_id": "16.2.1", "supported": True, "reason": "matches"}]})))
    good = _asyncio.run(rv.citation_support(AIMessage(content="Under clause 16.2.1 you must send an itemised list with proof of purchase."), msgs))
    assert good.ok and "16.2.1" in (good.reviewer_note or "")
    monkeypatch.setattr(rv, "_reviewer", lambda: Stub("not json at all"))
    assert _asyncio.run(rv.citation_support(AIMessage(content="Under clause 16.2.1 you must send an itemised list."), msgs)).ok  # the check failing is a pass


def test_a_rules_engine_result_naming_its_clauses_is_a_source_for_them():
    from falcon.services.runtime.review import retrieved_ids
    case = json.dumps({"request_id": 26, "status": "pending", "kind": "bereavement", "clauses": ["3.3.2", "10.2.1"], "if_approved": {"refund": "2425.00"}})
    assert retrieved_ids([("request_exception", case)]) == {"3.3.2", "10.2.1"}
    draft = AIMessage(content="Under clause 3.3.2 of the Conditions, the fare paid is refunded on a death certificate; case #26 is logged, 2425.00 AED if approved.")
    assert review_draft(draft, [HumanMessage(content="my father died"), ToolMessage(content=case, tool_call_id="1", name="request_exception")], lang="en").ok
    other = AIMessage(content="Under clause 8.1 of the Conditions your allowance is on the ticket.")
    assert "policy_without_source" in review_draft(other, [HumanMessage(content="bags?"), ToolMessage(content=case, tool_call_id="1", name="request_exception")], lang="en").problems


# ---------------------------------------------------------------- decision #41: a bare "yes" after a quote raises the card without the model
def test_a_bare_yes_after_an_open_quote_raises_the_card_without_the_model():
    from falcon.services.runtime.graph import confirmation_shortcut, is_bare_confirmation, open_quote
    for text in ["نعم", "نعم، تابع الإلغاء", "yes please", "ok go ahead", "موافق", "Confirm."]:
        assert is_bare_confirmation(text), text
    for text in ["نعم ولكن هل يمكن التغيير إلى رحلة أخرى؟", "yes, and also email me the ticket", "what about Flex?"]:
        assert not is_bare_confirmation(text), text
    quote = ToolMessage(content=QUOTE, tool_call_id="1", name="quote_cancellation")
    msgs = [HumanMessage(content="ألغي حجزي"), AIMessage(content="", tool_calls=[{"name": "quote_cancellation", "args": {"pnr": "K3P7QZ"}, "id": "1"}]), quote,
            AIMessage(content="الرسوم 100 درهم، هل أتابع؟"), HumanMessage(content="نعم")]
    assert open_quote(msgs)["quote_id"] == "Q-1"
    tc = confirmation_shortcut({"messages": msgs})
    assert tc["name"] == "cancel_booking" and tc["args"] == {"pnr": "K3P7QZ", "quote_id": "Q-1"}
    done = msgs + [AIMessage(content="", tool_calls=[tc]), ToolMessage(content=CANCELLED.replace("1291.00", "1291.00") + ' Q-1', tool_call_id=tc["id"], name="cancel_booking"), AIMessage(content="تم."), HumanMessage(content="نعم")]
    assert confirmation_shortcut({"messages": done}) is None  # the quote was consumed: a later "yes" is for the model
    # the graph raises the card: the run stops at the interrupt with the write tool call as the last message
    model = ScriptedModel(messages=iter([]))
    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "t-confirm", "tools": [], "llm": model, "default_lang": "ar"}}
    asyncio.run(graph.ainvoke({"messages": msgs, "role": "customer", "user_name": "Fatima", "preferred_lang": "ar"}, config))
    snap = asyncio.run(graph.aget_state(config))
    assert snap.values["messages"][-1].tool_calls[0]["name"] == "cancel_booking" and snap.tasks and snap.tasks[0].interrupts


# ---------------------------------------------------------------- decision #43: two unresolved replies in a row hand over to a human
def test_two_held_replies_in_a_row_log_a_handover_case_and_say_so():
    from langchain_core.tools import tool as make_tool

    from falcon.services.runtime.graph import ESCALATED

    calls: list[dict] = []

    @make_tool
    def handover_to_human(summary: str, reason: str = "customer_request") -> str:
        """Log a case for a human agent."""
        calls.append({"summary": summary, "reason": reason})
        return json.dumps({"request_id": 77, "status": "pending"})

    bad = ["المبلغ المسترد 900.00 درهم."] * (settings.review_max_rounds + 1)
    model = ScriptedModel(messages=iter([AIMessage(content=d) for d in bad * 2 + ["يمكنني تسعير الإلغاء لك، هل أتحقق؟"]]))
    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "t-escalate", "tools": [handover_to_human], "llm": model, "default_lang": "ar"}}
    base = {"role": "customer", "user_name": "Fatima", "preferred_lang": "ar"}
    asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="كم سأسترد إذا ألغيت؟")], **base}, config))
    first = asyncio.run(graph.aget_state(config)).values
    assert first["verification"]["status"] == "held" and first["unresolved"] == 1 and calls == []  # one miss is not a reason to hand over
    asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="كم المبلغ؟")]}, config))
    second = asyncio.run(graph.aget_state(config)).values
    assert calls and calls[0]["reason"] == "unresolved" and "كم المبلغ؟" in calls[0]["summary"]
    assert second["messages"][-1].content == ESCALATED["ar"].format(case=77)
    assert second["verification"]["status"] == "escalated" and second["verification"]["case"] == 77 and second["unresolved"] == 0
    assert second["turn_log"][-1]["tool"] == "handover_to_human" and second["turn_log"][-1]["escalation"] is True
    asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="طيب، ما خياراتي؟")]}, config))
    third = asyncio.run(graph.aget_state(config)).values
    assert third["verification"]["status"] == "verified" and third["unresolved"] == 0 and len(calls) == 1


def test_a_held_reply_after_a_good_one_does_not_hand_over():
    from langchain_core.tools import tool as make_tool

    calls: list[dict] = []

    @make_tool
    def handover_to_human(summary: str, reason: str = "customer_request") -> str:
        """Log a case for a human agent."""
        calls.append({"summary": summary, "reason": reason})
        return json.dumps({"request_id": 1})

    bad = ["You would get 900.00 AED back."] * (settings.review_max_rounds + 1)
    model = ScriptedModel(messages=iter([AIMessage(content=d) for d in bad + ["I can quote the exact refund: shall I check?"] + bad]))
    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "t-no-escalate", "tools": [handover_to_human], "llm": model, "default_lang": "en"}}
    base = {"role": "customer", "user_name": "Test", "preferred_lang": "en"}
    for text in ["what do I get back?", "and if I cancel today?", "how much exactly?"]:
        asyncio.run(graph.ainvoke({"messages": [HumanMessage(content=text)], **base}, config))
    values = asyncio.run(graph.aget_state(config)).values
    assert values["verification"]["status"] == "held" and values["unresolved"] == 1 and calls == []  # the streak restarted after the good reply


# ---------------------------------------------------------------- 19 Oct full run: the four fixes behind the failed and held tasks
def test_an_announced_confirm_button_after_a_quote_raises_the_card():
    from falcon.services.runtime.graph import card_shortcut
    quote = ToolMessage(content=QUOTE, tool_call_id="1", name="quote_cancellation")
    state = {"messages": [HumanMessage(content="ألغي حجزي"), AIMessage(content="", tool_calls=[{"name": "quote_cancellation", "args": {"pnr": "K3P7QZ"}, "id": "1"}]), quote]}
    draft = AIMessage(content="سأقوم بتطبيق التغيير الآن. اضغطي على زر التأكيد أدناه للمتابعة.")
    out = card_shortcut(state, draft)
    assert out is not None and out.tool_calls[0]["name"] == "cancel_booking" and out.tool_calls[0]["args"]["quote_id"] == "Q-1"
    assert card_shortcut({"messages": state["messages"][:1]}, draft) is None  # no quote on the table: nothing to raise
    assert card_shortcut(state, AIMessage(content="الرسوم 100 درهم، هل أتابع؟")) is None  # a question, not an announced button


def test_an_invented_case_number_is_a_false_claim_and_a_logged_case_is_evidence():
    v = f.check_reply("قمتي بإلغاء الحجز بنجاح. رقم الحالة هو EX12345.", [("get_booking", '{"pnr": "R8E2GN", "status": "confirmed"}')], [])
    assert v.false_claim
    v = f.check_reply("I've logged your question as Case #7 for a human agent.", [("handover_to_human", '{"request_id": 7, "status": "pending"}')], [])
    assert not v.false_claim
    v = f.check_reply("I've logged your question as Case #8 for a human agent.", [("handover_to_human", '{"request_id": 7, "status": "pending"}')], [])
    assert v.false_claim  # the number is not the one the system issued
    v = f.check_reply("I couldn't find a booking with reference M8T2LD on your account.", [("get_booking", "Error: No booking with that reference was found on your account.")], ["cancel my booking M8T2LD"])
    assert not v.false_claim  # a booking reference the customer gave is not a case number


def test_escalation_reuses_a_case_the_model_already_logged_this_turn():
    from langchain_core.tools import tool as make_tool

    from falcon.services.runtime.graph import ESCALATED, escalate

    calls: list[dict] = []

    @make_tool
    def handover_to_human(summary: str, reason: str = "customer_request") -> str:
        """Log a case for a human agent."""
        calls.append({"reason": reason})
        return json.dumps({"request_id": 99})

    held = AIMessage(content=HELD["en"], id="m1")
    state = {"messages": [HumanMessage(content="log a case"), held], "lang": "en", "unresolved": 2, "verification": {"status": "held", "rounds": []},
             "turn_log": [{"tool": "handover_to_human", "ok": True, "result_json": {"request_id": 41}}]}
    out = asyncio.run(escalate(state, {"configurable": {"tools": [handover_to_human]}}))
    assert calls == [] and out["messages"][0].content == ESCALATED["en"].format(case=41) and out["verification"]["case"] == 41


# ---------------------------------------------------------------- decisions #46-#47: topic-scoped prompt and tools; precomputed guard
def test_the_topic_picks_the_prompt_section_and_the_tool_menu():
    from falcon.services.runtime.personas import COMMON_TOOLS, TOPIC_TOOLS, tools_for_topic
    from falcon.services.runtime.prompts import ALL_SECTIONS, TOPIC_SECTIONS, system_prompt

    full = system_prompt("customer", lang="en", name="Test")
    for section in ALL_SECTIONS:
        assert section.strip() in full  # no topic: the whole prompt, as before the split
    cancel = system_prompt("customer", lang="ar", name="Fatima", topic="cancel_refund")
    assert "Bereavement and force majeure" in cancel and "New bookings: search_flights" not in cancel
    assert "Current date and time" in cancel and cancel.index("How to behave") < cancel.index("Current date and time")  # dynamic lines last
    assert "ROLE: customer" in cancel and "REPLY_LANGUAGE: ar" in cancel  # the scripted model still finds its markers
    assert set(TOPIC_SECTIONS) == {"bookings", "cancel_refund", "change", "baggage", "book_new", "disruption", "policy", "exception"} == set(TOPIC_TOOLS)

    class T:  # a bound tool is anything with a name
        def __init__(self, name):
            self.name = name
    every = [T(n) for n in ("search_policy", "quote_cancellation", "cancel_booking", "create_booking", "quote_booking", "add_extra_bag", "handover_to_human", "list_my_bookings")]
    names = lambda ts: {t.name for t in ts}  # noqa: E731
    assert names(tools_for_topic("cancel_refund", every)) == {"search_policy", "quote_cancellation", "cancel_booking", "handover_to_human", "list_my_bookings"}
    assert names(tools_for_topic("baggage", every)) == {"search_policy", "add_extra_bag", "handover_to_human", "list_my_bookings"}
    assert names(tools_for_topic(None, every)) == names(every) == names(tools_for_topic("general", every))
    assert COMMON_TOOLS <= names(tools_for_topic("book_new", every)) | {"check_travel_eligibility", "get_request_status", "get_booking"}


def test_the_agent_narrows_only_on_a_classifier_topic(monkeypatch):
    from falcon.services.runtime.graph import turn_topic

    monkeypatch.setattr(settings, "topic_routing", True)
    monkeypatch.setattr(settings, "llm_provider", "openrouter")
    assert turn_topic({"guardrail": {"source": "llm", "topic": "baggage"}}) == "baggage"
    assert turn_topic({"guardrail": {"source": "llm", "topic": "general"}}) is None
    assert turn_topic({"guardrail": {"source": "rules", "topic": "baggage"}}) is None  # a pattern decided: no classifier topic
    assert turn_topic({"guardrail": {"source": "rules_fallback"}}) is None
    assert turn_topic({}) is None
    monkeypatch.setattr(settings, "llm_provider", "fake")
    assert turn_topic({"guardrail": {"source": "llm", "topic": "baggage"}}) is None  # the scripted model keeps every tool
    monkeypatch.setattr(settings, "llm_provider", "openrouter")
    monkeypatch.setattr(settings, "topic_routing", False)
    assert turn_topic({"guardrail": {"source": "llm", "topic": "baggage"}}) is None


def test_the_guard_node_reuses_a_precomputed_verdict_for_the_same_text():
    from falcon.services.runtime.guardrails import Guardrail

    pre = Guardrail(label="task", action="allow", source="llm", topic="baggage")
    model = ScriptedModel(messages=iter([AIMessage(content="Your Saver fare includes 30 kg; shall I quote 10 kg more?")]))
    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "t-pre", "tools": [], "llm": model, "default_lang": "en",
                               "guard_precomputed": {"text": "add 10 kg to K3P7QZ", "clean": "add 10 kg to K3P7QZ", "guard": pre}}}
    asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="add 10 kg to K3P7QZ")], "role": "customer", "user_name": "Test", "preferred_lang": "en"}, config))
    values = asyncio.run(graph.aget_state(config)).values
    assert values["guardrail"]["source"] == "llm" and values["guardrail"]["topic"] == "baggage"  # the node did not classify again


def test_citation_support_judges_a_cited_parent_against_its_retrieved_children(monkeypatch):
    # FAILURE_DIARY #67: a reply citing "7.4" when only 7.4.1 was retrieved raised KeyError, and the turn became "model unavailable"
    import asyncio as _asyncio
    from falcon.services.runtime import review as rv
    clause = json.dumps({"results": [{"clause_id": "7.4.1", "text_en": "Expectant mothers may travel up to the end of the 36th week."}]})
    msgs = [HumanMessage(content="I'm pregnant, can I fly?"), ToolMessage(content=clause, tool_call_id="1", name="search_policy")]
    seen = {}

    class Stub:
        async def ainvoke(self, msgs_in):
            seen["items"] = json.loads(msgs_in[1].content)["items"]
            return AIMessage(content=json.dumps({"items": [{"clause_id": "7.4", "supported": True, "reason": "matches"}]}))
    monkeypatch.setattr(rv, "_reviewer", lambda: Stub())
    out = _asyncio.run(rv.citation_support(AIMessage(content="Under clause 7.4 you may fly until the end of week 36."), msgs))
    assert out.ok and seen["items"][0]["clause_id"] == "7.4" and "36th week" in seen["items"][0]["clause_text"]


def test_a_bare_yes_to_two_offers_raises_no_card():
    # FAILURE_DIARY #65: the agent quoted credit (1291, fee 100) and card (1091, fee 300) and asked which; a bare "نعم" raised the
    # card for the last quote, the dearer one. Two different offers on the table: the model must ask, no shortcut may choose.
    from falcon.services.runtime.graph import card_shortcut, confirmation_shortcut, open_quote
    credit = ToolMessage(content=json.dumps({"kind": "cancel", "pnr": "K3P7QZ", "allowed": True, "fee": "100.00", "refund": "1291.00", "refund_method": "credit", "currency": "AED", "quote_id": "Q-C"}), tool_call_id="1", name="quote_cancellation")
    card = ToolMessage(content=json.dumps({"kind": "cancel", "pnr": "K3P7QZ", "allowed": True, "fee": "300.00", "refund": "1091.00", "refund_method": "original", "currency": "AED", "quote_id": "Q-O"}), tool_call_id="2", name="quote_cancellation")
    calls = AIMessage(content="", tool_calls=[{"name": "quote_cancellation", "args": {"pnr": "K3P7QZ", "refund_method": "credit"}, "id": "1"},
                                              {"name": "quote_cancellation", "args": {"pnr": "K3P7QZ", "refund_method": "original"}, "id": "2"}])
    asked = AIMessage(content="رصيد سفر: 1291.00 AED برسوم 100.00، أو إلى البطاقة: 1091.00 AED برسوم 300.00. أي طريقة تفضلين؟")
    msgs = [HumanMessage(content="أريد إلغاء حجزي K3P7QZ"), calls, credit, card, asked, HumanMessage(content="نعم")]
    assert open_quote(msgs) is None
    assert confirmation_shortcut({"messages": msgs}) is None
    # the "press confirm" shortcut does not choose either
    assert card_shortcut({"messages": msgs[:4]}, AIMessage(content="اضغطي على زر التأكيد")) is None
    # the same offer quoted twice (a refreshed quote) is still one offer: the latest one is confirmed
    again = ToolMessage(content=json.dumps({"kind": "cancel", "pnr": "K3P7QZ", "allowed": True, "fee": "100.00", "refund": "1291.00", "refund_method": "credit", "currency": "AED", "quote_id": "Q-C2"}), tool_call_id="3", name="quote_cancellation")
    one = [HumanMessage(content="أريد إلغاء حجزي K3P7QZ"), calls, credit, again, AIMessage(content="هل تؤكدين؟"), HumanMessage(content="نعم")]
    assert confirmation_shortcut({"messages": one})["args"]["quote_id"] == "Q-C2"
