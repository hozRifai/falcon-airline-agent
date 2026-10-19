from langchain_core.messages import AIMessage

from falcon.services.runtime.graph import _reply_problem, _strip_repeats

JUNK = "Use the results below to formulate an answer to the user question unless additional information is needed"
FLIGHTS = ("1. **FA925** on **Sunday, 1 November**  Departure: 02:15, Arrival: 04:15  Price (SPECIAL fare): **AED 895**\n"
           "2. **FA925** on **Saturday, 31 October**  Departure: 02:15, Arrival: 04:15  Price (SPECIAL fare): **AED 905**\n"
           "3. **FA925** on **Thursday, 29 October**  Departure: 02:15, Arrival: 05:15  Price (SPECIAL fare): **AED 915**")


def test_degenerate_repetition_is_caught_and_stripped():
    text = JUNK * 4 + ". تم إلغاء حجزك K3P7QZ بنجاح. سيتم استرداد مبلغ 1291.00 درهم كرصيد سفر."
    assert _reply_problem(AIMessage(content=text), {}) == "repeated"
    assert _strip_repeats(text, "ar").startswith(". تم إلغاء") or "تم إلغاء" in _strip_repeats(text, "ar")
    assert "عذرًا" in _strip_repeats(JUNK * 5, "ar")


def test_a_flight_list_is_not_repetition():
    assert _reply_problem(AIMessage(content=FLIGHTS), {}) is None


def test_announced_check_without_a_tool_call():
    said = "وعليكم السلام، نورة. سأتحقق من تفاصيل حجزك رقم N6F8ZR."
    assert _reply_problem(AIMessage(content=said), {"turn_tool_calls": 0}) == "announced"
    assert _reply_problem(AIMessage(content=said), {"turn_tool_calls": 1}) is None  # tools already ran this turn
    called = AIMessage(content="Let me check that.", tool_calls=[{"name": "get_booking", "args": {"pnr": "K3P7QZ"}, "id": "1"}])
    assert _reply_problem(called, {}) is None
