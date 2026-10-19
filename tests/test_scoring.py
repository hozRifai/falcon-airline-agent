from falcon.eval.scoring import _normalise


def test_amounts_match_whatever_the_formatting():
    assert "1291" in _normalise("Travel credit refund: 1,291 AED")
    assert "1291" in _normalise("المبلغ المسترد: ١٬٢٩١ درهم")
    assert "1291" in _normalise("1 291 AED")
    assert "2275.00" in _normalise("2,275.00")


def test_clause_ids_and_lists_are_left_alone():
    assert _normalise("see 9.1.2 and 10.3") == "see 9.1.2 and 10.3"
    assert _normalise("flights 1, 2, 3") == "flights 1, 2, 3"
    assert _normalise("12,3456") == "12,3456"


def test_curly_quotes_read_as_straight():
    assert "couldn't find" in _normalise("I’m sorry, but I couldn’t find a booking")


def test_cited_clause_ids_survive_an_arabic_conjunction_and_arabic_digits():
    from falcon.eval.scoring import CLAUSE_ID_RE, _ARABIC_DIGITS
    assert CLAUSE_ID_RE.findall("بناءً على شروط النقل (البند 7.4.1 و7.4.3)، لا يُسمح بالسفر بعد الأسبوع 36.") == ["7.4.1", "7.4.3"]
    assert CLAUSE_ID_RE.findall("See clauses 7.4.1 and 7.4.3; version 10.2.1.5 is not a clause, 1,291 AED is not one either.") == ["7.4.1", "7.4.3", "10.2.1.5"]
    assert CLAUSE_ID_RE.findall("البند ٧.٤.٣".translate(_ARABIC_DIGITS)) == ["7.4.3"]


def test_grounding_is_cumulative_over_the_conversation():
    from falcon.eval.scoring import score
    from falcon.eval.tasks import Task
    task = Task(id="t", lang="en", persona="u-noura", goal="g", grounding=True)
    retrieved = {"tool": "check_travel_eligibility", "args": {}, "ok": True, "clause_ids": ["7.7.3", "7.7.2"], "result_json": {"clauses": [{"clause_id": "7.7.3"}]}}
    transcript = [{"user": "my 8 year old flies alone", "assistant": "Clause 7.7.3 applies.", "what_i_checked": [retrieved], "usage": {}, "latency_ms": 0},
                  {"user": "no escort then", "assistant": "Then the conditions of clause 7.7.3 must be met in full.", "what_i_checked": [], "usage": {}, "latency_ms": 0}]
    assert score(task, transcript).grounding_ok
    transcript[1]["assistant"] = "Clause 7.7.3 and clause 8.1 apply."
    r = score(task, transcript)
    assert not r.grounding_ok and "8.1" in r.reasons[0]
