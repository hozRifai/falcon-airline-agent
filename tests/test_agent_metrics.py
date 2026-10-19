"""Agent-level metrics (falcon.eval.metrics) on hand-built transcripts."""

from falcon.core.config import settings
from falcon.eval.metrics import amounts_in, compute, sequence_matches


def turn(user, assistant, *tools):
    return {"user": user, "assistant": assistant, "usage": {"input_tokens": 1000, "output_tokens": 100}, "latency_ms": 1200,
            "what_i_checked": [{"tool": name, "args": args, "ok": ok, "result_json": result} for name, args, ok, result in tools]}


def test_sequence_matching_with_alternatives_and_gaps():
    assert sequence_matches(["get_booking|list_my_bookings", "quote_cancellation", "cancel_booking"], ["list_my_bookings", "get_fare_rules", "quote_cancellation", "cancel_booking"])
    assert not sequence_matches(["quote_cancellation", "cancel_booking"], ["cancel_booking", "quote_cancellation"])
    assert sequence_matches([], ["anything"])


def test_amounts_are_read_in_both_languages_and_formats():
    assert amounts_in("The fee is **100.00 AED** and 1,291 AED comes back") == {"100.00", "1291.00"}
    assert amounts_in("سيتم استرداد ١٬٢٩١ درهم كرصيد") == {"1291.00"}
    assert amounts_in("within 24 hours, clause 10.2.1, flight FA923") == set()


def test_unsupported_amount_and_false_claim_are_caught():
    quote = {"allowed": True, "fee": "100.00", "refund": "1291.00", "quote_id": "Q-1"}
    good = [turn("cancel K3P7QZ as credit", "Fee 100.00 AED, refund 1,291 AED. Confirm?", ("quote_cancellation", {"pnr": "K3P7QZ"}, True, quote)),
            turn("yes", "Your booking K3P7QZ has been cancelled; 1291.00 AED will be refunded.", ("cancel_booking", {"pnr": "K3P7QZ", "quote_id": "Q-1"}, True, {"status": "cancelled", "refund_amount": "1291.00"}))]
    m = compute(good, ["quote_cancellation", "cancel_booking"])
    assert m.amount_faithful and not m.false_claim and m.sequence_ok and m.efficiency == 1.0 and m.contained
    invented = [turn("cancel K3P7QZ", "The fee is 250.00 AED.", ("quote_cancellation", {"pnr": "K3P7QZ"}, True, quote))]
    assert compute(invented).unsupported_amounts == ["250.00"]
    claimed = [turn("cancel it", "Your booking has been cancelled.", ("quote_cancellation", {"pnr": "K3P7QZ"}, True, quote))]
    assert compute(claimed).false_claim
    flight_fact = [turn("is my flight ok?", "Your flight FA927 has been cancelled by the airline; your booking is still confirmed.", ("get_booking", {"pnr": "P5D3VJ"}, True, {"flight_departed": False}))]
    assert not compute(flight_fact).false_claim
    promise_ar = [turn("ألغِ الحجز", "رسوم الإلغاء 100.00 درهم وسيتم استرداد 1291.00 درهم. هل تؤكد؟", ("quote_cancellation", {"pnr": "K3P7QZ"}, True, quote))]
    assert not compute(promise_ar).false_claim


def test_customer_quoted_amounts_are_not_hallucinations():
    t = [turn("I paid 2,425 AED, what do I get back?", "You paid 2,425 AED; the refund is 2275.00 AED.", ("quote_cancellation", {"pnr": "B4N6WX"}, True, {"refund": "2275.00"}))]
    assert compute(t).amount_faithful


def test_operations_metrics():
    t = [turn("help", "I will hand you to a colleague.", ("handover_to_human", {"summary": "x"}, True, {"ticket": 1}))]
    m = compute(t)
    assert m.handover and not m.contained and m.turn_latencies == [1200] and round(m.cost_usd, 6) == round(1000 / 1e6 * settings.llm_price_in_per_m + 100 / 1e6 * settings.llm_price_out_per_m, 6)
