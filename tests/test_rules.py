"""Rules engine tests: one test per persona rule, no database, frozen clock."""

from datetime import datetime, timedelta
from decimal import Decimal

from falcon.core.clock import GULF
from falcon.core.models import Booking, FareRules, FlightStatus, Passenger, Segment
from falcon.core.rules import bag_quote, cancellation_quote, change_quote, waiver_quote

NOW = datetime(2026, 10, 20, 9, 0, tzinfo=GULF)

SPECIAL = FareRules(code="SPECIAL", name_en="Special", name_ar="الخاصة", bag_kg=20, free_changes=0, max_changes=1, change_fee=Decimal("200"),
                  refund_credit_allowed=True, refund_credit_fee=Decimal("150"), refund_cash_allowed=False, refund_cash_fee=Decimal("0"),
                  deadline_hours=24, price_multiplier=Decimal("1.0"))
SAVER = FareRules(code="SAVER", name_en="Saver", name_ar="الموفرة", bag_kg=30, free_changes=1, max_changes=2, change_fee=Decimal("200"),
                  refund_credit_allowed=True, refund_credit_fee=Decimal("100"), refund_cash_allowed=True, refund_cash_fee=Decimal("300"),
                  deadline_hours=24, price_multiplier=Decimal("1.3"))
FLEX = FareRules(code="FLEX", name_en="Flex", name_ar="المرنة", bag_kg=35, free_changes=2, max_changes=3, change_fee=Decimal("100"),
                     refund_credit_allowed=True, refund_credit_fee=Decimal("0"), refund_cash_allowed=True, refund_cash_fee=Decimal("200"),
                     deadline_hours=8, price_multiplier=Decimal("1.7"))


def booking(pnr, fare, hours_ahead, paid, pax=1, status="confirmed", changes_used=0, fare_per_pax=None, segments=1):
    dep = NOW + timedelta(hours=hours_ahead)
    segs = [Segment(seq=i + 1, flight_instance_id=f"FA10{i}-X", flight_number=f"FA10{i}", origin="DXB", destination="CAI",
                    dep_at=dep + timedelta(hours=6 * i), arr_at=dep + timedelta(hours=6 * i + 4), fare_per_pax=Decimal(fare_per_pax or paid) / pax)
            for i in range(segments)]
    return Booking(pnr=pnr, customer_id="C-1", status=status, fare_family=fare.code, total_paid=Decimal(paid), changes_used=changes_used,
                   created_at=NOW - timedelta(days=10), passengers=[Passenger(id=i + 1, first_name="P", last_name=str(i)) for i in range(pax)], segments=segs)


def test_value_35h_can_cancel_credit_and_cash():
    b = booking("K3P7QZ", SAVER, 35, "728")
    q = cancellation_quote(b, SAVER, NOW, {}, "credit")
    assert q.allowed and q.fee == Decimal("100.00") and q.refund == Decimal("628.00")
    q = cancellation_quote(b, SAVER, NOW, {}, "cash")
    assert q.allowed and q.fee == Decimal("300.00") and q.refund == Decimal("428.00")


def test_value_17h_is_past_deadline_and_waiver_possible():
    b = booking("M8T2LD", SAVER, 17, "448")
    q = cancellation_quote(b, SAVER, NOW, {})
    assert not q.allowed and q.reason_code == "PAST_DEADLINE" and q.waiver_possible
    w = waiver_quote(b, SAVER, NOW, {}, "credit", waive_fee=False)
    assert w.allowed and w.refund == Decimal("348.00") and w.details["exception"]["ignore_deadline"] is True
    w2 = waiver_quote(b, SAVER, NOW, {}, "credit", waive_fee=True)
    assert w2.refund == Decimal("448.00")


def test_basic_credit_only():
    b = booking("B4N6WX", SPECIAL, 95, "480")
    assert cancellation_quote(b, SPECIAL, NOW, {}, "cash").reason_code == "CASH_NOT_ALLOWED"
    q = cancellation_quote(b, SPECIAL, NOW, {}, "credit")
    assert q.allowed and q.fee == Decimal("150.00") and q.refund == Decimal("330.00")


def test_flex_11h_allowed_free_credit():
    b = booking("R2Y9HC", FLEX, 11, "484")
    q = cancellation_quote(b, FLEX, NOW, {}, "credit")
    assert q.allowed and q.fee == Decimal("0.00") and q.refund == Decimal("484.00")
    assert cancellation_quote(b, FLEX, NOW, {}, "cash").fee == Decimal("200.00")


def test_airline_cancelled_full_refund_any_fare():
    b = booking("P5D3VJ", SPECIAL, 22, "495")
    statuses = {"FA100-X": FlightStatus(flight_instance_id="FA100-X", status="cancelled")}
    q = cancellation_quote(b, SPECIAL, NOW, statuses, "cash")  # cash normally not allowed on Basic
    assert q.allowed and q.reason_code == "AIRLINE_CANCELLED" and q.refund == Decimal("495.00") and q.fee == 0


def test_long_delay_full_refund_but_three_hours_is_not_enough():
    b = booking("R8E2GN", FLEX, 3, "442")
    three = {"FA100-X": FlightStatus(flight_instance_id="FA100-X", status="delayed", delay_minutes=180)}
    assert cancellation_quote(b, FLEX, NOW, three).reason_code == "PAST_DEADLINE"
    five = {"FA100-X": FlightStatus(flight_instance_id="FA100-X", status="delayed", delay_minutes=300)}
    assert cancellation_quote(b, FLEX, NOW, five).reason_code == "LONG_DELAY"


def test_family_of_four_fees_per_passenger():
    b = booking("N6F8ZR", SAVER, 190, "2208", pax=4)
    q = cancellation_quote(b, SAVER, NOW, {}, "credit")
    assert q.fee == Decimal("400.00") and q.refund == Decimal("1808.00") and q.details["passengers"] == 4


def test_already_cancelled_and_no_show():
    assert cancellation_quote(booking("A9C2PM", SAVER, -30, "377", status="cancelled"), SAVER, NOW, {}).reason_code == "ALREADY_CANCELLED"
    assert cancellation_quote(booking("T1M4XV", SAVER, -25, "448"), SAVER, NOW, {}).reason_code == "FLOWN"


def test_change_fee_after_free_change_used_and_max_changes():
    b = booking("S3L5KQ", SAVER, 143, "403", changes_used=1)
    q = change_quote(b, SAVER, NOW, {}, 1, "FA150-Y", NOW + timedelta(days=8), Decimal("450"))
    assert q.allowed and q.fee == Decimal("200.00") and q.details["fare_difference"] == "47.00" and q.amount_due == Decimal("247.00")
    b2 = booking("S3L5KQ", SAVER, 143, "403", changes_used=2)
    assert change_quote(b2, SAVER, NOW, {}, 1, "FA150-Y", NOW + timedelta(days=8), Decimal("450")).reason_code == "MAX_CHANGES"


def test_change_free_first_time_and_no_refund_of_negative_difference():
    b = booking("K3P7QZ", SAVER, 35, "728")
    q = change_quote(b, SAVER, NOW, {}, 1, "FA193-Y", NOW + timedelta(days=3), Decimal("600"))
    assert q.allowed and q.fee == 0 and q.amount_due == 0 and q.details["free_change_used"] is True


def test_change_after_airline_disruption_is_free_on_any_fare_and_does_not_count():
    # Mariam: Special fare (no free change, 200 fee) on a flight the airline cancelled; rebooking on the same route is free
    b = booking("M2A7HS", SPECIAL, 28, "965")
    cancelled = {b.segments[0].flight_instance_id: FlightStatus(flight_instance_id=b.segments[0].flight_instance_id, status="cancelled", delay_minutes=0)}
    q = change_quote(b, SPECIAL, NOW, cancelled, 1, "FA929-X", NOW + timedelta(hours=34), Decimal("1400"))
    assert q.allowed and q.reason_code == "AIRLINE_CANCELLED" and q.fee == 0 and q.amount_due == 0
    assert q.details["fare_difference"] == "0.00" and q.details["involuntary"] == "cancelled" and q.details["free_change_used"] is False
    # a 5-hour delay counts too, and it beats the deadline and the change cap; a 3-hour delay does not
    late = booking("R8E2GN", FLEX, 3, "820", changes_used=3)
    long_delay = {late.segments[0].flight_instance_id: FlightStatus(flight_instance_id=late.segments[0].flight_instance_id, status="delayed", delay_minutes=300)}
    assert change_quote(late, FLEX, NOW, long_delay, 1, "FA862-X", NOW + timedelta(hours=9), Decimal("900")).reason_code == "LONG_DELAY"
    short_delay = {late.segments[0].flight_instance_id: FlightStatus(flight_instance_id=late.segments[0].flight_instance_id, status="delayed", delay_minutes=180)}
    assert change_quote(late, FLEX, NOW, short_delay, 1, "FA862-X", NOW + timedelta(hours=9), Decimal("900")).reason_code == "PAST_DEADLINE"
    # the same flight and a departed flight are still refused, disruption or not
    assert change_quote(b, SPECIAL, NOW, cancelled, 1, b.segments[0].flight_instance_id, NOW + timedelta(hours=28), Decimal("965")).reason_code == "SAME_FLIGHT"
    assert change_quote(b, SPECIAL, NOW, cancelled, 1, "FA927-X", NOW - timedelta(hours=2), Decimal("965")).reason_code == "NEW_FLIGHT_DEPARTED"


def test_change_segment_of_connection():
    b = booking("H7Q4TK", SAVER, 120, "1177", fare_per_pax="1177", segments=2)
    q = change_quote(b, SAVER, NOW, {}, 2, "FA192-Z", NOW + timedelta(days=6), Decimal("700"))
    assert q.allowed and q.details["segment_seq"] == 2
    assert change_quote(b, SAVER, NOW, {}, 3, "FA192-Z", NOW + timedelta(days=6), Decimal("700")).reason_code == "NO_SUCH_SEGMENT"


def test_bag_pricing_and_cutoff():
    b = booking("B4N6WX", SPECIAL, 95, "480")
    q = bag_quote(b, SPECIAL, NOW, {}, 20, [1])
    assert q.allowed and q.amount_due == Decimal("90.00")
    assert bag_quote(b, SPECIAL, NOW, {}, 25, [1]).reason_code == "INVALID_WEIGHT"
    assert bag_quote(b, SPECIAL, NOW, {}, 20, [9]).reason_code == "NO_SUCH_PASSENGER"
    assert bag_quote(booking("X", SPECIAL, 1, "480"), SPECIAL, NOW, {}, 20, [1]).reason_code == "TOO_LATE"


def test_an_exception_quote_says_what_it_waives():
    b = booking("M8T2LD", SAVER, 17, "448")
    q = cancellation_quote(b, SAVER, NOW, {}, "credit", ignore_deadline=True)
    assert q.allowed and q.details["exception"] == {"ignore_deadline": True, "waive_fee": False}
    assert "exception" not in cancellation_quote(booking("K3P7QZ", SAVER, 35, "1391"), SAVER, NOW, {}).details


# ---------------------------------------------------------------- new bookings
from falcon.core.rules import booking_quote  # noqa: E402


def instance(hours_ahead=6 * 24, seats_sold=100, base="900"):
    dep = NOW + timedelta(hours=hours_ahead)
    return {"id": "FA951-2026-10-26", "flight_number": "FA951", "origin": "DXB", "destination": "BEY", "dep_at": dep, "arr_at": dep + timedelta(hours=4),
            "seats_total": 354, "seats_sold": seats_sold, "base_price": Decimal(base)}


def test_booking_price_is_the_search_price_times_passengers():
    q = booking_quote(instance(), SPECIAL, NOW, None, 2, ["Layla Saeed"])
    assert q.allowed and q.kind == "book" and q.amount_due == Decimal("1800.00") and q.details["fare_per_pax"] == "900.00"
    assert q.details["passenger_names"] == ["Layla Saeed"] and q.details["seats_available"] == 254
    saver = booking_quote(instance(), SAVER, NOW, None, 1)
    assert saver.amount_due == Decimal("1170.00")  # 900 x 1.3, whole dirhams, as search_flights shows it


def test_booking_refused_when_cancelled_closed_or_full():
    cancelled = FlightStatus(flight_instance_id="FA951-2026-10-26", status="cancelled", delay_minutes=0, reason="ops")
    assert booking_quote(instance(), SPECIAL, NOW, cancelled, 1).reason_code == "FLIGHT_CANCELLED"
    assert booking_quote(instance(hours_ahead=1), SPECIAL, NOW, None, 1).reason_code == "BOOKING_CLOSED"
    assert booking_quote(instance(hours_ahead=-3), SPECIAL, NOW, None, 1).reason_code == "FLIGHT_DEPARTED"
    assert booking_quote(instance(seats_sold=353), SPECIAL, NOW, None, 2).reason_code == "NO_SEATS"
    assert booking_quote(instance(), SPECIAL, NOW, None, 10).reason_code == "BAD_PASSENGERS"


def test_a_refusal_on_a_short_delay_states_the_five_hour_rule():
    # Reem (18 Oct delay task): Flex to Muscat in 3 h, delayed 3 h: past the 8 h deadline, no refund right yet, and the customer must hear the threshold
    b = booking("R8E2GN", FLEX, 3, "965")
    three = {"FA100-X": FlightStatus(flight_instance_id="FA100-X", status="delayed", delay_minutes=180)}
    q = cancellation_quote(b, FLEX, NOW, three, "cash")
    assert not q.allowed and q.reason_code == "PAST_DEADLINE" and q.waiver_possible
    assert "180 minutes" in q.reason_en and "5 hours or more" in q.reason_en and "5 ساعات" in q.reason_ar and q.details["delay_minutes"] == 180
    plain = cancellation_quote(b, FLEX, NOW, {}, "cash")
    assert "delay" not in plain.reason_en and "delay_minutes" not in plain.details


# ---------------------------------------------------------------- fitness to travel: Article 7 as thresholds
import pytest  # noqa: E402

from falcon.core.rules import travel_eligibility  # noqa: E402


def test_pregnancy_weeks_follow_clauses_7_4_1_and_7_4_3():
    (f,) = travel_eligibility(pregnancy_weeks=38)  # Layla, 18 Oct: "38th month" was her 38th week
    assert f.status == "not_carried" and f.clause_ids == ["7.4.3"] and "36th week" in f.rule_en and f.figures["last_week"] == 36
    (f,) = travel_eligibility(pregnancy_weeks=30)
    assert f.status == "document_required" and f.clause_ids == ["7.4.1", "7.4.3"] and f.figures["letter_from_week"] == 28 and "midwife" in f.rule_en
    (f,) = travel_eligibility(pregnancy_weeks=36)
    assert f.status == "document_required"  # the 36th week itself is still carried, with the letter
    (f,) = travel_eligibility(pregnancy_weeks=20)
    assert f.status == "ok" and "7.4.1" in f.rule_ar and "7.4.3" in f.rule_ar
    (f,) = travel_eligibility(pregnancy_weeks=34, multiple_pregnancy=True)
    assert f.status == "not_carried" and f.figures["last_week"] == 32 and "32nd week" in f.rule_en
    ok, clearance = travel_eligibility(pregnancy_weeks=20, complicated_pregnancy=True)
    assert ok.status == "ok" and clearance.status == "clearance_required" and clearance.clause_ids == ["7.4.2"] and "MEDIF" in clearance.rule_en


def test_months_are_not_weeks():
    (f,) = travel_eligibility(pregnancy_months=38)
    assert f.status == "needs_weeks" and set(f.clause_ids) == {"7.4.1", "7.4.3"} and "not a possible duration" in f.rule_en and "38th week" in f.rule_en
    assert f.figures["stated_months"] == 38 and "approx_weeks" not in f.figures
    (f,) = travel_eligibility(pregnancy_months=8)
    assert f.status == "needs_weeks" and f.figures["approx_weeks"] == 35 and "Ask which week" in f.rule_en
    (f,) = travel_eligibility(pregnancy_weeks=38 * 2)  # a figure no pregnancy reaches is read as months, not weeks
    assert f.status == "needs_weeks"


def test_newborns_and_children_alone_follow_clauses_7_4_5_and_7_7():
    (f,) = travel_eligibility(infant_age_days=3)
    assert f.status == "not_carried" and f.clause_ids == ["7.4.5"] and f.figures["min_days"] == 7 and "7 days" in f.rule_en
    (f,) = travel_eligibility(infant_age_days=10)
    assert f.status == "ok"
    (f,) = travel_eligibility(child_age_years=8, travelling_alone=True)
    assert f.status == "conditions_apply" and f.clause_ids == ["7.7.3", "7.7.2"] and f.figures["notice_hours"] == 12 and "12 hours" in f.rule_en
    (f,) = travel_eligibility(child_age_years=4, travelling_alone=True)
    assert f.status == "escort_required" and f.clause_ids == ["7.7.1"] and f.figures["escort_min_age"] == 16
    (f,) = travel_eligibility(child_age_years=4, travelling_alone=False)
    assert f.status == "conditions_apply" and f.clause_ids == ["7.7.1"]
    (f,) = travel_eligibility(child_age_years=14, travelling_alone=True)
    assert f.status == "conditions_apply" and f.clause_ids == ["7.7.2"]
    (f,) = travel_eligibility(child_age_years=8, travelling_alone=False)
    assert f.status == "ok"
    (f,) = travel_eligibility(child_age_years=17, travelling_alone=True)
    assert f.status == "ok"
    assert travel_eligibility() == []
    with pytest.raises(ValueError):
        travel_eligibility(pregnancy_weeks=-1)
    with pytest.raises(ValueError):
        travel_eligibility(child_age_years="eight")


# ---------------------------------------------------------------- What the Conditions grant on bereavement and force majeure
from falcon.core.rules import compassionate_quote  # noqa: E402


def test_bereavement_refunds_the_fare_paid_and_cites_the_clauses():
    b = booking("B4N6WX", SPECIAL, 96, "2425")
    q = compassionate_quote(b, SPECIAL, NOW, {}, "bereavement", "cash")
    assert q.allowed and q.fee == Decimal("0") and q.refund == Decimal("2425.00") and q.refund_method == "cash" and q.reason_code == "BEREAVEMENT"
    assert q.details["clauses"] == ["3.3.2", "10.2.1"] and "death certificate" in q.details["evidence_required"] and "3.3.2" in q.reason_en
    fm = compassionate_quote(b, SPECIAL, NOW, {}, "force_majeure", "cash")
    assert fm.allowed and fm.refund_method == "credit" and fm.details["clauses"] == ["3.3.1"] and "3.3.1" in fm.reason_ar
    flown = compassionate_quote(booking("X", SPECIAL, -3, "500"), SPECIAL, NOW, {}, "bereavement")
    assert not flown.allowed and flown.reason_code == "FLOWN"


def test_cancelling_some_of_the_passengers_refunds_their_fares_and_keeps_the_booking():
    """Decision #60: Hozayfa's wife cannot fly at 37 weeks; her seats are cancelled, his stay."""
    b = booking("H2R4BY", SAVER, 95, "2860", pax=2, fare_per_pax="1430", segments=2)  # two segments at 715 each per passenger
    q = cancellation_quote(b, SAVER, NOW, {}, "credit", passenger_ids=[2])
    assert q.allowed and q.fee == Decimal("100.00") and q.refund == Decimal("1330.00")  # her two fares (715 + 715) minus one fee
    assert q.details["passenger_ids"] == [2] and q.details["remaining"] == 1 and q.details["passengers"] == 1
    whole = cancellation_quote(b, SAVER, NOW, {}, "credit", passenger_ids=[1, 2])  # naming everyone is a whole-booking cancellation
    assert "passenger_ids" not in whole.details and whole.refund == Decimal("2660.00")
    assert cancellation_quote(b, SAVER, NOW, {}, "credit", passenger_ids=[9]).reason_code == "NO_SUCH_PASSENGER"
    b.passengers[1].cancelled_at = NOW  # once she is cancelled, a quote for the booking is for him alone
    q2 = cancellation_quote(b, SAVER, NOW, {}, "credit")
    assert q2.details["passengers"] == 1 and q2.fee == Decimal("100.00")
