"""The rules engine: deterministic money and deadlines.

One pure function per decision. No I/O, no clock reads, no model. The same functions are
used by (1) the quote tools the agent calls, (2) the write actions, which re-run the quote
before touching a booking, and (3) the evaluation harness, which computes the expected
answer for every task. One function, three uses: if it is wrong it is wrong everywhere,
and it is tested in `tests/test_rules.py` against every persona.

The LLM never sees these calculations, only their results, which it explains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal

from .models import Booking, FareRules, FlightStatus, Quote

LONG_DELAY_MINUTES = 300  # Falcon rule: an airline delay of 5 hours or more entitles a full refund
BAG_PRICES_AED = {10: Decimal("60"), 20: Decimal("90"), 30: Decimal("130")}  # per passenger, extra kg beyond the fare

TWO_PLACES = Decimal("0.01")


def money(value: Decimal | int | float | str) -> Decimal:
    return Decimal(str(value)).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


def hours_to_departure(booking: Booking, now: datetime) -> float:
    first = min(s.dep_at for s in booking.segments)
    return (first - now).total_seconds() / 3600.0


def deadline_at(booking: Booking, rules: FareRules) -> datetime:
    first = min(s.dep_at for s in booking.segments)
    return first - timedelta(hours=rules.deadline_hours)


def _segment_disruption(segment_instance: str, statuses: dict[str, FlightStatus]) -> str | None:
    """'cancelled' or 'long_delay' if the airline disrupted this segment, else None."""
    st = statuses.get(segment_instance)
    if st is None:
        return None
    if st.status == "cancelled":
        return "cancelled"
    if st.delay_minutes >= LONG_DELAY_MINUTES:
        return "long_delay"
    return None


def _short_delay_minutes(booking: Booking, statuses: dict[str, FlightStatus]) -> int:
    """The longest delay on the booking's segments when it is below the full-refund threshold; 0 when there is none."""
    delays = [statuses[s.flight_instance_id].delay_minutes for s in booking.segments if s.flight_instance_id in statuses]
    longest = max(delays, default=0)
    return longest if 0 < longest < LONG_DELAY_MINUTES else 0


def _airline_disruption(booking: Booking, statuses: dict[str, FlightStatus]) -> str | None:
    """Return 'cancelled' or 'long_delay' if the airline disrupted any segment, else None."""
    for seg in booking.segments:
        d = _segment_disruption(seg.flight_instance_id, statuses)
        if d:
            return d
    return None


def has_flown(booking: Booking, now: datetime, statuses: dict[str, FlightStatus]) -> bool:
    """True once any segment has departed (or its departure time has passed without a cancellation)."""
    for seg in booking.segments:
        st = statuses.get(seg.flight_instance_id)
        if st is not None and st.status in ("departed", "landed"):
            return True
        if seg.dep_at <= now and not (st and st.status == "cancelled"):
            return True
    return False


def travelling(booking: Booking) -> list:
    """The passengers still on the booking who pay a fare: not infants, not cancelled."""
    return [p for p in booking.passengers if p.pax_type != "infant" and p.cancelled_at is None]


def _pax_count(booking: Booking) -> int:
    return max(1, len(travelling(booking)))


def cancellation_quote(booking: Booking, rules: FareRules, now: datetime, statuses: dict[str, FlightStatus], refund_method: str = "credit",
                       *, ignore_deadline: bool = False, waive_fee: bool = False, passenger_ids: list[int] | None = None) -> Quote:
    """What a cancellation costs and refunds now. The two flags are what an approved exception case would grant (`waiver_quote`);
    nothing in the agent sets them for a real cancellation. `passenger_ids` cancels only those passengers (decision #60): the
    refund is what they paid in fares, the fee is per passenger, the booking stays for the others."""
    active = travelling(booking)
    chosen = [p for p in active if p.id in set(passenger_ids)] if passenger_ids else active
    partial = bool(passenger_ids) and 0 < len(chosen) < len(active)
    pax = max(1, len(chosen))
    paid = money(sum((s.fare_per_pax for s in booking.segments), Decimal(0)) * len(chosen)) if partial else money(booking.total_paid)
    who = {"passenger_ids": [p.id for p in chosen], "remaining": len(active) - len(chosen)} if partial else {}
    hrs = hours_to_departure(booking, now)
    deadline = deadline_at(booking, rules)
    base = dict(kind="cancel", pnr=booking.pnr, currency=booking.currency, refund_method=refund_method, hours_to_departure=round(hrs, 1), deadline_at=deadline)
    if passenger_ids and not chosen:
        return Quote(**base, allowed=False, reason_code="NO_SUCH_PASSENGER",
                     reason_en="None of those passenger ids is a paying passenger still on this booking.", reason_ar="لا يوجد راكب بهذه الأرقام على هذا الحجز.")

    if booking.status == "cancelled":
        return Quote(**base, allowed=False, reason_code="ALREADY_CANCELLED",
                     reason_en="This booking is already cancelled.", reason_ar="هذا الحجز ملغى بالفعل.")

    disruption = _airline_disruption(booking, statuses)
    if disruption:
        refund = paid
        return Quote(**base, allowed=True, reason_code="AIRLINE_CANCELLED" if disruption == "cancelled" else "LONG_DELAY",
                     reason_en=("The airline cancelled your flight, so you are entitled to a full refund by any method, with no fee."
                                if disruption == "cancelled" else
                                "Your flight is delayed by five hours or more, so you are entitled to a full refund with no fee."),
                     reason_ar=("ألغت شركة الطيران رحلتك، لذلك يحق لك استرداد كامل المبلغ بأي طريقة دون رسوم."
                                if disruption == "cancelled" else
                                "تأخرت رحلتك خمس ساعات أو أكثر، لذلك يحق لك استرداد كامل المبلغ دون رسوم."),
                     fee=money(0), refund=refund, details={"passengers": pax, "paid": str(paid), **who})

    if has_flown(booking, now, statuses):
        return Quote(**base, allowed=False, reason_code="FLOWN",
                     reason_en="The flight has already departed, so this booking can no longer be cancelled or refunded.",
                     reason_ar="أقلعت الرحلة بالفعل، لذلك لا يمكن إلغاء هذا الحجز أو استرداد قيمته.")

    if hrs < rules.deadline_hours and not ignore_deadline:
        # A flight delayed by less than five hours gives no refund right, but the customer should hear what would (the delay task of
        # 18 Oct expected the rule to be stated): the threshold is in the reason, with the delay as it stands.
        delay = _short_delay_minutes(booking, statuses)
        delay_en = (f" Your flight is currently delayed by {delay} minutes; a delay of {LONG_DELAY_MINUTES // 60} hours or more would entitle you to a full refund "
                    "with no fee.") if delay else ""
        delay_ar = f" رحلتك متأخرة حاليًا {delay} دقيقة؛ التأخير {LONG_DELAY_MINUTES // 60} ساعات أو أكثر يمنحك استرداد كامل المبلغ دون رسوم." if delay else ""
        return Quote(**base, allowed=False, reason_code="PAST_DEADLINE", waiver_possible=True,
                     reason_en=(f"The {rules.name_en} fare must be cancelled at least {rules.deadline_hours} hours before departure. "
                                f"Departure is in {hrs:.1f} hours. The customer-care team can review an exception case." + delay_en),
                     reason_ar=(f"يجب إلغاء درجة {rules.name_ar} قبل {rules.deadline_hours} ساعة على الأقل من الإقلاع. "
                                f"الإقلاع بعد {hrs:.1f} ساعة. يمكن لفريق خدمة العملاء مراجعة طلب استثناء." + delay_ar),
                     details={"passengers": pax, **({"delay_minutes": delay} if delay else {}), **who})

    if refund_method == "cash":
        allowed, fee_each = rules.refund_cash_allowed, rules.refund_cash_fee
        if not allowed:
            return Quote(**base, allowed=False, reason_code="CASH_NOT_ALLOWED", waiver_possible=False,
                         reason_en=f"The {rules.name_en} fare is not refundable to the original payment method. A refund as travel credit is possible.",
                         reason_ar=f"درجة {rules.name_ar} غير قابلة للاسترداد إلى وسيلة الدفع الأصلية. يمكن الاسترداد كرصيد سفر.",
                         details={"passengers": pax, "alternative": "credit"})
    else:
        allowed, fee_each = rules.refund_credit_allowed, rules.refund_credit_fee
        if not allowed:
            return Quote(**base, allowed=False, reason_code="CREDIT_NOT_ALLOWED",
                         reason_en=f"The {rules.name_en} fare is not refundable.", reason_ar=f"درجة {rules.name_ar} غير قابلة للاسترداد.")

    fee = money(0) if waive_fee else money(fee_each * pax)
    refund = max(money(0), paid - fee)
    label_en = "travel credit" if refund_method == "credit" else "the original payment method"
    label_ar = "رصيد سفر" if refund_method == "credit" else "وسيلة الدفع الأصلية"
    return Quote(**base, allowed=True, reason_code="OK",
                 reason_en=f"Cancellation is allowed. Fee {fee} {booking.currency} ({fee_each} per passenger x {pax}); refund {refund} {booking.currency} as {label_en}.",
                 reason_ar=f"الإلغاء مسموح. الرسوم {fee} {booking.currency} ({fee_each} لكل راكب × {pax})؛ المبلغ المسترد {refund} {booking.currency} كـ{label_ar}.",
                 fee=fee, refund=refund,
                 details={"passengers": pax, "paid": str(paid), "fee_per_passenger": str(money(fee_each)), **who,
                          **({"exception": {"ignore_deadline": ignore_deadline, "waive_fee": waive_fee}} if ignore_deadline or waive_fee else {})})


def change_quote(booking: Booking, rules: FareRules, now: datetime, statuses: dict[str, FlightStatus], segment_seq: int, new_instance_id: str,
                 new_dep_at: datetime, new_fare_per_pax: Decimal) -> Quote:
    pax = _pax_count(booking)
    hrs = hours_to_departure(booking, now)
    deadline = deadline_at(booking, rules)
    base = dict(kind="change", pnr=booking.pnr, currency=booking.currency, hours_to_departure=round(hrs, 1), deadline_at=deadline)
    segment = next((s for s in booking.segments if s.seq == segment_seq), None)

    if booking.status == "cancelled":
        return Quote(**base, allowed=False, reason_code="ALREADY_CANCELLED", reason_en="This booking is cancelled; it cannot be changed.", reason_ar="هذا الحجز ملغى ولا يمكن تغييره.")
    if segment is None:
        return Quote(**base, allowed=False, reason_code="NO_SUCH_SEGMENT", reason_en="That flight segment is not on this booking.", reason_ar="هذا القطاع غير موجود في الحجز.")
    if segment.flight_instance_id == new_instance_id:
        return Quote(**base, allowed=False, reason_code="SAME_FLIGHT", reason_en="That is the flight already on the booking.", reason_ar="هذه هي الرحلة الموجودة في الحجز بالفعل.")
    if has_flown(booking, now, statuses):
        return Quote(**base, allowed=False, reason_code="FLOWN", reason_en="The flight has already departed; the booking can no longer be changed.", reason_ar="أقلعت الرحلة بالفعل؛ لا يمكن تغيير الحجز.")
    if new_dep_at <= now:
        return Quote(**base, allowed=False, reason_code="NEW_FLIGHT_DEPARTED", reason_en="The requested flight has already departed.", reason_ar="الرحلة المطلوبة أقلعت بالفعل.")
    disruption = _segment_disruption(segment.flight_instance_id, statuses)
    if disruption:
        # Involuntary rebooking: the airline broke the contract, so the same-route rebooking is free whatever the
        # fare: no change fee, no fare difference, no deadline, and it does not use up one of the fare's changes.
        return Quote(**base, allowed=True, reason_code="AIRLINE_CANCELLED" if disruption == "cancelled" else "LONG_DELAY",
                     reason_en=("The airline cancelled this flight, so rebooking on the same route is free: no change fee and no fare difference, on any fare."
                                if disruption == "cancelled" else
                                "This flight is delayed by five hours or more, so rebooking on the same route is free: no change fee and no fare difference."),
                     reason_ar=("ألغت شركة الطيران هذه الرحلة، لذلك إعادة الحجز على المسار نفسه مجانية: بلا رسوم تغيير وبلا فرق سعر، لأي درجة."
                                if disruption == "cancelled" else
                                "تأخرت هذه الرحلة خمس ساعات أو أكثر، لذلك إعادة الحجز على المسار نفسه مجانية: بلا رسوم تغيير وبلا فرق سعر."),
                     fee=money(0), amount_due=money(0),
                     details={"passengers": pax, "segment_seq": segment_seq, "new_flight_instance_id": new_instance_id,
                              "old_fare_per_pax": str(money(segment.fare_per_pax)), "new_fare_per_pax": str(money(segment.fare_per_pax)),
                              "fare_difference": "0.00", "free_change_used": False, "changes_used": booking.changes_used, "max_changes": rules.max_changes,
                              "involuntary": disruption})
    if (segment.dep_at - now).total_seconds() / 3600 < rules.deadline_hours:
        return Quote(**base, allowed=False, reason_code="PAST_DEADLINE", waiver_possible=True,
                     reason_en=f"Changes to the {rules.name_en} fare must be made at least {rules.deadline_hours} hours before departure.",
                     reason_ar=f"يجب تغيير درجة {rules.name_ar} قبل {rules.deadline_hours} ساعة على الأقل من الإقلاع.")
    if booking.changes_used >= rules.max_changes:
        return Quote(**base, allowed=False, reason_code="MAX_CHANGES", waiver_possible=True,
                     reason_en=f"The {rules.name_en} fare allows {rules.max_changes} change(s); this booking has used {booking.changes_used}.",
                     reason_ar=f"تسمح درجة {rules.name_ar} بـ{rules.max_changes} تغيير؛ استخدم هذا الحجز {booking.changes_used}.")

    free = booking.changes_used < rules.free_changes
    fee_each = money(0) if free else money(rules.change_fee)
    fee = money(fee_each * pax)
    diff_each = max(money(0), money(new_fare_per_pax) - money(segment.fare_per_pax))
    diff = money(diff_each * pax)
    due = money(fee + diff)
    return Quote(**base, allowed=True, reason_code="OK",
                 reason_en=(f"Change allowed. Change fee {fee} {booking.currency} ({'free change' if fee_each == 0 else str(fee_each) + ' per passenger'}), "
                            f"fare difference {diff} {booking.currency}; total due {due} {booking.currency}."),
                 reason_ar=(f"التغيير مسموح. رسوم التغيير {fee} {booking.currency} ({'تغيير مجاني' if fee_each == 0 else str(fee_each) + ' لكل راكب'})، "
                            f"فرق السعر {diff} {booking.currency}؛ الإجمالي المستحق {due} {booking.currency}."),
                 fee=fee, amount_due=due,
                 details={"passengers": pax, "segment_seq": segment_seq, "new_flight_instance_id": new_instance_id,
                          "old_fare_per_pax": str(money(segment.fare_per_pax)), "new_fare_per_pax": str(money(new_fare_per_pax)),
                          "fare_difference": str(diff), "free_change_used": free, "changes_used": booking.changes_used, "max_changes": rules.max_changes})


def bag_quote(booking: Booking, rules: FareRules, now: datetime, statuses: dict[str, FlightStatus], kg: int, passenger_ids: list[int]) -> Quote:
    base = dict(kind="bag", pnr=booking.pnr, currency=booking.currency, hours_to_departure=round(hours_to_departure(booking, now), 1))
    if booking.status == "cancelled":
        return Quote(**base, allowed=False, reason_code="ALREADY_CANCELLED", reason_en="This booking is cancelled.", reason_ar="هذا الحجز ملغى.")
    if has_flown(booking, now, statuses):
        return Quote(**base, allowed=False, reason_code="FLOWN", reason_en="The flight has already departed.", reason_ar="أقلعت الرحلة بالفعل.")
    if kg not in BAG_PRICES_AED:
        return Quote(**base, allowed=False, reason_code="INVALID_WEIGHT", reason_en="Baggage can be added in 10, 20 or 30 kg.", reason_ar="يمكن إضافة الأمتعة بوزن 10 أو 20 أو 30 كجم.")
    known = {p.id for p in booking.passengers}
    if not passenger_ids or any(pid not in known for pid in passenger_ids):
        return Quote(**base, allowed=False, reason_code="NO_SUCH_PASSENGER", reason_en="Unknown passenger on this booking.", reason_ar="راكب غير معروف في هذا الحجز.")
    if (min(s.dep_at for s in booking.segments) - now).total_seconds() < 2 * 3600:
        return Quote(**base, allowed=False, reason_code="TOO_LATE", reason_en="Baggage can be added up to 2 hours before departure.", reason_ar="يمكن إضافة الأمتعة حتى ساعتين قبل الإقلاع.")
    due = money(BAG_PRICES_AED[kg] * len(passenger_ids) * len(booking.segments))
    return Quote(**base, allowed=True, reason_code="OK",
                 reason_en=f"{kg} kg for {len(passenger_ids)} passenger(s) on {len(booking.segments)} flight(s): {due} {booking.currency}.",
                 reason_ar=f"{kg} كجم لـ{len(passenger_ids)} راكب على {len(booking.segments)} رحلة: {due} {booking.currency}.",
                 amount_due=due, details={"kg": kg, "passenger_ids": passenger_ids, "included_kg": rules.bag_kg, "segments": len(booking.segments)})


def waiver_quote(booking: Booking, rules: FareRules, now: datetime, statuses: dict[str, FlightStatus], refund_method: str, waive_fee: bool,
                 passenger_ids: list[int] | None = None) -> Quote:
    """What the customer-care team would grant if it approved the exception case. Shown to the customer as 'if approved'."""
    return cancellation_quote(booking, rules, now, statuses, refund_method, ignore_deadline=True, waive_fee=waive_fee, passenger_ids=passenger_ids)


BOOKING_CLOSES_HOURS = 2  # a new booking is taken until two hours before departure


def booking_quote(instance: dict, rules: FareRules, now: datetime, status: FlightStatus | None, passengers: int, passenger_names: list[str] | None = None) -> Quote:
    """Price a new booking on one flight instance (a row of reservations.flight_instances) in one fare family.

    The fare is the same arithmetic search_flights shows (base price x the family's multiplier, whole dirhams), so the number the
    customer saw in the search is the number on the quote and on the card. Nothing here reads the model's words."""
    hrs = (instance["dep_at"] - now).total_seconds() / 3600
    base = dict(kind="book", pnr="", currency="AED", hours_to_departure=round(hrs, 1))
    if status is not None and status.status == "cancelled":
        return Quote(**base, allowed=False, reason_code="FLIGHT_CANCELLED", reason_en="This flight has been cancelled by the airline; it cannot be booked.",
                     reason_ar="ألغت شركة الطيران هذه الرحلة؛ لا يمكن حجزها.")
    if instance["dep_at"] <= now:
        return Quote(**base, allowed=False, reason_code="FLIGHT_DEPARTED", reason_en="This flight has already departed.", reason_ar="أقلعت هذه الرحلة بالفعل.")
    if hrs < BOOKING_CLOSES_HOURS:
        return Quote(**base, allowed=False, reason_code="BOOKING_CLOSED",
                     reason_en=f"Bookings close {BOOKING_CLOSES_HOURS} hours before departure; this flight departs in {hrs:.1f} hours.",
                     reason_ar=f"يُغلق الحجز قبل {BOOKING_CLOSES_HOURS} ساعتين من الإقلاع؛ تقلع هذه الرحلة بعد {hrs:.1f} ساعة.")
    if passengers < 1 or passengers > 9:
        return Quote(**base, allowed=False, reason_code="BAD_PASSENGERS", reason_en="A booking is for 1 to 9 passengers.", reason_ar="الحجز لراكب واحد إلى 9 ركاب.")
    seats = int(instance["seats_total"]) - int(instance["seats_sold"])
    if seats < passengers:
        return Quote(**base, allowed=False, reason_code="NO_SEATS", reason_en=f"Only {seats} seat(s) left on this flight.", reason_ar=f"لم يتبقَّ سوى {seats} مقعد على هذه الرحلة.")
    fare = money(Decimal(instance["base_price"]) * rules.price_multiplier).quantize(Decimal("1"))
    total = money(fare * passengers)
    names = [n for n in (passenger_names or []) if n][: max(0, passengers - 1)]
    return Quote(**base, allowed=True, reason_code="OK", amount_due=total,
                 reason_en=f"Booking available. {rules.name_en} fare {fare} AED per passenger x {passengers} = {total} AED, charged to the card on file.",
                 reason_ar=f"الحجز متاح. درجة {rules.name_ar} {fare} درهم للراكب × {passengers} = {total} درهم، تُحصَّل من البطاقة المسجلة.",
                 details={"flight_instance_id": instance["id"], "flight_number": instance["flight_number"], "origin": instance["origin"],
                          "destination": instance["destination"], "dep_at": instance["dep_at"].isoformat(), "arr_at": instance["arr_at"].isoformat(),
                          "fare_family": rules.code, "fare_per_pax": str(money(fare)), "passengers": passengers, "passenger_names": names, "seats_available": seats})


# ---------------------------------------------------------------- fitness to travel: Article 7 of the Conditions as thresholds
# The figures and clause ids below are the ones in data/policy/coc_en.pdf (the same text search_policy returns), fixed here so that
# "38 weeks" is decided by code and not by the model reading "36th week" correctly. One function, three uses again: the policy
# server's check_travel_eligibility tool (which attaches the clause text, so the reviewer can verify every figure the agent repeats),
# the scripted model, and the tests. No clock, no I/O, no model.
PREGNANCY_LETTER_FROM_WEEK = 28       # 7.4.1: after the 28th week, carried only with a medical report or letter
PREGNANCY_LAST_WEEK_SINGLE = 36       # 7.4.3(a): not carried after the 36th week of a single pregnancy
PREGNANCY_LAST_WEEK_MULTIPLE = 32     # 7.4.3(b): not carried after the 32nd week of a multiple pregnancy
PREGNANCY_MAX_WEEKS = 45              # above this a figure cannot be weeks of pregnancy: the customer spoke in months, or slipped
PREGNANCY_MAX_MONTHS = 10
WEEKS_PER_MONTH = 4.33
NEWBORN_MIN_DAYS = 7                  # 7.4.5: newborn infants generally not carried until 7 days after delivery
CHILD_ESCORT_UNDER_YEARS = 5          # 7.7.1: under 5 only with an escort aged 16 or more (18 or more if special attention is needed)
ESCORT_MIN_AGE_YEARS = 16
ESCORT_MIN_AGE_SPECIAL_YEARS = 18
CHILD_CONDITIONS_UNDER_YEARS = 12     # 7.7.3: more than 5 and less than 12, alone: every listed condition must be met
CHILD_ESCORT_SERVICE_TO_YEARS = 16    # 7.7.2: between 5 and 16 an escort can be arranged with at least 12 hours' notice
ESCORT_NOTICE_HOURS = 12
STOPOVER_HOURS_DXB, STOPOVER_HOURS_OTHER = 8, 4  # 7.7.3(d)
ELIGIBILITY_SEVERITY = {"ok": 0, "needs_weeks": 1, "conditions_apply": 2, "document_required": 3, "escort_required": 4, "clearance_required": 5, "not_carried": 6}


@dataclass
class Finding:
    topic: str                     # pregnancy | newborn | child_alone
    status: str                    # a key of ELIGIBILITY_SEVERITY
    clause_ids: list[str]          # the clauses the rule rests on; the tool attaches their text
    rule_en: str
    rule_ar: str
    figures: dict = field(default_factory=dict)  # every number the rule used, so a reply can be checked against them


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _whole(name: str, value, low: int, high: int) -> int | None:
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number.") from None
    if value < low or value > high:
        raise ValueError(f"{name} must be between {low} and {high}; got {value}.")
    return value


def travel_eligibility(*, pregnancy_weeks=None, pregnancy_months=None, multiple_pregnancy: bool = False, complicated_pregnancy: bool = False,
                       infant_age_days=None, child_age_years=None, travelling_alone: bool = False) -> list[Finding]:
    """What Article 7 of the Conditions says for the facts given, one finding per fact; [] when no fact was given.

    A pregnancy in months (or a week count no pregnancy can reach) is answered with the rules and `needs_weeks`: the Conditions
    count weeks, and "38th month" on 18 Oct was a customer in her 38th week, who could not be carried."""
    weeks = _whole("pregnancy_weeks", pregnancy_weeks, 0, 99)
    months = _whole("pregnancy_months", pregnancy_months, 0, 99)
    days = _whole("infant_age_days", infant_age_days, 0, 3650)
    age = _whole("child_age_years", child_age_years, 0, 120)
    out: list[Finding] = []
    last = PREGNANCY_LAST_WEEK_MULTIPLE if multiple_pregnancy else PREGNANCY_LAST_WEEK_SINGLE
    kind_en = "multiple" if multiple_pregnancy else "single"
    kind_ar = "بأكثر من جنين" if multiple_pregnancy else "بجنين واحد"
    base = {"letter_from_week": PREGNANCY_LETTER_FROM_WEEK, "last_week": last, "last_week_single": PREGNANCY_LAST_WEEK_SINGLE, "last_week_multiple": PREGNANCY_LAST_WEEK_MULTIPLE}
    if weeks is not None and weeks > PREGNANCY_MAX_WEEKS:
        months, weeks = weeks, None  # "38 months" passed as weeks, or a slip: handled as the months case
    if weeks is None and months is not None:
        approx = round(months * WEEKS_PER_MONTH) if months <= PREGNANCY_MAX_MONTHS else None
        if approx is None:
            said_en = (f"{months} months is not a possible duration of pregnancy, so the customer most likely means the {_ordinal(months)} week"
                       + (f", which is after the {_ordinal(last)}-week limit" if months > last else "") + ".")
            said_ar = f"{months} شهرًا ليست مدة حمل ممكنة، فالأرجح أن العميلة تقصد الأسبوع {months}" + (f"، وهو بعد حد الأسبوع {last}" if months > last else "") + "."
        else:
            said_en = f"{months} months is about {approx} weeks."
            said_ar = f"{months} أشهر تعادل نحو {approx} أسبوعًا."
        out.append(Finding("pregnancy", "needs_weeks", ["7.4.1", "7.4.3"],
                           rule_en=(f"The Conditions count weeks of pregnancy, not months: a medical report or letter from a doctor or midwife is required after the "
                                    f"{_ordinal(PREGNANCY_LETTER_FROM_WEEK)} week (clause 7.4.1), and carriage is not possible after the {_ordinal(PREGNANCY_LAST_WEEK_SINGLE)} week of a "
                                    f"single pregnancy or the {_ordinal(PREGNANCY_LAST_WEEK_MULTIPLE)} week of a multiple pregnancy (clause 7.4.3). {said_en} Ask which week of "
                                    "pregnancy the customer is in (or the due date) before saying whether they can travel."),
                           rule_ar=(f"تحسب شروط النقل الحمل بالأسابيع لا بالأشهر: يلزم تقرير أو رسالة طبية من طبيب أو قابلة بعد الأسبوع {PREGNANCY_LETTER_FROM_WEEK} (البند 7.4.1)، "
                                    f"ولا يمكن النقل بعد الأسبوع {PREGNANCY_LAST_WEEK_SINGLE} في الحمل بجنين واحد أو بعد الأسبوع {PREGNANCY_LAST_WEEK_MULTIPLE} في الحمل بأكثر من جنين "
                                    f"(البند 7.4.3). {said_ar} اسألي عن أسبوع الحمل (أو موعد الولادة المتوقع) قبل تأكيد إمكانية السفر."),
                           figures={**base, "stated_months": months, **({"approx_weeks": approx} if approx is not None else {})}))
    elif weeks is not None:
        if weeks > last:
            out.append(Finding("pregnancy", "not_carried", ["7.4.3"],
                               rule_en=(f"At {weeks} weeks of a {kind_en} pregnancy the customer cannot be carried: the Conditions do not allow carriage after the "
                                        f"{_ordinal(last)} week of a {kind_en} pregnancy (clause 7.4.3)."),
                               rule_ar=f"في الأسبوع {weeks} من الحمل {kind_ar} لا يمكن نقل العميلة: لا تسمح شروط النقل بالنقل بعد الأسبوع {last} في الحمل {kind_ar} (البند 7.4.3).",
                               figures={**base, "stated_weeks": weeks}))
        elif weeks > PREGNANCY_LETTER_FROM_WEEK:
            out.append(Finding("pregnancy", "document_required", ["7.4.1", "7.4.3"],
                               rule_en=(f"At {weeks} weeks the customer is carried only with a medical report or letter signed by a qualified doctor or midwife stating whether the "
                                        "pregnancy is single or multiple, that it is uncomplicated, the estimated date of delivery, the latest date they are expected to be fit to "
                                        f"travel, that they are in good health and that nothing known prevents them from flying (clause 7.4.1, after the {_ordinal(PREGNANCY_LETTER_FROM_WEEK)} week). "
                                        f"Carriage is not possible after the {_ordinal(last)} week of a {kind_en} pregnancy (clause 7.4.3)."),
                               rule_ar=(f"في الأسبوع {weeks} تُنقل العميلة فقط بتقرير أو رسالة طبية موقعة من طبيب أو قابلة مؤهلين تبيّن ما إذا كان الحمل بجنين واحد أو أكثر، وأنه خالٍ من "
                                        "التعقيدات، والتاريخ المتوقع للولادة، وآخر تاريخ يُتوقع أن تكون فيه مؤهلة للسفر، وأنها بصحة جيدة، وأنه لا يوجد سبب معروف يمنعها من الطيران "
                                        f"(البند 7.4.1، بعد الأسبوع {PREGNANCY_LETTER_FROM_WEEK}). ولا يمكن النقل بعد الأسبوع {last} في الحمل {kind_ar} (البند 7.4.3)."),
                               figures={**base, "stated_weeks": weeks}))
        else:
            out.append(Finding("pregnancy", "ok", ["7.4.1", "7.4.3"],
                               rule_en=(f"At {weeks} weeks no medical letter is required yet: one is required after the {_ordinal(PREGNANCY_LETTER_FROM_WEEK)} week (clause 7.4.1), "
                                        f"and carriage is not possible after the {_ordinal(last)} week of a {kind_en} pregnancy (clause 7.4.3)."),
                               rule_ar=(f"في الأسبوع {weeks} لا يلزم تقرير طبي بعد: يلزم بعد الأسبوع {PREGNANCY_LETTER_FROM_WEEK} (البند 7.4.1)، ولا يمكن النقل بعد الأسبوع {last} "
                                        f"في الحمل {kind_ar} (البند 7.4.3)."),
                               figures={**base, "stated_weeks": weeks}))
    if (weeks is not None or months is not None) and complicated_pregnancy:
        out.append(Finding("pregnancy", "clearance_required", ["7.4.2"],
                           rule_en=("A complicated pregnancy (single or multiple) needs a Medical Information Form (MEDIF) and prior clearance from the medical department at any "
                                    "week, even within the gestational limits (clause 7.4.2)."),
                           rule_ar=("الحمل الذي تتخلله تعقيدات (بجنين واحد أو أكثر) يحتاج إلى استمارة المعلومات الطبية MEDIF وتصريح مسبق من القسم الطبي في أي أسبوع، حتى ضمن حدود "
                                    "عمر الحمل (البند 7.4.2)."),
                           figures={}))
    if days is not None:
        if days < NEWBORN_MIN_DAYS:
            out.append(Finding("newborn", "not_carried", ["7.4.5"],
                               rule_en=(f"A newborn aged {days} days cannot travel yet: carriage of newly born infants is generally not permitted until {NEWBORN_MIN_DAYS} days after "
                                        f"delivery (clause 7.4.5). From day {NEWBORN_MIN_DAYS} the infant can be carried; ask us or our agent for the details."),
                               rule_ar=(f"لا يمكن سفر المولود البالغ من العمر {days} أيام بعد: لا يُسمح عادةً بنقل الأطفال حديثي الولادة إلا بعد {NEWBORN_MIN_DAYS} أيام من الولادة "
                                        f"(البند 7.4.5). اعتبارًا من اليوم {NEWBORN_MIN_DAYS} يمكن نقله؛ اسأل الشركة أو وكيلها عن التفاصيل."),
                               figures={"stated_days": days, "min_days": NEWBORN_MIN_DAYS}))
        else:
            out.append(Finding("newborn", "ok", ["7.4.5"],
                               rule_en=f"An infant aged {days} days is past the {NEWBORN_MIN_DAYS}-day minimum for newborns (clause 7.4.5); nothing in that clause prevents the trip.",
                               rule_ar=f"المولود البالغ من العمر {days} يومًا تجاوز الحد الأدنى البالغ {NEWBORN_MIN_DAYS} أيام لحديثي الولادة (البند 7.4.5)؛ لا شيء في هذا البند يمنع الرحلة.",
                               figures={"stated_days": days, "min_days": NEWBORN_MIN_DAYS}))
    if age is not None:
        fig = {"stated_age": age, "escort_under": CHILD_ESCORT_UNDER_YEARS, "escort_min_age": ESCORT_MIN_AGE_YEARS, "escort_min_age_special": ESCORT_MIN_AGE_SPECIAL_YEARS,
               "conditions_under": CHILD_CONDITIONS_UNDER_YEARS, "escort_service_to": CHILD_ESCORT_SERVICE_TO_YEARS, "notice_hours": ESCORT_NOTICE_HOURS}
        if age < CHILD_ESCORT_UNDER_YEARS:
            escort_en = (f"accompanied on the same flight and in the same compartment by an escort aged {ESCORT_MIN_AGE_YEARS} or more ({ESCORT_MIN_AGE_SPECIAL_YEARS} or more "
                         "if the child needs special attention) (clause 7.7.1)")
            escort_ar = (f"برفقة مرافق على الرحلة نفسها وفي المقصورة نفسها يبلغ من العمر {ESCORT_MIN_AGE_YEARS} عامًا أو أكثر ({ESCORT_MIN_AGE_SPECIAL_YEARS} عامًا أو أكثر إذا كان "
                         "الطفل يحتاج إلى رعاية خاصة) (البند 7.7.1)")
            if travelling_alone:
                out.append(Finding("child_alone", "escort_required", ["7.7.1"],
                                   rule_en=f"A child aged {age} cannot travel alone: under {CHILD_ESCORT_UNDER_YEARS} a child is not accepted for carriage unless {escort_en}.",
                                   rule_ar=f"لا يمكن لطفل عمره {age} سنوات السفر وحده: دون سن {CHILD_ESCORT_UNDER_YEARS} سنوات لا يُقبل الطفل للنقل إلا {escort_ar}.",
                                   figures=fig))
            else:
                out.append(Finding("child_alone", "conditions_apply", ["7.7.1"],
                                   rule_en=f"A child aged {age} must be {escort_en}.",
                                   rule_ar=f"يجب أن يكون الطفل البالغ {age} سنوات {escort_ar}.",
                                   figures=fig))
        elif travelling_alone and CHILD_ESCORT_UNDER_YEARS < age < CHILD_CONDITIONS_UNDER_YEARS:
            out.append(Finding("child_alone", "conditions_apply", ["7.7.3", "7.7.2"],
                               rule_en=(f"A child aged {age} travelling alone is accepted only when all the conditions of clause 7.7.3 are met: a parent, guardian or responsible adult "
                                        "accompanies the child at check-in and stays at the airport with a member of our staff until the aircraft has taken off, and proves that the "
                                        "child will be met at the destination (or stopover) by a named parent, guardian or responsible adult; that adult completes and provides all "
                                        "the documentation; the reservation is confirmed; and the journey involves no change of airport en route and no scheduled stopover over "
                                        f"{STOPOVER_HOURS_DXB} hours at Dubai International or {STOPOVER_HOURS_OTHER} hours elsewhere unless an adult takes charge for the whole stopover. "
                                        f"An escort can be arranged for a child between {CHILD_ESCORT_UNDER_YEARS} and {CHILD_ESCORT_SERVICE_TO_YEARS}, with notice at least "
                                        f"{ESCORT_NOTICE_HOURS} hours before travel (clause 7.7.2)."),
                               rule_ar=(f"يُقبل نقل طفل عمره {age} سنوات يسافر وحده فقط عند استيفاء شروط البند 7.7.3 كاملة: أن يرافقه أحد الوالدين أو وصي أو شخص بالغ مسؤول عند إنجاز "
                                        "إجراءات السفر ويبقى في المطار مع أحد موظفينا حتى إقلاع الطائرة، ويثبت أن الطفل سيُستقبل في مطار الوصول (أو محطة التوقف) من قبل والد أو وصي أو "
                                        "شخص بالغ مسؤول محدد؛ وأن يستكمل ذلك الشخص جميع الوثائق اللازمة ويقدمها؛ وأن يكون الحجز مؤكدًا؛ وألا يتضمن المسار تغيير مطار أو محطة توقف تتجاوز "
                                        f"{STOPOVER_HOURS_DXB} ساعات في مطار دبي الدولي أو {STOPOVER_HOURS_OTHER} ساعات في أي مطار آخر ما لم يتولَّ شخص بالغ مسؤوليته طوال التوقف. "
                                        f"ويمكن ترتيب خدمة مرافقة للطفل بين {CHILD_ESCORT_UNDER_YEARS} و{CHILD_ESCORT_SERVICE_TO_YEARS} عامًا بإشعار قبل {ESCORT_NOTICE_HOURS} ساعة "
                                        "على الأقل من السفر (البند 7.7.2)."),
                               figures={**fig, "stopover_hours_dxb": STOPOVER_HOURS_DXB, "stopover_hours_other": STOPOVER_HOURS_OTHER}))
        elif travelling_alone and age <= CHILD_ESCORT_SERVICE_TO_YEARS:
            out.append(Finding("child_alone", "conditions_apply", ["7.7.2"],
                               rule_en=(f"For a child aged {age} travelling alone an escort can be arranged, provided we receive notice at least {ESCORT_NOTICE_HOURS} hours before "
                                        f"travel and subject to the unaccompanied-minor conditions (clause 7.7.2, ages {CHILD_ESCORT_UNDER_YEARS} to {CHILD_ESCORT_SERVICE_TO_YEARS})."),
                               rule_ar=(f"لطفل عمره {age} عامًا يسافر وحده يمكن ترتيب خدمة مرافقة، بشرط إشعارنا قبل {ESCORT_NOTICE_HOURS} ساعة على الأقل من السفر ووفق شروط القصّر غير "
                                        f"المصحوبين (البند 7.7.2، من {CHILD_ESCORT_UNDER_YEARS} إلى {CHILD_ESCORT_SERVICE_TO_YEARS} عامًا)."),
                               figures=fig))
        elif travelling_alone:
            out.append(Finding("child_alone", "ok", ["7.7.2"],
                               rule_en=f"A passenger aged {age} is above the age range of the unaccompanied-children rules, which end at {CHILD_ESCORT_SERVICE_TO_YEARS} (clause 7.7.2).",
                               rule_ar=f"المسافر البالغ {age} عامًا يتجاوز الفئة العمرية لقواعد الأطفال دون مرافق التي تنتهي عند {CHILD_ESCORT_SERVICE_TO_YEARS} عامًا (البند 7.7.2).",
                               figures=fig))
        else:
            out.append(Finding("child_alone", "ok", ["7.7.2"],
                               rule_en=(f"A child aged {age} travelling with an adult: Article 7.7 (unaccompanied children) does not apply; its escort service covers children between "
                                        f"{CHILD_ESCORT_UNDER_YEARS} and {CHILD_ESCORT_SERVICE_TO_YEARS} travelling alone (clause 7.7.2)."),
                               rule_ar=(f"الطفل البالغ {age} سنوات المسافر مع شخص بالغ: لا تنطبق المادة 7.7 (الأطفال دون مرافق)؛ تغطي خدمة المرافقة فيها الأطفال بين "
                                        f"{CHILD_ESCORT_UNDER_YEARS} و{CHILD_ESCORT_SERVICE_TO_YEARS} عامًا المسافرين وحدهم (البند 7.7.2)."),
                               figures=fig))
    return out


# What the Conditions of Carriage grant on evidence, as a pure function: clause 3.3.2 (a member of the Immediate
# Family dies before travel: a refund under 10.2.1(a), the fare paid, on a death certificate) and clause 3.3.1 (Force Majeure prevents
# travel: a credit for the non-refundable part of the fare, on adequate evidence). Logged as a case; the care team requests the document by email; the
# care team checks the document and approves. The agent cites the clause and quotes this, never a normal cancellation with a fee.
COMPASSIONATE_KINDS = {
    "bereavement": {"clauses": ["3.3.2", "10.2.1"], "evidence": "a valid death certificate of the family member", "evidence_ar": "شهادة وفاة سارية لفرد العائلة"},
    "force_majeure": {"clauses": ["3.3.1"], "evidence": "adequate evidence of the force majeure event (for example a medical report)", "evidence_ar": "دليل كافٍ على الظرف القاهر (مثل تقرير طبي)"},
}


def compassionate_quote(booking: Booking, rules: FareRules, now: datetime, statuses: dict[str, FlightStatus], kind: str, refund_method: str = "credit") -> Quote:
    if kind not in COMPASSIONATE_KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    spec = COMPASSIONATE_KINDS[kind]
    pax = _pax_count(booking)
    hrs = hours_to_departure(booking, now)
    method = "credit" if kind == "force_majeure" else refund_method  # 3.3.1 is a credit by its own words; 3.3.2 refunds the fare paid
    base = dict(kind="cancel", pnr=booking.pnr, currency=booking.currency, refund_method=method, hours_to_departure=round(hrs, 1), deadline_at=deadline_at(booking, rules))
    if booking.status == "cancelled":
        return Quote(**base, allowed=False, reason_code="ALREADY_CANCELLED", reason_en="This booking is already cancelled.", reason_ar="هذا الحجز ملغى بالفعل.")
    if has_flown(booking, now, statuses):
        return Quote(**base, allowed=False, reason_code="FLOWN", reason_en="Clauses 3.3.1 and 3.3.2 apply before the carriage has begun; this flight has already departed.",
                     reason_ar="ينطبق البندان 3.3.1 و3.3.2 قبل بدء الرحلة؛ أقلعت هذه الرحلة بالفعل.")
    refund = money(booking.total_paid)
    if kind == "bereavement":
        reason_en = (f"Clause 3.3.2: a member of the Immediate Family has died before the carriage began, so the fare paid is refunded under clause 10.2.1(a): "
                     f"{refund} {booking.currency}, no fee, on {spec['evidence']}.")
        reason_ar = f"البند 3.3.2: وفاة أحد أفراد العائلة المباشرة قبل بدء الرحلة، فيُرد المبلغ المدفوع وفق البند 10.2.1(أ): {refund} {booking.currency} دون رسوم، مقابل {spec['evidence_ar']}."
    else:
        reason_en = (f"Clause 3.3.1: Force Majeure prevented the travel before the carriage began, so the non-refundable part of the fare becomes a credit "
                     f"for a ticket within 12 months: {refund} {booking.currency}, no fee, on {spec['evidence']}.")
        reason_ar = f"البند 3.3.1: منع ظرف قاهر السفر قبل بدء الرحلة، فيتحول الجزء غير القابل للاسترداد من الأجرة إلى رصيد لتذكرة خلال 12 شهرًا: {refund} {booking.currency} دون رسوم، مقابل {spec['evidence_ar']}."
    return Quote(**base, allowed=True, reason_code=kind.upper(), fee=money(0), refund=refund,
                 reason_en=reason_en, reason_ar=reason_ar,
                 details={"passengers": pax, "paid": str(money(booking.total_paid)), "clauses": spec["clauses"], "evidence_required": spec["evidence"],
                          "decided_by": "customer-care team on the document it requests"})
