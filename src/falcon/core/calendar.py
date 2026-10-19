"""Relative periods, computed from the frozen clock (FAILURE_DIARY #39).

"Next week", "this weekend", "the first week of next month" are date arithmetic, and date arithmetic is the model's weak
spot: asked for the cheapest flight "next week" it searched one day, a week from today. So the calendar is computed here,
printed into the system prompt as exact dates, and the model's only job is to copy the right pair into search_flights.
Weeks run Monday to Sunday; the UAE weekend is Saturday and Sunday.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .clock import now

WEEKDAY_EN = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKDAY_AR = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]


@dataclass(frozen=True)
class Period:
    key: str
    en: str
    ar: str
    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


def _month_start(d: date) -> date:
    return d.replace(day=1)


def _next_month_start(d: date) -> date:
    return (_month_start(d) + timedelta(days=32)).replace(day=1)


def periods(today: date) -> list[Period]:
    """The periods a customer names in words, as first and last day. Deterministic; the tests pin the boundaries."""
    monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6)
    saturday = monday + timedelta(days=5)
    next_mon = monday + timedelta(days=7)
    month_end = _next_month_start(today) - timedelta(days=1)
    nm_start = _next_month_start(today)
    nm_end = _next_month_start(nm_start) - timedelta(days=1)
    P = Period  # noqa: N806
    return [
        P("today", "today", "اليوم", today, today),
        P("tomorrow", "tomorrow", "غدًا", today + timedelta(days=1), today + timedelta(days=1)),
        P("this_week", "this week (the rest of it)", "هذا الأسبوع (ما تبقى منه)", today, sunday),
        P("this_weekend", "this weekend", "نهاية هذا الأسبوع", max(saturday, today), sunday),
        P("next_week", "next week", "الأسبوع القادم", next_mon, next_mon + timedelta(days=6)),
        P("next_weekend", "next weekend", "نهاية الأسبوع القادم", next_mon + timedelta(days=5), next_mon + timedelta(days=6)),
        P("week_after_next", "the week after next", "الأسبوع الذي يلي القادم", next_mon + timedelta(days=7), next_mon + timedelta(days=13)),
        P("rest_of_month", "the rest of this month", "بقية هذا الشهر", today, month_end),
        P("next_month", "next month (whole)", "الشهر القادم (كاملًا)", nm_start, nm_end),
        P("first_week_next_month", "the first week of next month", "الأسبوع الأول من الشهر القادم", nm_start, nm_start + timedelta(days=6)),
        P("second_week_next_month", "the second week of next month", "الأسبوع الثاني من الشهر القادم", nm_start + timedelta(days=7), nm_start + timedelta(days=13)),
        P("last_week_next_month", "the last week of next month", "الأسبوع الأخير من الشهر القادم", nm_end - timedelta(days=6), nm_end),
        P("last_two_weeks_next_month", "the last two weeks of next month", "الأسبوعان الأخيران من الشهر القادم", nm_end - timedelta(days=13), nm_end),
    ]


def _fmt(d: date) -> str:
    return f"{WEEKDAY_EN[d.weekday()]} {d.isoformat()}"


def calendar_line(reference: datetime | None = None) -> str:
    """One paragraph for the system prompt: every period with its exact first and last day, in both languages."""
    today = (reference or now()).date()
    parts = []
    for p in periods(today):
        span = _fmt(p.start) if p.start == p.end else f"{_fmt(p.start)} to {_fmt(p.end)}"
        parts.append(f"{p.en} / {p.ar} = {span}")
    return ("Calendar, computed from the clock (weeks run Monday to Sunday; the UAE weekend is Saturday and Sunday): "
            + "; ".join(parts) + ".")
