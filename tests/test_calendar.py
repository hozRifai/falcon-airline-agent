"""Relative periods are computed, not guessed (FAILURE_DIARY #39): the boundaries are pinned here for the frozen clock and the edges."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from falcon.core.calendar import calendar_line, periods


def _p(today: date) -> dict:
    return {p.key: p for p in periods(today)}


def test_frozen_clock_tuesday_20_october():
    p = _p(date(2026, 10, 20))  # the demo's "now"
    assert (p["today"].start, p["today"].end) == (date(2026, 10, 20), date(2026, 10, 20))
    assert (p["this_week"].start, p["this_week"].end) == (date(2026, 10, 20), date(2026, 10, 25))
    assert (p["this_weekend"].start, p["this_weekend"].end) == (date(2026, 10, 24), date(2026, 10, 25))
    assert (p["next_week"].start, p["next_week"].end) == (date(2026, 10, 26), date(2026, 11, 1))  # Monday to Sunday, not "seven days from now"
    assert (p["next_weekend"].start, p["next_weekend"].end) == (date(2026, 10, 31), date(2026, 11, 1))
    assert (p["week_after_next"].start, p["week_after_next"].end) == (date(2026, 11, 2), date(2026, 11, 8))
    assert (p["rest_of_month"].start, p["rest_of_month"].end) == (date(2026, 10, 20), date(2026, 10, 31))
    assert (p["next_month"].start, p["next_month"].end) == (date(2026, 11, 1), date(2026, 11, 30))
    assert (p["first_week_next_month"].start, p["first_week_next_month"].end) == (date(2026, 11, 1), date(2026, 11, 7))
    assert (p["last_two_weeks_next_month"].start, p["last_two_weeks_next_month"].end) == (date(2026, 11, 17), date(2026, 11, 30))
    assert p["next_month"].days == 30 and p["next_week"].days == 7 and p["next_weekend"].days == 2


def test_edges_sunday_and_month_end():
    sunday = _p(date(2026, 10, 25))
    assert (sunday["this_week"].start, sunday["this_week"].end) == (date(2026, 10, 25), date(2026, 10, 25))  # the rest of the week is today
    assert (sunday["this_weekend"].start, sunday["this_weekend"].end) == (date(2026, 10, 25), date(2026, 10, 25))  # Saturday is gone
    assert sunday["next_week"].start == date(2026, 10, 26)
    dec = _p(date(2026, 12, 31))
    assert (dec["rest_of_month"].start, dec["rest_of_month"].end) == (date(2026, 12, 31), date(2026, 12, 31))
    assert (dec["next_month"].start, dec["next_month"].end) == (date(2027, 1, 1), date(2027, 1, 31))  # year boundary
    feb = _p(date(2027, 1, 15))
    assert feb["next_month"].end == date(2027, 2, 28)
    for d in range(60):  # every period is a non-empty range with start <= end, whatever the day
        for p in periods(date(2026, 10, 1) + timedelta(days=d)):
            assert p.start <= p.end and p.days >= 1, p


def test_calendar_line_names_both_languages_and_exact_dates():
    line = calendar_line(datetime(2026, 10, 20, 9, tzinfo=timezone(timedelta(hours=4))))
    assert line.startswith("Calendar, computed from the clock")
    assert "next week / الأسبوع القادم = Mon 2026-10-26 to Sun 2026-11-01" in line
    assert "the first week of next month / الأسبوع الأول من الشهر القادم = Sun 2026-11-01 to Sat 2026-11-07" in line
    assert "today / اليوم = Tue 2026-10-20;" in line
    assert "{" not in line and "}" not in line  # it is inserted into a str.format template
