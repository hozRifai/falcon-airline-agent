"""Frozen clock.

Every deadline rule ("24 hours before departure") depends on "now". If "now" drifts, a
booking that is cancellable today is not cancellable tomorrow, and pass^k numbers measured
on Monday cannot be reproduced on Thursday. So every service, the seed, the rules engine and
the evaluation harness call `now()` from here, and `now()` returns FALCON_NOW.

Set FALCON_NOW=wallclock to use the real clock (never during evaluation).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .config import settings

GULF = timezone(timedelta(hours=4), name="Asia/Dubai")


def now() -> datetime:
    value = settings.falcon_now
    if value == "wallclock":
        return datetime.now(tz=GULF)
    return datetime.fromisoformat(value).astimezone(GULF)
