"""Two confirmations of the same quote at the same moment execute it once (FAILURE_DIARY #73).

A double click, a network retry, or the voice "نعم" and the button together send the same write twice. Both requests used to pass
the "quote not consumed" check before either marked it, so the booking was cancelled once but the refund was written twice. Live:
needs the reservations MCP server running (skipped otherwise); the database is re-seeded first.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys

import psycopg
import pytest

from falcon.core.config import REPO_ROOT, settings
from tests.test_reservations_live import CUSTOMER_ROLES, _up, call, tools_for, user

pytestmark = pytest.mark.skipif(not (_up(settings.legacy_api_url + "/health") and _up(settings.mcp_reservations_url)), reason="services not running")


@pytest.fixture(scope="module", autouse=True)
def reseed():
    subprocess.run([sys.executable, str(REPO_ROOT / "db" / "apply.py"), "--seed-only"], check=True, capture_output=True)


def refund_rows(pnr: str) -> int:
    with psycopg.connect(settings.dsn_admin) as conn:
        return conn.execute("SELECT count(*) FROM reservations.transactions WHERE pnr = %s AND kind LIKE 'refund%%'", (pnr,)).fetchone()[0]


def test_two_simultaneous_confirms_of_one_quote_refund_once():
    async def go():
        t = await tools_for(user("u-fatima", "C-1001"), CUSTOMER_ROLES, ["reservations"])
        q = await call(t, "quote_cancellation", pnr="K3P7QZ", refund_method="credit")
        assert q["allowed"] and q["refund"] == "1291.00"
        before = refund_rows("K3P7QZ")
        a, b = await asyncio.gather(*(call(t, "cancel_booking", pnr="K3P7QZ", quote_id=q["quote_id"]) for _ in range(2)))
        assert sorted([a["already_done"], b["already_done"]]) == [False, True], (a, b)
        assert a["reference"] == b["reference"]  # the second answer is the first one's result, not a new cancellation
        assert refund_rows("K3P7QZ") == before + 1
    asyncio.run(go())
