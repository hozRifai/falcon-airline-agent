"""End-to-end tests through the runtime HTTP API with the scripted model (no API key needed).

Requires: scripts/dev.sh start   (all services). Reseeds the database first.
"""

from __future__ import annotations

import json
import subprocess
import sys

import httpx
import pytest

from falcon.core.config import REPO_ROOT, settings

B = settings.runtime_public_url


def _up() -> bool:
    try:
        return httpx.get(f"{B}/api/health", timeout=2).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _up(), reason="runtime not running")


@pytest.fixture(scope="module", autouse=True)
def reseed():
    subprocess.run([sys.executable, str(REPO_ROOT / "db" / "apply.py"), "--seed-only"], check=True, capture_output=True)


ADMIN = {"X-Admin-Token": settings.admin_token}  # the tests are admin tooling: exempt from the per-IP rate limit, like the eval harness


def login(client: httpx.Client, user_id: str) -> dict:
    r = client.post(f"{B}/api/auth/demo-login", json={"user_id": user_id, "access_code": settings.demo_access_code}, headers=ADMIN)
    r.raise_for_status()
    return {"Authorization": f"Bearer {r.json()['token']}", **ADMIN}


def sse(resp) -> list[tuple[str, dict]]:
    events, ev = [], None
    for line in resp.iter_lines():
        if line.startswith("event:"):
            ev = line[6:].strip()
        elif line.startswith("data:"):
            events.append((ev, json.loads(line[5:].strip())))
    return events


def chat(client, headers, message):
    with client.stream("POST", f"{B}/api/chat", headers=headers, json={"message": message}) as resp:
        ev = sse(resp)
    return [d for e, d in ev if e == "done"][0], ev


def confirm(client, headers, approved, note=None):
    with client.stream("POST", f"{B}/api/chat/confirm", headers=headers, json={"approved": approved, "note": note}) as resp:
        ev = sse(resp)
    return [d for e, d in ev if e == "done"][0]


def test_wrong_access_code_and_bad_token():
    with httpx.Client(timeout=30) as c:
        if settings.demo_access_code:
            assert c.post(f"{B}/api/auth/demo-login", json={"user_id": "u-fatima", "access_code": "nope"}).status_code == 403
        else:  # no code configured (the demo's default): any code is accepted, an unknown persona is not
            assert c.post(f"{B}/api/auth/demo-login", json={"user_id": "u-fatima", "access_code": "nope"}).status_code == 200
            assert c.post(f"{B}/api/auth/demo-login", json={"user_id": "u-nobody"}).status_code == 404
        assert c.get(f"{B}/api/bookings", headers={"Authorization": "Bearer junk"}).status_code == 401


def test_customer_cancel_flow_with_decline_then_approve():
    with httpx.Client(timeout=60) as c:
        H = login(c, "u-fatima")
        b = c.get(f"{B}/api/bookings", headers=H).json()["bookings"]
        assert b[0]["pnr"] == "K3P7QZ" and b[0]["status"] == "confirmed"
        done, _ = chat(c, H, "I want to cancel my Cairo flight")
        assert [t["tool"] for t in done["what_i_checked"]] == ["list_my_bookings", "quote_cancellation"]
        assert "1291.00" in done["message"] and done["pending"] is None
        done, ev = chat(c, H, "yes")
        assert done["pending"] and done["pending"]["actions"][0]["tool"] == "cancel_booking"
        assert done["pending"]["actions"][0]["quote"]["refund"] == "1291.00"  # the confirmation card carries the quoted amount
        assert any(e == "confirm" for e, _ in ev)
        done = confirm(c, H, False, "changed my mind")
        assert any(t.get("declined") for t in done["what_i_checked"])
        assert c.get(f"{B}/api/bookings", headers=H).json()["bookings"][0]["status"] == "confirmed"  # nothing changed
        done, _ = chat(c, H, "ok yes cancel it")
        assert done["pending"]
        done = confirm(c, H, True)
        assert "cancelled" in done["message"].lower() and [t["tool"] for t in done["what_i_checked"]][-1] == "cancel_booking"
        bk = c.get(f"{B}/api/bookings", headers=H).json()["bookings"][0]
        assert bk["status"] == "cancelled" and bk["refund_amount"] == "1291.00"
        h = c.get(f"{B}/api/chat/history", headers=H).json()
        assert len(h["messages"]) >= 6 and h["pending"] is None


def test_language_switch_mid_conversation():
    with httpx.Client(timeout=60) as c:
        H = login(c, "u-omar")
        done, _ = chat(c, H, "Show my bookings")
        assert done["lang"] == "en" and done["what_i_checked"][0]["tool"] == "list_my_bookings" and done["message"].startswith("Here are")
        done, _ = chat(c, H, "اعرض حجوزاتي")
        assert done["lang"] == "ar" and done["message"].startswith("هذه حجوزاتك")
        done, _ = chat(c, H, "ok and my booking B4N6WX cancel to credit")
        assert done["lang"] == "en"


def test_past_deadline_persona_gets_a_logged_case():
    with httpx.Client(timeout=60) as c:
        H = login(c, "u-khalid")
        done, _ = chat(c, H, "ألغي حجزي من فضلك")
        assert done["lang"] == "ar" and "لا يمكن" in done["message"] and "استثناء" in done["message"]
        done, _ = chat(c, H, "طلب استثناء: حالة طبية طارئة")
        assert done["pending"] and done["pending"]["actions"][0]["tool"] == "request_exception"
        done = confirm(c, H, True)
        assert "#" in done["message"] and "البريد" in done["message"]  # a case number and the email promise
        # staff endpoints no longer exist (the SPA fallback answers with the web page, not JSON): the case is decided outside the agent
        for path in ("/api/supervisor/requests", "/api/ops/disruptions"):
            r = c.get(f"{B}{path}", headers=H)
            assert r.status_code in (404, 405) or "text/html" in r.headers.get("content-type", "")
        personas = c.get(f"{B}/api/personas").json()["personas"]
        assert {p["role"] for p in personas} == {"customer"}


def test_airline_cancelled_flight_in_the_seed_gives_a_full_refund():
    with httpx.Client(timeout=90) as c:
        M = login(c, "u-mariam")
        bk = c.get(f"{B}/api/bookings", headers=M).json()["bookings"][0]
        assert bk["segments"][0]["live_status"] == "cancelled"
        done, _ = chat(c, M, "cancel my booking, refund to my card")
        assert "0.00" in done["message"] and "965.00" in done["message"]


def test_admin_reset_and_metrics():
    with httpx.Client(timeout=120) as c:
        assert c.post(f"{B}/api/admin/reset").status_code == 403
        r = c.post(f"{B}/api/admin/reset", headers={"X-Admin-Token": settings.admin_token}).json()
        assert r["ok"]
        m = c.get(f"{B}/api/metrics/summary", headers={"X-Admin-Token": settings.admin_token}).json()
        assert "turns" in m and isinstance(m["tools"], list)
