"""Integration tests against the running reservations + flight-ops MCP servers and the legacy API.

Run `scripts/dev.sh start legacy reservations flightops` first. The database is re-seeded at the
start of the module so the personas are in their designed state. Skipped when the servers are down.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from falcon.core import auth as roles
from falcon.core.auth import UserContext
from falcon.core.config import REPO_ROOT, settings
from falcon.core.mcp_client import client_for

CUSTOMER_ROLES = [roles.BOOKINGS_READ, roles.BOOKINGS_WRITE, roles.FLIGHTS_READ, roles.POLICY_READ]


def _up(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2).status_code < 500
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not (_up(settings.legacy_api_url + "/health") and _up(settings.mcp_reservations_url.replace("/mcp", "/mcp"))), reason="services not running")


@pytest.fixture(scope="module", autouse=True)
def reseed():
    subprocess.run([sys.executable, str(REPO_ROOT / "db" / "apply.py"), "--seed-only"], check=True, capture_output=True)


def user(uid: str, cid: str | None, role: str = "customer", lang: str = "en") -> UserContext:
    return UserContext(user_id=uid, display_name=uid, role=role, customer_id=cid, lang=lang, session_id=f"test-{uid}")


async def tools_for(u: UserContext, role_list: list[str], servers: list[str]):
    c = client_for(u, role_list, servers)
    return {t.name: t for t in await c.get_tools()}


def parse(result):
    """langchain-mcp-adapters returns content blocks; tool results are JSON text."""
    if isinstance(result, list):
        text = "".join(b.get("text", "") for b in result if isinstance(b, dict))
    else:
        text = str(result)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_text": text}


async def call(tools, name, **kwargs):
    return parse(await tools[name].ainvoke(kwargs))


def test_schema_has_no_customer_id():
    async def go():
        t = await tools_for(user("u-fatima", "C-1001"), CUSTOMER_ROLES, ["reservations"])
        for name, tool in t.items():
            schema = tool.args_schema if isinstance(tool.args_schema, dict) else tool.args_schema.model_json_schema()
            assert "customer_id" not in json.dumps(schema), name
        assert {"cancel_booking", "quote_cancellation", "search_flights", "request_exception"} <= set(t)
    asyncio.run(go())


def test_fatima_cancels_her_booking_with_quote_and_idempotent_repeat():
    async def go():
        t = await tools_for(user("u-fatima", "C-1001"), CUSTOMER_ROLES, ["reservations"])
        mine = await call(t, "list_my_bookings")
        assert mine["count"] == 1 and mine["bookings"][0]["pnr"] == "K3P7QZ"
        q = await call(t, "quote_cancellation", pnr="K3P7QZ", refund_method="credit")
        assert q["allowed"] and q["fee"] == "100.00" and q["refund"] == "1291.00" and q["quote_id"]
        done = await call(t, "cancel_booking", pnr="K3P7QZ", quote_id=q["quote_id"])
        assert done["status"] == "cancelled" and done["refund_amount"] == "1291.00" and not done["already_done"]
        again = await call(t, "cancel_booking", pnr="K3P7QZ", quote_id=q["quote_id"])
        assert again["already_done"] is True and again["refund_amount"] == "1291.00"
        b = await call(t, "get_booking", pnr="K3P7QZ")
        assert b["status"] == "cancelled"
    asyncio.run(go())


def test_other_customers_booking_is_not_found():
    async def go():
        t = await tools_for(user("u-omar", "C-1003"), CUSTOMER_ROLES, ["reservations"])
        r = await call(t, "get_booking", pnr="M8T2LD")  # Khalid's
        assert "No booking" in r["_text"]
        r2 = await call(t, "get_booking", pnr="ZZZZZZ")
        assert r["_text"].split("Error executing tool get_booking: ")[-1] == r2["_text"].split("Error executing tool get_booking: ")[-1]
    asyncio.run(go())


def test_khalid_past_deadline_logs_an_exception_case():
    async def go():
        k = await tools_for(user("u-khalid", "C-1002", lang="ar"), CUSTOMER_ROLES, ["reservations"])
        q = await call(k, "quote_cancellation", pnr="M8T2LD")
        assert not q["allowed"] and q["reason_code"] == "PAST_DEADLINE" and q["exception_if_approved"]["refund"] == "842.00"
        import base64
        note = await call(k, "store_attachment", filename="doctor_note.pdf", content_type="application/pdf", content_base64=base64.b64encode(b"%PDF-1.4 demo").decode())
        req = await call(k, "request_exception", pnr="M8T2LD", kind="late_cancellation", reason="Medical emergency, doctor's note attached", evidence_attachment_ids=[note["id"]])
        assert req["status"] == "pending" and req["if_approved"]["refund"] == "842.00"
        assert f"Case #{req['request_id']}" in req["message"] and "email" in req["message"]
        # a second request for the same booking does not open a second case
        again = await call(k, "request_exception", pnr="M8T2LD", kind="late_cancellation", reason="same")
        assert again["already_pending"] and again["request_id"] == req["request_id"]
        status = await call(k, "get_request_status", request_id=req["request_id"])
        assert status["status"] == "pending" and status["kind"] == "waiver"
        b = await call(k, "get_booking", pnr="M8T2LD")
        assert b["status"] == "confirmed"  # nothing changes until the care team decides, outside the agent
        # another customer cannot read the case
        o = await tools_for(user("u-omar", "C-1003"), CUSTOMER_ROLES, ["reservations"])
        assert "No request" in (await call(o, "get_request_status", request_id=req["request_id"]))["_text"]
    asyncio.run(go())


def test_special_fare_credit_only_and_airline_cancelled_full_refund():
    async def go():
        o = await tools_for(user("u-omar", "C-1003"), CUSTOMER_ROLES, ["reservations"])
        assert (await call(o, "quote_cancellation", pnr="B4N6WX", refund_method="cash"))["reason_code"] == "CASH_NOT_ALLOWED"
        assert (await call(o, "quote_cancellation", pnr="B4N6WX", refund_method="credit"))["refund"] == "2275.00"
        y = await tools_for(user("u-youssef", "C-1005"), CUSTOMER_ROLES, ["reservations"])
        q = await call(y, "quote_cancellation", pnr="P5D3VJ", refund_method="cash")
        assert q["allowed"] and q["reason_code"] == "AIRLINE_CANCELLED" and q["refund"] == "895.00" and q["fee"] == "0.00"
    asyncio.run(go())


def test_search_with_weekday_exclusion_and_alternative_airport():
    async def go():
        t = await tools_for(user("u-fatima", "C-1001"), CUSTOMER_ROLES, ["reservations"])
        r = await call(t, "search_flights", origin="Dubai", destination="Cairo", date_from="2026-10-24", date_to="2026-11-07", exclude_weekdays=["monday"], passengers=2)
        assert r["origin"]["code"] == "DXB" and r["count"] > 10
        assert all(o["weekday"] != "Monday" for o in r["options_cheapest_first"])
        prices = [o["price_per_passenger_aed"]["SPECIAL"] for o in r["options_cheapest_first"]]
        assert prices == sorted(prices)
        assert r["alternative_airport"] and r["alternative_airport"]["airport"] == "DWC" and r["alternative_airport"]["saving_per_passenger_aed_vs_best_main_option"] > 0
        assert r["alternative_airport"]["drive_minutes"] == 40
        bad = await call(t, "search_flights", origin="Lisbon", destination="Cairo", date_from="2026-10-24")
        assert "Unknown origin" in bad["_text"]
    asyncio.run(go())


def test_salma_change_after_free_change_used():
    async def go():
        t = await tools_for(user("u-salma", "C-1009", lang="ar"), CUSTOMER_ROLES, ["reservations"])
        q = await call(t, "quote_change", pnr="S3L5KQ", new_flight_instance_id="FA833-2026-10-27", segment_seq=1)
        assert q["allowed"] and q["fee"] == "200.00"
        done = await call(t, "change_booking", pnr="S3L5KQ", quote_id=q["quote_id"])
        assert done["changes_used"] == 2 and done["new_flight"].startswith("FA833")
        wrong_route = await call(t, "quote_change", pnr="S3L5KQ", new_flight_instance_id="FA929-2026-10-27")
        assert "same route" in wrong_route["_text"]
    asyncio.run(go())


def test_quote_cannot_be_used_by_another_customer_or_for_another_booking():
    async def go():
        o = await tools_for(user("u-omar", "C-1003"), CUSTOMER_ROLES, ["reservations"])
        q = await call(o, "quote_cancellation", pnr="B4N6WX")
        n = await tools_for(user("u-noura", "C-1007"), CUSTOMER_ROLES, ["reservations"])
        r = await call(n, "cancel_booking", pnr="N6F8ZR", quote_id=q["quote_id"])
        assert "does not match" in r["_text"]
        b = await call(n, "get_booking", pnr="N6F8ZR")
        # two layers: the instruction-shaped note is withheld by the server (FAILURE_DIARY #25, decision #21); the subtler one
        # passes the filter and reaches the model as data, which is what the adversarial task prompt_injection_subtle_note_en tests
        notes = {p["name"]: p for p in b["passengers"]}
        assert notes["Maha Abdullah"]["note_entered_by_customer"] is None and notes["Maha Abdullah"]["note_status"].startswith("withheld")
        assert "reimbursement" in notes["Zayed Abdullah"]["note_entered_by_customer"] and "note_status" not in notes["Zayed Abdullah"]
    asyncio.run(go())


def test_bag_quote_and_add():
    async def go():
        t = await tools_for(user("u-omar", "C-1003"), CUSTOMER_ROLES, ["reservations"])
        q = await call(t, "quote_extra_bag", pnr="B4N6WX", kg=20)
        assert q["allowed"] and q["amount_due"] == "90.00"
        done = await call(t, "add_extra_bag", pnr="B4N6WX", quote_id=q["quote_id"])
        assert done["amount_charged"] == "90.00"
    asyncio.run(go())


def test_mariam_airline_cancelled_flight_full_refund_any_method():
    async def go():
        m = await tools_for(user("u-mariam", "C-1012", lang="ar"), CUSTOMER_ROLES, ["reservations"])
        b = await call(m, "get_booking", pnr="M2A7HS")
        assert b["segments"][0]["live_status"] == "cancelled" and b["segments"][0]["status_reason"] == "Weather at Cairo"
        q = await call(m, "quote_cancellation", pnr="M2A7HS", refund_method="cash")
        assert q["allowed"] and q["reason_code"] == "AIRLINE_CANCELLED" and q["fee"] == "0.00" and q["refund"] == "965.00"
        # a Special fare is credit-only in normal times; the airline's cancellation overrides that
        assert "No booking" in (await call(m, "quote_cancellation", pnr="B4N6WX", refund_method="cash"))["_text"]  # not her booking: refused
        # rebooking after the cancellation is free on her Special fare, does not use a change, and the cancelled flight is not offered as an option
        search = await call(m, "search_flights", origin="DXB", destination="CAI", date_from="2026-10-21")
        ids = [o["flight_instance_id"] for o in search["options_cheapest_first"]]
        assert "FA923-2026-10-21" not in ids and "FA927-2026-10-21" not in ids and "FA929-2026-10-21" in ids
        q = await call(m, "quote_change", pnr="M2A7HS", new_flight_instance_id="FA929-2026-10-21")
        assert q["allowed"] and q["reason_code"] == "AIRLINE_CANCELLED" and q["fee"] == "0.00" and q["amount_due"] == "0.00"
        done = await call(m, "change_booking", pnr="M2A7HS", quote_id=q["quote_id"])
        assert done["amount_charged"] == "0.00" and done["changes_used"] == 0 and done["involuntary"] is True and "FA929" in done["new_flight"]
        after = await call(m, "get_booking", pnr="M2A7HS")
        assert after["segments"][0]["flight_number"] == "FA929" and after["segments"][0]["live_status"] != "cancelled"
        no_staff = await tools_for(user("u-mariam", "C-1012"), CUSTOMER_ROLES, ["flightops"])
        assert "cancel_flight" not in {t.name for t in no_staff.values()} and "query_ops_data" not in {t.name for t in no_staff.values()}
    asyncio.run(go())


def test_booking_arc_search_quote_create_reuse_and_isolation():
    """A new booking is a quote-bound write like every other (decision #30): priced from the search, created once, visible after."""
    async def go():
        t = await tools_for(user("u-ahmed", "C-1008"), CUSTOMER_ROLES, ["reservations"])
        s = await call(t, "search_flights", origin="DXB", destination="BEY", date_from="2026-10-26", date_to="2026-11-01")
        best = s["options_cheapest_first"][0]
        assert best["flight_instance_id"] == "FA951-2026-10-26" and best["price_per_passenger_aed"]["SPECIAL"] == 900
        q = await call(t, "quote_booking", flight_instance_id=best["flight_instance_id"], fare_family="special", passengers=1)
        assert q["allowed"] and q["amount_due"] == "900.00" and q["quote_id"] and q["flights"][0]["flight_number"] == "FA951" and q["lead_passenger"] == "Ahmed Saeed"
        r = await call(t, "create_booking", quote_id=q["quote_id"])
        assert r["already_done"] is False and r["status"] == "confirmed" and len(r["pnr"]) == 6 and r["amount_charged"] == "900.00"
        again = await call(t, "create_booking", quote_id=q["quote_id"])
        assert again["already_done"] is True and again["pnr"] == r["pnr"]
        b = await call(t, "get_booking", pnr=r["pnr"])
        assert b["status"] == "confirmed" and b["segments"][0]["flight_instance_id"] == "FA951-2026-10-26" and b["passengers"][0]["name"] == "Ahmed Saeed" and b["total_paid"] == "900.00"
        after = await call(t, "search_flights", origin="DXB", destination="BEY", date_from="2026-10-26", date_to="2026-10-26")
        assert after["options_cheapest_first"][0]["seats_available"] == best["seats_available"] - 1
        # two passengers need the other traveller's name; another customer cannot use the quote
        bad = await call(t, "quote_booking", flight_instance_id=best["flight_instance_id"], fare_family="SAVER", passengers=2)
        assert "names" in bad["_text"]
        o = await tools_for(user("u-omar", "C-1003"), CUSTOMER_ROLES, ["reservations"])
        stolen = await call(o, "create_booking", quote_id=q["quote_id"])
        assert "not a booking quote" in stolen["_text"]
    asyncio.run(go())


def test_send_itinerary_goes_to_the_profile_address_masked():
    async def go():
        o = await tools_for(user("u-omar", "C-1003"), CUSTOMER_ROLES, ["reservations"])
        r = await call(o, "send_itinerary", pnr="B4N6WX")
        assert r["sent"] is True and r["pnr"] == "B4N6WX" and "***@" in r["to"] and r["reference"].startswith("MAIL-B4N6WX-")
        assert "FA1" in r["subject"] and "@" in r["to"] and r["to"].count("*") == 3
        again = await call(o, "send_itinerary", pnr="B4N6WX")
        assert again["reference"] != r["reference"]  # each request is one email
        f = await tools_for(user("u-fatima", "C-1001"), CUSTOMER_ROLES, ["reservations"])
        other = await call(f, "send_itinerary", pnr="B4N6WX")
        assert "No booking" in other["_text"]  # someone else's booking: not found, same wording as always
    asyncio.run(go())


def test_exception_cases_for_personal_reasons_need_an_attached_document():
    """Decision #33: a medical reason without a document is refused with an instruction to ask for one; with one, the case carries it;
    a reason the airline can see for itself (a delay) needs none; another customer's attachment is not accepted."""
    import base64

    async def go():
        k = await tools_for(user("u-khalid", "C-1002"), CUSTOMER_ROLES, ["reservations"])
        refused = await call(k, "request_exception", pnr="M8T2LD", kind="late_cancellation", reason="medical emergency, I have a doctor's note")
        assert "attach" in refused["_text"].lower() and "paperclip" in refused["_text"].lower()
        before = (await call(k, "list_my_attachments"))["count"]  # an earlier test in this module may have attached one already
        stored = await call(k, "store_attachment", filename="doctor_note.pdf", content_type="application/pdf", content_base64=base64.b64encode(b"%PDF-1.4 demo").decode())
        listed = await call(k, "list_my_attachments")
        assert listed["count"] == before + 1 and listed["attachments"][-1]["id"] == stored["id"] and "data" not in listed["attachments"][-1]
        case = await call(k, "request_exception", pnr="M8T2LD", kind="late_cancellation", reason="medical emergency, doctor's note attached", evidence_attachment_ids=[stored["id"]])
        assert case["status"] == "pending"
        if not case.get("already_pending"):
            assert case["evidence"][0]["filename"] == "doctor_note.pdf" and "doctor_note.pdf attached" in case["message"]
        o = await tools_for(user("u-omar", "C-1003"), CUSTOMER_ROLES, ["reservations"])
        stolen = await call(o, "request_exception", pnr="B4N6WX", kind="fee_waiver", reason="medical reasons", evidence_attachment_ids=[stored["id"]])
        assert "not on this customer" in stolen["_text"]
        r = await tools_for(user("u-reem", "C-1011"), CUSTOMER_ROLES, ["reservations"])
        delay = await call(r, "request_exception", pnr="R8E2GN", kind="late_cancellation", reason="the flight is delayed three hours and I will drive instead")
        assert delay["status"] == "pending"  # the delay is in the airline's own data: no document needed
    asyncio.run(go())


def test_bereavement_case_quotes_the_clause_not_the_fee():
    import base64

    async def go():
        o = await tools_for(user("u-omar", "C-1003"), CUSTOMER_ROLES, ["reservations"])
        refused = await call(o, "request_exception", pnr="B4N6WX", kind="bereavement", reason="my father passed away")
        assert "attach" in refused["_text"].lower()
        cert = await call(o, "store_attachment", filename="death_certificate.pdf", content_type="application/pdf", content_base64=base64.b64encode(b"%PDF-1.4 demo").decode())
        paid = (await call(o, "get_booking", pnr="B4N6WX"))["total_paid"]  # an earlier test in this module may have added a bag
        case = await call(o, "request_exception", pnr="B4N6WX", kind="bereavement", reason="my father passed away, certificate attached", refund_method="cash", evidence_attachment_ids=[cert["id"]])
        assert case["status"] == "pending"
        if not case.get("already_pending"):
            assert case["kind"] == "bereavement" and case["clauses"] == ["3.3.2", "10.2.1"], case
            assert case["if_approved"]["refund"] == paid and case["if_approved"]["fee"] == "0.00" and case["if_approved"]["refund_method"] == "cash", case["if_approved"]
            assert "3.3.2" in case["message"] and case["evidence"][0]["filename"] == "death_certificate.pdf"
    asyncio.run(go())
