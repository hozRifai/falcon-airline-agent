"""Data access for the reservations system (role res_rw). Every query that touches a booking
is filtered by the customer id that arrived in the signed user context, never by a value the
model supplied. A booking that belongs to someone else is simply "not found".
"""

from __future__ import annotations

import hashlib
import json
import secrets
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from falcon.core.arabic import normalize_arabic
from falcon.core.clock import GULF, now
from falcon.core.db import Database
from falcon.core.models import Booking, Extra, FareRules, FlightStatus, Override, Passenger, Quote, Segment
from falcon.core.rules import cancellation_quote, money

QUOTE_TTL_MINUTES = 10


def _wall() -> datetime:
    """Wall clock for quote expiry (a session concern), unlike the frozen business clock."""
    return datetime.now(tz=timezone.utc)


class ReservationsRepo:
    def __init__(self, db: Database):
        self.db = db
        self._fares: dict[str, FareRules] | None = None
        self._airports: list[dict] | None = None

    # ------------------------------------------------------------ reference data
    async def fare_rules(self) -> dict[str, FareRules]:
        if self._fares is None:
            rows = await self.db.fetch_all("SELECT * FROM reservations.fare_families ORDER BY sort_order")
            self._fares = {r["code"]: FareRules(**{k: v for k, v in r.items() if k != "sort_order"}) for r in rows}
        return self._fares

    async def airports(self) -> list[dict]:
        if self._airports is None:
            self._airports = await self.db.fetch_all("SELECT * FROM reservations.airports ORDER BY code")
        return self._airports

    async def resolve_airport(self, text: str) -> dict | None:
        """'DXB', 'Dubai', 'sharjah', 'القاهرة' -> airport row. Sharjah and Ajman map to DXB (the nearest Falcon airport)."""
        key = normalize_arabic(text.strip().lower())
        for a in await self.airports():
            if key.upper() == a["code"] or key == a["city_en"].lower() or key == normalize_arabic(a["city_ar"]) or key in [normalize_arabic(c) for c in a["serves_cities"]]:
                return a
        return None

    # ------------------------------------------------------------ customers and bookings
    async def customer(self, customer_id: str) -> dict | None:
        return await self.db.fetch_one("SELECT * FROM reservations.customers WHERE customer_id = %s", (customer_id,))

    async def list_bookings(self, customer_id: str) -> list[dict]:
        rows = await self.db.fetch_all(
            """SELECT b.pnr, b.status, b.fare_family, b.total_paid, b.currency, b.changes_used, b.created_at,
                      (SELECT count(*) FROM reservations.passengers p WHERE p.pnr = b.pnr) AS passengers,
                      (SELECT json_agg(json_build_object('seq', s.seq, 'flight_number', fi.flight_number, 'flight_instance_id', fi.id,
                               'origin', fi.origin, 'destination', fi.destination, 'dep_at', fi.dep_at, 'arr_at', fi.arr_at) ORDER BY s.seq)
                         FROM reservations.booking_segments s JOIN reservations.flight_instances fi ON fi.id = s.flight_instance_id WHERE s.pnr = b.pnr) AS segments
               FROM reservations.bookings b WHERE b.customer_id = %s ORDER BY b.created_at DESC""",
            (customer_id,),
        )
        return rows

    async def booking(self, pnr: str, customer_id: str) -> Booking | None:
        b = await self.db.fetch_one("SELECT * FROM reservations.bookings WHERE pnr = %s AND customer_id = %s", (pnr.upper().strip(), customer_id))
        if not b:
            return None
        segs = await self.db.fetch_all(
            """SELECT s.seq, s.flight_instance_id, s.fare_per_pax, fi.flight_number, fi.origin, fi.destination, fi.dep_at, fi.arr_at
               FROM reservations.booking_segments s JOIN reservations.flight_instances fi ON fi.id = s.flight_instance_id
               WHERE s.pnr = %s ORDER BY s.seq""", (b["pnr"],))
        pax = await self.db.fetch_all("SELECT id, first_name, last_name, pax_type, special_request FROM reservations.passengers WHERE pnr = %s ORDER BY id", (b["pnr"],))
        extras = await self.db.fetch_all("SELECT id, passenger_id, kind, kg, price FROM reservations.booking_extras WHERE pnr = %s ORDER BY id", (b["pnr"],))
        return Booking(
            pnr=b["pnr"], customer_id=b["customer_id"], status=b["status"], fare_family=b["fare_family"], currency=b["currency"],
            total_paid=b["total_paid"], changes_used=b["changes_used"], created_at=b["created_at"], cancelled_at=b["cancelled_at"],
            refund_method=b["refund_method"], refund_amount=b["refund_amount"],
            passengers=[Passenger(**p) for p in pax],
            segments=[Segment(seq=s["seq"], flight_instance_id=s["flight_instance_id"], flight_number=s["flight_number"], origin=s["origin"],
                              destination=s["destination"], dep_at=s["dep_at"], arr_at=s["arr_at"], fare_per_pax=s["fare_per_pax"]) for s in segs],
            extras=[Extra(**e) for e in extras],
        )

    async def transactions(self, pnr: str) -> list[dict]:
        return await self.db.fetch_all("SELECT kind, amount, currency, reference, created_at FROM reservations.transactions WHERE pnr = %s ORDER BY id", (pnr,))

    async def instance(self, instance_id: str) -> dict | None:
        return await self.db.fetch_one("SELECT * FROM reservations.flight_instances WHERE id = %s", (instance_id,))

    # ------------------------------------------------------------ search
    async def search(self, origin: str, destination: str, date_from: date, date_to: date, exclude_weekdays: list[int], passengers: int) -> list[dict]:
        rows = await self.db.fetch_all(
            """SELECT fi.*, (fi.seats_total - fi.seats_sold) AS seats_available, ao.tz AS org_tz, ad.tz AS dst_tz
               FROM reservations.flight_instances fi
               JOIN reservations.airports ao ON ao.code = fi.origin JOIN reservations.airports ad ON ad.code = fi.destination
               WHERE fi.origin = %s AND fi.destination = %s AND fi.service_date BETWEEN %s AND %s AND fi.dep_at > %s
               ORDER BY fi.base_price, fi.dep_at""",
            (origin, destination, date_from, date_to, now()),
        )
        fares = await self.fare_rules()
        out = []
        for r in rows:
            if r["service_date"].weekday() in exclude_weekdays or r["seats_available"] < passengers:
                continue
            prices = {code: int((r["base_price"] * f.price_multiplier).quantize(Decimal("1"))) for code, f in fares.items()}
            out.append({
                "flight_instance_id": r["id"], "flight_number": r["flight_number"], "date": r["service_date"].isoformat(),
                "weekday": r["service_date"].strftime("%A"), "origin": r["origin"], "destination": r["destination"],
                "departure_local": r["dep_at"].astimezone(ZoneInfo(r["org_tz"])).strftime("%Y-%m-%d %H:%M"),
                "arrival_local": r["arr_at"].astimezone(ZoneInfo(r["dst_tz"])).strftime("%Y-%m-%d %H:%M"),
                "duration_minutes": int((r["arr_at"] - r["dep_at"]).total_seconds() // 60),
                "seats_available": r["seats_available"], "price_per_passenger_aed": prices,
            })
        return out

    # ------------------------------------------------------------ quotes
    async def store_quote(self, quote: Quote, customer_id: str, request: dict[str, Any]) -> str:
        quote_id = "Q-" + uuid.uuid4().hex[:10].upper()
        await self.db.execute(
            "INSERT INTO reservations.quotes(quote_id,pnr,customer_id,kind,request,result,amount,created_at,expires_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (quote_id, quote.pnr or None, customer_id, quote.kind, json.dumps(request, default=str), quote.model_dump_json(),
             quote.refund if quote.kind == "cancel" else quote.amount_due, _wall(), _wall() + timedelta(minutes=QUOTE_TTL_MINUTES)),
        )
        return quote_id

    async def quote(self, quote_id: str, customer_id: str) -> dict | None:
        return await self.db.fetch_one("SELECT * FROM reservations.quotes WHERE quote_id = %s AND customer_id = %s", (quote_id, customer_id))

    # ------------------------------------------------------------ write actions (each one transaction)
    async def _claim(self, conn, quote_id: str) -> None:
        """First statement of every write: mark the quote used, only if nobody has. Postgres locks the row, so a second confirmation
        arriving at the same moment waits for the first to commit, then finds it used (FAILURE_DIARY #73: two simultaneous confirms
        both passed the earlier "not consumed" check and refunded twice)."""
        cur = await conn.execute("UPDATE reservations.quotes SET consumed_at=%s WHERE quote_id=%s AND consumed_at IS NULL", (_wall(), quote_id))
        if cur.rowcount != 1:
            raise QuoteAlreadyUsed(quote_id)

    async def execute_cancellation(self, booking: Booking, quote: Quote, quote_id: str | None, override: Override | None = None) -> dict:
        ts = now()
        pax = max(1, len([p for p in booking.passengers if p.pax_type != "infant"]))
        kind = "refund_credit" if quote.refund_method == "credit" else "refund_cash"
        reference = f"{'CR' if kind == 'refund_credit' else 'RF'}-{booking.pnr}-{uuid.uuid4().hex[:6].upper()}"
        result = {"pnr": booking.pnr, "status": "cancelled", "refund_amount": str(quote.refund), "refund_method": quote.refund_method,
                  "fee_charged": str(quote.fee), "reference": reference, "cancelled_at": ts.isoformat(), "currency": booking.currency,
                  "override": override.model_dump(exclude_none=True) if override else None}
        async with self.db.connection() as conn:
            async with conn.transaction():
                if quote_id:
                    await self._claim(conn, quote_id)
                await conn.execute(
                    "UPDATE reservations.bookings SET status='cancelled', cancelled_at=%s, cancel_reason=%s, refund_method=%s, refund_amount=%s WHERE pnr=%s AND status='confirmed'",
                    (ts, quote.reason_code, quote.refund_method, quote.refund, booking.pnr))
                await conn.execute("INSERT INTO reservations.transactions(pnr,kind,amount,currency,reference,created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                                   (booking.pnr, kind, quote.refund, booking.currency, reference, ts))
                if quote.fee > 0:
                    await conn.execute("INSERT INTO reservations.transactions(pnr,kind,amount,currency,reference,created_at) VALUES (%s,'fee',%s,%s,%s,%s)",
                                       (booking.pnr, quote.fee, booking.currency, f"FEE-{booking.pnr}", ts))
                for seg in booking.segments:
                    await conn.execute("UPDATE reservations.flight_instances SET seats_sold = greatest(0, seats_sold - %s) WHERE id = %s", (pax, seg.flight_instance_id))
                if quote_id:
                    await conn.execute("UPDATE reservations.quotes SET consumed_at=%s, consumed_result=%s WHERE quote_id=%s", (_wall(), json.dumps(result), quote_id))
        return result

    async def execute_change(self, booking: Booking, quote: Quote, quote_id: str) -> dict:
        ts = now()
        d = quote.details
        pax = max(1, len([p for p in booking.passengers if p.pax_type != "infant"]))
        seq, new_id = int(d["segment_seq"]), d["new_flight_instance_id"]
        old = next(s for s in booking.segments if s.seq == seq)
        new_inst = await self.instance(new_id)
        used = 0 if d.get("involuntary") else 1  # an airline disruption does not consume one of the fare's changes
        result = {"pnr": booking.pnr, "status": "confirmed", "changed_segment": seq, "old_flight": f"{old.flight_number} {old.dep_at.astimezone(GULF).isoformat(timespec='minutes')}",
                  "new_flight": f"{new_inst['flight_number']} {new_inst['dep_at'].astimezone(GULF).isoformat(timespec='minutes')}",
                  "amount_charged": str(quote.amount_due), "currency": booking.currency, "reference": f"CHG-{booking.pnr}-{uuid.uuid4().hex[:6].upper()}",
                  "changes_used": booking.changes_used + used, "involuntary": bool(d.get("involuntary"))}
        async with self.db.connection() as conn:
            async with conn.transaction():
                await self._claim(conn, quote_id)
                await conn.execute("UPDATE reservations.booking_segments SET flight_instance_id=%s, fare_per_pax=%s WHERE pnr=%s AND seq=%s",
                                   (new_id, money(d["new_fare_per_pax"]), booking.pnr, seq))
                await conn.execute("UPDATE reservations.bookings SET changes_used = changes_used + %s, total_paid = total_paid + %s WHERE pnr=%s", (used, quote.amount_due, booking.pnr))
                if quote.fee > 0:
                    await conn.execute("INSERT INTO reservations.transactions(pnr,kind,amount,currency,reference,created_at) VALUES (%s,'fee',%s,%s,%s,%s)", (booking.pnr, quote.fee, booking.currency, f"FEE-{booking.pnr}", ts))
                diff = money(quote.amount_due) - money(quote.fee)
                if diff > 0:
                    await conn.execute("INSERT INTO reservations.transactions(pnr,kind,amount,currency,reference,created_at) VALUES (%s,'payment',%s,%s,%s,%s)", (booking.pnr, diff, booking.currency, result["reference"], ts))
                await conn.execute("UPDATE reservations.flight_instances SET seats_sold = greatest(0, seats_sold - %s) WHERE id = %s", (pax, old.flight_instance_id))
                await conn.execute("UPDATE reservations.flight_instances SET seats_sold = seats_sold + %s WHERE id = %s", (pax, new_id))
                await conn.execute("UPDATE reservations.quotes SET consumed_at=%s, consumed_result=%s WHERE quote_id=%s", (_wall(), json.dumps(result), quote_id))
        return result

    PNR_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I: a reference is read aloud at the airport

    async def _new_pnr(self, conn) -> str:
        for _ in range(20):
            pnr = "".join(secrets.choice(self.PNR_ALPHABET) for _ in range(6))
            cur = await conn.execute("SELECT 1 FROM reservations.bookings WHERE pnr = %s", (pnr,))
            if await cur.fetchone() is None:
                return pnr
        raise RuntimeError("could not allocate a booking reference")

    async def execute_booking(self, customer: dict, quote: Quote, quote_id: str) -> dict:
        """Create the booking a quote priced: booking, segment, passengers, payment, the seats, the consumed quote; one transaction.
        Raises ValueError when the seats are gone between the quote and now (the transaction rolls back)."""
        ts = now()
        d = quote.details
        n, fare = int(d["passengers"]), money(d["fare_per_pax"])
        names = [f"{customer['first_name']} {customer['last_name']}"] + list(d.get("passenger_names") or [])
        async with self.db.connection() as conn:
            async with conn.transaction():
                await self._claim(conn, quote_id)
                pnr = await self._new_pnr(conn)
                taken = await conn.execute("UPDATE reservations.flight_instances SET seats_sold = seats_sold + %s WHERE id = %s AND seats_total - seats_sold >= %s",
                                           (n, d["flight_instance_id"], n))
                if taken.rowcount != 1:
                    raise ValueError("the seats are no longer available on this flight")
                await conn.execute("INSERT INTO reservations.bookings(pnr,customer_id,status,fare_family,currency,total_paid,changes_used,created_at) VALUES (%s,%s,'confirmed',%s,'AED',%s,0,%s)",
                                   (pnr, customer["customer_id"], d["fare_family"], quote.amount_due, ts))
                await conn.execute("INSERT INTO reservations.booking_segments(pnr,seq,flight_instance_id,fare_per_pax) VALUES (%s,1,%s,%s)", (pnr, d["flight_instance_id"], fare))
                for name in names[:n]:
                    first, _, last = name.strip().partition(" ")
                    await conn.execute("INSERT INTO reservations.passengers(pnr,first_name,last_name,pax_type) VALUES (%s,%s,%s,'adult')", (pnr, first, last or "-"))
                reference = f"PAY-{pnr}"
                await conn.execute("INSERT INTO reservations.transactions(pnr,kind,amount,currency,reference,created_at) VALUES (%s,'payment',%s,'AED',%s,%s)", (pnr, quote.amount_due, reference, ts))
                dep = datetime.fromisoformat(d["dep_at"]).astimezone(GULF)
                result = {"pnr": pnr, "status": "confirmed", "flight": f"{d['flight_number']} {d['origin']}-{d['destination']} {dep.isoformat(timespec='minutes')}",
                          "flight_instance_id": d["flight_instance_id"], "fare_family": d["fare_family"], "passengers": names[:n], "fare_per_passenger": str(fare),
                          "amount_charged": str(quote.amount_due), "currency": "AED", "payment": "card on file", "reference": reference, "created_at": ts.isoformat()}
                await conn.execute("UPDATE reservations.quotes SET consumed_at=%s, consumed_result=%s, pnr=%s WHERE quote_id=%s", (_wall(), json.dumps(result), pnr, quote_id))
        return result

    async def execute_bag(self, booking: Booking, quote: Quote, quote_id: str) -> dict:
        ts = now()
        d = quote.details
        result = {"pnr": booking.pnr, "added_kg": d["kg"], "passenger_ids": d["passenger_ids"], "amount_charged": str(quote.amount_due), "currency": booking.currency,
                  "reference": f"BAG-{booking.pnr}-{uuid.uuid4().hex[:6].upper()}"}
        async with self.db.connection() as conn:
            async with conn.transaction():
                await self._claim(conn, quote_id)
                for pid in d["passenger_ids"]:
                    await conn.execute("INSERT INTO reservations.booking_extras(pnr,passenger_id,kind,kg,price,created_at) VALUES (%s,%s,'bag',%s,%s,%s)",
                                       (booking.pnr, pid, d["kg"], money(quote.amount_due) / len(d["passenger_ids"]), ts))
                await conn.execute("UPDATE reservations.bookings SET total_paid = total_paid + %s WHERE pnr=%s", (quote.amount_due, booking.pnr))
                await conn.execute("INSERT INTO reservations.transactions(pnr,kind,amount,currency,reference,created_at) VALUES (%s,'payment',%s,%s,%s,%s)", (booking.pnr, quote.amount_due, booking.currency, result["reference"], ts))
                await conn.execute("UPDATE reservations.quotes SET consumed_at=%s, consumed_result=%s WHERE quote_id=%s", (_wall(), json.dumps(result), quote_id))
        return result

    # ------------------------------------------------------------ attachments (decision #33)
    async def store_attachment(self, customer_id: str, filename: str, content_type: str, data: bytes) -> dict:
        return await self.db.fetch_one(
            "INSERT INTO reservations.attachments(customer_id,filename,content_type,size_bytes,sha256,data,uploaded_at) VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id, filename, content_type, size_bytes, uploaded_at",
            (customer_id, filename, content_type, len(data), hashlib.sha256(data).hexdigest(), data, now()))

    async def attachments(self, customer_id: str, ids: list[int] | None = None) -> list[dict]:
        """The customer's attachments (name, type, size, time; never the bytes), all of them or the given ids that are theirs."""
        if ids is None:
            return await self.db.fetch_all("SELECT id, filename, content_type, size_bytes, uploaded_at FROM reservations.attachments WHERE customer_id = %s ORDER BY id", (customer_id,))
        if not ids:
            return []
        return await self.db.fetch_all("SELECT id, filename, content_type, size_bytes, uploaded_at FROM reservations.attachments WHERE customer_id = %s AND id = ANY(%s) ORDER BY id", (customer_id, ids))

    # ------------------------------------------------------------ email (decision #32)
    async def queue_email(self, customer_id: str, pnr: str | None, to_email: str, subject: str, body: str) -> dict:
        return await self.db.fetch_one(
            "INSERT INTO reservations.email_outbox(pnr,customer_id,to_email,subject,body,status,created_at) VALUES (%s,%s,%s,%s,%s,'queued',%s) RETURNING id, status, created_at",
            (pnr, customer_id, to_email, subject, body, now()))

    # ------------------------------------------------------------ approvals (human in the loop)
    async def create_request(self, customer_id: str, pnr: str | None, kind: str, reason: str, proposed: dict | None) -> dict:
        row = await self.db.fetch_one(
            "INSERT INTO reservations.approval_requests(pnr,customer_id,kind,reason,proposed,status,created_at) VALUES (%s,%s,%s,%s,%s,'pending',%s) RETURNING id, status, created_at",
            (pnr, customer_id, kind, reason, json.dumps(proposed, default=str) if proposed else None, now()))
        return row

    async def request(self, request_id: int, customer_id: str | None = None) -> dict | None:
        if customer_id:
            return await self.db.fetch_one("SELECT * FROM reservations.approval_requests WHERE id = %s AND customer_id = %s", (request_id, customer_id))
        return await self.db.fetch_one("SELECT * FROM reservations.approval_requests WHERE id = %s", (request_id,))

    async def pending_requests(self) -> list[dict]:
        return await self.db.fetch_all(
            """SELECT r.*, c.first_name || ' ' || c.last_name AS customer_name, c.preferred_lang
               FROM reservations.approval_requests r JOIN reservations.customers c ON c.customer_id = r.customer_id
               WHERE r.status = 'pending' ORDER BY r.created_at""")

    async def recent_requests(self, limit: int = 20) -> list[dict]:
        return await self.db.fetch_all(
            """SELECT r.*, c.first_name || ' ' || c.last_name AS customer_name FROM reservations.approval_requests r
               JOIN reservations.customers c ON c.customer_id = r.customer_id ORDER BY r.created_at DESC LIMIT %s""", (limit,))


class QuoteAlreadyUsed(Exception):
    """Another request consumed this quote first; the caller answers with that request's stored result."""


def quote_for_record(record: dict) -> Quote:
    return Quote.model_validate_json(record["result"]) if isinstance(record["result"], str) else Quote.model_validate(record["result"])


def recompute_cancellation(booking: Booking, rules: FareRules, statuses: dict[str, FlightStatus], request: dict) -> Quote:
    override = Override.model_validate(request["override"]) if request.get("override") else None
    return cancellation_quote(booking, rules, now(), statuses, request.get("refund_method", "credit"), override)
