"""Personas: who can sign in to the demo, and what the agent may do for them.

One persona, the customer (decision #25). Three things are fixed per persona, and the model is not involved in any of them:
1. the service roles minted into the bearer token (what the MCP servers will allow),
2. the MCP servers the session connects to,
3. the tool allow-list (what the model is shown at all).
The servers enforce 1 regardless of 3; the allow-list just keeps the model's menu honest. Staff work (deciding
exception cases, disrupting flights) happens outside the agent: cases are rows in `reservations.requests`,
disruptions are events in the flight-ops system.
"""

from __future__ import annotations

from falcon.core import auth as roles
from falcon.core.db import Database

ROLE_PROFILES = {
    "customer": {
        "service_roles": [roles.BOOKINGS_READ, roles.BOOKINGS_WRITE, roles.FLIGHTS_READ, roles.POLICY_READ],
        "servers": ["reservations", "flightops", "policy"],
        "tools": {"get_my_profile", "list_my_bookings", "get_booking", "get_fare_rules", "search_flights", "quote_booking", "quote_cancellation", "quote_change", "quote_extra_bag",
                  "create_booking", "cancel_booking", "change_booking", "add_extra_bag", "request_exception", "send_itinerary", "list_my_attachments", "handover_to_human", "get_request_status",
                  "get_flight_status", "list_disruptions", "search_policy", "get_clause", "list_articles", "get_article", "get_glossary", "check_travel_eligibility"},
    },
}

# Decision #46: the agent is bound to the tools its topic needs, not to all 26 on every call. COMMON is on every menu (the document floor,
# the eligibility check and the way out to a human must never be missing); an unknown topic gets everything.
COMMON_TOOLS = {"search_policy", "check_travel_eligibility", "handover_to_human", "get_request_status", "list_my_bookings", "get_booking"}
TOPIC_TOOLS: dict[str, set[str]] = {
    "bookings": {"get_my_profile", "get_flight_status", "list_disruptions", "send_itinerary", "get_fare_rules"},
    "cancel_refund": {"get_fare_rules", "quote_cancellation", "cancel_booking", "list_my_attachments", "request_exception", "get_flight_status", "get_clause"},
    "change": {"get_fare_rules", "search_flights", "quote_change", "change_booking", "get_flight_status"},
    "baggage": {"get_fare_rules", "quote_extra_bag", "add_extra_bag", "get_clause"},
    "book_new": {"get_fare_rules", "search_flights", "quote_booking", "create_booking", "send_itinerary"},
    "disruption": {"get_flight_status", "list_disruptions", "search_flights", "quote_change", "change_booking", "quote_cancellation", "cancel_booking", "get_fare_rules"},
    "policy": {"get_clause", "get_article", "list_articles", "get_glossary", "get_fare_rules", "list_my_attachments", "request_exception", "quote_cancellation"},
    "exception": {"list_my_attachments", "request_exception", "quote_cancellation", "get_fare_rules", "get_clause"},
}


def tools_for_topic(topic: str | None, tools: list) -> list:
    """The subset of the bound tools a topic needs; everything when the topic is unknown."""
    allowed = TOPIC_TOOLS.get(topic or "")
    if allowed is None:
        return list(tools)
    keep = allowed | COMMON_TOOLS
    return [t for t in tools if t.name in keep]


# Tools the app calls on the customer's behalf (the upload endpoint) that the model never sees on its menu (decision #33).
API_TOOLS = {"store_attachment"}

# Tools that change the world: the runtime interrupts before these and asks the signed-in person to confirm.
WRITE_TOOLS = {"create_booking", "cancel_booking", "change_booking", "add_extra_bag", "request_exception"}


async def list_personas(db: Database) -> list[dict]:
    rows = await db.fetch_all("SELECT user_id, display_name, display_name_ar, role, customer_id, lang, persona_note_en, persona_note_ar, sort_order FROM agent.users ORDER BY sort_order")
    return [dict(r) for r in rows]


async def get_persona(db: Database, user_id: str) -> dict | None:
    return await db.fetch_one("SELECT * FROM agent.users WHERE user_id = %s", (user_id,))
