"""The customer persona: what the agent may do for the signed-in customer, fixed in code, never by the model.

1. the service roles minted into the bearer token (what the MCP servers will allow),
2. the MCP servers the session connects to,
3. the tool allow-list (what the model is shown at all), narrowed per topic.
The servers enforce 1 whatever the model asks; the allow-list keeps the model's menu honest. Staff work (deciding exception
cases, disrupting flights) happens outside the agent: cases are rows in `reservations.approval_requests`, disruptions are
events in the flight-ops system (decision #25).
"""

from __future__ import annotations

from falcon.core import auth as roles
from falcon.core.db import Database

SERVICE_ROLES = [roles.BOOKINGS_READ, roles.BOOKINGS_WRITE, roles.FLIGHTS_READ, roles.POLICY_READ]
SERVERS = ["reservations", "flightops", "policy"]
CUSTOMER_TOOLS = {"get_my_profile", "list_my_bookings", "get_booking", "get_fare_rules", "search_flights", "quote_booking", "quote_cancellation", "quote_change",
                  "quote_extra_bag", "create_booking", "cancel_booking", "change_booking", "add_extra_bag", "request_exception", "send_itinerary", "handover_to_human",
                  "get_request_status", "get_flight_status", "list_disruptions", "search_policy", "get_clause", "list_articles", "get_article", "get_glossary",
                  "check_travel_eligibility"}

# The agent is bound to the tools its topic needs, not to all 25 on every call (decision #46). COMMON is on every menu: the
# document floor, the eligibility check and the way out to a human must never be missing. An unknown topic gets everything.
COMMON_TOOLS = {"search_policy", "check_travel_eligibility", "handover_to_human", "get_request_status", "list_my_bookings", "get_booking"}
TOPIC_TOOLS: dict[str, set[str]] = {
    "bookings": {"get_my_profile", "get_flight_status", "list_disruptions", "send_itinerary", "get_fare_rules"},
    "cancel_refund": {"get_fare_rules", "quote_cancellation", "cancel_booking", "request_exception", "get_flight_status", "get_clause"},
    # A change or a new booking often comes with a bag in the same message ("move me to Tuesday and add 20 kg"): the guard names one
    # topic, so the bag tools ride on both menus, or the agent says it cannot add baggage (FAILURE_DIARY #80).
    "change": {"get_fare_rules", "search_flights", "quote_change", "change_booking", "get_flight_status", "quote_extra_bag", "add_extra_bag"},
    "baggage": {"get_fare_rules", "quote_extra_bag", "add_extra_bag", "get_clause"},
    "book_new": {"get_fare_rules", "search_flights", "quote_booking", "create_booking", "send_itinerary", "quote_extra_bag", "add_extra_bag"},
    "disruption": {"get_flight_status", "list_disruptions", "search_flights", "quote_change", "change_booking", "quote_cancellation", "cancel_booking", "get_fare_rules"},
    "policy": {"get_clause", "get_article", "list_articles", "get_glossary", "get_fare_rules", "request_exception", "quote_cancellation"},
    "exception": {"request_exception", "quote_cancellation", "get_fare_rules", "get_clause"},
}


def tools_for_topic(topic: str | None, tools: list) -> list:
    """The subset of the bound tools a topic needs; everything when the topic is unknown."""
    allowed = TOPIC_TOOLS.get(topic or "")
    if allowed is None:
        return list(tools)
    keep = allowed | COMMON_TOOLS
    return [t for t in tools if t.name in keep]


async def list_personas(db: Database) -> list[dict]:
    rows = await db.fetch_all("SELECT user_id, display_name, display_name_ar, customer_id, lang, persona_note_en, persona_note_ar FROM agent.users WHERE shown ORDER BY sort_order")
    return [dict(r) for r in rows]


async def get_persona(db: Database, user_id: str) -> dict | None:
    return await db.fetch_one("SELECT user_id, display_name, display_name_ar, customer_id, lang, persona_note_en, persona_note_ar FROM agent.users WHERE user_id = %s", (user_id,))
