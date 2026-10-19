"""Flight-ops MCP server: the agent's window into the legacy flight API.

Read-only tools (role Flights.Read): live flight status and current disruptions. Disruptions are events in the
operations system (seeded for the demo); the agent reads them and never writes them (decision #25: one persona).

Run:  python -m falcon.services.mcp_flightops.server   (port 8002)
"""

from __future__ import annotations

import logging
from datetime import date

import uvicorn
from mcp.server.fastmcp.exceptions import ToolError

from falcon.core import auth as roles
from falcon.core.clock import GULF, now
from datetime import datetime
from falcon.core.config import settings
from falcon.core.db import Database
from falcon.core.legacy_client import LegacyFlightApiError, LegacyFlightClient
from falcon.core.mcp_server import Audit, audited, build_server, require_role, wrap_app

log = logging.getLogger("falcon.flightops")
PORT = 8002

legacy = LegacyFlightClient()
audit = Audit("flightops", Database(settings.dsn_audit, max_size=2))


mcp = build_server(
    "falcon-flightops",
    audience=settings.entra_aud_flightops,
    public_url=settings.mcp_flightops_url,
    port=PORT,
    instructions="Live flight status and disruptions from Falcon's flight operations system.",
)


def _present(item: dict) -> dict:
    """Translate the legacy API's terse record into something the model can read."""
    local = lambda s: datetime.fromisoformat(s).astimezone(GULF).isoformat(timespec="minutes")  # noqa: E731
    return {
        "flight_instance_id": item["fid"], "flight_number": item["fno"], "date": item["dt"], "origin": item["org"], "destination": item["dst"],
        "scheduled_departure_gulf_time": local(item["std"]), "scheduled_arrival_gulf_time": local(item["sta"]), "status": item["st"],
        "delay_minutes": item["dly"], "reason": item.get("rsn"), "gate": item.get("gate"), "updated_at": local(item["upd"]),
    }


@mcp.tool()
@audited(audit, "get_flight_status")
async def get_flight_status(flight_number: str, date: str) -> dict:
    """Live status of a Falcon flight on a given date (YYYY-MM-DD): on time, delayed (with minutes and reason), cancelled, departed, landed.
    Use the flight number from the booking (for example FA929)."""
    require_role(roles.FLIGHTS_READ)
    try:
        items = await legacy.find(flight_number.strip().upper(), date)
    except LegacyFlightApiError as exc:
        raise ToolError(f"The flight operations system is not responding ({exc}). Tell the customer status is temporarily unavailable; do not guess.") from exc
    if not items:
        return {"found": False, "message": f"No flight {flight_number.upper()} on {date}."}
    return {"found": True, "flights": [_present(i) for i in items], "as_of": now().isoformat()}


@mcp.tool()
@audited(audit, "list_disruptions")
async def list_disruptions(date: str | None = None) -> dict:
    """Flights that are currently delayed or cancelled, optionally for one date (YYYY-MM-DD)."""
    require_role(roles.FLIGHTS_READ)
    try:
        items = await legacy.disruptions(date)
    except LegacyFlightApiError as exc:
        raise ToolError(f"The flight operations system is not responding ({exc}).") from exc
    return {"count": len(items), "flights": [_present(i) for i in items][:50]}


app = wrap_app(mcp)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
