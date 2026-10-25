"""Client for the legacy flight API, shared by the flight-ops MCP server and the reservations server.

What a wrapper around an existing API has to do: authenticate (API key from config, never from the model), time out (3 s),
retry transient failures with backoff, translate the vendor's terse field names into our own model, and fail closed: if the
API cannot be reached the caller gets an explicit error, not a guess.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import httpx

from .config import settings
from .models import FlightStatus

RETRY_STATUSES = {429, 502, 503, 504}


class LegacyFlightApiError(RuntimeError):
    pass


class LegacyFlightClient:
    def __init__(self, *, timeout: float = 3.0, retries: int = 3):
        self.retries = retries
        self._client = httpx.AsyncClient(base_url=settings.legacy_api_url.rstrip("/"), headers={"X-API-Key": settings.legacy_api_key}, timeout=timeout)

    async def _get(self, path: str, params: dict) -> dict:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = await self._client.get(path, params=params)
                if resp.status_code in RETRY_STATUSES:
                    last = LegacyFlightApiError(f"{resp.status_code} from legacy API")
                elif resp.status_code >= 400:
                    raise LegacyFlightApiError(f"legacy API {resp.status_code}: {resp.text[:200]}")
                else:
                    return resp.json()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
            await asyncio.sleep(0.25 * (2 ** attempt))
        raise LegacyFlightApiError(f"legacy API unreachable after {self.retries + 1} attempts: {last}")

    @staticmethod
    def _to_status(item: dict) -> FlightStatus:
        return FlightStatus(flight_instance_id=item["fid"], status=item["st"], delay_minutes=item["dly"], reason=item.get("rsn"), updated_at=datetime.fromisoformat(item["upd"]))

    async def statuses(self, fids: list[str]) -> dict[str, FlightStatus]:
        """Live status of several flights at once (the rules engine asks for every segment of a booking)."""
        if not fids:
            return {}
        data = await self._get("/v1/flights/status", {"ids": ",".join(fids)})
        return {item["fid"]: self._to_status(item) for item in data["items"]}

    async def find(self, flight_number: str, service_date: str) -> list[dict]:
        return (await self._get("/v1/flights", {"flight_number": flight_number, "dt": service_date}))["items"]

    async def disruptions(self, service_date: str | None = None) -> list[dict]:
        items, page = [], 1
        while True:
            data = await self._get("/v1/disruptions", {"page": page, "page_size": 100, **({"dt": service_date} if service_date else {})})
            items += data["items"]
            if len(data["items"]) < 100:
                return items
            page += 1
