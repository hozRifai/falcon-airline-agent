"""Client for the legacy flight API, shared by the flight-ops MCP server and the reservations server.

What a wrapper around an existing API has to do, and this one does:
- authenticate (API key from config, never from the model),
- time out (3 s) and retry transient failures (503/429/timeouts) with backoff,
- cache short-lived reads (statuses change rarely; quotes ask repeatedly),
- translate the vendor's terse field names into our own model (`FlightStatus`),
- fail closed: if the API cannot be reached, callers get an explicit error, not a guess.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime

import httpx

from .config import settings
from .models import FlightStatus

RETRY_STATUSES = {429, 502, 503, 504}


class LegacyFlightApiError(RuntimeError):
    pass


class LegacyFlightClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, *, timeout: float = 3.0, retries: int = 3, cache_ttl: float = 10.0):
        self.base_url = (base_url or settings.legacy_api_url).rstrip("/")
        self.api_key = api_key or settings.legacy_api_key
        self.timeout = timeout
        self.retries = retries
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, dict]] = {}
        self._client = httpx.AsyncClient(base_url=self.base_url, headers={"X-API-Key": self.api_key}, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = await self._client.request(method, path, **kwargs)
                if resp.status_code in RETRY_STATUSES:
                    last = LegacyFlightApiError(f"{resp.status_code} from legacy API")
                else:
                    if resp.status_code >= 400:
                        raise LegacyFlightApiError(f"legacy API {resp.status_code}: {resp.text[:200]}")
                    return resp.json()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
            await asyncio.sleep(0.25 * (2 ** attempt))
        raise LegacyFlightApiError(f"legacy API unreachable after {self.retries + 1} attempts: {last}")

    @staticmethod
    def _to_status(item: dict) -> FlightStatus:
        return FlightStatus(flight_instance_id=item["fid"], status=item["st"], delay_minutes=item["dly"], reason=item.get("rsn"),
                            updated_at=datetime.fromisoformat(item["upd"]))

    async def flight(self, fid: str) -> dict:
        cached = self._cache.get(fid)
        if cached and time.time() - cached[0] < self.cache_ttl:
            return cached[1]
        data = await self._request("GET", f"/v1/flights/{fid}")
        self._cache[fid] = (time.time(), data)
        return data

    async def statuses(self, fids: list[str]) -> dict[str, FlightStatus]:
        """Batch lookup used by the rules engine. Cache-aware: only misses hit the API."""
        out: dict[str, FlightStatus] = {}
        misses = []
        for fid in fids:
            cached = self._cache.get(fid)
            if cached and time.time() - cached[0] < self.cache_ttl:
                out[fid] = self._to_status(cached[1])
            else:
                misses.append(fid)
        if misses:
            data = await self._request("GET", "/v1/flights/status", params={"ids": ",".join(misses)})
            for item in data["items"]:
                self._cache[item["fid"]] = (time.time(), item)
                out[item["fid"]] = self._to_status(item)
        return out

    async def find(self, flight_number: str, service_date: str) -> list[dict]:
        data = await self._request("GET", "/v1/flights", params={"flight_number": flight_number, "dt": service_date})
        return data["items"]

    async def disruptions(self, service_date: str | None = None) -> list[dict]:
        items, page = [], 1
        while True:
            params = {"page": page, "page_size": 100}
            if service_date:
                params["dt"] = service_date
            data = await self._request("GET", "/v1/disruptions", params=params)
            items += data["items"]
            if len(data["items"]) < 100:
                return items
            page += 1
