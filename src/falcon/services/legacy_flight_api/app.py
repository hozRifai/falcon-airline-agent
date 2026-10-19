"""The "legacy" flight operations REST API.

This stands in for a system the airline already has (an ops control or flight-information
system from another vendor). It is deliberately plain: API-key auth, terse JSON, paginated
lists, its own rate limit, and an optional failure-injection knob so the MCP wrapper's
retries can be demonstrated. The agent never talks to this API directly; the flight-ops
MCP server does, which is the normal FDE pattern (wrap what exists, don't replace it).

Run:  uvicorn falcon.services.legacy_flight_api.app:app --port 8010
"""

from __future__ import annotations

import os
import random
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from falcon.core.clock import now
from falcon.core.config import settings
from falcon.core.db import Database

FLAKY_PCT = float(os.environ.get("LEGACY_FLAKY_PCT", "0"))  # 0..100: percent of requests that fail with 503
RATE_LIMIT_PER_MIN = int(os.environ.get("LEGACY_RATE_LIMIT_PER_MIN", "240"))

db = Database(settings.dsn_flightops)
_bucket: dict[str, list[float]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.open()
    yield
    await db.close()


app = FastAPI(title="Falcon legacy flight ops API", version="1.0", lifespan=lifespan)
from falcon.core.telemetry import setup_tracing  # noqa: E402

setup_tracing("falcon-legacy-flight-api", app)  # decision #27


async def api_key(x_api_key: Annotated[str | None, Header()] = None) -> str:
    if x_api_key != settings.legacy_api_key:
        raise HTTPException(status_code=401, detail={"code": "E401", "msg": "invalid api key"})
    return x_api_key


@app.middleware("http")
async def rate_limit_and_chaos(request: Request, call_next):
    key = request.headers.get("x-api-key", "anon")
    window = _bucket.setdefault(key, [])
    cutoff = time.time() - 60
    window[:] = [t for t in window if t > cutoff]
    if len(window) >= RATE_LIMIT_PER_MIN:
        return JSONResponse({"code": "E429", "msg": "rate limit"}, status_code=429, headers={"Retry-After": "5"})
    window.append(time.time())
    if FLAKY_PCT and request.method == "GET" and random.random() * 100 < FLAKY_PCT:
        return JSONResponse({"code": "E503", "msg": "upstream unavailable"}, status_code=503)
    return await call_next(request)


def _row(r: dict) -> dict:
    return {
        "fid": r["flight_instance_id"], "fno": r["flight_number"], "dt": r["service_date"].isoformat(),
        "org": r["origin"], "dst": r["destination"], "std": r["dep_at"].isoformat(), "sta": r["arr_at"].isoformat(),
        "ac": r["aircraft"], "cap": r["seats_total"], "st": r["status"], "dly": r["delay_minutes"], "rsn": r["reason"], "gate": r["gate"], "upd": r["updated_at"].isoformat(),
    }


BASE_SQL = """SELECT f.flight_instance_id, f.flight_number, f.service_date, f.origin, f.destination, f.dep_at, f.arr_at, f.aircraft, f.seats_total,
                     s.status, s.delay_minutes, s.reason, s.gate, s.updated_at
              FROM flightops.flights f JOIN flightops.flight_status s ON s.flight_instance_id = f.flight_instance_id"""


@app.get("/v1/flights")
async def list_flights(flight_number: str | None = None, dt: date | None = None, org: str | None = None, dst: str | None = None,
                       page: int = 1, page_size: int = Query(50, le=200), _: str = Depends(api_key)):
    where, params = [], []
    if flight_number:
        where.append("f.flight_number = %s"); params.append(flight_number.upper())
    if dt:
        where.append("f.service_date = %s"); params.append(dt)
    if org:
        where.append("f.origin = %s"); params.append(org.upper())
    if dst:
        where.append("f.destination = %s"); params.append(dst.upper())
    sql = BASE_SQL + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY f.dep_at LIMIT %s OFFSET %s"
    rows = await db.fetch_all(sql, params + [page_size, (page - 1) * page_size])
    return {"page": page, "page_size": page_size, "items": [_row(r) for r in rows]}


@app.get("/v1/flights/status")
async def batch_status(ids: str, _: str = Depends(api_key)):
    wanted = [i.strip() for i in ids.split(",") if i.strip()][:100]
    rows = await db.fetch_all(BASE_SQL + " WHERE f.flight_instance_id = ANY(%s)", (wanted,))
    return {"items": [_row(r) for r in rows]}


@app.get("/v1/flights/{fid}")
async def get_flight(fid: str, _: str = Depends(api_key)):
    row = await db.fetch_one(BASE_SQL + " WHERE f.flight_instance_id = %s", (fid,))
    if not row:
        raise HTTPException(404, detail={"code": "E404", "msg": "unknown flight"})
    return _row(row)


@app.get("/v1/disruptions")
async def disruptions(dt: date | None = None, page: int = 1, page_size: int = Query(50, le=200), _: str = Depends(api_key)):
    sql = BASE_SQL + " WHERE s.status IN ('delayed','cancelled')" + (" AND f.service_date = %s" if dt else "") + " ORDER BY f.dep_at LIMIT %s OFFSET %s"
    params = ([dt] if dt else []) + [page_size, (page - 1) * page_size]
    rows = await db.fetch_all(sql, params)
    return {"page": page, "page_size": page_size, "items": [_row(r) for r in rows]}


@app.get("/health")
async def health():
    return {"ok": True, "service": "legacy-flight-api", "now": now().isoformat()}
