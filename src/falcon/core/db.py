"""Database access: one async pool per service, each opened with that service's own role.

Why per-role pools: the reservations server physically cannot read policy or flight-ops
tables, and the ops query role can only see aggregate views. That is the guardrail the
panel can verify in `db/schema.sql`, independent of anything the model does.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


class Database:
    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 8, search_path: str | None = None):
        options = f"-c search_path={search_path}" if search_path else None
        kwargs: dict[str, Any] = {"row_factory": dict_row}
        if options:
            kwargs["options"] = options
        self.pool = AsyncConnectionPool(dsn, min_size=min_size, max_size=max_size, open=False, kwargs=kwargs)
        self._opened = False

    async def open(self) -> None:
        if not self._opened:
            await self.pool.open()
            self._opened = True

    async def ensure_open(self) -> None:
        """Lazy open: stateless MCP servers run their FastMCP lifespan per request, so pools are opened on first use instead."""
        await self.open()

    async def close(self) -> None:
        await self.pool.close()

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[psycopg.AsyncConnection]:
        await self.ensure_open()
        async with self.pool.connection() as conn:
            yield conn

    async def fetch_all(self, sql: str, params: Any = None) -> list[dict]:
        await self.ensure_open()
        async with self.pool.connection() as conn:
            cur = await conn.execute(sql, params)
            return await cur.fetchall()

    async def fetch_one(self, sql: str, params: Any = None) -> dict | None:
        await self.ensure_open()
        async with self.pool.connection() as conn:
            cur = await conn.execute(sql, params)
            return await cur.fetchone()

    async def execute(self, sql: str, params: Any = None) -> None:
        await self.ensure_open()
        async with self.pool.connection() as conn:
            await conn.execute(sql, params)


def sync_connect(dsn: str, **kwargs: Any) -> psycopg.Connection:
    """Plain synchronous connection for scripts (seed, ingest, reset)."""
    return psycopg.connect(dsn, row_factory=dict_row, **kwargs)
