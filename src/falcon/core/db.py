"""Database access: one async pool per service, opened with that service's own role (the reservations server physically
cannot read policy or flight-ops tables: the guardrail the panel can verify in db/schema.sql, independent of the model)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .config import settings


def connection_options(search_path: str | None = None, statement_timeout_ms: int | None = None) -> str:
    """libpq `options` for every connection: the statement timeout (settings.db_statement_timeout_ms unless given) and a search path."""
    timeout = settings.db_statement_timeout_ms if statement_timeout_ms is None else statement_timeout_ms
    return " ".join(([f"-c statement_timeout={timeout}"] if timeout else []) + ([f"-c search_path={search_path}"] if search_path else []))


class Database:
    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 8, search_path: str | None = None, statement_timeout_ms: int | None = None):
        kwargs: dict[str, Any] = {"row_factory": dict_row}
        if options := connection_options(search_path, statement_timeout_ms):
            kwargs["options"] = options
        # check: a connection is tested before it is handed out, so one the server dropped (a restart, a failover) is replaced
        self.pool = AsyncConnectionPool(dsn, min_size=min_size, max_size=max_size, open=False, kwargs=kwargs, check=AsyncConnectionPool.check_connection)

    async def open(self) -> None:
        """Lazy: the stateless MCP servers have no start-up hook, so the pool opens on first use."""
        if self.pool.closed:
            await self.pool.open()

    async def close(self) -> None:
        await self.pool.close()

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[psycopg.AsyncConnection]:
        await self.open()
        async with self.pool.connection() as conn:
            yield conn

    async def fetch_all(self, sql: str, params: Any = None) -> list[dict]:
        async with self.connection() as conn:
            cur = await conn.execute(sql, params)
            return await cur.fetchall()

    async def fetch_one(self, sql: str, params: Any = None) -> dict | None:
        async with self.connection() as conn:
            cur = await conn.execute(sql, params)
            return await cur.fetchone()

    async def execute(self, sql: str, params: Any = None) -> None:
        async with self.connection() as conn:
            await conn.execute(sql, params)
