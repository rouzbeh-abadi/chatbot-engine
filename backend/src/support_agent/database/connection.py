"""Database engine and session lifecycle.

Built lazily on purpose. Creating the engine at import time means anything that
merely imports this module -- a test collector, a CLI, the MCP tool server on a
machine with no database -- fails before it can do anything useful.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from support_agent.settings import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    """The process-wide connection pool.

    `pool_pre_ping` costs one round trip per checkout and saves the first query
    after a database restart or an idle-timeout kill from failing.
    """
    return create_async_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=get_engine(),
        expire_on_commit=False,
    )


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """One session per request or per tool call. Usable as a FastAPI dependency."""
    async with get_session_factory()() as session:
        yield session


async def dispose_engine() -> None:
    """Close the pool. Call on shutdown, and between tests that swap databases.

    Never raises: this runs in `finally` blocks, where an exception would mask
    whatever actually went wrong.
    """
    if get_engine.cache_info().currsize:
        try:
            await get_engine().dispose()
        except Exception:  # noqa: BLE001 - cleanup must not shadow the real error
            pass
    get_engine.cache_clear()
    get_session_factory.cache_clear()
