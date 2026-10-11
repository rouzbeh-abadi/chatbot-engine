"""The threads indexing runs on, apart from the event loop's default pool.

Reading a document can block a thread for its whole deadline, and embedding
for as long as the provider takes. On the default pool (CPUs + 4 threads),
a few large uploads left no thread for any chat's retrieval, which runs there
too (docs/review-2026-10.md, INGEST-5). Indexing gets `ENGINE_INDEX_CONCURRENCY`
threads of its own instead; past that, documents wait their turn and chats
do not.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import threading
import weakref
from collections.abc import AsyncIterator, Callable, Hashable
from concurrent.futures import ThreadPoolExecutor

from chatbot_engine.settings import get_settings

_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _index_pool() -> ThreadPoolExecutor:
    global _pool

    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ThreadPoolExecutor(
                    max_workers=get_settings().index_concurrency,
                    thread_name_prefix="index",
                )
    return _pool


async def run_indexing[T](fn: Callable[..., T], /, *args: object) -> T:
    """`asyncio.to_thread`, on indexing's own threads: `fn(*args)` with the
    caller's context (its request id and trace), off the event loop."""
    loop = asyncio.get_running_loop()
    call = functools.partial(contextvars.copy_context().run, fn, *args)
    return await loop.run_in_executor(_index_pool(), call)


class KeyedLocks:
    """An asyncio lock per key, gone once nothing holds or waits for it."""

    def __init__(self) -> None:
        self._locks: weakref.WeakValueDictionary[Hashable, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    def __call__(self, key: Hashable) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = self._locks[key] = asyncio.Lock()
        return lock


class Gate:
    """Many at once, or one alone.

    What reads and writes a project's vectors shares the gate; what replaces
    or drops the collection they live in (a move, a rebuild, a purge) has it
    alone, once everyone sharing it is done. One waiting to be alone goes
    before anyone who arrives after it, so it is never kept waiting by a
    stream of searches.
    """

    def __init__(self) -> None:
        self._turn = asyncio.Condition()
        self._sharing = 0
        self._alone = False
        self._waiting = 0

    @contextlib.asynccontextmanager
    async def shared(self) -> AsyncIterator[None]:
        async with self._turn:
            await self._turn.wait_for(lambda: not self._alone and not self._waiting)
            self._sharing += 1
        try:
            yield
        finally:
            async with self._turn:
                self._sharing -= 1
                self._turn.notify_all()

    @contextlib.asynccontextmanager
    async def alone(self) -> AsyncIterator[None]:
        async with self._turn:
            self._waiting += 1
            try:
                await self._turn.wait_for(lambda: not self._alone and not self._sharing)
            finally:
                self._waiting -= 1
                self._turn.notify_all()
            self._alone = True
        try:
            yield
        finally:
            async with self._turn:
                self._alone = False
                self._turn.notify_all()


class KeyedGates:
    """A `Gate` per key, gone once nothing holds or waits for it."""

    def __init__(self) -> None:
        self._gates: weakref.WeakValueDictionary[Hashable, Gate] = (
            weakref.WeakValueDictionary()
        )

    def __call__(self, key: Hashable) -> Gate:
        gate = self._gates.get(key)
        if gate is None:
            gate = self._gates[key] = Gate()
        return gate
