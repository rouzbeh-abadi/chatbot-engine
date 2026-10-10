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
import contextvars
import functools
import threading
import weakref
from collections.abc import Callable, Hashable
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
