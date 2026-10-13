"""The threads indexing runs on, apart from the event loop's default pool.

Reading a document can block a thread for its whole deadline, and embedding
for as long as the provider takes. On the default pool (CPUs + 4 threads),
a few large uploads left no thread for any chat's retrieval, which runs there
too (docs/review-2026-10.md, INGEST-5). Indexing gets `ENGINE_INDEX_CONCURRENCY`
threads of its own instead; past that, documents wait their turn and chats
do not.

What indexing writes to a project's collection it writes behind the
project's gate (`Gate`), taken once one of these threads is free for the
write, not before (`run_indexing_shared`): held while waiting for a thread,
the gate would keep a rebuild or a drop of the collection, and every search
of the project behind that, waiting for other projects' documents to be read
and embedded.
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
from typing import cast

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


async def run_indexing_shared[T](
    gate: Gate, fn: Callable[..., T], /, *args: object
) -> T:
    """`run_indexing`, sharing `gate` while `fn` runs, and only then.

    The gate is taken once one of indexing's threads is free to run `fn`:
    the thread waits for it, rather than the gate for a thread. And it is
    held until `fn` has returned, even when the caller stops waiting, since
    the thread cannot be stopped: what it writes must not land in a
    collection that a rebuild or a drop has replaced meanwhile.
    """
    loop = asyncio.get_running_loop()
    call = functools.partial(contextvars.copy_context().run, fn, *args)
    ours = asyncio.Event()
    go = threading.Event()
    wanted = True

    def turn() -> T | None:
        loop.call_soon_threadsafe(ours.set)
        # A bounded wait, so a thread whose caller went with the event loop
        # (an engine stopping) does not keep the interpreter from exiting.
        while not go.wait(1.0):
            if loop.is_closed():
                return None
        return call() if wanted else None

    done = loop.run_in_executor(_index_pool(), turn)
    try:
        await ours.wait()
        async with gate.shared():
            go.set()
            return cast(T, await finish(done))
    finally:
        if not go.is_set():
            # Stopped before `fn` began: the thread is let go without it.
            wanted = False
            go.set()


async def in_thread[T](fn: Callable[..., T], /, *args: object) -> T:
    """`asyncio.to_thread`, waited for to the end (`finish`): for what runs
    with a project's gate alone, which must not be let go while the thread
    is still at work on the project's collections."""
    return await finish(asyncio.ensure_future(asyncio.to_thread(fn, *args)))


async def finish[T](work: asyncio.Future[T]) -> T:
    """`work`'s result, waited for even when the caller is cancelled: a
    thread cannot be stopped, so whoever holds a gate around one holds it
    until the thread is done. The cancellation is raised after that."""
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        while not work.done():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait([work])
        if not work.cancelled():
            # Looked at, so a failure nobody waits for any more is not
            # reported again as never retrieved.
            work.exception()
        raise


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
    or drops the collection they live in (a move, the end of a rebuild, a
    purge) has it alone, once everyone sharing it is done. One waiting to be
    alone goes before anyone who arrives after it, so it is never kept
    waiting by a stream of searches; those searches wait for it instead, so
    the gate is had alone only when there is something to replace or drop,
    and never by one still waiting for anything but the gate itself.
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
