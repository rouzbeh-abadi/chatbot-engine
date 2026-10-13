"""The gate a project's vectors are read, written and replaced behind.

Searches and writes share it; a move, a rebuild or a drop of the project's
collection has it alone, once everyone sharing it is done, and goes before
anyone who arrives after it asked.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest

from chatbot_engine.rag.workers import Gate, KeyedGates


async def test_one_alone_waits_for_everyone_sharing_to_finish() -> None:
    gate = Gate()
    order: list[str] = []
    reading = asyncio.Event()
    done_reading = asyncio.Event()

    async def search() -> None:
        async with gate.shared():
            reading.set()
            await done_reading.wait()
            order.append("search")

    async def rebuild() -> None:
        async with gate.alone():
            order.append("rebuild")

    searching = asyncio.create_task(search())
    await reading.wait()
    rebuilding = asyncio.create_task(rebuild())
    await asyncio.sleep(0.01)
    assert order == [], "the rebuild started while a search was reading"

    done_reading.set()
    await asyncio.gather(searching, rebuilding)
    assert order == ["search", "rebuild"]


async def test_one_waiting_to_be_alone_goes_before_those_who_come_after() -> None:
    """Otherwise a stream of searches would keep a rebuild waiting for ever."""
    gate = Gate()
    order: list[str] = []
    release = asyncio.Event()

    async def first() -> None:
        async with gate.shared():
            await release.wait()
            order.append("first search")

    async def rebuild() -> None:
        async with gate.alone():
            order.append("rebuild")

    async def later() -> None:
        async with gate.shared():
            order.append("later search")

    tasks = [asyncio.create_task(first())]
    await asyncio.sleep(0)
    tasks.append(asyncio.create_task(rebuild()))
    await asyncio.sleep(0)
    tasks.append(asyncio.create_task(later()))
    await asyncio.sleep(0.01)
    release.set()
    await asyncio.gather(*tasks)

    assert order == ["first search", "rebuild", "later search"]


async def test_many_share_it_at_once() -> None:
    gate = Gate()
    inside = 0
    most = 0

    async def search() -> None:
        nonlocal inside, most
        async with gate.shared():
            inside += 1
            most = max(most, inside)
            await asyncio.sleep(0.01)
            inside -= 1

    await asyncio.gather(*(search() for _ in range(5)))
    assert most == 5


async def test_one_that_stops_waiting_does_not_hold_the_others() -> None:
    """Cancelled while waiting to be alone, it no longer stands in the way."""
    gate = Gate()
    release = asyncio.Event()

    async def holder() -> None:
        async with gate.shared():
            await release.wait()

    async def alone() -> None:
        async with gate.alone():
            pass

    holding = asyncio.create_task(holder())
    await asyncio.sleep(0)
    waiting = asyncio.create_task(alone())
    await asyncio.sleep(0)
    waiting.cancel()
    await asyncio.sleep(0)

    async with asyncio.timeout(1):
        async with gate.shared():
            pass
    release.set()
    await holding


def test_one_gate_per_key_while_it_is_used() -> None:
    gates = KeyedGates()
    gate = gates("p")
    assert gates("p") is gate
    assert gates("q") is not gate


# --- the gate, taken once indexing has a thread ---------------------------------


@pytest.fixture
def one_index_thread(monkeypatch: pytest.MonkeyPatch) -> Iterator[ThreadPoolExecutor]:
    """Indexing's pool, one thread wide, so a test can keep it busy."""
    from chatbot_engine.rag import workers

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="index")
    monkeypatch.setattr(workers, "_pool", pool)
    yield pool
    pool.shutdown(wait=True)


async def test_a_write_waiting_for_a_thread_does_not_hold_its_gate(
    one_index_thread: ThreadPoolExecutor,
) -> None:
    """Another project's document has indexing's only thread. A write queued
    behind it must not hold its project's gate meanwhile: a rebuild waiting
    for the gate, and every search behind the rebuild, would wait for the
    other project's document."""
    from chatbot_engine.rag.workers import run_indexing_shared

    gate = Gate()
    busy = threading.Event()
    other_project = one_index_thread.submit(busy.wait, 10)
    ran: list[str] = []
    queued = asyncio.create_task(run_indexing_shared(gate, ran.append, "written"))
    await asyncio.sleep(0.1)

    async with asyncio.timeout(1):
        async with gate.alone():
            assert ran == [], "precondition: the write is still waiting for a thread"

    busy.set()
    await queued
    other_project.result()
    assert ran == ["written"]


async def test_a_write_holds_its_gate_until_its_thread_is_done_even_when_cancelled(
    one_index_thread: ThreadPoolExecutor,
) -> None:
    """The thread cannot be stopped, so a caller that stops waiting must not
    let the gate go before the write has landed: a drop or a rebuild of the
    collection would race it."""
    from chatbot_engine.rag.workers import run_indexing_shared

    gate = Gate()
    writing, release = threading.Event(), threading.Event()

    def write() -> None:
        writing.set()
        release.wait(10)

    task = asyncio.create_task(run_indexing_shared(gate, write))
    await asyncio.to_thread(writing.wait, 5)
    task.cancel()
    alone = asyncio.create_task(_alone(gate))
    await asyncio.sleep(0.2)
    assert not alone.done(), "the gate was let go while the write was running"

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with asyncio.timeout(1):
        await alone


async def test_a_write_cancelled_before_it_had_a_thread_never_runs(
    one_index_thread: ThreadPoolExecutor,
) -> None:
    from chatbot_engine.rag.workers import run_indexing_shared

    gate = Gate()
    busy = threading.Event()
    one_index_thread.submit(busy.wait, 10)
    ran: list[str] = []
    task = asyncio.create_task(run_indexing_shared(gate, ran.append, "written"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    busy.set()
    await asyncio.to_thread(one_index_thread.submit(lambda: None).result, 5)
    assert ran == []


async def _alone(gate: Gate) -> None:
    async with gate.alone():
        pass
