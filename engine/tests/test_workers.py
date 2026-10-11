"""The gate a project's vectors are read, written and replaced behind.

Searches and writes share it; a move, a rebuild or a drop of the project's
collection has it alone, once everyone sharing it is done, and goes before
anyone who arrives after it asked.
"""

from __future__ import annotations

import asyncio

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
