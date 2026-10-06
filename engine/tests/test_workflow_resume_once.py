"""A paused turn is resumed once.

Two requests with the same `thread_id` (a double click, a client that
retried) both found the pause and both resumed it, running the rest of the
turn twice: two bookings, two tickets. Now the request that resumes claims
the pause before anything runs, and the other is told it is no longer
waiting (`resume_expired`).
"""

from __future__ import annotations

import asyncio

import pytest
from test_workflow_ask import PROJECT, Tools, _asked, _request, _run

from chatbot_engine.models.events import DoneEvent, ErrorEvent

pytest.importorskip("langgraph_agent.pauses")
from langgraph_agent.pauses import Pauses


def _stores(tmp_path) -> list[Pauses]:
    return [
        Pauses.memory(),
        Pauses.sqlite(tmp_path / "checkpoints.sqlite3", ttl_s=3600),
    ]


@pytest.mark.parametrize("store", ["memory", "sqlite"])
async def test_two_resumes_of_one_question_run_the_rest_of_the_turn_once(
    store, tmp_path
):
    pauses = dict(zip(["memory", "sqlite"], _stores(tmp_path), strict=True))[store]
    tools = Tools()
    thread = _asked(await _run(_request(PROJECT), pauses, tools)).thread_id
    answer = {"thread_id": thread, "value": "x.com"}

    first, second = await asyncio.gather(
        _run(_request(PROJECT, "x.com", answer), pauses, tools),
        _run(_request(PROJECT, "x.com", answer), pauses, tools),
    )

    assert tools.calls == [("book", {"name": "Project X", "url": "https://x.com"})]
    refused = [
        events
        for events in (first, second)
        if isinstance(events[0], ErrorEvent) and events[0].code == "resume_expired"
    ]
    assert len(refused) == 1
    assert isinstance(refused[0][-1], DoneEvent)
    assert refused[0][-1].finish_reason == "error"


async def test_a_resume_after_the_turn_went_on_finds_nothing_waiting():
    pauses, tools = Pauses.memory(), Tools()
    thread = _asked(await _run(_request(PROJECT), pauses, tools)).thread_id
    answer = {"thread_id": thread, "value": "x.com"}

    await _run(_request(PROJECT, "x.com", answer), pauses, tools)
    again = await _run(_request(PROJECT, "x.com", answer), pauses, tools)

    assert isinstance(again[0], ErrorEvent) and again[0].code == "resume_expired"
    assert len(tools.calls) == 1


async def test_a_resume_from_elsewhere_does_not_use_up_the_pause():
    """A request from another conversation is refused without taking the
    pause, so the visitor it belongs to can still answer."""
    pauses, tools = Pauses.memory(), Tools()
    thread = _asked(await _run(_request(PROJECT), pauses, tools)).thread_id
    answer = {"thread_id": thread, "value": "x.com"}

    stranger = await _run(
        _request(PROJECT, "x.com", answer, session_id="someone-else"), pauses, tools
    )
    owner = await _run(_request(PROJECT, "x.com", answer), pauses, tools)

    assert isinstance(stranger[0], ErrorEvent)
    assert not any(isinstance(e, ErrorEvent) for e in owner)
    assert tools.calls == [("book", {"name": "Project X", "url": "https://x.com"})]


@pytest.mark.parametrize("store", ["memory", "sqlite"])
async def test_a_claim_is_given_once(store, tmp_path):
    pauses = dict(zip(["memory", "sqlite"], _stores(tmp_path), strict=True))[store]
    thread = _asked(await _run(_request(PROJECT), pauses, Tools())).thread_id

    assert await pauses.claim(thread, "support", "other") is None
    assert await pauses.claim(thread, "support", "s1") is not None
    assert await pauses.claim(thread, "support", "s1") is None


async def test_a_turn_stopped_on_its_way_keeps_no_state():
    """A resumed turn the caller abandons (or the deadline stops) is never
    resumed again, so its saved state goes with it."""
    pauses, tools = Pauses.memory(), Tools()
    thread = _asked(await _run(_request(PROJECT), pauses, tools)).thread_id
    forgotten: list[str] = []
    forget = pauses.forget

    async def recording(thread_id: str) -> None:
        forgotten.append(thread_id)
        await forget(thread_id)

    pauses.forget = recording  # type: ignore[method-assign]

    class Hangs(Tools):
        async def call_tool(self, *, name, arguments, **kwargs):
            await asyncio.sleep(30)

    task = asyncio.ensure_future(
        _run(
            _request(PROJECT, "x.com", {"thread_id": thread, "value": "x.com"}),
            pauses,
            Hangs(),
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert forgotten == [thread]
    assert await pauses.find(thread) is None
