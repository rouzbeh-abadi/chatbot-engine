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
from langgraph_agent.workflow import thread_of


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
    pause_id = _asked(await _run(_request(PROJECT), pauses, Tools())).thread_id
    thread, mark = thread_of(pause_id), pause_id.rsplit(".", 1)[1]

    assert await pauses.claim(thread, "support", "other", mark) is None
    assert await pauses.claim(thread, "support", "s1", mark) is not None
    assert await pauses.claim(thread, "support", "s1", mark) is None


@pytest.mark.parametrize("store", ["memory", "sqlite"])
async def test_a_claim_takes_only_the_pause_its_id_names(store, tmp_path):
    """The pause id carries the checkpoint the turn paused at, and the record
    keeps it: an answer meant for an earlier question of the turn, however it
    is timed against the resume before it, finds nothing to claim
    (WORKFLOW-3)."""
    pauses = dict(zip(["memory", "sqlite"], _stores(tmp_path), strict=True))[store]
    pause_id = _asked(await _run(_request(PROJECT), pauses, Tools())).thread_id
    thread, mark = thread_of(pause_id), pause_id.rsplit(".", 1)[1]

    assert await pauses.claim(thread, "support", "s1", "0" * 32) is None
    # Nor does an id given out before 0.1.31, which names no checkpoint.
    assert await pauses.claim(thread, "support", "s1") is None
    claimed = await pauses.claim(thread, "support", "s1", mark)
    assert claimed is not None and claimed.mark == mark


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

    assert forgotten == [thread_of(thread)]
    assert await pauses.find(thread_of(thread)) is None


async def test_the_question_a_resume_answers_is_not_a_second_visit():
    """`max_steps` counts the workflow's own steps in a request. A resumed
    request starts by running the question it answers again, which was
    counted when it was asked: with `max_steps: 1` the step after the
    question still runs (WORKFLOW-9)."""
    spec = {
        "start": "name",
        "max_steps": 1,
        "nodes": [
            {
                "id": "name",
                "type": "ask",
                "prompt": "Name?",
                "input": "text",
                "understand": False,
                "var": "name",
            },
            {
                "id": "book",
                "type": "tool",
                "tool": "book",
                "arguments": {"name": "{{vars.name}}"},
                "var": "booked",
            },
        ],
        "edges": [{"from": "name", "to": "book"}],
    }
    pauses, tools = Pauses.memory(), Tools()
    asked = _asked(await _run(_request(spec), pauses, tools))
    resumed = await _run(
        _request(spec, "Ann", {"thread_id": asked.thread_id, "value": "Ann"}),
        pauses,
        tools,
    )

    assert tools.calls == [("book", {"name": "Ann"})]
    assert resumed[-1].finish_reason == "stop"


async def test_an_engine_before_0_1_31_can_still_record_a_pause_in_the_file(tmp_path):
    """A rollback, or two versions on one file during a deploy: the older
    engine writes its record with five values, so the checkpoint each pause
    keeps since 0.1.31 lives in a table of its own, not a sixth column."""
    import sqlite3

    path = tmp_path / "checkpoints.sqlite3"
    pauses = Pauses.sqlite(path, ttl_s=3600)
    _asked(await _run(_request(PROJECT), pauses, Tools()))
    await pauses.close()

    older = sqlite3.connect(path)
    older.execute(
        "INSERT OR REPLACE INTO workflow_pauses VALUES (?, ?, ?, ?, ?)",
        ("support:older", "support", "s1", "hash", 0.0),
    )
    older.commit()
    older.close()
