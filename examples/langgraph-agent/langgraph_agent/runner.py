"""Run a compiled LangGraph as a task and stream the events its nodes push.

Both agents in this package push events onto a queue from inside their
nodes, rather than reconstructing them from LangGraph's own stream: the nodes
know exactly what happened, and the output stays identical to the engine's
loop agent. This is the one place that drains that queue.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from chatbot_engine.models.events import Event

#: Marks the end of the event stream, so the drain loop knows the graph finished.
_DONE = object()


async def run_graph(
    graph: Any,
    state: dict[str, Any],
    config: dict[str, Any],
    events: asyncio.Queue[Any],
) -> AsyncIterator[Event]:
    """Invoke `graph` with `state` and `config`, yielding what its nodes put on `events`.

    The graph runs as a task while this drains the queue, so events reach the
    caller as they happen. Once drained, anything the graph raised is raised
    here, after the events that preceded it have been delivered.
    """

    async def drive() -> None:
        try:
            await graph.ainvoke(state, config)
        finally:
            await events.put(_DONE)

    task = asyncio.create_task(drive())
    try:
        while True:
            item = await events.get()
            if item is _DONE:
                break
            yield item
        await task
    finally:
        if not task.done():
            task.cancel()
