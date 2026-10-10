"""Pick the agent a request asked for.

Which agent runs a turn is part of the assistant config, so it is decided per
request rather than at wiring time. One engine can therefore serve the built-in
loop, the graph, and any agent an adopter has installed, at the same time.

This is itself an `Agent`. Everything upstream -- the service, the route, the
event stream -- is unchanged and unaware there is a choice at all.

It is also where every turn gets its deadline, whichever agent runs it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextvars import ContextVar

from chatbot_engine.agent.client import (
    price_usage,
    turn_meter,
    unavailable_message,
    usage_event,
)
from chatbot_engine.agent.registry import available_agents, build_agent
from chatbot_engine.models.chat import ChatRequest
from chatbot_engine.models.events import DoneEvent, Event, TokenEvent, UsageEvent
from chatbot_engine.ports.agent import Agent, ToolProvider
from chatbot_engine.settings import get_settings

logger = logging.getLogger(__name__)

#: Set by a caller that must tell a turn stopped at its deadline from one
#: that ended on its own, such as the evaluation's judge: `within_deadline`
#: adds the deadline's seconds to the list when it stops a turn. The stream
#: is the same either way, so `/chat` is unaffected.
DEADLINE_PASSED: ContextVar[list[float] | None] = ContextVar(
    "deadline_passed", default=None
)


class AgentRouter:
    """Builds and runs whichever agent the assistant config names."""

    def __init__(self, tools: ToolProvider) -> None:
        self._tools = tools
        #: Built once per name and reused. An agent holds no per-turn state --
        #: everything about a turn arrives with the request -- so rebuilding one
        #: for every message would be waste, not isolation.
        self._built: dict[str, Agent] = {}

    def run(self, request: ChatRequest) -> AsyncIterator[Event]:
        """Run one turn on the requested agent, or the engine's default.

        The agent is built here, before the stream starts, so an unknown name
        is still a status code; the turn then runs within its deadline.
        """
        settings = get_settings()
        name = request.project.agent or settings.agent

        if name not in self._built:
            self._built[name] = build_agent(name, self._tools)

        return within_deadline(
            self._built[name].run(request), request, settings.turn_deadline_s
        )

    async def forget(self, project_id: str, session_id: str | None = None) -> int:
        """Forget what any agent keeps between turns for a project, or for one
        of its sessions (a workflow's turns paused on a question, with the
        answers they held); how many turns. Every installed agent is asked,
        built if no turn has built it yet, since what it kept is on disk."""
        forgotten = 0
        for name in available_agents():
            if name not in self._built:
                try:
                    self._built[name] = build_agent(name, self._tools)
                except Exception as exc:
                    # An agent that cannot be built ran no turn here, so
                    # kept nothing to forget.
                    logger.warning(
                        "agent %r could not be built to forget: %s", name, exc
                    )
                    continue
            forget = getattr(self._built[name], "forget", None)
            if forget is not None:
                forgotten += await forget(project_id, session_id)
        return forgotten


async def within_deadline(
    events: AsyncIterator[Event], request: ChatRequest, seconds: float
) -> AsyncIterator[Event]:
    """A turn's events until its deadline, and its end when it passes.

    Past `seconds` (`ENGINE_TURN_DEADLINE_S`) the agent is stopped where it
    is, whatever it was waiting on: a provider being retried, a tool server
    that does not answer. The visitor is told the assistant's
    `unavailable_message`, as a workflow's tool step says it when a call
    fails, and the turn ends: with a `usage` event for what it had spent when
    it had not reported that yet, then `done`.
    """
    deadline = asyncio.get_running_loop().time() + seconds
    spoke = reported = False
    try:
        while True:
            timeout = asyncio.timeout_at(deadline)
            try:
                async with timeout:
                    event = await anext(events)
            except StopAsyncIteration:
                return
            except TimeoutError:
                if not timeout.expired():
                    # The agent's own timeout, not the turn's: its failure.
                    raise
                break
            if isinstance(event, TokenEvent) and event.text:
                spoke = True
            elif isinstance(event, UsageEvent):
                reported = True
            yield event
    finally:
        # Stopped at the deadline, the agent has already unwound; closed early
        # by the caller, it unwinds here.
        close = getattr(events, "aclose", None)
        if close is not None:
            await close()

    logger.warning(
        "turn passed its deadline of %.0fs and was stopped; it ends with the "
        "unavailable message",
        seconds,
    )
    if (passed := DEADLINE_PASSED.get()) is not None:
        passed.append(seconds)
    yield TokenEvent(
        text=("\n\n" if spoke else "") + unavailable_message(request.project)
    )
    meter = turn_meter()
    if not reported and meter is not None and meter["total_tokens"] > 0:
        model = request.project.model or get_settings().chat_model
        yield usage_event(price_usage(meter, model))
    yield DoneEvent(finish_reason="stop")
