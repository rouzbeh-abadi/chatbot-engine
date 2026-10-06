"""A turn has a deadline, and a streamed call is retried in one layer only.

The provider client retried three times and `stream_reply` retried the whole
call three more, with no bound on the turn: a provider that kept failing
could hold a request for many minutes. Now a model that streams is built
with no retries of its own, `stream_reply` is the one layer, and every turn,
whichever agent runs it, ends at `ENGINE_TURN_DEADLINE_S` with the
assistant's unavailable message.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk
from test_agent_parity import (
    AGENTS,
    FakeTools,
    ScriptedModel,
    _no_retrieval,
    _request,
    _usage_chunk,
)

from chatbot_engine.agent import registry
from chatbot_engine.agent.chat_agent import ChatAgent
from chatbot_engine.agent.client import (
    DEFAULT_UNAVAILABLE_MESSAGE,
    add_usage,
    empty_totals,
    start_meter,
)
from chatbot_engine.agent.router import AgentRouter
from chatbot_engine.models.events import (
    DoneEvent,
    TokenEvent,
    ToolCallFinishedEvent,
    UsageEvent,
)
from chatbot_engine.models.workflow import WorkflowSpec
from chatbot_engine.settings import get_settings
from langgraph_agent.agent import LangGraphAgent
from langgraph_agent.workflow import WorkflowAgent


class Slow:
    """An agent that waits on something that never answers."""

    def __init__(self, *, speak: str = "", spend: int = 0) -> None:
        self.speak = speak
        self.spend = spend
        self.unwound = False

    def run(self, request):
        async def events():
            if self.speak:
                yield TokenEvent(text=self.speak)
            if self.spend:
                add_usage(empty_totals(), _usage_chunk(self.spend, 1))
            try:
                await asyncio.sleep(30)
            finally:
                self.unwound = True
            yield TokenEvent(text="never said")

        return events()


def _deadline(monkeypatch, seconds: str) -> None:
    monkeypatch.setenv("ENGINE_TURN_DEADLINE_S", seconds)
    get_settings.cache_clear()


async def _route(agent, request=None) -> list:
    with patch.object(registry, "available_agents", lambda: {"loop": lambda t: agent}):
        router = AgentRouter(tools=object())
        return [event async for event in router.run(request or _request())]


async def test_a_turn_past_its_deadline_ends_with_the_unavailable_message(
    monkeypatch,
):
    _deadline(monkeypatch, "0.05")
    agent = Slow()

    events = await _route(agent)

    assert [type(e).__name__ for e in events] == ["TokenEvent", "DoneEvent"]
    assert events[0].text == DEFAULT_UNAVAILABLE_MESSAGE
    assert events[-1].finish_reason == "stop"
    assert agent.unwound, "the agent was stopped, not left running"


async def test_the_judge_reports_a_case_stopped_at_its_deadline_as_an_error(
    monkeypatch,
):
    """Its words are the unavailable message, not the prompt's answer:
    graded, they would lower the score for a slow provider."""
    from chatbot_engine.eval.prompt_evaluation import generate_answer
    from chatbot_engine.models.evals import EvalCase

    _deadline(monkeypatch, "0.05")
    case = EvalCase(
        id="c1", category="shipping", question="Do you ship?", expected="Says yes."
    )
    with patch.object(registry, "available_agents", lambda: {"loop": lambda t: Slow()}):
        answer = await generate_answer(
            AgentRouter(tools=object()), _request().project, case
        )

    assert answer.error == "the turn passed its deadline of 0s"
    assert answer.text == DEFAULT_UNAVAILABLE_MESSAGE


async def test_after_words_the_message_is_its_own_paragraph_in_the_assistants_words(
    monkeypatch,
):
    _deadline(monkeypatch, "0.05")
    request = _request()
    request.project.unavailable_message = "Not now, sorry."

    events = await _route(Slow(speak="Let me check."), request)

    assert "".join(e.text for e in events if isinstance(e, TokenEvent)) == (
        "Let me check.\n\nNot now, sorry."
    )


async def test_what_the_turn_spent_is_reported_when_it_had_not_been(monkeypatch):
    _deadline(monkeypatch, "0.05")
    start_meter()

    events = await _route(Slow(spend=120))

    usage = next(e for e in events if isinstance(e, UsageEvent))
    assert (usage.input_tokens, usage.output_tokens) == (120, 1)
    assert isinstance(events[-1], DoneEvent)


async def test_an_agents_own_timeout_is_its_failure_not_the_deadline(monkeypatch):
    _deadline(monkeypatch, "30")

    class TimesOut:
        def run(self, request):
            async def events():
                raise TimeoutError("the agent's own")
                yield  # pragma: no cover

            return events()

    with pytest.raises(TimeoutError, match="the agent's own"):
        await _route(TimesOut())


def test_a_chat_turn_gets_the_deadline(
    client: TestClient, project: dict[str, object], monkeypatch
) -> None:
    """Through `/chat`, whichever agent the engine builds."""
    _deadline(monkeypatch, "0.05")

    with patch.object(
        registry, "available_agents", lambda: {"loop": lambda tools: Slow()}
    ):
        response = client.post("/chat", json={"project": project, "message": "hi"})

    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert [e["type"] for e in events] == ["token", "done"]
    assert events[0]["text"] == DEFAULT_UNAVAILABLE_MESSAGE


# --- one layer of retries ---------------------------------------------------


def _rate_limited() -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://x"))
    return openai.RateLimitError("slow down", response=response, body=None)


class BreaksOnce(ScriptedModel):
    """Streams the first round's chunks, then breaks; plays the rounds after."""

    broke: bool = False

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        if not self.broke:
            self.broke = True
            for chunk in self.rounds[0]:
                yield _generation(chunk)
            raise _rate_limited()
        async for chunk in super()._astream(messages, stop, run_manager, **kwargs):
            yield chunk


def _generation(chunk):
    from langchain_core.outputs import ChatGenerationChunk

    return ChatGenerationChunk(message=chunk)


def _built_without_retries(built) -> bool:
    return bool(built.call_args_list) and all(
        call.kwargs.get("max_retries") == 0 for call in built.call_args_list
    )


@pytest.mark.parametrize("which", AGENTS)
async def test_a_retried_call_runs_its_tool_once_and_builds_no_retrying_client(
    which: str,
) -> None:
    """The stream sends a tool call, breaks, and is retried: the reply is
    summed afresh, so the tool runs once; and the model behind it was built
    with no retries of its own."""
    asks = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"name": "get_booking_status", "args": "{}", "id": "c1", "index": 0}
        ],
    )
    model = BreaksOnce(rounds=[[asks], [AIMessageChunk(content="Delayed.")]], seen=[])
    tools = FakeTools()
    module = "chatbot_engine.agent" if which == "loop" else "langgraph_agent"
    agent = ChatAgent(tools=tools) if which == "loop" else LangGraphAgent(tools=tools)
    builder = (
        "chatbot_engine.agent.client.build_chat_model"
        if which == "loop"
        else "langgraph_agent.agent.build_chat_model"
    )

    with (
        patch(builder, return_value=model) as built,
        patch(
            f"{module}.{'chat_agent' if which == 'loop' else 'agent'}.retrieve_with_usage",
            new=_no_retrieval,
        ),
    ):
        events = [event async for event in agent.run(_request())]

    assert tools.calls == ["get_booking_status"]
    assert [e.call_id for e in events if isinstance(e, ToolCallFinishedEvent)] == ["c1"]
    assert "".join(e.text for e in events if isinstance(e, TokenEvent)) == "Delayed."
    assert _built_without_retries(built)


async def test_every_streamed_workflow_call_builds_no_retrying_client():
    spec = {
        "start": "kind",
        "nodes": [
            {
                "id": "kind",
                "type": "condition",
                "question": "Booking?",
                "branches": {"yes": "answer", "no": "answer"},
            },
            {"id": "answer", "type": "model", "tools": False},
        ],
    }
    request = _request()
    request.project = request.project.model_copy(
        update={"workflow": WorkflowSpec.model_validate(spec), "agent": "workflow"}
    )
    model = ScriptedModel(
        rounds=[[AIMessageChunk(content="yes")], [AIMessageChunk(content="On time.")]],
        seen=[],
    )

    with (
        patch("langgraph_agent.workflow.build_chat_model", return_value=model) as built,
        patch("langgraph_agent.workflow.retrieve_with_usage", new=_no_retrieval),
    ):
        events = [e async for e in WorkflowAgent(tools=FakeTools()).run(request)]

    assert "".join(e.text for e in events if isinstance(e, TokenEvent)) == "On time."
    assert len(built.call_args_list) == 2
    assert _built_without_retries(built)
