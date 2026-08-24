"""The agent reports token usage and cost, summed across every model call.

A turn that calls a tool is several model calls; the usage must cover all of
them, not just the last. A scripted model stands in for the provider so the
token counts are known exactly.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from chatbot_engine.agent.chat_agent import ChatAgent
from chatbot_engine.models.chat import AssistantConfig, ChatRequest
from chatbot_engine.models.events import DoneEvent, TokenEvent, UsageEvent


class ScriptedModel(BaseChatModel):
    """Replays one scripted list of chunks per round, like a real stream."""

    rounds: list
    model_name: str = "test/model"

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError

    def bind_tools(self, tools, **kwargs):
        return self

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for chunk in self.rounds.pop(0):
            yield ChatGenerationChunk(message=chunk)


class FakeTools:
    """One tool the model may call; records what it was asked to run."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def list_tools(self, config):
        return [
            {
                "server": "s",
                "name": "get_booking_status",
                "description": "d",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]

    async def call_tool(self, *, name, **kwargs):
        self.calls.append(name)
        return '{"status": "delayed"}'


async def _no_retrieval(_request):
    return []


def _request(model="test/model") -> ChatRequest:
    return ChatRequest(
        project=AssistantConfig(
            project_id="support", name="S", system_prompt="p", model=model
        ),
        message="is my flight delayed?",
    )


async def _run(rounds, tools, model_name="test/model"):
    model = ScriptedModel(rounds=rounds, model_name=model_name)
    with (
        patch("chatbot_engine.agent.client.build_chat_model", return_value=model),
        patch("chatbot_engine.agent.chat_agent.retrieve", new=_no_retrieval),
    ):
        return [
            event async for event in ChatAgent(tools=tools).run(_request(model_name))
        ]


def _usage_chunk(prompt: int, completion: int) -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        usage_metadata={
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": prompt + completion,
        },
    )


@pytest.mark.asyncio
async def test_usage_and_cost_summed_across_tool_rounds() -> None:
    # Round 1 asks for a tool (30/10); round 2 answers (50/20).
    rounds = [
        [
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "get_booking_status", "args": "{}", "id": "c1", "index": 0}
                ],
            ),
            _usage_chunk(30, 10),
        ],
        [AIMessageChunk(content="Delayed."), _usage_chunk(50, 20)],
    ]
    tools = FakeTools()

    events = await _run(rounds, tools, model_name="openai/gpt-5-mini")

    assert tools.calls == ["get_booking_status"]
    usage = next(e for e in events if isinstance(e, UsageEvent))
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (80, 30, 110)
    # 80 in @ $0.25/1M + 30 out @ $2.00/1M = $0.00008.
    assert usage.cost_usd == pytest.approx(0.00008)
    assert usage.model == "openai/gpt-5-mini"


@pytest.mark.asyncio
async def test_usage_event_comes_before_done() -> None:
    rounds = [[AIMessageChunk(content="Hi."), _usage_chunk(5, 2)]]

    events = await _run(rounds, FakeTools())
    types = [type(e).__name__ for e in events]

    assert "".join(e.text for e in events if isinstance(e, TokenEvent)) == "Hi."
    assert types.index("UsageEvent") < types.index("DoneEvent")
    assert isinstance(events[-1], DoneEvent)


@pytest.mark.asyncio
async def test_cost_is_none_for_an_unpriced_model() -> None:
    rounds = [[AIMessageChunk(content="Hi."), _usage_chunk(5, 2)]]

    events = await _run(rounds, FakeTools(), model_name="mystery/model")

    usage = next(e for e in events if isinstance(e, UsageEvent))
    assert usage.cost_usd is None
