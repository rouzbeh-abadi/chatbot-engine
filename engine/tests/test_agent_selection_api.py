"""Choosing the agent over HTTP.

The parity tests drive both agents directly. This proves the other half: that
`agent` on the request actually reaches the router, that the selected agent runs
the turn, and that the events come back over the wire the same either way.

The model and retrieval are patched, so no provider is called; everything else
is the real route, the real dependency wiring, and the real router.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk


class OneWordModel(BaseChatModel):
    """Answers "Hi." and reports its token usage, without calling anything."""

    model_name: str = "openai/gpt-5-mini"

    @property
    def _llm_type(self) -> str:
        return "one-word"

    def _generate(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError

    def bind_tools(self, tools, **kwargs):
        return self

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(content="Hi."))
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                usage_metadata={
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "total_tokens": 7,
                },
            )
        )


async def _no_retrieval(_request):
    return []


def _events(body: str) -> list[dict]:
    """The NDJSON stream, one object per line."""
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def _ask(client: TestClient, project: dict, agent: str | None) -> list[dict]:
    """Post one turn through the real route, with the model stubbed out."""
    config = {**project, "model": "openai/gpt-5-mini"}
    if agent is not None:
        config["agent"] = agent

    model = OneWordModel()
    with (
        patch("chatbot_engine.agent.client.build_chat_model", return_value=model),
        patch("chatbot_engine.agent.graph_agent.build_chat_model", return_value=model),
        patch("chatbot_engine.agent.chat_agent.retrieve", new=_no_retrieval),
        patch("chatbot_engine.agent.graph_agent.retrieve", new=_no_retrieval),
    ):
        response = client.post("/chat", json={"project": config, "message": "hello"})

    assert response.status_code == 200, response.text
    return _events(response.text)


@pytest.mark.parametrize("agent", ["loop", "graph"])
def test_either_agent_answers_over_http(
    client: TestClient, project: dict[str, object], agent: str
) -> None:
    events = _ask(client, project, agent)
    types = [event["type"] for event in events]

    assert "token" in types
    assert types[-1] == "done"
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert answer == "Hi."


def test_omitting_the_agent_still_answers(
    client: TestClient, project: dict[str, object]
) -> None:
    """An assistant config that says nothing gets the engine's default agent."""
    events = _ask(client, project, None)

    assert [e["type"] for e in events][-1] == "done"


def test_both_agents_return_the_same_stream_over_http(
    client: TestClient, project: dict[str, object]
) -> None:
    """Switching `agent` must not change what the browser receives."""
    loop = _ask(client, project, "loop")
    graph = _ask(client, project, "graph")

    assert [e["type"] for e in loop] == [e["type"] for e in graph]

    def usage(events: list[dict]) -> dict:
        return next(e for e in events if e["type"] == "usage")

    assert usage(loop)["total_tokens"] == usage(graph)["total_tokens"]
    assert usage(loop)["cost_usd"] == usage(graph)["cost_usd"]


def test_an_unknown_agent_is_rejected(
    client: TestClient, project: dict[str, object]
) -> None:
    """`agent` is an open string on the wire, because the valid set depends on
    what is installed. The registry checks it and answers 422."""
    response = client.post(
        "/chat",
        json={"project": {**project, "agent": "banana"}, "message": "hello"},
    )

    assert response.status_code == 422
