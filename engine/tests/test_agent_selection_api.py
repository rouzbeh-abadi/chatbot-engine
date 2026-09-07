"""The real plugin, selected over HTTP.

The parity tests construct both agents directly. This proves the other half:
`agent` on the request reaches the router through the real wiring, and the
bundled plugin answers a turn over the wire. The model and retrieval are
patched, so no provider is called.
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
    return [], {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


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
        patch("langgraph_agent.agent.build_chat_model", return_value=model),
        patch("chatbot_engine.agent.chat_agent.retrieve_with_usage", new=_no_retrieval),
        patch("langgraph_agent.agent.retrieve_with_usage", new=_no_retrieval),
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
