"""Which agent runs a turn."""

from __future__ import annotations

from unittest.mock import patch

from chatbot_engine.agent import registry
from chatbot_engine.agent.router import AgentRouter
from chatbot_engine.models.chat import AssistantConfig, ChatRequest


class Marker:
    """Stands in for an agent, and says which one it is when run."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, request):
        async def events():
            yield self.name

        return events()


def _request(agent: str | None) -> ChatRequest:
    return ChatRequest(
        project=AssistantConfig(
            project_id="p", name="n", system_prompt="s", agent=agent
        ),
        message="hi",
    )


def _named(*names: str):
    """A registry offering exactly `names`, each returning its own marker."""
    return patch.object(
        registry,
        "available_agents",
        lambda: {n: (lambda tools, n=n: Marker(n)) for n in names},
    )


async def _ran(agent: str | None) -> str:
    router = AgentRouter(tools=object())
    return await anext(router.run(_request(agent)))


async def test_the_config_chooses_the_agent() -> None:
    with _named("loop", "graph"):
        assert await _ran("graph") == "graph"
        assert await _ran("loop") == "loop"


async def test_no_choice_falls_back_to_the_engine_default() -> None:
    """An assistant that says nothing gets the engine's configured agent."""
    with _named("loop", "graph"):
        assert await _ran(None) == "loop"


async def test_an_agent_is_built_once_and_reused() -> None:
    """Agents hold no per-turn state, so rebuilding per message is waste."""
    built: list[str] = []

    def factory(tools):
        built.append("x")
        return Marker("loop")

    with patch.object(registry, "available_agents", lambda: {"loop": factory}):
        router = AgentRouter(tools=object())
        for _ in range(3):
            [item async for item in router.run(_request("loop"))]

    assert built == ["x"]
