"""Which agents this engine can run, including ones it did not ship with.

An agent is the thing that runs a chat turn. The engine ships two, but the
interesting case is the third one: an adopter who wants their own graph, their
own tool loop, or a different framework entirely, without forking the engine.

They register it as a Python entry point. In their own package:

    # pyproject.toml
    [project.entry-points."chatbot_engine.agents"]
    my-graph = "my_package.agent:build"

    # my_package/agent.py
    def build(tools: ToolProvider) -> Agent:
        return MyGraphAgent(tools=tools)

Install that package alongside the engine and `agent: my-graph` in the assistant
config selects it. Nothing in the engine changes, and no fork is involved.

The factory takes the `ToolProvider` because that is the one thing an agent
cannot construct for itself: it is how the engine reaches the application's
tools. Everything else about a turn arrives with the request.

This is why `agent` is an open string on the wire rather than a fixed set. The
valid values depend on what is installed, so they are checked here, at the point
where that is known, and the error names what this engine actually has.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from chatbot_engine.errors import EngineError

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime otherwise
    from chatbot_engine.ports.agent import Agent, ToolProvider

#: Entry-point group third-party agents register under.
ENTRY_POINT_GROUP = "chatbot_engine.agents"

#: Builds an agent from the one dependency it cannot make for itself.
AgentFactory = Callable[["ToolProvider"], "Agent"]


class UnknownAgentError(EngineError):
    """The assistant asked for an agent this engine does not have installed.

    A caller's mistake rather than a broken engine, so the API maps it to 422
    and lists the agents that are available.
    """


def _builtin() -> dict[str, AgentFactory]:
    """The two agents the engine ships with.

    Imported lazily: `graph` pulls in LangGraph, which is an optional extra, and
    listing the available agents must not require it.
    """

    def loop(tools: "ToolProvider") -> "Agent":
        from chatbot_engine.agent.chat_agent import ChatAgent

        return ChatAgent(tools=tools)

    def graph(tools: "ToolProvider") -> "Agent":
        from chatbot_engine.agent.graph_agent import LangGraphAgent

        return LangGraphAgent(tools=tools)

    return {"loop": loop, "graph": graph}


def available_agents() -> dict[str, AgentFactory]:
    """Every agent this engine can run: the built-ins, plus installed plugins.

    A plugin may replace a built-in by registering the same name. That is
    deliberate: swapping the default `loop` for your own implementation should
    not require the engine to grant permission.
    """
    factories = _builtin()

    for point in entry_points(group=ENTRY_POINT_GROUP):
        factories[point.name] = point.load()

    return factories


def build_agent(name: str, tools: "ToolProvider") -> "Agent":
    """Construct the named agent, or say which names this engine knows.

    The list in the message is the installed set, not a hardcoded one, so it
    stays true when a plugin is added or removed.
    """
    factories = available_agents()

    factory = factories.get(name)
    if factory is None:
        raise UnknownAgentError(
            f"unknown agent {name!r}; this engine has {sorted(factories)}. "
            f"Register your own under the {ENTRY_POINT_GROUP!r} entry-point "
            "group, or pick one of the above."
        )

    return factory(tools)
