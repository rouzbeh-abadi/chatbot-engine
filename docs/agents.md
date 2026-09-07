# Agents

An agent runs one chat turn: retrieve context, call the model, run any tools it
asks for, and stream the result back as events.

The engine ships exactly one, and it is not the interesting part. What matters is
that an agent is a *plugin*: you install your own, name it in the assistant
config, and the engine runs it without knowing anything about how it works.

| Agent | Where it comes from | Selected as |
| --- | --- | --- |
| `loop` | the engine. A `for` loop over model calls and tool rounds | `agent: loop` |
| `graph` | the `examples/langgraph-agent` package, installed as a plugin | `agent: graph` |
| *yours* | your own package | `agent: your-name` |

## Why the engine ships only one

Choosing LangGraph, or any other framework, is an application decision. An
engine that bundled one would be making that decision for every adopter and
carrying the dependency whether or not they wanted it.

So the engine defines two things and stops:

- the **`Agent` port** in `ports/agent.py`: one method, `run(request)`, yielding events
- the **registry** in `agent/registry.py`: how an agent is discovered and named

Everything else lives outside it. `grep langgraph engine/src` returns nothing.

## Choosing one

```yaml
# backend/src/support_agent/projects/support.yaml
agent: loop     # or graph, or the name of a plugin you installed
```

Omit it and the engine's default applies (`ENGINE_AGENT`, itself `loop`). The
example UI also offers a dropdown, and a request may override the project's
choice per turn, so two agents can be compared without redeploying.

## What crosses the wire

Nothing but a name. A graph is code, and code does not travel over HTTP:

```text
backend  --POST /chat-->  {"project": {..., "agent": "graph"}, ...}
                                              |
engine   registry.build_agent("graph") -------+
             |
             +--> the entry point of an installed package
```

The backend **selects** an agent; whoever deploys the engine **provides** it, by
installing the package. That is the same division as a database and its
extensions: the application picks, the deployment installs.

## Installing your own

**1. Write it.** Anything with a `run` method that yields the engine's events:

```python
# my_package/agent.py
class MyAgent:
    def __init__(self, tools):        # the engine's MCP tool provider
        self._tools = tools

    def run(self, request):
        async def events():
            yield TokenEvent(text="...")
            yield DoneEvent(finish_reason="stop")
        return events()

def build(tools) -> MyAgent:          # the factory the entry point names
    return MyAgent(tools)
```

**2. Register it** in your own `pyproject.toml`:

```toml
[project.entry-points."chatbot_engine.agents"]
my-agent = "my_package.agent:build"
```

**3. Install it into the environment the engine runs in.** For a container, that
means building your own engine image:

```dockerfile
FROM your-engine-image
COPY my-agent /opt/my-agent
RUN pip install /opt/my-agent
```

This repository does exactly that for its own plugin, in
[`docker/engine-with-plugins.Dockerfile`](../docker/engine-with-plugins.Dockerfile).
`engine/Dockerfile` stays framework-free and still builds on its own.

**4. Select it** with `agent: my-agent`. It appears in `GET /agents` and the
example UI's dropdown automatically, because both read the installed set rather
than a hardcoded list.

A plugin may register a name a built-in already uses, which replaces it.
Swapping out `loop` for your own implementation needs no permission from the
engine.

## What your agent owes the caller

The engine does not police this, but the example backend and UI both assume it:
emit the same events in the same order.

| Event | When |
| --- | --- |
| `retrieval` | once, before the answer, if you retrieved anything |
| `token` | repeatedly, as the answer streams |
| `tool_call_started` / `tool_call_finished` | around each tool call |
| `usage` | once, after the answer |
| `done` | last |

`engine/tests/test_agent_parity.py` asserts that the engine's `loop` and the
plugin's `graph` are indistinguishable on all of it. An agent written outside
the engine is not a second-class citizen, and that test is what keeps it true.

## The bundled example

[`examples/langgraph-agent/`](../examples/langgraph-agent) is a complete, working
plugin: its own `pyproject.toml`, its own LangGraph dependency, and a graph of
four nodes with one conditional edge.

```text
START -> retrieve -> model -+-(tool calls)-> tools -+
                            |                       | (back to model)
                            +-(none)-> finish -> END
```

Copy it as the starting point for your own. It is installed by the demo stack
and exercised by the test suite, so it cannot quietly rot.

**A graph is worth the machinery** once a turn stops being a straight line:
pausing for human approval mid-turn, resuming a half-finished turn from a
checkpointer, or branching on the kind of question. For a single question and
answer, the engine's plain loop is simpler and does the same job, which is why
it stays the default.

## When the name is not installed

Selecting an agent the engine does not have is a `422`, and the message lists
what it does have:

```json
{
  "detail": "unknown agent 'my-agent'; this engine has ['graph', 'loop']. Register your own under the 'chatbot_engine.agents' entry-point group, or pick one of the above."
}
```

The list is the real installed set, so it stays true as plugins come and go.
