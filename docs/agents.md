# Agents

An agent runs one chat turn: retrieve context, call the model, run any tools it
asks for, and stream the result back as events. The engine ships two, and the
assistant config picks per request. The set is open: you can install your own
and select it by name without forking the engine, which is what
[Installing your own agent](#installing-your-own-agent) covers.

Both produce the same events, in the same order, with the same token counts.
Switching between them changes how the turn is *organised*, not what the caller
receives.

| Agent | What it is | Needs |
| --- | --- | --- |
| `loop` | A `for` loop over model calls and tool rounds. The default. | nothing |
| `graph` | The same turn as a LangGraph state machine. | the `graph` extra |
| *yours* | Whatever you register under the entry-point group. | your package |

## Choosing one

```yaml
# backend/src/support_agent/projects/support.yaml
agent: loop     # loop | graph
```

Omit it and the engine's own default applies (`ENGINE_AGENT`, itself `loop`).
Because the choice travels with the request, one engine can serve both, which is
what makes them comparable without redeploying.

## `loop`, the default

`agent/chat_agent.py` and `agent/client.py`. Retrieval, then a bounded loop: call
the model, and if it asked for tools, run them, feed the results back, and go
round again until it answers in prose. `max_tool_iterations` bounds it.

It is the default because it is the smaller thing. A single question and answer
is a straight line, and a straight line reads better as a loop than as a graph.

## `graph`

`agent/graph_agent.py`. The identical turn as three nodes and one conditional
edge:

```
START -> retrieve -> model -+-(tool calls)-> tools -+
                            |                       | (back to model)
                            +-(none)-> finish -> END
```

State carries the messages, the running token usage, and the model's name. The
one branch in a turn, "did the model ask for a tool?", becomes a conditional
edge rather than an `if` inside a loop.

## When the graph is worth it

Today it is a faithful re-expression of the loop: same inputs, same events, same
answer, more machinery. On that comparison alone the loop wins.

The graph earns its place when a turn stops being a straight line:

- **Interrupts and human-in-the-loop.** Pause before a tool runs, wait for
  approval, resume. A loop has nowhere to pause.
- **Persistence and resumption.** A LangGraph checkpointer can save the turn's
  state and resume it after a crash, or let a conversation be reloaded later.
- **Branching.** Route a question to different paths, or run a sub-agent, without
  the loop growing conditionals.

If none of that is on your roadmap, stay on `loop`.

## Installing the extra

LangGraph is optional, so the base install does not carry it:

```bash
pip install "chatbot-engine[graph]"
```

Selecting `agent: graph` without it fails with a message naming the extra rather
than an import traceback. LangGraph is imported only when a request actually
selects the graph agent, so the base install pays nothing for its existence.

## Keeping the two honest

Two agents are only useful if a caller cannot tell them apart. Three things keep
them aligned, and the suite checks it:

- **Tools run through the same code.** The graph's tool node calls the same
  `run_tool_calls` the loop uses, rather than reimplementing it.
- **Cost is priced by the same helper**, so the two can never disagree about
  what a turn cost.
- **Events are emitted by the nodes themselves**, onto a queue, rather than
  reconstructed from LangGraph's stream. The nodes know what happened; a stream
  of graph updates has to be interpreted, and the interpretation shifts between
  LangGraph versions.

`engine/tests/test_agent_parity.py` runs the same scripted conversation through
both and asserts the event sequences, answer text, tool calls and usage match.
`engine/tests/test_agent_selection_api.py` does it again over HTTP.

## Installing your own agent

The two built-ins are not special. An agent is anything with one method, and the
engine discovers third-party ones through a Python entry point, so you can add
yours without forking the engine or editing its code.

**1. Write it.** Anything satisfying the `Agent` port in `ports/agent.py`:

```python
# my_package/agent.py
from collections.abc import AsyncIterator

class MyGraphAgent:
    def __init__(self, tools):        # the engine's MCP tool provider
        self._tools = tools

    def run(self, request) -> AsyncIterator[Event]:
        ...                            # yield the same events the built-ins do

def build(tools) -> MyGraphAgent:
    return MyGraphAgent(tools)
```

**2. Register the factory** in your own package:

```toml
[project.entry-points."chatbot_engine.agents"]
my-graph = "my_package.agent:build"
```

**3. Install it alongside the engine**, and name it:

```yaml
agent: my-graph
```

The factory receives the `ToolProvider` because that is the one thing an agent
cannot construct for itself: it is how the engine reaches your application's
tools over MCP. Everything else about a turn arrives with the request.

A plugin may register a name a built-in already uses, which replaces it.
Swapping out the default `loop` for your own implementation should not require
the engine's permission.

### What your agent owes the caller

The engine does not police this, but the frontend and the backend both assume
it: emit the same events the built-ins emit, in the same order. `retrieval`
first if you retrieved, `token` as the answer streams, `tool_call_started` and
`tool_call_finished` around each tool, `usage`, then `done` last.
`engine/tests/test_agent_parity.py` shows what that looks like asserted.

### When the name is not installed

Selecting an agent this engine does not have is a `422`, and the message lists
what it does have:

```json
{
  "detail": "unknown agent 'my-graph'; this engine has ['graph', 'loop']. Register your own under the 'chatbot_engine.agents' entry-point group, or pick one of the above."
}
```

The list is the real installed set rather than a hardcoded one, so it stays true
as plugins come and go.
