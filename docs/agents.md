# Agents

An agent runs one chat turn: it retrieves context, calls the model, runs the
tools the model asks for, and streams the result back as events.

The engine ships one agent and defines the contract for others. Any agent is a
plugin: a package installed into the engine's environment, registered under a
Python entry point, and selected by name in the assistant configuration.

| Agent | Provided by | Selected as |
| --- | --- | --- |
| `loop` | the engine: a loop over model calls and tool rounds | `agent: loop` |
| `graph` | `examples/langgraph-agent`, installed as a plugin | `agent: graph` |
| any other | your own package | `agent: <name>` |

## Scope of the engine

The engine defines two things:

- the `Agent` protocol in `ports/agent.py`: one method, `run(request)`, that
  returns an async iterator of events
- the registry in `agent/registry.py`: how agents are discovered and named

It depends on no agent framework. `grep langgraph engine/src` returns nothing.
The choice of framework is made by whoever writes an agent, not by the engine.

## Selection

```yaml
# examples/backend/src/support_agent/projects/support.yaml
agent: loop
```

When the field is omitted, `ENGINE_AGENT` applies, which defaults to `loop`. A
request may override the project's choice per turn; the example UI exposes this
as a dropdown populated from `GET /agents`.

Only the name crosses the wire. The backend selects an agent; the deployment
provides it by installing the package:

```text
backend  --POST /chat-->  {"project": {..., "agent": "graph"}, ...}
                                              |
engine   registry.build_agent("graph") -------+
             |
             +--> the entry point of an installed package
```

## Writing an agent

**1. Implement the protocol.** Any object with a `run` method that yields the
engine's events:

```python
# my_package/agent.py
class MyAgent:
    def __init__(self, tools):  # the engine's ToolProvider
        self._tools = tools

    def run(self, request):
        async def events():
            yield TokenEvent(text="...")
            yield DoneEvent(finish_reason="stop")

        return events()


def build(tools) -> MyAgent:  # the factory the entry point names
    return MyAgent(tools)
```

`tools` is the engine's MCP tool provider, the one dependency an agent cannot
construct for itself. Everything else about a turn arrives in `request`.

**2. Register the factory** in the package's `pyproject.toml`:

```toml
[project.entry-points."chatbot_engine.agents"]
my-agent = "my_package.agent:build"
```

**3. Install the package into the engine's environment.** For a container, build
an image on top of the engine image:

```dockerfile
FROM ghcr.io/rouzbeh-abadi/chatbot-engine/engine:0.1.5
COPY my-agent /opt/my-agent
RUN pip install /opt/my-agent
```

The published engine image carries no plugins. This repository builds its demo
engine the same way, in
[`docker/engine-with-plugins.Dockerfile`](../docker/engine-with-plugins.Dockerfile),
which is why `agent: graph` is available under `docker compose` and not from
the published image alone.

**4. Select it** with `agent: my-agent`. It appears in `GET /agents` and in the
example UI without further configuration, because both read the installed set.

A plugin may register a name the engine already uses. The plugin wins, so
`loop` itself can be replaced without modifying the engine.

## The agent contract

The engine does not enforce the following, but the example backend and UI
depend on it.

**Events.** Emit the same events in the same order as `loop`:

| Event | When |
| --- | --- |
| `retrieval` | once, before the answer, when anything was retrieved |
| `token` | repeatedly, as the answer streams |
| `tool_call_started`, `tool_call_finished` | around each tool call |
| `usage` | once, after the answer |
| `done` | last |

**System prompt.** Send `request.project.system_prompt` to the model ahead of
the conversation. This is not visible in the event stream: an agent that omits
it still emits a well-formed stream, and only the content of the answer
changes. The persona, the grounding rules and any notes the backend appended
to the prompt are all lost.

**Tool execution.** Use `chatbot_engine.agent.client.run_tool_calls`. It
forwards the caller's `user_id` and `session_id` to the tool server, reports a
failed tool as `ok=false` rather than ending the turn, and feeds the result back
as a `ToolMessage`. Two agents that ran tools differently would report them
differently.

**Cost.** Retrieve with `retrieve_with_usage()`, which returns the hits and
the token counts of retrieval's own model calls, and pass those counts to
`stream_completion(..., prior=...)`, so the turn's `usage` covers every call
it made. Price with `chatbot_engine.agent.client.price_usage`, so both agents
price a turn identically.

`engine/tests/test_agent_parity.py` asserts all four points for `loop` and
`graph`.

## The bundled plugin

[`examples/langgraph-agent/`](../examples/langgraph-agent) is a complete plugin:

```text
examples/langgraph-agent/
├── pyproject.toml                    the entry point
└── langgraph_agent/
    ├── __init__.py
    └── agent.py                      the graph
```

The entry point is the only connection to the engine:

```toml
[project.entry-points."chatbot_engine.agents"]
graph = "langgraph_agent.agent:build"
```

The graph it builds:

```python
graph = StateGraph(_State)
graph.add_node("retrieve", retrieve_node)
graph.add_node("model", model_node)
graph.add_node("tools", tools_node)
graph.add_node("finish", finish_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "model")
graph.add_conditional_edges("model", next_step, {"tools": "tools", "finish": "finish"})
graph.add_edge("tools", "model")
graph.add_edge("finish", END)

return graph.compile()
```

```text
START -> retrieve -> model -+-(tool calls)-> tools -+
                            |                       | (back to model)
                            +-(none)-> finish -> END
```

Installation is what makes it appear:

```console
$ python -c "from chatbot_engine.agent.registry import available_agents; print(sorted(available_agents()))"
['loop']

$ pip install ./examples/langgraph-agent

$ python -c "from chatbot_engine.agent.registry import available_agents; print(sorted(available_agents()))"
['graph', 'loop']
```

A graph is the appropriate structure when a turn stops being a straight line:
pausing for approval mid-turn, resuming a partial turn from a checkpointer, or
branching on the kind of question. For a single question and answer the
engine's loop does the same work with less, which is why it is the default.

## Errors

Selecting an agent the engine does not have is a `422` whose message lists the
installed set:

```json
{
  "detail": "unknown agent 'my-agent'; this engine has ['graph', 'loop']. Register your own under the 'chatbot_engine.agents' entry-point group, or pick one of the above."
}
```

The list is read from the installed packages at request time.

## Workflows

An assistant can describe its turn as a graph instead of relying on an
agent's built-in shape. Set `agent: workflow` and send a `workflow` block
with the assistant config: nodes from a fixed library, edges between them,
and a start node. The `workflow` agent (in the LangGraph plugin) builds a
LangGraph from it for each request and streams the same events as every
other agent, so the caller's UI needs nothing new.

```json
"agent": "workflow",
"workflow": {
  "start": "retrieve",
  "nodes": [
    {"id": "retrieve", "type": "retrieve"},
    {"id": "kind", "type": "condition",
     "question": "Is the visitor asking about a specific order?",
     "branches": {"order": "lookup", "other": "answer"}},
    {"id": "lookup", "type": "tool", "tool": "get_booking_status",
     "arguments": {"reference": "{{message}}"}, "var": "booking"},
    {"id": "answer", "type": "model", "prompt": "Booking data: {{vars.booking}}"}
  ],
  "edges": [{"from": "retrieve", "to": "kind"}, {"from": "lookup", "to": "answer"}]
}
```

| Node | Does |
| --- | --- |
| `retrieve` | searches the knowledge base; later model steps see the passages |
| `model` | calls the assistant's model with the prompt, the conversation and the retrieved passages, streams the reply as the answer, and runs the tools it asks for, up to `max_tool_iterations` rounds (`tools: false` disables them); `prompt` appends instructions for this step; `var` stores the reply in a variable instead of speaking it |
| `condition` | asks the utility model (`ENGINE_UTILITY_MODEL`, temperature 0) one question, expecting one of the branch labels, and follows that branch; the answer is matched exactly, then as a whole word, and the first label is the fallback |
| `tool` | calls one tool, allowlisted on one of the assistant's `mcp_servers`, with templated arguments, and stores the result text in `var`; the call is reported as `tool_call_started` and `tool_call_finished` events like any other |
| `reply` | streams a fixed, templated text as the answer |
| `handoff` | streams a message, sets `vars.handed_off` to `true`, and, when `tool` is named, calls it with the transcript so a ticket or an email can be raised |
| `end` | finishes the turn; the same as a node with no outgoing edge |

Templates in `prompt`, `text`, `message` and tool arguments may use
`{{message}}`, `{{user_id}}`, `{{session_id}}` and `{{vars.<name>}}`; a
condition also sets `vars.condition_<id>` to the label it chose. A node with
no outgoing edge ends the turn, after which the agent emits the `usage` and
`done` events. The schema refuses unknown node ids,
unreachable nodes, a condition with edges, and a node with two outgoing
edges; `max_steps` (default 30) caps the visits in one turn. Every node
appears as a step in the trace when tracing is on. Without a `workflow`, the
agent runs retrieve then model, the same shape as the `graph` agent.
