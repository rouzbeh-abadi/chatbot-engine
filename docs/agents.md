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
FROM ghcr.io/rouzbeh-abadi/chatbot-engine/engine:0.1.20
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
| `done` | last; `finish_reason` is `stop`, `length` (cut at `max_output_tokens`) or `tool_limit` (still asking for tools after `max_tool_iterations` rounds) |

**System prompt.** Send `request.project.system_prompt` to the model ahead of
the conversation. This is not visible in the event stream: an agent that omits
it still emits a well-formed stream, and only the content of the answer
changes. The persona, the grounding rules and any notes the backend appended
to the prompt are all lost.

**Model calls.** Stream through `chatbot_engine.agent.client.stream_reply`.
It retries a stream that fails before
its first token on rate limits, provider 5xx and dropped connections, and
passes the failure on once text has been shown. Build the model with
`build_chat_model`, which applies the assistant's `max_output_tokens`, and
read the reason a reply ended with `finish_reason_of`.

**Tool execution.** Use `chatbot_engine.agent.client.run_tool_calls`. It
forwards the caller's `user_id` and `session_id` to the tool server, reports a
failed tool as `ok=false` rather than ending the turn, and feeds the result back
as a `ToolMessage`. Two agents that ran tools differently would report them
differently.

**Tool rounds.** After `max_tool_iterations` rounds with the model still
asking for tools, end the turn with `usage` and `done` saying `tool_limit`.
What the model said so far has streamed; failing at that point would leave
the caller with text and an error.

**Cost.** Retrieve with `retrieve_with_usage()`, which returns the hits and
the token counts of retrieval's own model calls, and pass those counts to
`stream_completion(..., prior=...)`, so the turn's `usage` covers every call
it made. Price with `chatbot_engine.agent.client.price_usage`, so both agents
price a turn identically.

`engine/tests/test_agent_parity.py` asserts these points for `loop` and
`graph`, including the `length` and `tool_limit` endings;
`engine/tests/test_workflow_agent.py` covers the same for `workflow`.

## The bundled plugin

[`examples/langgraph-agent/`](../examples/langgraph-agent) is a complete plugin:

```text
examples/langgraph-agent/
├── pyproject.toml                    the entry points
└── langgraph_agent/
    ├── __init__.py
    ├── agent.py                      the fixed graph: retrieve, model, tools
    ├── workflow.py                   a graph built per request from the assistant's workflow
    └── runner.py                     runs a compiled graph and streams what its nodes emit
```

The entry points are the only connection to the engine:

```toml
[project.entry-points."chatbot_engine.agents"]
graph = "langgraph_agent.agent:build"
workflow = "langgraph_agent.workflow:build"
```

What the plugin does not do is reimplement the engine. Its nodes call the
engine's helpers: `stream_reply` for every model call (so the first-token
retry applies), `run_tool_calls` for every tool call (so timing, failure
handling and the started and finished events are the engine's), `retrieve_with_usage`
and `price_usage`. The plugin's own code is the graph shape and `runner.py`,
which runs the compiled graph as a task and drains the queue its nodes push
events onto. Two agents that streamed, retried or ran tools differently
would be a bug, and the parity tests would catch it.

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
| `model` | calls the assistant's model with the prompt, the conversation and the retrieved passages, streams the reply as the answer, and runs the tools it asks for, up to `max_tool_iterations` rounds (`tools: false` disables them; running out ends the turn with `tool_limit`); the reply is capped by `max_output_tokens`; `prompt` appends instructions for this step; `var` stores the reply in a variable instead of speaking it |
| `condition` | asks the utility model (`ENGINE_UTILITY_MODEL`, temperature 0) one question, expecting one of the branch labels, and follows that branch; the answer is matched exactly, then as a whole word, and the first label is the fallback |
| `tool` | calls one tool, allowlisted on one of the assistant's `mcp_servers`, with templated arguments, through the same runner as a model's own tool calls, and stores the result text in `var`; reported as `tool_call_started` and `tool_call_finished` with the real duration. When the call fails or the tool is not offered right now (its server is down, or no longer has it), `on_error: "stop"` (the default) streams the assistant's `unavailable_message` and ends the turn, so steps that assume the call worked never run; `on_error: "continue"` goes on with `var` empty |
| `reply` | streams a fixed, templated text as the answer |
| `handoff` | streams a message, sets `vars.handed_off` to `true`, and, when `tool` is named, calls it with `reason` (templated, with a default) and the transcript through the same runner, so a ticket or an email can be raised and the call shows in the log |
| `ask` | pauses the turn to ask the visitor one thing (`input`: `text`, `phone`, `email`, `url`, or `choice` with `options` or `options_from` a variable holding a JSON list), and continues with the answer in `var` (and a choice's label in `<var>_label`); `optional` allows skipping, which leaves both empty. With `understand` (on by default) the reply is read first: an answer in other words keeps only the value, a visitor who declines goes to `on_decline` (or hears `decline_reply`), and a reply that does not answer is replied to and asked again up to `retries` times, then goes to `on_other`. See "Asking the visitor" below |
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

A tool argument that renders empty (a skipped question, an unset variable) is
left out of the call, so the tool sees it as not given.

### Asking the visitor

The `ask` step is LangGraph's human-in-the-loop. Inside the step,
`interrupt()` stops the graph; a checkpointer (SQLite, at
`ENGINE_CHECKPOINT_DB`) keeps its state; the response ends with an
`input_required` event describing the question. The visitor's answer arrives
as a new request carrying `resume`, the agent rebuilds the same graph, and
`Command(resume=...)` continues it: the step runs again from the top,
`interrupt()` returns the answer this time, and the graph goes on.

```json
{"id": "phone", "type": "ask", "prompt": "What number should we call?", "input": "phone", "var": "phone"},
{"id": "slots", "type": "tool", "tool": "list_callback_slots", "var": "slots"},
{"id": "when", "type": "ask", "prompt": "When suits you?", "input": "choice", "options_from": "slots", "var": "slot"},
{"id": "book", "type": "tool", "tool": "request_callback",
 "arguments": {"phone": "{{vars.phone}}", "slot": "{{vars.slot}}", "reason": "{{message}}"}, "var": "ticket"}
```

What the step guarantees:

- **The answer is checked where it is used.** A phone number must have 6 to
  15 digits (spaces, dashes and brackets are dropped), an email must look
  like one, a web address gains `https://` when it has no scheme, and a
  choice must be one of the options offered. A refused answer asks again with
  `error` set.
- **Nothing is counted twice.** Usage up to the pause is reported when the
  turn pauses; the resumed part reports only its own.
- **`{{message}}` stays the message that started the turn,** not the answer.
  A model step after the pause reads the conversation the resumed request
  sends, plus only what the graph produced after the pause.
- **A resume is scoped.** It must come from the same project and session,
  within `ENGINE_PAUSE_TTL_S`, and for the same workflow; a finished turn's
  state is deleted, and a workflow with no `ask` step never touches the
  checkpointer.
- **A choice with no options is skipped,** leaving the variable empty, so a
  tool that found no slots does not produce an empty question.
- **A reply is read before it is kept** (`understand`, on by default), since
  people answer in their own words. A reply that already fits (a valid phone
  number, one of the options) is kept as it is, without a model call. Any
  other goes to the utility model, which says what it is:
  - **an answer**, in any words or language: only the value is kept ("call it
    Apollo please" gives `Apollo`; "the afternoon one" picks that option);
  - **a no**: the visitor declines, cancels or changes their mind. The turn
    goes to `on_decline`, or says `decline_reply` (or, without one, a short
    acknowledgement in the visitor's language that the model wrote) and ends,
    so "no, I don't want a new project" never becomes a project's name;
  - **something else**, such as a question back or another topic: it is
    replied to from the knowledge base and the question is asked again, up to
    `retries` times (1 by default, 0 to 3). After that the turn goes to
    `on_other`, or replies once more and leaves the question.

  A text reply is always read; a valid phone number, address or option is
  kept without a call, and so is an option named in other letters ("no" for
  a "No" option) and the skip by its name. An answer that is an attempt but
  does not fit ("+44 20" for a phone number, a digit short) is asked again
  with the reason, as often as it takes, and does not count against
  `retries`; a refusal ("my number is secret") is a no. When the question may
  be skipped, a reply that says there is nothing to give skips it, as pressing
  the skip does. If the reading cannot be had, the question is asked again
  and keeps waiting; if it cannot be had twice in a row, the reply is taken
  as it stands, as with `understand` off, so the question can still be
  answered. `understand: false` keeps every reply that
  passes the check, as it is. Reading a reply is one small call on the
  utility model, counted in the turn's usage as the utility part; the
  `input_required` event carries `understand`, so a client knows whether
  typed words are read.

```json
{"id": "name", "type": "ask", "prompt": "What should the project be called?", "input": "text",
 "var": "name", "retries": 1, "on_decline": "no_project", "on_other": "handoff"}
```
