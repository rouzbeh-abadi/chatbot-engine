# Connecting a backend to the chatbot engine

The engine is a separate HTTP service. It retrieves, prompts a model, and calls
tools. It holds no knowledge of users, products or domains; the backend
supplies all of that on every request and relays the answer to its own client.

```text
Browser
   ↓  your API
Your backend        :8000
   ↓  HTTP + NDJSON
Chatbot engine      :8100
   ↓  MCP
Your tool server    :8200
```

The engine never reads the backend's database and never stores its
configuration. The one call in the other direction is MCP: the engine invokes
tools that run in the backend, against the backend's data.

**Contents**

1. [Prerequisites](#1-prerequisites)
2. [Endpoints](#2-endpoints)
3. [Sending a chat message](#3-sending-a-chat-message)
4. [Reading the answer](#4-reading-the-answer)
5. [Uploading documents](#5-uploading-documents)
6. [Handling failure](#6-handling-failure)
7. [Exposing tools](#7-exposing-tools)
8. [A complete example](#8-a-complete-example)
9. [Reference implementation](#9-reference-implementation)

---

## 1. Prerequisites

The engine must be running and reachable.

```bash
make engine          # http://localhost:8100
curl localhost:8100/health
```

```json
{"status": "ok", "service": "chatbot-engine", "version": "0.1.0"}
```

Two settings on the backend side:

| Setting | Value | Notes |
| --- | --- | --- |
| `BACKEND_ENGINE_URL` | `http://localhost:8100` | `http://engine:8100` under Docker Compose |
| `BACKEND_ENGINE_API_KEY` | matches the engine's `ENGINE_API_KEY` | Both blank on localhost |

When the engine has a key configured, send it as `X-API-Key` on every request
except `/health`. Without a key the engine is open. `ENGINE_ENV=production`
refuses to start in that state, because the engine holds the model provider's
credentials.

The engine also accepts named keys, for deployments with more than one caller:

```
ENGINE_API_KEYS=web:s3cret,batch:0ther
```

Rate limits are counted per name, and a key can be rotated by running the old
and new names side by side. The sending side is unchanged: one `X-API-Key`
header.

`429` is a normal response on the routes that spend provider credits. It
carries `Retry-After` in seconds.

**A browser must not call the engine directly.** The engine has no notion of
end-user permissions. The backend decides who may ask.

---

## 2. Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/chat` | Ask a question and stream the answer |
| `GET` | `/agents` | The agent names this engine can run |
| `PUT` | `/documents` | Add or replace a document in the knowledge base |
| `GET` | `/documents?project_id=…` | List what is indexed |
| `DELETE` | `/documents/{doc_id}?project_id=…` | Remove a document |

Two more for operations:

| Method | Path | Reports |
| --- | --- | --- |
| `GET` | `/health` | The process is up |
| `GET` | `/health/ready` | Whether a turn can be served, and which agents are installed |

```json
{"ready": true, "model_provider": true, "vector_store": true, "agents": ["graph", "loop"]}
```

`ready` is false when no model provider key is configured, or when the vector
store does not answer. Without a key the engine still accepts and chunks
documents, and answers `501` to `/chat`.

---

## 3. Sending a chat message

`POST /chat` with a JSON body. The engine stores no configuration, so the
request carries the whole assistant definition:

```json
{
  "project": {
    "project_id": "support",
    "name": "Customer Support Assistant",
    "system_prompt": "You are a customer support assistant. Be concise.",
    "model": "openai/gpt-5-mini",
    "temperature": 0.2,
    "max_output_tokens": 800,
    "top_k": 5,
    "max_tool_iterations": 6,
    "mcp_servers": [
      {
        "name": "support-tools",
        "url": "http://localhost:8200/mcp",
        "allowed_tools": ["get_booking_status", "get_flight_status"]
      }
    ]
  },
  "message": "Is my flight delayed? My booking is AB12CD.",
  "session_id": "conv-1731",
  "user_id": "alice",
  "history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi, how can I help?"}
  ]
}
```

### Fields

| Field | Required | Meaning |
| --- | --- | --- |
| `project` | yes | The assistant: prompt, model, retrieval settings, tools. `agent` selects the agent (see [agents.md](agents.md)); `embedding_model` and the chunking fields describe its knowledge base; `retrieval`, `rerank` and `retrieval_candidates` configure how chunks are found (see [retrieval.md](retrieval.md)); `workflow` describes the turn as a graph of steps for `agent: workflow` (see the Workflows section of agents.md); `tracing` names the assistant's own trace destination (below); `max_output_tokens` caps one reply, and the `done` event says `length` when it did |
| `message` | yes | The user's message. Must not be empty |
| `session_id` | no | The conversation id. Forwarded to the tool server as `X-Session-Id` |
| `user_id` | no | Opaque. Forwarded to the tool server as `X-User-Id` so it can scope reads and writes |
| `history` | no | Earlier turns, oldest first |

### Tracing per assistant

Add a `tracing` block to `project` to send that assistant's traces to its own
Langfuse, a cloud project or a self-hosted instance, instead of the engine's
`ENGINE_TRACING` destination (engine 0.1.3+):

```json
"tracing": {
  "provider": "langfuse",
  "public_key": "pk-lf-…",
  "secret_key": "sk-lf-…",
  "host": "https://cloud.langfuse.com"
}
```

The keys travel with every request, so this assumes what the deployment
guide already requires: the engine is reached only by the backend, over TLS.
Traces carry the request, project, session and user ids either way.

### A workflow per assistant

With `agent: workflow` (the LangGraph plugin, in the `engine-langgraph`
image), `project.workflow` describes the turn as nodes and edges from a fixed
library: retrieve, model, condition, tool, reply, handoff, end. The schema is
validated at the API boundary, so a malformed graph is a `422` before anything
runs; a tool step may only name a tool that one of `mcp_servers` allowlists.
Without a `workflow`, the agent runs retrieve then model. The node table and
an example are in the Workflows section of [agents.md](agents.md).

### Why the whole configuration is sent every time

The engine then holds no state that needs migrating, and the backend remains
the single source of truth for what its assistant is. A prompt change takes
effect on the next request. One engine can serve several applications, each
sending its own configuration.

### Two constraints

**`allowed_tools` must be non-empty.** Tool names and descriptions come from the
tool server and are placed in the model's prompt. An open list would let a
tool server inject instructions. List the names explicitly.

**The frontend must not supply `project`.** A browser that can send a
`system_prompt` can rewrite the assistant. Accept a small request from the
browser and build `project` server-side from the backend's own configuration.
The reference backend does this in
[`api/chat.py`](../examples/backend/src/support_agent/api/chat.py): the
browser sends a project *name*, and the definition is loaded from
[`projects/support.yaml`](../examples/backend/src/support_agent/projects/support.yaml).

### Unknown fields are rejected

Every model uses `extra="forbid"`. A misspelled field is a `422` that names it.

---

## 4. Reading the answer

`POST /chat` returns `application/x-ndjson`: one JSON object per line, sent as
the answer is produced.

```text
{"type":"retrieval","query":"Is my flight delayed?","sources":[...]}
{"type":"tool_call_started","call_id":"c1","tool":"get_booking_status",...}
{"type":"tool_call_finished","call_id":"c1","tool":"get_booking_status","ok":true}
{"type":"token","text":"Your "}
{"type":"token","text":"flight "}
{"type":"token","text":"is delayed."}
{"type":"usage","input_tokens":812,"output_tokens":24,"total_tokens":836}
{"type":"done","finish_reason":"stop"}
```

Read it line by line and switch on `type`.

### Events

| `type` | Fields | Use |
| --- | --- | --- |
| `retrieval` | `query`, `sources[]` | Render sources. Arrives before the answer |
| `token` | `text` | Append to the answer |
| `tool_call_started` | `call_id`, `tool`, `server`, `arguments` | Show progress |
| `tool_call_finished` | `call_id`, `tool`, `ok`, `duration_ms`, `error` | Pair with `started` by `call_id` |
| `usage` | `input_tokens`, `output_tokens`, `total_tokens`, `cost_usd`, `model` | Display cost. Tokens cover every model call in the turn; `cost_usd` is null unless the engine's `ENGINE_PRICING` lists the model |
| `error` | `code`, `message` | The turn failed after the response started |
| `done` | `finish_reason` | Always last. `stop`; `length` when `max_output_tokens` cut the answer (show the visitor it was shortened); `tool_limit` when the model was still asking for tools after `max_tool_iterations` rounds (what it said so far has streamed); `error` after an `error` event |

Each `sources[]` entry has `doc_id`, `source`, `score`, and optionally
`heading`, `page` and `excerpt`. `heading` is present when the document was
chunked by headings, `page` when it was chunked by page; see
[chunking.md](chunking.md).

### Stream rules

**A run always ends with `done`,** including on failure. A failure after the
response has started arrives as an `error` event followed by `done` with
`finish_reason: "error"`, because the `200` status has already been sent.

**Ignore an unrecognised `type`.** Event types will be added.

### Converting to SSE for a browser

NDJSON is for service-to-service use. The translation to server-sent events
belongs in the backend:

```python
def sse_frame(event) -> str:
    # The trailing blank line terminates the frame.
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"
```

Full version:
[`api/streaming.py`](../examples/backend/src/support_agent/api/streaming.py).

### Without streaming

Collect the events and fold them into one object. `collect()` in the same file
does this; the reference backend exposes it as `POST /chat/sync`.

---

## 5. Uploading documents

Send raw bytes over multipart. Extraction is the engine's job; the original
file carries page numbers and layout that extracted text has lost.

```bash
curl -X PUT localhost:8100/documents \
  -F project_id=support \
  -F external_id=baggage.md \
  -F "file=@examples/backend/knowledge/baggage.md;type=text/markdown"
```

| Field | Meaning |
| --- | --- |
| `project_id` | The knowledge base this belongs to |
| `external_id` | The backend's own identifier for the document: a path, a row id, anything stable |
| `file` | The file |
| `embedding_model` | Optional. Must match what queries use; the engine keeps one collection per model |
| `chunking_strategy` | Optional. `size`, `headings`, or `page`. See [chunking.md](chunking.md) |
| `chunk_size`, `chunk_overlap` | Optional. The size cap every strategy applies |

The optional fields default to the engine's settings. Send them from the
project configuration so a document is chunked and embedded consistently. They
apply at ingest; changing them means re-indexing.

Returns `201` and a record:

```json
{
  "doc_id": "89ad9185...",
  "external_id": "baggage.md",
  "project_id": "support",
  "filename": "baggage.md",
  "mimetype": "text/markdown",
  "size_bytes": 4725,
  "content_hash": "3cf89ca9...",
  "status": "indexed",
  "chunk_count": 10,
  "error": null,
  "created_at": "2026-08-19T09:12:44Z",
  "updated_at": "2026-08-19T09:12:44Z"
}
```

`status` is one of `received`, `indexed`, `unchanged`, `failed`. `received`
means the engine has no provider key and could not embed.

### Idempotency

`external_id` is the key. The same id replaces the document; identical bytes
skip the work and answer `unchanged`. A sync over a whole corpus can therefore
be re-run safely.
[`scripts/seed_knowledge.py`](../examples/backend/scripts/seed_knowledge.py)
walks a folder and uses each file's relative path as its `external_id`.

### Listing and deleting

```bash
curl "localhost:8100/documents?project_id=support"
curl -X DELETE "localhost:8100/documents/89ad9185...?project_id=support"
```

`project_id` is required on both.

### Validation

The engine rejects an empty file with `400` and one over 25 MB with `413`.
Validate on the backend side as well, to save a round trip and to report the
error in the backend's own vocabulary.

---

## 6. Handling failure

The engine is a separate process, and its being unavailable is a normal
condition. The backend should distinguish:

| Condition | Engine returns | Backend should return |
| --- | --- | --- |
| Engine not running | connection refused | `503` |
| Engine missing configuration (no provider key) | `501` | `501`, passed through |
| Request malformed | `4xx` | `502`; the backend's own bug |
| Document unusable (`415`, `422`) | `4xx` | the same code; the caller's document |
| Engine failed | `5xx` | `502` |

The `501` body names the variable to set:

```json
{"detail": "no ENGINE_OPENROUTER_API_KEY is set -- put your OpenRouter key in .env, or set it in the engine's environment"}
```

### Validate before calling

Perform validation, authentication and configuration lookup first, so a bad
field is still a `422` and an unknown project is still a `404` when the engine
is down.

### Streaming

Start the request and check the status before beginning the backend's own
response. With a lazy generator the HTTP call happens on first iteration, after
the backend has committed to `200`, and an engine error reaches the browser as
an empty success. The reference client makes `start_chat()` an awaited call
that returns an iterator; see
[`engine_client/client.py`](../examples/backend/src/support_agent/engine_client/client.py).

---

## 7. Exposing tools

Tools stay in the backend. The engine learns their name, description and input
schema over MCP, and invokes them by name.

```text
once per turn
  engine → tools/list
  engine ← [{name, description, input_schema}, ...]
  engine   drops anything not in allowed_tools
  engine   passes the rest to the model

per tool call
  model  → call get_booking_status with {booking_reference: "AB12CD"}
  engine → tools/call, with X-User-Id and X-Session-Id
  backend runs the tool against its own data
  engine ← the result
  engine   feeds it back to the model and continues
```

### Steps

**1. Run an MCP server.** The reference one is
[`mcp_tools.py`](../examples/backend/src/support_agent/mcp_tools.py): three
domain tools and the memory tool.

```bash
make tools     # http://localhost:8200/mcp
```

**2. Declare it** in the `mcp_servers` block of `project`. Under Docker Compose
the URL must use the service name (`http://mcp-tools:8200/mcp`).

**3. Write docstrings for the model.** The name, type hints and docstring are
the schema the model reads when deciding whether to call a tool.

```python
@mcp.tool()
async def get_booking_status(booking_reference: str) -> dict[str, str]:
    """Look up one booking by its reference.

    Returns the passenger, route, travel date, fare type, baggage allowance,
    flight number, and current status. Use this whenever a customer mentions a
    booking reference.

    Args:
        booking_reference: Six-character booking reference, e.g. "AB12CD".
    """
```

### Conventions

**Return a missing record as data, not an exception.**

```python
return {
    "booking_reference": ref,
    "status": "not_found",
    "message": "No booking was found with this reference.",
}
```

A tool that raises for "not found" teaches the model that the tool is broken.

**Return values in words a person would recognise.** `"cabin baggage only"`
rather than `null`.

**Return what the next call needs.** `get_booking_status` returns the flight
number, which lets the model call `get_flight_status` in the same turn.

### Failures

A tool that raises, or returns an MCP result flagged `isError`, does not end
the turn. The engine reports it as a `tool_call_finished` with `ok: false` and
feeds the error text back to the model, which can explain or try another way.

### Caller context

Every tool call carries `X-User-Id` and `X-Session-Id`, taken from the chat
request. The engine attaches no meaning to either; a tool that scopes what it
reads or writes takes them from the headers, never from an argument the model
supplies. See [memory.md](memory.md) for a tool built on this.

### Tool results are untrusted

Whatever a tool returns is placed in the model's context. Treat it as data, the
same as retrieved document text.

---

## 8. A complete example

Engine and tool server running:

```bash
curl -N -X POST localhost:8100/chat \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: dev-key' \
  -d '{
    "project": {
      "project_id": "support",
      "name": "Support",
      "system_prompt": "You are a travel support assistant.",
      "mcp_servers": [{
        "name": "support-tools",
        "url": "http://localhost:8200/mcp",
        "allowed_tools": ["get_booking_status", "get_flight_status"]
      }]
    },
    "message": "Is my flight delayed? Booking AB12CD.",
    "user_id": "alice"
  }'
```

In Python:

```python
import httpx


async def ask(question: str) -> None:
    payload = {
        "project": load_my_assistant_config(),  # server-side
        "message": question,
        "user_id": "alice",
    }
    async with httpx.AsyncClient(base_url="http://localhost:8100") as client:
        async with client.stream("POST", "/chat", json=payload) as response:
            response.raise_for_status()  # 4xx / 5xx surface here
            async for line in response.aiter_lines():
                if line.strip():
                    handle(json.loads(line))  # switch on event["type"]
```

`raise_for_status()` before the loop is what keeps a `501` from being read as an
empty answer.

---

## 9. Reference implementation

[`examples/backend/`](../examples/backend) is a working backend. In reading
order:

| File | Shows |
| --- | --- |
| [`engine_client/client.py`](../examples/backend/src/support_agent/engine_client/client.py) | The HTTP client: streaming, error taxonomy, connection cleanup |
| [`engine_client/models.py`](../examples/backend/src/support_agent/engine_client/models.py) | The wire contract as a client sees it |
| [`api/chat.py`](../examples/backend/src/support_agent/api/chat.py) | Building a safe request from an untrusted one |
| [`api/streaming.py`](../examples/backend/src/support_agent/api/streaming.py) | NDJSON to SSE, and folding a run into one object |
| [`api/documents.py`](../examples/backend/src/support_agent/api/documents.py) | Upload validation and forwarding raw bytes |
| [`mcp_tools.py`](../examples/backend/src/support_agent/mcp_tools.py) | The tools the engine can call |
| [`projects/support.yaml`](../examples/backend/src/support_agent/projects/support.yaml) | An assistant definition |

The backend copies the engine's models rather than importing its package: two
services that share a Python package are one deployable. A schema parity test
at the repository root keeps the two copies identical.
