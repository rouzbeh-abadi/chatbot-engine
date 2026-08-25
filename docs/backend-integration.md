# Connecting a backend to the chatbot engine

The engine is a separate HTTP service. It knows how to retrieve, prompt a model,
and call tools — and nothing about your users, your product, or your domain.

Your backend supplies all of that on every request, then relays the answer to
whatever is in front of it.

```text
Browser
   ↓  your API
Your backend        :8000
   ↓  HTTP + NDJSON
Chatbot engine      :8100
   ↓  MCP
Your tool server    :8200
```

The engine never calls your database and never stores your configuration. The one
loop back is MCP: the engine calls tools that live in *your* backend, because your
tools read your data with your user's permissions.

**Contents**

1. [Before you start](#1-before-you-start)
2. [The four things the engine exposes](#2-the-four-things-the-engine-exposes)
3. [Sending a chat message](#3-sending-a-chat-message)
4. [Reading the answer](#4-reading-the-answer)
5. [Uploading documents](#5-uploading-documents)
6. [Handling failure](#6-handling-failure)
7. [Letting the engine call your tools](#7-letting-the-engine-call-your-tools)
8. [A complete example](#8-a-complete-example)
9. [Reference implementation](#9-reference-implementation)

---

## 1. Before you start

You need the engine running and its address.

```bash
make engine          # http://localhost:8100
curl localhost:8100/health
```

```json
{"status": "ok", "service": "chatbot-engine", "version": "0.1.0"}
```

Two settings on your side:

| Setting | Value | Notes |
| --- | --- | --- |
| `BACKEND_ENGINE_URL` | `http://localhost:8100` | `http://engine:8100` under Docker Compose |
| `BACKEND_ENGINE_API_KEY` | matches the engine's `ENGINE_API_KEY` | Leave both blank on localhost |

If the engine has `ENGINE_API_KEY` set, send it as an `X-API-Key` header on every
request except `/health`. If it is unset, the engine is open — fine locally, not
fine in a deployment, because the engine holds the model provider's credentials.

**Do not let a browser call the engine directly.** It has no notion of end-user
permissions and will happily answer anyone who can reach it. Your backend is the
thing that decides who is allowed to ask.

---

## 2. The four things the engine exposes

| Method | Path | You use it to |
| --- | --- | --- |
| `POST` | `/chat` | Ask a question and stream the answer |
| `PUT` | `/documents` | Add or replace a document in the knowledge base |
| `GET` | `/documents?project_id=…` | List what is indexed |
| `DELETE` | `/documents/{doc_id}?project_id=…` | Remove a document |

Plus two you will use while developing:

| Method | Path | Tells you |
| --- | --- | --- |
| `GET` | `/health` | The engine is up |
| `GET` | `/health/ready` | Which capabilities have an implementation |

`/health/ready` is worth knowing about. It answers the question "why am I getting
a 501":

```json
{"chat": false, "documents": false}
```

`false` means nobody has registered an implementation in the engine's
`api/deps.py` yet, so that route will return 501 no matter what you send.

---

## 3. Sending a chat message

`POST /chat` with a JSON body. The important idea: **the engine stores no
configuration, so you send the whole assistant definition every time.**

```json
{
  "project": {
    "project_id": "support",
    "name": "Customer Support Assistant",
    "system_prompt": "You are a customer support assistant. Be concise.",
    "model": "openai/gpt-5-mini",
    "temperature": 0.2,
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

### The fields

| Field | Required | Meaning |
| --- | --- | --- |
| `project` | yes | The whole assistant: prompt, model, retrieval settings, tools |
| `message` | yes | What the user just said. Must not be empty |
| `session_id` | no | Your conversation id. The engine passes it through |
| `user_id` | no | Opaque. Sent to your MCP server as an `X-User-Id` header so *it* can authorise |
| `history` | no | Earlier turns, oldest first |

### Why the whole config, every time

Because it means the engine holds no state you have to migrate, and you stay the
single source of truth for what your assistant is. Change a prompt in your own
config file and the next request uses it — no engine restart, no deployment.

It also means one engine can serve several applications, each sending its own
configuration.

### Two things to be careful about

**`allowed_tools` must be non-empty.** Tool names and descriptions come back from
your tool server and end up inside the model's prompt. If you let a server offer
whatever it likes, a compromised or careless tool server can inject instructions.
List the names explicitly.

**Never let your frontend supply `project`.** If a browser can send a
`system_prompt`, anyone can rewrite your assistant. Accept a small request from
the browser — a message and maybe a session id — and build the `project` block
server-side from your own configuration.

The reference backend does exactly this: [`api/chat.py`](../backend/src/support_agent/api/chat.py)
accepts `{message, session_id, project, history}` where `project` is only a
*name*, then loads the real definition from
[`projects/support.yaml`](../backend/src/support_agent/projects/support.yaml).

### Unknown fields are rejected

Every model uses `extra="forbid"`, so a typo comes back as a `422` naming the
field rather than being silently ignored. `"tempreature": 0.2` will fail loudly.

---

## 4. Reading the answer

`POST /chat` returns `application/x-ndjson`: **one JSON object per line**, sent as
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

Read it line by line. Each line is complete JSON; switch on `type`.

### The seven events

| `type` | Fields | What to do with it |
| --- | --- | --- |
| `retrieval` | `query`, `sources[]` | Show the sources panel. Arrives before the answer, on purpose |
| `token` | `text` | Append to the answer as it arrives |
| `tool_call_started` | `call_id`, `tool`, `server`, `arguments` | Show "checking your booking…" |
| `tool_call_finished` | `call_id`, `tool`, `ok`, `duration_ms`, `error` | Pair with `started` by `call_id` |
| `usage` | `input_tokens`, `output_tokens`, `total_tokens`, `cost_usd`, `model` | Display cost |
| `error` | `code`, `message` | Something failed mid-answer |
| `done` | `finish_reason` | Always last. Stop reading |

Each `sources[]` entry has `doc_id`, `source`, `score`, and optionally `heading`
and `excerpt` — enough to render a citation.

### Two rules for consuming the stream

**A run always ends with `done`**, including when it fails. If a failure happens
after the response has started, you get an `error` event followed by a `done` with
`finish_reason: "error"`. The engine cannot change the status code at that point —
`200` has already been sent — so the failure arrives as data.

**Handle a `type` you do not recognise by ignoring it.** New event types will be
added. Skipping unknown lines keeps your backend working against a newer engine.

### Turning it into SSE for a browser

NDJSON is for service-to-service. A browser wants server-sent events, and that
translation belongs in your backend, which is the thing that knows a browser is
on the other end:

```python
def sse_frame(event) -> str:
    # The trailing blank line matters -- without it a browser buffers forever.
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"
```

Full version: [`api/streaming.py`](../backend/src/support_agent/api/streaming.py).

### If you do not want to stream

Collect the events and fold them into one object — answer, sources, tool calls,
usage. `collect()` in the same file does this, and the reference backend exposes
it as `POST /chat/sync` for smoke tests and simple clients.

---

## 5. Uploading documents

Send **raw bytes** over multipart. Do not extract the text yourself: the original
file carries page numbers and layout that plain text has already thrown away, and
extraction is the engine's job.

```bash
curl -X PUT localhost:8100/documents \
  -F project_id=support \
  -F external_id=baggage.md \
  -F "file=@backend/knowledge/baggage.md;type=text/markdown"
```

| Field | Meaning |
| --- | --- |
| `project_id` | Which knowledge base this belongs to |
| `external_id` | **Your** identifier for the document — a path, a row id, anything stable |
| `file` | The file itself |

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

`status` is one of `received`, `indexed`, `unchanged`, `failed`.

### Uploads are idempotent

`external_id` is the key. Send the same id again and the document is **replaced**,
not duplicated. Send identical bytes and the engine skips the work entirely and
answers `unchanged` — so you can safely re-run a sync over your whole corpus.

That is what makes bulk loading simple:
[`scripts/seed_knowledge.py`](../backend/scripts/seed_knowledge.py) walks a folder
and uses each file's relative path as its `external_id`.

### Listing and deleting

```bash
curl "localhost:8100/documents?project_id=support"
curl -X DELETE "localhost:8100/documents/89ad9185...?project_id=support"
```

`project_id` is required on both — the engine keeps knowledge bases separate.

### Validate before you forward

The engine rejects an empty file with `400` and one over 25 MB with `413`, but
check on your side too. It saves a network round trip, and it lets you return an
error in your own vocabulary.

---

## 6. Handling failure

The engine is a separate process. It being down is a normal thing that happens,
and your backend should say which of these went wrong:

| What happened | Engine returns | Your backend should return |
| --- | --- | --- |
| Engine is not running | *connection refused* | **503** — try again shortly |
| Capability not implemented yet | `501` | **501** — pass it through |
| Your request was malformed | `4xx` | **502** — your bug, not the caller's |
| Engine broke | `5xx` | **502** |

Keeping these apart matters. "The engine is not running" and "this feature does
not exist yet" need very different reactions, and collapsing both into a `500`
makes debugging much harder than it needs to be.

The `501` body names the exact function to implement:

```json
{
  "detail": "no Agent is registered -- implement one under chatbot_engine/agent/ and return it from chatbot_engine.api.deps.get_agent()"
}
```

### Validate before you call

Do your own validation, authentication, and configuration lookup *first*. Then a
bad field is still a `422` and an unknown project is still a `404` even when the
engine is completely down. Everything you can answer without the engine, answer
without the engine.

### One subtlety for streaming

Start the request and check the status **before** you begin your own response.
If you use a lazy generator, the HTTP call does not happen until the first
iteration — by which point you have already committed to `200`, and a `501` reaches
the browser as an empty success.

The reference client makes `start_chat()` an awaited call that returns an iterator,
so the status check happens while the status line can still change. See
[`engine_client/client.py`](../backend/src/support_agent/engine_client/client.py).

---

## 7. Letting the engine call your tools

Your tools stay in your backend. The engine only learns their **name, description
and input schema** — never their code.

```text
once per turn
  engine → tools/list          "what do you have?"
  engine ← [{name, description, input_schema}, ...]
  engine   drops anything not in allowed_tools
  engine   passes the survivors to the model

per tool call
  model  → "call get_booking_status with {booking_reference: 'AB12CD'}"
  engine → tools/call
  YOUR CODE RUNS, against your database, as your user
  engine ← the result
  engine   feeds it back to the model and continues
```

### What you have to do

**1. Run an MCP server.** The reference one is
[`mcp_tools.py`](../backend/src/support_agent/mcp_tools.py) — about 200 lines for
three tools:

```bash
make tools     # http://localhost:8200/mcp
```

**2. Tell the engine where it is,** in the `mcp_servers` block of your `project`
config. Under Docker Compose this must be the service name (`http://mcp-tools:8200/mcp`),
not `localhost` — containers do not share a network namespace.

**3. Write the docstrings for the model, not for a developer.** The name, type
hints and docstring *are* the schema the model reads when deciding whether to call
a tool. Vague docstrings are the most common reason tool calling behaves badly.

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

### Three conventions worth copying

**Return a missing record as data, not an exception.**

```python
return {"booking_reference": ref, "status": "not_found",
        "message": "No booking was found with this reference."}
```

A tool that raises for "not found" teaches the model that the tool is broken, and
it stops trying to use it.

**Return values in words a person would recognise.** `"cabin baggage only"` rather
than `null`. The model relays these to a human, and a bare null invites it to
invent an explanation.

**Give the model something to chain.** `get_booking_status` returns the flight
number, which is what lets the model then ask `get_flight_status` about it in the
same turn. Without that link, a perfectly reasonable question — "is my flight
delayed?" — cannot be answered.

### Tool results are untrusted

Whatever a tool returns goes into the model's context. Treat it like retrieved
document text: data, never instructions. The engine is responsible for wrapping
it safely, but do not put anything in a tool result that you would not want a
model to act on.

---

## 8. A complete example

Engine up, tool server up, then:

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

In Python, with the streaming and cleanup handled:

```python
import httpx

async def ask(question: str) -> None:
    payload = {
        "project": load_my_assistant_config(),   # yours, server-side
        "message": question,
        "user_id": "alice",
    }
    async with httpx.AsyncClient(base_url="http://localhost:8100") as client:
        async with client.stream("POST", "/chat", json=payload) as response:
            response.raise_for_status()          # 501 / 4xx / 5xx surface here
            async for line in response.aiter_lines():
                if line.strip():
                    handle(json.loads(line))     # switch on event["type"]
```

`raise_for_status()` before the loop is the important part — it is what stops a
`501` from being mistaken for an empty answer.

---

## 9. Reference implementation

A working backend lives in [`backend/`](../backend). The parts worth reading, in
order:

| File | What it shows |
| --- | --- |
| [`engine_client/client.py`](../backend/src/support_agent/engine_client/client.py) | The HTTP client: streaming, error taxonomy, connection cleanup |
| [`engine_client/models.py`](../backend/src/support_agent/engine_client/models.py) | The wire contract as a client sees it |
| [`api/chat.py`](../backend/src/support_agent/api/chat.py) | Building a safe request from an untrusted one |
| [`api/streaming.py`](../backend/src/support_agent/api/streaming.py) | NDJSON to SSE, and folding a run into one object |
| [`api/documents.py`](../backend/src/support_agent/api/documents.py) | Upload validation and forwarding raw bytes |
| [`mcp_tools.py`](../backend/src/support_agent/mcp_tools.py) | Three tools the engine can call |
| [`projects/support.yaml`](../backend/src/support_agent/projects/support.yaml) | An assistant definition |

That backend copies the engine's models rather than importing its package, on
purpose: two services that share a Python package are one deployable wearing a
disguise. A schema parity test keeps the two copies honest.
