# chatbot-engine

A standalone RAG and MCP tool-calling service.

The engine answers questions from a knowledge base, calls tools when it needs live
data, and streams the answer back token by token as it is produced. It handles the
retrieval, the prompt, the model call, and the loop that runs tools and feeds their
results back to the model.

What it deliberately does not handle: who your users are, how they log in, what
your product is called, or what any of your data means. It has no database of
yours and no configuration of yours.

Instead, everything specific to an application arrives **with each request** — the
system prompt, the model, how many chunks to retrieve, and which tools may be
used. That has three consequences worth knowing up front:

- the engine holds no state you would ever have to migrate;
- you change a prompt in your own config and the next request uses it, with no
  engine restart and no deployment;
- one engine can serve several applications at once, each sending its own
  configuration.

It runs as its own HTTP service on port `8100`. A chat turn comes back as a stream
of typed events — retrieved sources first, then tokens as they are generated, tool
progress as tools run, and finally token usage and cost.

```bash
uv run uvicorn chatbot_engine.app:app --port 8100 --reload
```

Interactive API docs: http://localhost:8100/docs

## Using it from a backend

The engine is reached over plain HTTP with JSON, so **it works with a backend
written in any language**. Python, TypeScript, Go, Ruby, Java, PHP — if it can make
an HTTP request and read a response line by line, it can drive the engine. There is
no SDK to install and no library to depend on.

Your backend keeps the parts that are genuinely yours:

| Your backend owns | Why it cannot live in the engine |
| --- | --- |
| Authentication and user identity | The engine has no idea who your users are |
| The assistant's configuration | It is your product's voice, not the engine's |
| Your domain tools | They read your data, with your user's permissions |
| The browser-facing transport | Only you know a browser is on the other end |

The tools are the one place the engine calls back to you. It connects to a tool
server that you run, over MCP, and can only use the tool names you explicitly
allow. A tool that looks up a booking has to execute where the booking data and
its permissions are — which is your backend, not the engine.

```text
Your backend      →  HTTP   →  chatbot engine
chatbot engine    →  MCP    →  your tool server
```

Everything a backend needs in order to do this — the endpoints, the exact request
shape, how to read the streamed answer, how to upload documents, how failures are
reported, and how to expose your own tools over MCP — is written up in full here:

**→ [Connecting a backend to the chatbot engine](docs/backend-integration.md)**

It is written for someone who has never seen the engine before, and it includes a
complete worked example in `curl` and in Python.

### The example backend in this repository

`backend/` is a working reference: a travel-support assistant for a fictional
airline, with a Postgres database, three domain tools served over MCP, and a
knowledge base of support documents. It exists so the engine has something
realistic to be tested against.

It happens to be written in Python with FastAPI, but nothing about that is
required — it is one example of the integration, not the only shape it can take.

```text
Frontend (React + TS)      :5173
   ↓  HTTP + SSE
Example backend        :8000    users, config, documents, domain tools
   ↓  HTTP + NDJSON
Chatbot engine         :8100    retrieval, prompts, model, tool loop
   ↓  MCP
Backend tool server    :8200    bookings, flights, support tickets
   ↓
Postgres               :5432
```

## Running the whole thing

```bash
make setup      # once: install dependencies and create .env
make dev        # engine and example backend together
make frontend   # the React UI on http://localhost:5173
make smoke      # in another terminal
```

`make` on its own lists every command. `make up` brings up the full stack in
Docker, including Postgres and the tool server.

## Layout

```text
engine/     the chatbot engine service     -- see engine/README.md
backend/    the example backend            -- see backend/README.md
frontend/   React + TypeScript chat UI      -- see frontend/README.md
docs/       the integration guide, and the project brief
tests/      the contract-parity test, the one place both services meet
```

## Where things stand

The engine's HTTP surface, event contract, document extraction, chunking, storage
and MCP client are written. The example backend is complete: it serves its API,
owns its database, and exposes three working tools.

The frontend is built too: a React and TypeScript chat interface that streams
tokens, renders Markdown answers, shows retrieved sources, shows tool calls as
they run, and reports token cost.

Still to build in the engine: the ingestion pipeline, embeddings and the vector
store, retrieval, prompt construction, the chat-model client, and the agent run
loop. Until each is registered in `engine/src/chatbot_engine/api/deps.py`, the
routes that depend on it answer `501` naming the exact function to fill in — and
the UI shows that state rather than looking broken.

To see what is live:

```bash
curl localhost:8100/health/ready
```

## Tests

```bash
make test
```