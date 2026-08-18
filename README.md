# chatbot-engine

A domain-specialised travel-support assistant, built as two independent services:
an **application backend** that owns the product, and an **AI engine** that owns
the AI.

## Architecture

```text
Frontend
   ↓  HTTP + SSE
Application Backend        :8000    users, auth, config, documents, tools
   ↓  HTTP (NDJSON)
AI Engine Service          :8100    retrieval, prompts, model, tool loop
   ↓  MCP
Backend MCP Tool Server    :8200    get_booking_status, get_flight_status, …
```

The engine is a **separate service**, not a library: the backend calls it over
HTTP and never imports its Python package. The loop back to the backend is
deliberate — the tools read this application's data, and the permissions that
guard that data live in the backend.

> **Status: the boundary is built, the AI is not.** Every AI-backed route answers
> `501` naming the function to implement. Routing, validation, streaming,
> configuration and the MCP tool schemas all work today, with 78 tests.

## Run it

```bash
make setup      # once: uv sync + create .env
make dev        # both services in one terminal
make smoke      # in another terminal
```

`make` on its own lists every command.

| Command | What it does |
| --- | --- |
| `make engine` | AI engine only → http://localhost:8100/docs |
| `make backend` | App backend only → http://localhost:8000/docs |
| `make tools` | MCP tool server on `:8200` |
| `make test` | 78 tests, no services needed |
| `make seed` | Load `backend/knowledge/` through the backend |
| `make up` / `make down` | Both services in Docker |

## Endpoints

The frontend talks only to the backend:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness |
| `POST` | `/chat` | Chat. Returns an SSE stream. |
| `POST` | `/chat/sync` | Non-streaming variant |
| `PUT` | `/documents` | Upsert a document by `external_id` |
| `GET` | `/documents` | What the assistant knows |
| `DELETE` | `/documents/{doc_id}` | Remove a document |

The engine's own API is documented in [engine/README.md](engine/README.md).

## Layout

```text
engine/     the AI engine service     -- see engine/README.md
backend/    the application backend   -- see backend/README.md
frontend/   not started
docs/       architecture.md, the brief
tests/      the one test that imports both services (contract parity)
```

## What is yours to write

Nothing in the repo pre-empts an AI decision. In dependency order — full table in
[the architecture doc](docs/architecture.md#10-where-to-implement-each-piece):

1. Blob store and document registry → `engine/src/chatbot_engine/rag/`
2. Extraction, chunking, embeddings, vector store → same folder
3. The ingest pipeline, registered at `deps.get_ingest_pipeline()`
4. Retriever, chat-model client, prompt construction
5. The agent / run loop, registered at `deps.get_agent()`
6. MCP discovery and invocation → `engine/src/chatbot_engine/mcp/client.py`
7. The three tool bodies → `backend/src/support_agent/mcp_tools.py`
8. Real authentication (`X-User-Id` is a placeholder) and the frontend

`curl localhost:8100/health/ready` tells you whether chat and document ingestion
have implementations registered — the two things that turn a 501 into a real
answer.

## Test

```bash
make test
```
