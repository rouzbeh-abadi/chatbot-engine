# chatbot-engine

A standalone RAG and MCP tool-calling service. It owns retrieval, prompt
assembly, the model call and the tool loop. It owns nothing about your users,
your domain or your business logic.

It does **not** depend on the application backend, and the backend does not
depend on it. They share an HTTP contract and nothing else.

> **Status: the boundary is built, the AI is not.** Every endpoint answers
> `501` with the name of the thing to implement and where to register it. See
> [Not implemented yet](#not-implemented-yet).

## Run it

```bash
uv run uvicorn chatbot_engine.app:app --port 8100 --reload
```

Interactive API docs: http://localhost:8100/docs

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness. Unauthenticated. |
| `GET` | `/health/ready` | Which capabilities have an implementation registered |
| `POST` | `/chat` | One turn. Streams NDJSON events. |
| `PUT` | `/documents` | Upsert a document by `external_id`. Multipart, raw bytes. |
| `GET` | `/documents?project_id=` | What is indexed for a project |
| `DELETE` | `/documents/{doc_id}?project_id=` | Remove a document |

Set `ENGINE_API_KEY` and every route except the two `/health` routes requires a
matching `X-API-Key`. Left unset the engine is open, which is fine on localhost and not
fine anywhere else — it holds the provider credentials and has no notion of
end-user permissions.

### Why NDJSON and not SSE

`POST /chat` streams `application/x-ndjson`: one JSON event per line. The
engine's client is an application backend, not a browser, so SSE framing belongs
to whichever service is actually talking to the browser. NDJSON stays readable
from `curl` and from any HTTP client.

Event types: `retrieval` · `token` · `usage` · `error` · `done`.

A run always terminates with `done`, including when it fails part-way — an
`error` event then a `done` with `finish_reason: "error"`, because the 200 status
has already been sent by that point.

## Layout

```text
src/chatbot_engine/
├── app.py         the service: routing, auth, error mapping
├── api/           the HTTP surface -- thin, delegates to services/
│   ├── chat.py        POST /chat
│   ├── documents.py   PUT/GET/DELETE /documents
│   ├── health.py      GET /health, /health/ready
│   ├── streaming.py   NDJSON framing
│   └── deps.py        ← register your implementations here
├── models/        request and response contracts (pure pydantic)
├── ports/         the interfaces future AI logic must satisfy
├── services/      the boundary between HTTP and AI logic
├── mcp/           MCP client connectivity: config and transport
├── agent/         YOURS -- the run loop
├── rag/           YOURS -- chunking, embedding, vector store, retriever
└── llm/           YOURS -- the chat-model client and embedder
```

`models/` and `ports/` are framework-free, and `tests/test_layering.py` enforces
that plus two more rules: contracts never import `api/` or `services/`, and the
engine never imports the backend.

## The five ports

| Port | Responsibility |
| --- | --- |
| `Agent` | One turn: retrieve, prompt, call tools, stream events |
| `ToolProvider` | Tool discovery and invocation over MCP |
| `IngestPipeline` | Raw bytes in, an indexed document out |
| `BlobStore` | The original uploaded files |
| `DocumentRegistry` | What is indexed and whether it is current |

## Not implemented yet

Everything behind those ports. `api/deps.py` returns `None` for each, which makes
the service layer raise and the API answer 501.

| Write it in | Register it as |
| --- | --- |
| `agent/` | `deps.get_agent()` |
| `rag/` (ingestion) | `deps.get_ingest_pipeline()` |
| `rag/` (storage) | `deps.get_registry()` |
| `mcp/client.py` | already registered; `list_tools` / `call_tool` raise |

Also unwritten: chunking, document parsing, embeddings, the vector store,
retrieval, query translation, prompt construction, the tool-calling loop, and
the chat-model client.

## Design notes worth keeping

- **Configuration arrives with every request.** The engine stores no assistant
  config, which is why it can be restarted or scaled out without migrating
  anything, and why one engine can serve several applications.
- **`allowed_tools` is required and non-empty.** Tool names and descriptions come
  from an MCP server and land inside the prompt, so an open list lets a server
  inject instructions.
- **Callers send raw bytes, not text.** The original bytes carry page numbers and
  layout that extracted text has already discarded, so extraction is the
  engine's job.
- **Three things are worth persisting per document**: the original file (so
  re-chunking never needs a re-upload), the chunks and vectors (what search
  reads), and one registry row (what a vector store answers badly).
- **Retrieved chunks and tool results are untrusted data.** Wrap them in explicit
  delimiters under a user role; never concatenate them into the system prompt.
- **Readiness is checked before the response starts.** `ChatService.stream()` is a
  plain function returning an iterator, not an async generator, so a missing
  implementation surfaces as 501 rather than a 200 with an empty body.

## Optional adapters

`chroma` and `mcp` are declared extras, so the core depends on neither. To pull
them into the dev workspace, change the root `pyproject.toml` dependency to
`chatbot-engine[chroma,mcp]` and re-run `uv sync`.

`pyproject.toml` declares Apache-2.0; add a `LICENSE` file if you publish the
service.
