# support-agent — application backend

The product API. Owns users, assistant configuration, the document endpoints, the
domain tools, and the browser-facing transport. Calls the AI engine over HTTP and
**never imports its Python package**.

## What it owns

- FastAPI routes for the app, and request validation
- Authentication and the user identity boundary (`X-User-Id` is a placeholder)
- Assistant configuration — `projects/*.yaml`, validated on load
- Document upload / list / delete, forwarding raw bytes to the engine
- The domain tools, and the MCP server that exposes them to the engine
- SSE formatting for the frontend
- `engine_client/` — the only code that talks to the engine

It owns no prompts, no retrieval, no chunking and no model calls.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | The app, `/health`, and engine-error → HTTP status mapping |
| `api/chat.py` | `POST /chat`, `/chat/sync` |
| `api/documents.py` | Upload, list and delete the knowledge base |
| `api/streaming.py` | SSE framing and folding, for the browser |
| `engine_client/client.py` | `EngineClient` and its error taxonomy |
| `engine_client/models.py` | The wire contract, mirrored deliberately |
| `engine.py` | The `EngineClient` FastAPI dependency |
| `assistant.py` | Loads and validates `projects/*.yaml` |
| `mcp_tools.py` | The three domain tools, served over MCP |
| `settings.py` | `BACKEND_*` environment variables |

## Run it

```bash
make backend    # :8000 -- needs the engine on :8100
make tools      # :8200 -- the MCP tool server
```

## Configuration

`projects/support.yaml` is the assistant: prompt, model, `top_k`, and which MCP
tools it may use. It is validated against the wire contract on load, so a typo
fails here with our error message rather than as a 422 from the engine.

`load_project` is cached — **restart after editing the YAML.**

## The tools

`mcp_tools.py` exposes three tools over MCP. Their signatures and docstrings are
complete and discoverable — the schema is what the model reads when deciding
whether to call one — and their bodies raise `NotImplementedError`.

They run *here*, not in the engine: they read this application's data and must
execute with the calling user's permissions, which the engine cannot evaluate.
Only allowlisted names in `support.yaml` are ever exposed.

## Why the contract is duplicated

`engine_client/models.py` is a copy of `chatbot_engine/models/`, not an import.
Depending on the engine's package would put the two services back in one
deployable. The copy is guarded by `tests/test_contract_parity.py` at the
repository root, which fails when the two drift.

## Tests

```bash
make test
```

Backend tests replace the engine through a FastAPI dependency override, so the
routes, validation and SSE framing under test are the real ones. `test_boundaries.py`
enforces the three rules that keep the split intact: no `chatbot_engine` import,
no AI libraries, and HTTP to the engine only from `engine_client/`.
