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

`mcp_tools.py` exposes three working tools over MCP, all reading the application
database:

| Tool | Returns |
| --- | --- |
| `get_booking_status` | passenger, route, fare, baggage, **flight number**, status |
| `get_flight_status` | on time / delayed / cancelled, times, gate |
| `create_support_ticket` | a ticket row, after checking the booking exists |

`get_booking_status` returns the flight number on purpose: it is what lets the
model chain a second call — "my booking is AB12CD, is my flight delayed?" — which
is the conversation worth demonstrating.

They run *here*, not in the engine: they read this application's data and must
execute with the calling user's permissions, which the engine cannot evaluate.
Only allowlisted names in `support.yaml` are ever exposed.

Two conventions, both aimed at the model: every value is a string in words a
customer would recognise ("cabin baggage only" rather than `null`), and a missing
record is returned as data rather than raised — a tool that raises for "not found"
teaches the model the tool is broken.

## The database

The backend owns bookings, flights and support tickets. The engine never touches
them; it reaches them only by calling the tools above.

```bash
make up          # postgres, engine, backend, mcp-tools
make migrate     # apply Alembic migrations
make seed-db     # load the demo data
```

The seed is idempotent and its dates are **relative to today**, so the check-in
window case stays inside the window whenever you run it. Six bookings, each
covering a case the knowledge base discusses:

| Booking | Case | Documents it exercises |
| --- | --- | --- |
| `AB12CD` | Flexible fare, refundable, inside the check-in window | refunds, check_in |
| `XY34ZT` | Basic fare, non-refundable, already cancelled | refunds, cancellations |
| `RF77KL` | Airline cancelled the flight — involuntary refund | cancellations, refunds |
| `MS55TR` | Two-leg itinerary, partial refund scope | refunds, booking_changes |
| `BG88QP` | Cabin baggage only, no checked allowance | baggage |
| `PS22WD` | Travel already completed — refund window closed | refunds |

Fare names match the knowledge base (`Flexible` / `Standard` / `Basic`) so a
retrieved policy and a tool result never contradict each other.

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
