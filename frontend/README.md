# frontend

Not started. The framework choice is open (the brief allows Next.js or
Streamlit), so nothing is scaffolded here yet.

## What it talks to

The backend, at `http://localhost:8000` — never the engine directly. The engine
holds the provider credentials and has no notion of end-user permissions.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/chat` | Chat. Returns an SSE stream. |
| `POST` | `/chat/sync` | Non-streaming fallback |
| `GET` | `/documents` | What the assistant knows, for a sources panel |

These answer `501` until the engine has an `Agent`, and `503` if the engine is not
running — worth handling separately in the UI, since they mean different things.

## Handling the stream

`POST /chat` returns `text/event-stream`. Each frame has an `event:` name and a
JSON `data:` payload:

| Event | Render as |
| --- | --- |
| `retrieval` | The sources panel and citations (`sources[]`) |
| `tool_call_started` | A progress indicator naming the tool |
| `tool_call_finished` | Tool result or error; pair with `started` via `call_id` |
| `token` | Append `text` to the answer |
| `usage` | Token count and cost |
| `error` | An error state — may arrive mid-stream, after a 200 |
| `done` | Stop; `finish_reason` says why |

`EventSource` cannot issue a POST, so use `fetch` with a `ReadableStream` reader,
or the `@microsoft/fetch-event-source` package.

These four cover the brief's UI requirements outright: `retrieval` gives you
sources, `tool_call_*` gives you tool results and progress indicators, and `usage`
gives you the token/cost display. `/chat/sync` returns the same information folded
into one object, including a `tool_calls` list.
