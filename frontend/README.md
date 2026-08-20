# frontend

A React + TypeScript chat interface for the support assistant. It talks only to
the application backend on port `8000` — never to the engine directly, which holds
the model provider's credentials and has no notion of end-user permissions.

## Running it

```bash
make frontend        # installs and starts Vite on http://localhost:5173
```

Or directly:

```bash
cd frontend && npm install && npm run dev
```

You need the backend running (`make backend`) and, for real answers, the engine
too (`make engine`).

| Command | Does |
| --- | --- |
| `npm run dev` | Dev server with hot reload |
| `npm run build` | Typecheck, then a production build into `dist/` |
| `npm run typecheck` | Types only |
| `npm test` | 35 tests: the SSE reader, the event loop, markdown rendering |

## How it reaches the backend

Vite proxies `/api/*` to `http://localhost:8000`, so the browser only ever talks
to one origin and the backend needs no CORS middleware. If you deploy the built
files somewhere that cannot proxy, either put a reverse proxy in front of both or
add CORS to the backend and set `VITE_API_BASE`.

## What it does with the stream

`POST /chat` returns server-sent events. `EventSource` cannot issue a POST, so
[`api/client.ts`](src/api/client.ts) reads the response body directly. Two details
that matters:

- **Frames are reassembled across chunks.** A network chunk can end mid-frame, so
  anything after the last blank line stays buffered for the next read.
- **The response status is checked before the body is consumed**, so a `501` or
  `503` throws instead of looking like an answer that never arrived.

Each event maps to something on screen:

| Event | Rendered as |
| --- | --- |
| `retrieval` | A collapsible source list with scores, headings and excerpts |
| `token` | Appended to the answer, with a blinking caret while streaming |
| `tool_call_started` | An amber pulsing chip — the progress indicator |
| `tool_call_finished` | The chip turns teal with its duration, or red on failure |
| `usage` | Token count, cost and model under the answer |
| `error` | An inline problem card; the stream is already committed to `200` by then |
| `done` | Streaming stops |

An unrecognised `type` is ignored rather than treated as an error, so the UI keeps
working against a newer backend.

## Showing the real state instead of looking broken

The engine's agent is not implemented yet, so `/chat` answers `501` today. Rather
than a generic failure, the UI distinguishes:

| Situation | What you see |
| --- | --- |
| `501` from the engine | "The engine has no agent yet", with the function to implement |
| `503` — engine not running | "The engine service is not running. Start it with `make engine`." |
| Backend unreachable | "Could not reach the backend… start it with `make backend`." |
| You pressed Stop | "Stopped" |

The knowledge-base panel no longer hits that state: `GET /documents` is wired, so
it lists what the engine holds with each document's chunk count and status. A
document that failed to ingest shows as `failed` there rather than disappearing.

## Files

```text
src/
├── main.tsx              mounts the app
├── App.tsx               layout, conversation state, the event loop
├── api/
│   ├── types.ts          the wire contract, mirrored from the backend
│   └── client.ts         streaming, SSE parsing, error taxonomy
├── components/
│   ├── Message.tsx       one turn: tools, answer, sources, usage
│   ├── Answer.tsx        the assistant's markdown, rendered safely
│   ├── Sources.tsx       collapsible citations
│   ├── ToolCalls.tsx     the tool chips
│   ├── Composer.tsx      the input; Enter sends, Shift+Enter newlines
│   └── Knowledge.tsx     what the assistant knows
└── styles.css            one token set, light and dark from the OS
```

`api/types.ts` is a deliberate copy of the backend's event contract, not a
generated client — the frontend is a separate deployable. Keep it in step with
`backend/src/support_agent/engine_client/models.py`.

## Notes

- Answers render as **Markdown** (`react-markdown` + `remark-gfm`), so lists,
  emphasis, tables and inline code come out formatted. Your system prompt can let
  the model answer naturally.
  **Raw HTML is deliberately not rendered.** Answer text is assembled partly from
  retrieved documents and tool output; neither is trusted enough to execute in the
  page, so `rehype-raw` is intentionally absent. Generated links open in a new tab
  with the opener severed.
  The user's own message stays literal — their line breaks are meaningful and
  their asterisks are not formatting.
- Conversation history is sent from the browser on each turn, which is fine for a
  demo. A real deployment should keep transcripts server-side, since a client can
  otherwise fabricate earlier assistant turns.
