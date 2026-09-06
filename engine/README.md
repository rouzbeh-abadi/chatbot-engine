# chatbot-engine

A standalone RAG and MCP tool-calling service.

The engine answers questions from a knowledge base, calls tools when it needs
live data, and streams the result back token by token. It knows how to retrieve,
how to prompt a model, and how to run a tool loop, and nothing at all about your
users, your product, or your domain.

That split is deliberate. Everything specific to an application stays in the
application: who the user is, what the assistant is called, which prompt it uses,
which tools it may reach. All of it arrives with each request, so the engine keeps
no state of yours and one engine can serve several applications.

It runs as its own HTTP service. Your backend calls it over HTTP; it calls your
tools back over MCP, because a tool that reads your data has to run where your
data and your permissions are.

```text
Your backend        →  HTTP    →  chatbot engine
chatbot engine      →  MCP     →  your tool server
```

## Architecture

**[Engine architecture](../docs/engine-architecture.md)** maps the entry points,
the layers a request passes through, and the file responsible for each step.
Read it before the source.

## Running it

```bash
uv run uvicorn chatbot_engine.app:app --port 8100 --reload
```

Interactive API docs: http://localhost:8100/docs

## Authentication and limits

The engine holds your provider credentials and has no notion of end users. It
cannot tell a cheap question from an expensive one, because both look identical
on the wire. So it authenticates *callers*, not people, and meters what they
spend.

```
ENGINE_API_KEYS=web:s3cret,batch:0ther
```

Every route but `/health` then needs a matching `X-API-Key`. Keys are compared in
constant time; a rejected one is logged with the path and client address, and
never with the key itself.

Names are not decoration. They buy two things.

**Throttling.** Rate limits are counted per name, so one runaway caller can be
slowed without turning the engine off for everyone else.

**Rotation.** Run the old and new key side by side until every caller has moved
across, then withdraw the old one. No synchronised restart.

`ENGINE_API_KEY` is the single-key shorthand, and is simply named `default`.

The limits below are per caller and generous by default: they stop runaway loops,
not normal use. Zero disables one.

| Variable | Default | Applies to |
| --- | --- | --- |
| `ENGINE_CHAT_RATE_LIMIT_PER_MINUTE` | 60 | `POST /chat` |
| `ENGINE_EVAL_RATE_LIMIT_PER_HOUR` | 20 | `POST /judge`, `POST /eval/rag` |
| `ENGINE_INGEST_RATE_LIMIT_PER_MINUTE` | 20 | `PUT /documents` |

Buckets live in the process's memory, so two replicas mean twice the effective
limit. That is a real limitation and the limit is still worth having. For an
exact global one, back `_Bucket` in `api/rate_limit.py` with Redis.

Set `ENGINE_ENV=production` and the engine refuses to start without a key at
all, rather than serving openly. See [DEPLOYMENT.md](../DEPLOYMENT.md).

**What the engine deliberately does not do: end-user identity.** An engine that
decided *which person* may ask something would need your user model, and it has
no business having one. That check belongs in the service calling it.

## Agents

A turn runs either as the built-in tool loop or as a LangGraph state machine,
chosen per assistant:

```yaml
agent: loop     # loop | graph
```

Both emit the same events, so the choice is invisible to the caller. You can
also install your own agent and select it by name, without forking the engine.
The bundled graph needs an optional extra:

```bash
pip install "chatbot-engine[graph]"
```

The loop is the default, and is the right answer until a turn needs to pause,
branch, or resume. **[Agents](../docs/agents.md)** explains the trade.

## Connecting a backend to it

Everything a backend needs is written up in one guide: the endpoints, the
request shape, how to read the streamed answer, how to upload documents, how to
handle the engine being down, and how to expose your own tools over MCP.

**→ [Connecting a backend to the chatbot engine](../docs/backend-integration.md)**

Start there. It includes a complete worked example and points at a working
reference backend.
