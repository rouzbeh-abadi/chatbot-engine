# Deployment

The defaults in this repository are for `make up` on a development machine:
every service is published to the host, the database password is in the compose
file, and nothing is authenticated. This document covers what changes for a
deployment other people can reach.

## Published images

Every version tag publishes multi-architecture images (amd64 and arm64) to this
repository's container registry:

```
ghcr.io/rouzbeh-abadi/chatbot-engine/engine:0.1.7
ghcr.io/rouzbeh-abadi/chatbot-engine/engine-langgraph:0.1.7
ghcr.io/rouzbeh-abadi/chatbot-engine/backend:0.1.7
ghcr.io/rouzbeh-abadi/chatbot-engine/frontend:0.1.7
```

`engine` carries the loop agent only. `engine-langgraph` is the same engine
with the LangGraph plugin installed, which adds the `graph` agent and the
`workflow` agent that runs a graph described in the request. Pick the second
when an assistant names either.

Each is also tagged `0.1`, `0` and `latest`. Pin to a full version in
production.

The Python images install from `uv.lock`, so two builds of one commit produce
the same image. Both are built with the repository root as context:

```bash
docker build -f engine/Dockerfile .
docker build -f examples/backend/Dockerfile .
```

**Which engine image.** `engine` runs the built-in `loop` agent only.
`engine-langgraph` is built from `docker/engine-with-plugins.Dockerfile`,
which installs `examples/langgraph-agent` on top of the engine; it is what
the demo stack in `docker-compose.yml` builds locally, and what a deployment
that names `agent: graph` or `agent: workflow` should pull. A deployment with
its own plugin builds its own image the same way; see
[docs/agents.md](docs/agents.md).

To run the published images, replace each service's `build:` block with the
corresponding `image:`.

## Cutting a release

```bash
git tag v0.1.7 && git push origin v0.1.7
```

The Release workflow runs the full test suite, publishes the three images, and
creates a GitHub release with generated notes. The suite runs again because a
tag can point at any commit.

Running the workflow manually from the Actions tab builds everything and
publishes nothing.

## Production mode

Set both services to production:

```
BACKEND_ENV=production
ENGINE_ENV=production
```

In this mode each service refuses to start on any default that is only safe
locally, and names every variable it requires:

```
RuntimeError: refusing to start with BACKEND_ENV=production:
  - BACKEND_ADMIN_KEY is not set, so /admin -- every booking, every ticket, ...
  - BACKEND_DATABASE_URL still carries the demo credentials (support_agent:...
```

A refused start fails the health check, so the container never enters a load
balancer.

## Secrets

Generate real values; `openssl rand -hex 32` is sufficient for all three.

| Variable | Guards |
| --- | --- |
| `ENGINE_API_KEY` | The engine, which holds the model provider credentials. Set the same value as `BACKEND_ENGINE_API_KEY`. For more than one caller use `ENGINE_API_KEYS`; see below. |
| `BACKEND_ADMIN_KEY` | `/admin` and the document write routes: bookings, tickets, evaluation runs, and the knowledge base. |
| `POSTGRES_PASSWORD` | The database. The demo value is committed to this repository. |

Pass secrets through the platform's secret mechanism. Nothing in this
repository reads a secret at build time.

## Network shape

Only the frontend is reachable. The engine, the tool server and Postgres are
internal:

```
        internet
           │
           ▼
   [ TLS terminator ]         provided by the deployment
           │
           ▼
   [ frontend :80 ]           nginx: static files + /api proxy
           │
           ▼
   [ backend :8000 ]
        │        │
        ▼        ▼
 [ engine ]  [ mcp-tools ]  [ postgres ]
```

The production overlay applies this:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

It unpublishes every port but the frontend's, binds that one to `127.0.0.1`,
requires each secret from the environment with no fallback, and sets both
services to production.

## TLS

Nothing in this repository terminates TLS. Put Caddy, Traefik, nginx or a cloud
load balancer in front of the frontend port and set `Strict-Transport-Security`
there. The frontend container speaks plain HTTP to whatever is in front of it,
so an HSTS header set inside it would be ignored.

## Authentication

**Guarded by `BACKEND_ADMIN_KEY`:** `/admin/*` and the document write routes
(`PUT /documents`, `DELETE /documents/{id}`). One shared operator secret,
compared in constant time. It establishes that a caller is an operator, not
which one, so there is no audit trail.

**Open:** `GET /health`, `GET /models`, `GET /agents`, `GET /documents`,
`GET /memory`, `DELETE /memory`, and the chat routes. Chat is rate limited but
unauthenticated: anyone who reaches the frontend can spend model credits within
that limit.

**User identity is not implemented.** `api/identity.py` is the seam. By default
every caller is `anonymous`. With `BACKEND_TRUST_USER_HEADER=true` the backend
uses the `X-User-Id` header instead, which is only safe when a proxy in front
authenticates the user and sets the header itself. The bundled nginx clears
`X-User-Id` on every request for this reason. To add accounts, replace
`resolve_user_id`; every route that takes `UserIdDep` follows.

**Long-term memory has its own identity rule.** Without authentication, memory
is partitioned by `X-Client-Id`, an id the browser generates and keeps. It
passes through the proxy untouched and is never treated as authentication.
It is a partition, not a permission: any client can send another client's id.
This is acceptable only while memory is the sole thing keyed on it. See
[docs/memory.md](docs/memory.md) before serving real users.

**Authorisation is not implemented.** Nothing checks that a user may use a
project, or that a booking belongs to the person asking about it. That check
belongs in the backend, next to identity.

## Engine keys

`ENGINE_API_KEY` is the single-key form. For more than one caller, or for
rotation, use named keys:

```
ENGINE_API_KEYS=web:s3cret,batch:0ther
```

Both forms combine; the single key is named `default`. Named keys provide:

- **Rotation without downtime.** Issue a new name, let both work while callers
  move across, then withdraw the old one.
- **Attribution.** The name appears in logs and is what rate limits are counted
  against, so one caller can be throttled without affecting the others.

Keys are compared in constant time. A rejected key is logged with the path and
client address, never with the key itself.

## Rate limits

Both services meter independently, per caller. The engine's limits are not
redundant with the backend's: the engine is where provider credits are spent,
and it cannot verify that a caller applies its own limit.

| Variable | Default | Applies to |
| --- | --- | --- |
| `BACKEND_CHAT_RATE_LIMIT_PER_MINUTE` | 30 | `POST /chat`, `POST /chat/sync` |
| `BACKEND_EVAL_RATE_LIMIT_PER_HOUR` | 20 | `POST /admin/eval/*` |
| `ENGINE_CHAT_RATE_LIMIT_PER_MINUTE` | 60 | the engine's `POST /chat` |
| `ENGINE_EVAL_RATE_LIMIT_PER_HOUR` | 20 | the engine's `POST /judge`, `POST /eval/rag` |
| `ENGINE_INGEST_RATE_LIMIT_PER_MINUTE` | 20 | `PUT /documents`; listing and deleting are unmetered |

Zero disables a limit. The backend buckets by authenticated user id when there
is one and by client address otherwise; the engine buckets by the name of the
key that authenticated the call.

Because the client address decides the bucket, the containers run uvicorn with
`--proxy-headers`. If the backend port is reachable from anywhere other than
the proxy, set `FORWARDED_ALLOW_IPS` to the proxy's address: `X-Forwarded-For`
is trivially forged, and a client that can forge it chooses its own bucket.

By default the buckets are held in process memory: exact for one replica, and
multiplied by the replica count for several. Set `ENGINE_REDIS_URL` and every
engine replica charges one shared bucket per caller, atomically. See
[Scaling out](#scaling-out).

## Observability

Three things the engine reports about itself, all on by default.

**A request id on every log line.** The backend mints one per request, or
keeps a well-formed `X-Request-Id` the caller sent, returns it in the
response, and sends it to the engine; the engine keeps it and forwards it to
the tool server. One id therefore follows a turn through all three processes,
and a failure in one can be found in the logs of the others by that id. Both
services accept and return the header, so a proxy or a browser that already
tags requests sees its own id come back.

**One line per chat turn**, from the logger `chatbot_engine.turn`: caller,
agent, model, tokens in and out, cost, tool calls and how many failed, the
outcome, and the duration. A turn the client abandoned is logged as
`cancelled`. With `ENGINE_LOG_FORMAT=json` every line is one JSON object with
those fields, for a collector that indexes them; the default is text for a
terminal. The backend has the same switch, `BACKEND_LOG_FORMAT`.

**Metrics** at the engine's `GET /metrics`, in Prometheus format,
unauthenticated like `/health`: turns by caller, agent and outcome; turn
latency; tokens and cost by model; tool calls by result.
`ENGINE_METRICS_ENABLED=false` removes the route.

Not included: distributed tracing. The request id gives the correlation;
spans and a trace backend are a deployment's own choice.

### Tracing

`ENGINE_TRACING` records every model call of a turn, with its prompt,
response, latency and tokens, where an operator can open it:

| Value | Needs | Where traces go |
| --- | --- | --- |
| `off` | nothing | nowhere (default) |
| `langsmith` | `ENGINE_LANGSMITH_API_KEY`, optional `ENGINE_LANGSMITH_PROJECT` | LangChain's hosted LangSmith |
| `langfuse` | `ENGINE_LANGFUSE_PUBLIC_KEY`, `ENGINE_LANGFUSE_SECRET_KEY`, optional `ENGINE_LANGFUSE_HOST`; the image has the `tracing` extra | Langfuse cloud, or a self-hosted Langfuse at the host you name |

Every trace carries the request id, the project id, the session id and the
user id, the same ids as the log lines, so a trace, a log line and a
conversation in the calling application all meet on one id. A misconfigured
destination stops the engine at startup rather than recording nothing.

Traces contain prompts and retrieved text. For a product that handles other
people's documents, Langfuse on a host you control is the option that keeps
that data where you can answer for it. An assistant can also name its own
destination in the request (`project.tracing`, see the integration guide),
which is how a multi-tenant product lets each customer keep their traces
on their own Langfuse.

## Scaling out

One engine process is the default shape, and four things in it belong to a
single process or host: the embedded Chroma files, the rate-limit buckets, the
SQLite document registry, and the directory of stored originals. Running
several replicas means moving each into a shared service, selected by one
setting:

| Setting | Moves | Service |
| --- | --- | --- |
| `ENGINE_CHROMA_URL` | the vector store | a Chroma server; `ENGINE_CHROMA_TOKEN` when it requires a credential |
| `ENGINE_REDIS_URL` | the rate-limit buckets | Redis |
| `ENGINE_REGISTRY_URL` | the document registry | Postgres |
| `ENGINE_BLOB_S3_BUCKET` | the stored originals | S3, or MinIO via `ENGINE_BLOB_S3_ENDPOINT_URL` |
| `BACKEND_REDIS_URL` | the backend's own rate-limit buckets | Redis |

Nothing above any of these seams changes. The published images include every
driver, so each setting works without a rebuild.

The scale overlay applies all of them and runs two engine replicas:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  -f docker-compose.scale.yml up -d
```

It requires `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` in addition to the
production overlay's secrets, and points the engine's registry at the demo's
Postgres server, where the engine keeps its own table. A real deployment gives
the engine its own database and bucket.

Readiness reflects the shared vector store: `GET /health/ready` reports
`vector_store: false`, and therefore `ready: false`, when the Chroma server
does not answer.

## Database migrations

Run migrations as a release step, before the new backend takes traffic:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  run --rm backend alembic upgrade head
```

Do not run `make seed-db` against a real database; it loads the demo bookings.

## Operational notes

- **Unprivileged containers.** Both services run as uid 10001 and the overlay
  sets `no-new-privileges`. A named volume created by an older image is owned
  by root, and Docker only assigns ownership when it initialises an empty
  volume. An engine that starts and then cannot write vectors is this case:

  ```bash
  docker compose run --rm --user root engine chown -R 10001:10001 /var/lib/chatbot-engine
  ```

  A bind mount needs the same `chown 10001`.
- **Health checks.** `GET /health` on both services is a liveness check and
  needs no authentication. The engine's `GET /health/ready` reports `ready`,
  `model_provider`, `vector_store` and the installed `agents`; `ready` is
  false when no provider key is set or the vector store does not answer. A
  readiness probe should gate on that field. The status code stays 200 so the
  body is readable.
- **Logs** go to stdout at `ENGINE_LOG_LEVEL` (default `INFO`). Anything the
  startup check found but did not block on is logged as a warning at boot.
- **Streaming.** Chat is server-sent events. A proxy in front must not buffer
  `/api/chat`; the bundled nginx config shows the three settings involved.
- **Provider retries and timeouts.** A model stream that fails before its
  first token (rate limit, provider 5xx, dropped connection) is retried up
  to `ENGINE_PROVIDER_MAX_RETRIES` times (default 3), doubling from half a
  second; a call that has streamed text is not retried, and the failure
  reaches the caller as an `error` event. `ENGINE_PROVIDER_TIMEOUT_S`
  (default 60) bounds one provider call. Set the retries to 0 if a proxy in
  front of OpenRouter already retries, or the two will compound.
