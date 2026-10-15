# Deployment

The defaults in this repository are for `make up` on a development machine:
every service is published to the host, the database password is in the compose
file, and nothing is authenticated. This document covers what changes for a
deployment other people can reach.

## Published images

Every version tag publishes multi-architecture images (amd64 and arm64) to this
repository's container registry:

```
ghcr.io/rouzbeh-abadi/chatbot-engine/engine:0.1.34
ghcr.io/rouzbeh-abadi/chatbot-engine/engine-langgraph:0.1.34
ghcr.io/rouzbeh-abadi/chatbot-engine/backend:0.1.34
ghcr.io/rouzbeh-abadi/chatbot-engine/frontend:0.1.34
```

`engine` carries the loop agent only. `engine-langgraph` is the same engine
with the LangGraph plugin installed, which adds the `graph` agent and the
`workflow` agent that runs a graph described in the request. Pick the second
when an assistant names either.

Each is also tagged `0.1`, `0` and `latest`. Pin to a full version in
production.

The Python images install from `uv.lock`, on a base pinned by digest, so two
builds of one commit install the same packages on the same base. Each
published image carries an SBOM and build provenance. Both are built with the repository root as context:

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
git tag v0.1.34 && git push origin v0.1.34
```

The Release workflow runs the full test suite, publishes the four images, and
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

In production the engine also serves no interactive docs or schema (`/docs`,
`/redoc`, `/openapi.json`), which are for building against it on a laptop,
and `GET /metrics` needs the API key; see [Observability](#observability).

## Secrets

Generate real values; `openssl rand -hex 32` is sufficient for all four.

| Variable | Guards |
| --- | --- |
| `ENGINE_API_KEY` | The engine, which holds the model provider credentials. Set the same value as `BACKEND_ENGINE_API_KEY`. For more than one caller use `ENGINE_API_KEYS`; see below. |
| `BACKEND_ADMIN_KEY` | `/admin` and the document write routes: bookings, tickets, evaluation runs, and the knowledge base. |
| `POSTGRES_PASSWORD` | The database. The demo value is committed to this repository. |
| `MCP_TOOLS_TOKEN` | The tool server, which takes the customer's id from each call. The production overlay gives it to the tool server and to the backend, which has the engine send it as a bearer token to that server alone. |

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

The terminator must add the visitor's address to `X-Forwarded-For` (Caddy and
Traefik do by default; nginx needs `proxy_set_header X-Forwarded-For
$proxy_add_x_forwarded_for`). The frontend's nginx believes that header only
from private addresses (`set_real_ip_from` in `examples/frontend/nginx.conf`),
so any machine or container on a private network in front of it may say who
the visitor is, and choose its own rate-limit bucket. Narrow the ranges to
where the terminator's requests come from: with the production overlay's
`127.0.0.1:8080`, that is the compose network's subnet (`docker network
inspect` names it; its gateway can change when the network is made again),
not `127.0.0.1`. A terminator on a public address is not believed at all, so
every visitor then shares its bucket: list its address there.

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
| `ENGINE_CHAT_RATE_LIMIT_PER_MINUTE` | 60 | the engine's `POST /chat`, and `POST /extract` for an image |
| `ENGINE_EVAL_RATE_LIMIT_PER_HOUR` | 20 | the engine's `POST /judge`, `POST /eval/rag` |
| `ENGINE_INGEST_RATE_LIMIT_PER_MINUTE` | 20 | `PUT /documents`; listing and deleting are unmetered |
| `ENGINE_EXTRACT_RATE_LIMIT_PER_MINUTE` | 120 | `POST /extract` for a document; an image is a model call and counts on the chat limit |

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

**Metrics** at the engine's `GET /metrics`, in Prometheus format: turns by
caller, agent and outcome; turn latency; tokens and cost by model; tool calls
by tool and result. A tool is counted by name only when the turn offered it,
up to 64 characters of the name; a call to a name the model made up is
counted under `not_offered`, so what a visitor steers the model into writing
neither adds a series nor is served here. Locally the route is open, like `/health`. Under
`ENGINE_ENV=production` it needs `X-API-Key` like any other caller, since its
counts name the callers and the models, so a scraper sends a key of its own.
`ENGINE_METRICS_PUBLIC=true` opens it for a scraper on a private network that
cannot send a header. `ENGINE_METRICS_ENABLED=false` removes the route and,
since 0.1.31, stops the counting too; turns are still logged.

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

Every trace carries the request id, the project id and the session id, the
same ids as the log lines, so a trace, a log line and a conversation in the
calling application all meet on one id. The user id is traced as a pseudonym
(`user-` and a keyed hash), since it can be a phone number or an email
address: traces still group by person, and the hash is keyed with the
engine's API keys, so it changes when they do. A misconfigured
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

Embedded Chroma belongs to one process: it keeps each collection's index in
that process's memory and writes it back from there, so a second engine on
the same `ENGINE_CHROMA_DIR` (another container on the volume, or uvicorn's
workers with `WEB_CONCURRENCY` above 1) would write its indexes over the
first one's. Since 0.1.33 a second engine on a directory another one holds
refuses to start, and says so; the lock (`.engine.lock` in that directory)
goes with the process that held it, however it ends. Several processes need
`ENGINE_CHROMA_URL`.

What one process keeps in memory, and replicas therefore do not share: the
turns a document's uploads and re-indexes take, a delete cancelling the
uploads of its document under way, a project's uploads refused for ten
minutes after it is purged, and the gate each project's searches, writes and
deletes share, which replacing or dropping one of its collections has to
itself. Each holds within one replica. Across replicas, send a document's
writes, and its delete, to one replica (by the project, at the load
balancer), or let a caller send one at a time, as ChatFrom's knowledge pages
do. A deleted document also stays keyword-searchable on the other replicas
for up to a minute, until each builds its keyword index again.

Because that gate is one process's, since 0.1.33 the engine does not rebuild
or drop a project's collections against a Chroma server as their chunks are
deleted,
as it does embedded (Operational notes): a rebuild racing another replica's
writes and deletes would lose live chunks and bring deleted ones back, and a
server keeps a dropped collection's files anyway, so every rebuild would add
a copy to its disk. On a server the room deleted chunks took stays in the
project's index, and a project whose last document goes keeps an empty
collection until it is purged, which drops it. Give the server's volume room
for everything ever indexed: the files the engine removes after a drop when
Chroma is embedded are the server's own to remove.

The workflow agent's paused turns stay in a file, `ENGINE_CHECKPOINT_DB`, on
the engine's volume: replicas share them only when they share that volume,
as the scale overlay's do on one host. On separate hosts, a question asked by
one replica is answered `resume_expired` by another; send a conversation's
turns to one replica.

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

- **Stopping.** Since 0.1.31 the engine closes a workflow's store of paused
  turns as it shuts down, so a stop after a turn that paused no longer waits
  for Docker's kill. Run the container with an init all the same (`init:
  true` in compose, or `docker run --init`): uvicorn ends its shutdown by
  raising the signal again, which a process running as PID 1 ignores, so
  anything left running would keep the container up until the kill; and an
  init reaps the processes the engine starts to read PDFs.
- **One collection per project, and the room a delete gives back.** Since
  0.1.30 each project has a collection of its own for each embedding model
  it uses. Chroma 1.5 keeps a deleted vector's room in its collection's
  index, and keeps a dropped collection's files on disk, so the engine
  gives the room back itself: a project's collection is dropped when its
  last document goes or the project is purged, and rebuilt from its live
  chunks once the chunks deleted from it (re-uploads, re-crawls) outnumber
  them and pass 2,000. The files a dropped collection leaves are then
  removed: every folder in `ENGINE_CHROMA_DIR` named like a segment that
  Chroma's own catalogue (`chroma.sqlite3`) no longer names. Keep that
  directory for Chroma alone (with the engine's `.engine.lock`). Since
  0.1.33 a rebuild copies while the project's searches, uploads and deletes
  go on, a
  thousand chunks at a time, and holds them only at the end, while it
  catches up with what changed during the copy and takes the old
  collection's place; most updates and deletes have nothing to tidy and
  hold no search at all. Against a Chroma server (`ENGINE_CHROMA_URL`)
  nothing is rebuilt or dropped but by a purge, and the server's files are
  not touched (Scaling out). `chroma.sqlite3` keeps the size it once
  reached, but reuses the room freed in it, so it no longer grows with what
  was deleted.
- **The first start of 0.1.30 moves every project's chunks** out of the one
  collection per model that every project shared before, into its own: in
  the background, and a project used before its turn is moved first. The
  shared collection is dropped once it is empty, and only then gives its
  room back: until the last project has moved it keeps its full size, so
  the volume needs room for a second copy of the whole store, not of the
  largest project. With less than `ENGINE_MIN_FREE_MB` free the move stops,
  logs why, and goes on when the project is next used. A project not moved
  yet can still be searched where it is, and since 0.1.33 its documents
  deleted and the project purged there: a delete needs no room.
- **What a delete leaves in Chroma's files for a while.** Embedded Chroma
  keeps each write in a log inside `chroma.sqlite3` until the collection's
  index takes it in, about every 1,000 records. A deleted chunk's text and
  vector can stay in that log until then, so a backup or a copy of the
  volume taken meanwhile still holds them; a project's whole knowledge goes
  from it at once when its collection is dropped. On a busy engine that is
  soon; on a quiet one it can be days. `chroma utils vacuum` on the stopped
  volume, or a Chroma server (`ENGINE_CHROMA_URL`) whose own settings say
  when its log is purged, shortens it.
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
  startup check found but did not block on is logged as a warning at boot. A
  tool server that cannot be reached is logged by its `name`, and an address
  in the error is cut to its host, since a path may carry a credential. For
  the same reason the HTTP clients' own request lines (`httpx`, `httpx2`,
  `httpcore`), which carry whole addresses, are logged only at `WARNING` and
  above, whatever the level. Since 0.1.33 an error's text is logged and sent
  in a failed tool call's event cut to its first 2,000 characters, as a tool
  server's message can run to megabytes.
- **Streaming.** Chat is server-sent events. A proxy in front must not buffer
  `/api/chat`; the bundled nginx config shows the three settings involved.
- **Which host serves a model.** OpenRouter serves a model from every host
  that offers it, and falls back to another when one fails: a Claude model
  from Amazon Bedrock or Google Vertex, a GPT model from Azure. Since 0.1.32
  every call the engine makes (the answer, the rewrite, the rerank, an image
  read, embeddings, the judge and RAGAS) names its model's maker as the only
  host, with no fallback: `anthropic`, `openai`, Google's own two
  (`google-ai-studio`, `google-vertex`), `mistral` and `xai`. A maker that
  serves none of its models on OpenRouter (DeepSeek's V4 models had only
  other companies' hosts in October 2026) is not pinned: those calls go by
  OpenRouter's defaults unless a request's `project.provider` names hosts.
  Nor are a listed maker's open-weight models (`openai/gpt-oss-*`,
  `google/gemma-*`, Mistral's 7B and Mixtral), which other companies serve
  and the maker may not, nor since 0.1.33 a model's variant (an id ending in
  `:free`, or any other `:variant`), which OpenRouter serves from the hosts it
  has for that variant. `ENGINE_PROVIDER_ROUTING=any` turns the pinning off.
  A maker's own outage then fails a call rather than moving it to another
  company. A request's `project.provider` is followed for its chatbot's model
  (the engine's `ENGINE_CHAT_MODEL` when it names none); what it asks without
  naming hosts (`data_collection: "deny"`, say) is added to the maker's
  pinning, and the calls on another model (the utility model, a judge,
  RAGAS) keep only that part. The embeddings (of documents and of the
  question) and an image read for `POST /extract` carry no request's
  routing, only the maker's pinning: a request's `data_collection` does not
  reach them. What a host may keep of the calls it serves is set on the
  OpenRouter account (its privacy settings), for every call those included,
  or per request with `data_collection`, for the calls that carry it.
- **Provider retries and timeouts.** A model stream that fails before its
  first token (rate limit, provider 5xx, dropped connection, an error the
  provider sends inside the stream) is retried up to
  `ENGINE_PROVIDER_MAX_RETRIES` times (default 3), doubling from half a
  second; a call that has streamed text to the caller is not retried, and the
  failure reaches the caller as an `error` event. A call whose text nobody is
  shown (a workflow's condition, the reading of a reply, a step that keeps its
  reply in a variable) is retried after its first token too, since 0.1.33.
  That is the only layer that retries
  a streamed call; the provider client retries the calls that are not
  streamed (the query rewrite, the rerank, the judge), with the same count.
  `ENGINE_PROVIDER_TIMEOUT_S` (default 60) bounds one provider call. Set the
  retries to 0 if a proxy in front of OpenRouter already retries, or the two
  will compound. Reading an image for `POST /extract` is never retried and
  takes at most 45 s, start to end, within the caller's patience; one cut off
  there answers `502` with an estimated `usage`, since the provider may still
  bill it. The RAGAS judge of `POST /eval/rag` keeps to the same timeout and
  retries (before 0.1.31 it had the SDK's ten minutes and six retries), and
  each of its cases is answered within `ENGINE_TURN_DEADLINE_S`; since 0.1.33
  a case that cannot be answered leaves that case unscored, not the run lost,
  and the judge's calls go through the proxy the environment names
  (`HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`, `NO_PROXY`), as every other
  provider call does.
- **Turn deadline.** `ENGINE_TURN_DEADLINE_S` (default 120) is the longest one
  chat turn may take, retrieval, model calls, retries and tool calls
  together. Past it the turn stops where it is and ends with the assistant's
  `unavailable_message`, a `usage` event and `done` with `stop`, so a provider
  that keeps failing or a tool server that does not answer cannot hold a
  request open for minutes.
- **Prompt and tool result sizes.** `ENGINE_PROMPT_CHARS` (default 400,000,
  about 100,000 tokens of English) is the most one model call's prompt may
  hold. Past it the oldest turns of the history are left out first; when no
  history at all is still too much, the last extracts go, then each file is
  cut to an even share. The system prompt and the message are never cut.
  Lower it for a model with a smaller context window.
  `ENGINE_TOOL_RESULT_CHARS` (default 20,000) is the most of one tool result
  the model reads, since a result is sent again with every later model call
  of the turn; the rest is cut, with a line saying how much was left out.
  Since 0.1.29 the turn's tool results count against `ENGINE_PROMPT_CHARS`
  too: a result is cut to the room the prompt has left once the history has
  given way to it (the oldest turns going first, in every agent since
  0.1.33), and a call there is no room for is answered as not run without
  running. A turn keeps its tool results until it ends, so since 0.1.33 it
  keeps no more than `ENGINE_TURN_TOOL_RESULT_CHARS` (default 1,000,000) of
  them, counting what a model reads of each and what a workflow's variables
  keep; past it a call is answered as not run, and a workflow's Tool Call
  step fails (its `on_error` applies). The loop and graph agents keep no
  more than their prompt holds, so with the defaults this binds on
  workflows, whose tool steps can loop. `ENGINE_WORKFLOW_VAR_CHARS` (default
  100,000) is the most one workflow variable holds: a Tool Call step keeps
  more of a result than a model reads, so a list of options stays whole, and
  past this it is cut, with a line saying how much was left out. Nothing
  else of a result is kept: before 0.1.33 every call kept the whole answer
  (up to the 4 MiB below) for the rest of the turn.
  `ENGINE_MAX_TOOL_CALLS_PER_REPLY` (default 10) is the most calls one model
  reply has run; the rest are answered as not run, and the model may ask
  again in its next round.
  `ENGINE_MCP_MAX_RESPONSE_BYTES` (default 4 MiB) is the most one answer from
  a tool server may be once inflated. Past it the answer is not read further
  and the call fails, so neither a large answer nor a small compressed one can
  fill the engine's memory.
- **Request size.** `ENGINE_MAX_BODY_BYTES` (default 8 MiB) is the largest
  JSON body the engine reads. A larger one is refused with 413 before it is
  read, at once when its `Content-Length` says so and otherwise as soon as it
  grows past the cap. `PUT /documents` and `POST /extract` keep their own
  limit of 25 MB, with 64 KB of room for the form around the file, held the
  same way since 0.1.31: a larger upload is refused before it is read, and
  so is one whose API key is wrong (`401`). An evaluation (`POST /judge`,
  `POST /eval/rag`) whose caller goes away is stopped, rather than going on
  answering and grading on the caller's key; nobody reads its `499`.
- **Reading files for a chat.** `ENGINE_EXTRACT_TIMEOUT_S` (default 20)
  bounds the reading of one document for `POST /extract`: it runs in a
  process of its own, bounded in CPU time and memory where the system allows,
  and is killed at the deadline, the file refused with 422.
  `ENGINE_EXTRACT_CONCURRENCY` (default 2) is how many such readings may run
  at once. Past it a reading waits up to 5 seconds for its turn, with at most
  two in line for each place, and is refused with 503 and `Retry-After` after
  that, so files cannot fill the box. Since 0.1.33 each caller (the API key,
  and the form's `project_id` when it sends one) reads one file at a time
  and may have at most half the line, so one caller's burst cannot refuse
  everyone else's files: send `project_id` when one key serves many
  chatbots. `ENGINE_EXTRACT_PARSE_MB` (default 4) is
  how much page content a PDF is parsed for. Past it the rest is not read, and
  a page that can show no text is not parsed at all, so a small file of
  drawing cannot hold a place for its whole deadline.
- **Reading knowledge documents.** A PDF uploaded to the knowledge base is
  read the same way, in a process of its own, with bounds sized for indexing:
  `ENGINE_INDEX_READ_TIMEOUT_S` (default 60) for the reading, and
  `ENGINE_INDEX_MAX_CHARS` (default 2,000,000) for the text one document may
  give. A document past either is a `failed` record that says why, and the
  upload answers 422. Plain text and Markdown are decoded in place, within the
  same character bound. `ENGINE_INDEX_MAX_CHUNKS` (default 20,000) is the most
  chunks one document may make; one over it is refused with 422 before
  anything is embedded. Chunks are embedded and written
  `ENGINE_INDEX_BATCH_SIZE` (default 128) at a time, so a document's vectors
  are never all in memory at once, and indexing runs on
  `ENGINE_INDEX_CONCURRENCY` (default 2) threads of its own, so uploads never
  hold up another project's chats: since 0.1.33 a write takes its project's
  gate only once it has one of those threads, and since 0.1.34 a large
  document's reading, cutting and scan for instructions to an AI no longer
  slow every other answer while they run: the threads hand the interpreter
  back every millisecond, and the scan reads 32 KB at a time. A write to
  embedded Chroma still holds every request for as long as it takes, so a
  smaller batch shortens that wait. A project's own chats wait while one of
  its collections is replaced or dropped (the end of a rebuild, a purge),
  and while it moves once, on the first start of 0.1.30.
- **Free space.** `ENGINE_MIN_FREE_MB` (default 512) is the room an upload
  leaves on the volumes the engine writes to. Below it the upload is refused
  with 503 before anything is written: a full volume can freeze embedded
  Chroma for every tenant and leave its index unreadable after a restart.
  The move of 0.1.30 and a rebuild keep to the floor too, and since 0.1.33
  the refusal says which was refused; deletes and purges never do, since
  they are how room is made. Zero turns the check off, and so brings that
  freeze back: since 0.1.33, with embedded Chroma, an engine under
  `ENGINE_ENV=production` refuses to start with it. Watch the volume all the same; the floor only keeps
  indexing from filling it. Since 0.1.32 `GET /health` says
  `"disk": "low"` once a volume has less than twice the floor free, for a
  monitor to warn on while uploads still work.
- **Keyword indexes.** `ENGINE_KEYWORD_INDEX_MB` (default 512) is the memory
  the keyword indexes of hybrid retrieval may hold together. Past it the one
  searched longest ago is dropped and built again when it is next needed.
