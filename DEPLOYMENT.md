# Deployment

The defaults in this repository are for `make up` on a development machine:
every service is published to the host, the database password is in the compose
file, and nothing is authenticated. This document covers what changes for a
deployment other people can reach.

## Published images

Every version tag publishes multi-architecture images (amd64 and arm64) to this
repository's container registry:

```
ghcr.io/rouzbeh-abadi/chatbot-engine/engine:0.1.0
ghcr.io/rouzbeh-abadi/chatbot-engine/backend:0.1.0
ghcr.io/rouzbeh-abadi/chatbot-engine/frontend:0.1.0
```

Each is also tagged `0.1`, `0` and `latest`. Pin to a full version in
production.

The Python images install from `uv.lock`, so two builds of one commit produce
the same image. Both are built with the repository root as context:

```bash
docker build -f engine/Dockerfile .
docker build -f examples/backend/Dockerfile .
```

**The published engine image carries no agent plugins.** It runs the built-in
`loop` agent only. The demo stack in `docker-compose.yml` does not use it; it
builds `docker/engine-with-plugins.Dockerfile`, which installs
`examples/langgraph-agent` on top of the engine, so that `agent: graph` is
selectable. A deployment that needs a plugin builds its own image the same way;
see [docs/agents.md](docs/agents.md).

To run the published images, replace each service's `build:` block with the
corresponding `image:`.

## Cutting a release

```bash
git tag v0.1.0 && git push origin v0.1.0
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

## Scaling out

One engine process is the default shape, and two things in it belong to a
single process: the embedded Chroma files under `ENGINE_CHROMA_DIR`, and the
rate-limit buckets. Running several replicas requires moving both into shared
services, each selected by one setting:

| Setting | Moves | Service |
| --- | --- | --- |
| `ENGINE_CHROMA_URL` | the vector store | a Chroma server, `http://host:8000` |
| `ENGINE_REDIS_URL` | the rate-limit buckets | Redis, `redis://host:6379/0` |

Nothing above either seam changes: per-model collections, the deletion sweep
and the limit arithmetic are identical embedded or shared. The published engine
image includes the `redis` extra, so the setting works without a rebuild.

The scale overlay applies both and runs two engine replicas:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  -f docker-compose.scale.yml up -d
```

Two components still belong to one writer after this: the document registry
(SQLite) and the stored originals, both under the engine volume. They are
written only by document uploads, which are an operator action, so a deployment
that ingests from one place is unaffected. A deployment that uploads from
several replicas at once must put both on shared storage or replace them;
`DocumentRegistry` and `BlobStore` are the ports.

Readiness reflects the shared store: `GET /health/ready` reports
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
