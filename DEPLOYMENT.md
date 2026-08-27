# Deploying this

The defaults in this repository are tuned for `make up` on a laptop: every
service is published to the host, the database password is in the compose file,
and nothing is authenticated. That is the right shape for reading the code and
the wrong shape for anything other people can reach.

This document is the difference between the two.

## Running the published images

Every version tag publishes multi-architecture images (amd64 and arm64) to this
repository's container registry, so a deployment does not have to build
anything:

```
ghcr.io/rouzbeh-abadi/chatbot-engine/engine:0.1.0
ghcr.io/rouzbeh-abadi/chatbot-engine/backend:0.1.0
ghcr.io/rouzbeh-abadi/chatbot-engine/frontend:0.1.0
```

Each is tagged `0.1.0`, `0.1`, `0` and `latest`, so you can pin as tightly as you
want. Pin to a full version in production — `latest` moves under you.

To run the published images instead of building locally, point the compose
services at them with `image:` and drop their `build:` blocks.

## Cutting a release

```bash
git tag v0.1.0 && git push origin v0.1.0
```

That runs the full test suite again (a tag can point at any commit, including
one that never went through a pull request), publishes the three images, and
opens a GitHub release with generated notes. Running the *Release* workflow
manually from the Actions tab builds everything but publishes nothing, which is
how to rehearse it without moving a tag.

## The one thing to do first

Set both services to production. They will then refuse to start on a default
that is only safe locally, instead of serving with one:

```
BACKEND_ENV=production
ENGINE_ENV=production
```

A refused start looks like this, and names every variable it wants:

```
RuntimeError: refusing to start with BACKEND_ENV=production:
  - BACKEND_ADMIN_KEY is not set, so /admin -- every booking, every ticket, ...
  - BACKEND_DATABASE_URL still carries the demo credentials (support_agent:...
```

This is deliberately a startup failure rather than a warning. A container that
would leak on its first request should fail its health check and never enter the
load balancer.

## Secrets

Generate real values. `openssl rand -hex 32` is a fine source for all three.

| Variable | Guards |
| --- | --- |
| `ENGINE_API_KEY` | The engine. It holds your provider credentials and has no notion of end users, so only the backend may reach it. Set the same value as `BACKEND_ENGINE_API_KEY`. For more than one caller, use `ENGINE_API_KEYS` instead — see below. |
| `BACKEND_ADMIN_KEY` | `/admin` and the document write routes: every booking, every ticket, the evaluation runs, and what the assistant knows. |
| `POSTGRES_PASSWORD` | The database. The demo value is in this repository, so it is not a password. |

Keep them out of the image and out of git — pass them through your platform's
secret mechanism. Nothing in this repo reads a secret at build time.

## The network shape

Only the frontend should be reachable. The engine, the MCP tool server and
Postgres have no business being on the internet:

```
        internet
           │
           ▼
   [ TLS terminator ]         you provide this
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

With Docker Compose, the production overlay does this for you:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

It unpublishes every port but the frontend's, binds that one to `127.0.0.1`,
requires each secret from the environment with no fallback, and sets both
services to production. Read it before you run it — it is short, and it is the
executable version of this section.

## TLS

Nothing here terminates TLS, on purpose: it is the one piece that depends
entirely on where you run. Put Caddy, Traefik, nginx, or a cloud load balancer
in front of the frontend port, and set `Strict-Transport-Security` there — the
frontend container speaks plain HTTP to whatever is in front of it, so an HSTS
header set inside it would be ignored.

## What is authenticated, and what is not

Be clear-eyed about this before you put it in front of users.

**Guarded by `BACKEND_ADMIN_KEY`** — `/admin/*` (bookings, tickets, evaluation
runs) and the document write routes (`PUT /documents`, `DELETE /documents/{id}`).
One shared operator secret, compared in constant time. It tells you someone is
an operator; it cannot tell you *which* operator, so there is no audit trail.

**Open** — `GET /health`, `GET /models`, `GET /documents`, and the chat routes.
Chat is rate limited but unauthenticated: anyone who reaches the frontend can
spend model credits within that limit. If that is not acceptable for your
deployment, chat is where to put your authentication.

**Not implemented: user accounts.** `api/identity.py` is the seam. Today it
returns `anonymous` unless `BACKEND_TRUST_USER_HEADER=true`, in which case it
believes the `X-User-Id` header — which is only correct when a proxy in front
authenticates the user and *overwrites* that header, as the bundled nginx config
does. Replace `resolve_user_id` with your own (a session cookie, a validated
JWT) and every route that takes `UserIdDep` follows.

**Not implemented: authorisation.** Nothing checks that a given user may use a
given project, or that a booking belongs to the person asking about it. For a
multi-tenant product that check belongs in the backend, next to identity — the
engine only ever sees an opaque id.

## The engine's keys

`ENGINE_API_KEY` is the single-key shorthand. For anything with more than one
caller, or that you intend to rotate, use named keys instead:

```
ENGINE_API_KEYS=web:s3cret,batch:0ther
```

Both forms combine; the single key is simply named `default`.

Naming buys two things a lone secret cannot give you:

- **Rotation without downtime.** Issue `web-next`, let both work while callers
  move across, then withdraw `web`. With one key every rotation is a
  synchronised restart of everything that calls the engine, which is why
  rotations quietly stop happening.
- **Attribution.** The name is what appears in logs and what rate limits are
  counted against, so one runaway caller can be found and throttled without
  turning the engine off for everybody.

Keys are matched in constant time against every configured value. A rejected key
is logged with the path and the client address — and never with the key itself.

## Rate limits

Per caller, per process:

Both services meter independently, per caller. The engine's limits are not
redundant with the backend's: the engine is where provider credits are actually
spent, and a caller's own limiter is a courtesy, not a control the engine can
verify exists.

| Variable | Default | Applies to |
| --- | --- | --- |
| `BACKEND_CHAT_RATE_LIMIT_PER_MINUTE` | 30 | `POST /chat`, `POST /chat/sync` |
| `BACKEND_EVAL_RATE_LIMIT_PER_HOUR` | 20 | `POST /admin/eval/*` |
| `ENGINE_CHAT_RATE_LIMIT_PER_MINUTE` | 60 | the engine's `POST /chat` |
| `ENGINE_EVAL_RATE_LIMIT_PER_HOUR` | 20 | the engine's `POST /judge`, `POST /eval/rag` |
| `ENGINE_INGEST_RATE_LIMIT_PER_MINUTE` | 20 | `PUT /documents` — listing and deleting are free, and unmetered |

The backend buckets by user id or client address; the engine buckets by the name
of the key that authenticated the call.

Zero disables a limit. Callers are bucketed by user id when one is
authenticated, and by client address otherwise — which is why the containers run
uvicorn with `--proxy-headers`, and why you must set `FORWARDED_ALLOW_IPS` to
your proxy's address if the backend port is reachable from anywhere else.
`X-Forwarded-For` is trivially forged, so believing it from an untrusted source
lets one client spread its usage across as many buckets as it likes.

The buckets live in the process's memory. Two replicas means twice the effective
limit, and a restart forgets everything. That is a real limitation, and the
limit is still worth having — it stops runaway clients and accidental loops. For
an exact global limit, back `_Bucket` in `api/rate_limit.py` with Redis; nothing
above `RateLimiter.check` changes.

## Database migrations

Run them as a release step, before the new backend takes traffic:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  run --rm backend alembic upgrade head
```

Do not run `make seed-db` against a real database — it loads the demo bookings.

## Operational notes

- **Both containers run as an unprivileged user** (uid 10001) and the overlay
  sets `no-new-privileges`. This matters on upgrade: a named volume created by
  an *older* image is owned by root, and Docker only takes ownership from the
  image when it initialises an empty volume. An engine that starts and then
  cannot write vectors is this. Fix it once, from the host:

  ```bash
  docker compose run --rm --user root engine chown -R 10001:10001 /var/lib/chatbot-engine
  ```

  A bind mount you provide yourself needs the same `chown 10001` treatment.
- **Health checks.** `GET /health` on both services is a liveness check and
  requires no authentication. The engine also has `GET /health/ready`, which
  reports which capabilities are wired up.
- **Logs** go to stdout at `ENGINE_LOG_LEVEL` (default `INFO`). Anything the
  startup check found but did not block on is logged as a warning at boot —
  worth alerting on, since it means something is misconfigured.
- **Streaming.** Chat is server-sent events. Any proxy you add must not buffer
  `/api/chat`, or the answer arrives in one lump at the end. The bundled nginx
  config shows the three settings involved.
