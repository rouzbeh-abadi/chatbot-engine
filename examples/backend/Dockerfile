# The example backend, installed from the lockfile. Built with the repository
# root as context, like the engine:
#
#   docker build -f examples/backend/Dockerfile .
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# `copy`, because the cache below is a mount that is gone at run time.
ENV UV_LINK_MODE=copy
WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY engine/pyproject.toml engine/README.md engine/LICENSE engine/NOTICE engine/
COPY examples/backend/pyproject.toml examples/backend/README.md examples/backend/
COPY examples/langgraph-agent/pyproject.toml examples/langgraph-agent/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package support-agent

COPY examples/backend/src examples/backend/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package support-agent

# Migrations and the demo seed have to be runnable in the deployed container,
# not just on a laptop.
COPY examples/backend/alembic.ini examples/backend/
COPY examples/backend/alembic examples/backend/alembic
COPY examples/backend/scripts examples/backend/scripts
COPY examples/backend/knowledge examples/backend/knowledge

# Drop root. A web process that never needs to write to its own image should not
# be able to: it turns a file-write bug into a failed syscall instead of a
# modified application. /app stays root-owned for the same reason, and because
# a chown of the install would copy every file into a new layer.
RUN useradd --system --uid 10001 --create-home app
USER app

WORKDIR /app/examples/backend
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
EXPOSE 8000

# `--proxy-headers` makes uvicorn read the client address from X-Forwarded-For
# rather than seeing every request as coming from the proxy, which is what the
# rate limiter buckets on. It is only honoured from addresses named in
# FORWARDED_ALLOW_IPS, which must be set to your proxy: the header is trivially
# forged, so trusting it from anywhere would let one client spread its usage
# across as many buckets as it likes.
CMD ["uvicorn", "support_agent.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers"]
