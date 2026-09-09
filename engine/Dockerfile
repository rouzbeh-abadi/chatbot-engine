# The engine, installed from the lockfile.
#
# Built with the repository root as context, because the lockfile lives there
# and is what makes two builds of the same commit produce the same image:
#
#   docker build -f engine/Dockerfile .
#
# The other workspace members' manifests are copied so uv can read the
# workspace; none of them is installed.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# `copy`, because the cache below is a mount that is gone at run time.
ENV UV_LINK_MODE=copy
WORKDIR /app

# Dependencies first, so a source edit does not invalidate the install layer.
COPY pyproject.toml uv.lock ./
COPY engine/pyproject.toml engine/README.md engine/LICENSE engine/NOTICE engine/
COPY examples/backend/pyproject.toml examples/backend/README.md examples/backend/
COPY examples/langgraph-agent/pyproject.toml examples/langgraph-agent/
# `eval` (RAGAS) so the admin dashboard can run the RAG evaluation live;
# `redis`, `postgres` and `s3` so the shared-service settings work without a
# rebuild. All are imported lazily, so normal serving loads none of them.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace \
    --package chatbot-engine --extra eval --extra redis --extra postgres --extra s3 --extra tracing

COPY engine/src engine/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package chatbot-engine --extra eval --extra redis --extra postgres --extra s3 --extra tracing

# Drop root, and own the data directory: the engine writes vectors, the document
# registry, and uploaded blobs, so its volume has to belong to the same user.
# /app stays root-owned and read-only to the process: a chown of the install
# would copy every file into a new layer, doubling the image.
RUN useradd --system --uid 10001 --create-home engine \
    && mkdir -p /var/lib/chatbot-engine \
    && chown engine:engine /var/lib/chatbot-engine
USER engine

ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
EXPOSE 8100

CMD ["uvicorn", "chatbot_engine.app:app", \
     "--host", "0.0.0.0", "--port", "8100", \
     "--proxy-headers"]
