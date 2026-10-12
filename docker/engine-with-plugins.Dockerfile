# The demo's engine image: the engine, plus this repository's agent plugin.
#
# This is the pattern an adopter follows. The published engine image carries no
# agent framework; you add the agents you want by installing your own packages
# into it. Keeping that step visible here is the point: the demo stack offers
# `agent: graph` for exactly the same reason your deployment would.
#
# Built from the repository root, from the lockfile, like engine/Dockerfile.
# By digest as well as tag, so a rebuild of one commit starts from the same
# base (docs/review-2026-10.md, INFRA-6); Dependabot proposes the next one.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim@sha256:531f855bda2c73cd6ef67d56b733b357cea384185b3022bd09f05e002cd144ca

# `copy`, because the cache below is a mount that is gone at run time.
ENV UV_LINK_MODE=copy
# A large wheel such as pandas can take minutes to download on a runner building
# two architectures at once; uv's 30 s default failed the 0.1.22 release.
ENV UV_HTTP_TIMEOUT=300
WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY engine/pyproject.toml engine/README.md engine/LICENSE engine/NOTICE engine/
COPY examples/backend/pyproject.toml examples/backend/README.md examples/backend/
COPY examples/langgraph-agent/pyproject.toml examples/langgraph-agent/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace \
    --package chatbot-engine --extra eval --extra redis --extra postgres --extra s3 --extra tracing

COPY engine/src engine/src
COPY examples/langgraph-agent examples/langgraph-agent
# The engine with its extras, then the plugin on top. `--inexact` keeps what
# the first sync installed rather than removing it as extraneous.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package chatbot-engine --extra eval --extra redis --extra postgres --extra s3 --extra tracing \
    && uv sync --frozen --no-dev --inexact --package langgraph-agent

# /app stays root-owned and read-only to the process; see engine/Dockerfile.
RUN useradd --system --uid 10001 --create-home engine \
    && mkdir -p /var/lib/chatbot-engine \
    && chown engine:engine /var/lib/chatbot-engine
USER engine

ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
# The engine's data lives on one path, so a single volume mount keeps it.
ENV ENGINE_CHROMA_DIR=/var/lib/chatbot-engine/chroma \
    ENGINE_REGISTRY_DB=/var/lib/chatbot-engine/documents.sqlite3 \
    ENGINE_BLOB_DIR=/var/lib/chatbot-engine/blobs \
    ENGINE_CHECKPOINT_DB=/var/lib/chatbot-engine/checkpoints.sqlite3
EXPOSE 8100

CMD ["uvicorn", "chatbot_engine.app:app", \
     "--host", "0.0.0.0", "--port", "8100", \
     "--proxy-headers"]
