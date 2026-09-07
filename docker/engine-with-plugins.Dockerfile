# The demo's engine image: the engine, plus this repository's agent plugin.
#
# This is the pattern an adopter follows. The published engine image carries no
# agent framework; you add the agents you want by installing your own packages
# into it. Keeping that step visible here is the point -- the demo stack offers
# `agent: graph` for exactly the same reason your deployment would.
#
# Built from the repository root, because it needs both the engine and the
# plugin. `engine/Dockerfile` stays buildable on its own and produces the plain,
# framework-free engine.

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY engine/pyproject.toml engine/README.md engine/LICENSE engine/NOTICE ./
COPY engine/src ./src
# `eval` (RAGAS) so the admin dashboard can run the RAG evaluation live, and
# `redis` so ENGINE_REDIS_URL can be set without rebuilding. Both are imported
# lazily, so normal serving loads neither.
RUN uv pip install --system --no-cache ".[eval,redis]"

# The agent plugin. `--no-deps` because the engine it depends on is already
# installed above and is not published to PyPI; LangGraph comes with it.
COPY examples/langgraph-agent /opt/langgraph-agent
RUN uv pip install --system --no-cache "langgraph>=0.2" \
    && uv pip install --system --no-cache --no-deps /opt/langgraph-agent

# Drop root, and own the data directory: the engine writes vectors, the document
# registry, and uploaded blobs, so its volume has to belong to the same user.
RUN useradd --system --uid 10001 --create-home engine \
    && mkdir -p /var/lib/chatbot-engine \
    && chown -R engine:engine /app /var/lib/chatbot-engine
USER engine

ENV PYTHONUNBUFFERED=1
EXPOSE 8100

CMD ["uvicorn", "chatbot_engine.app:app", \
     "--host", "0.0.0.0", "--port", "8100", \
     "--proxy-headers"]
