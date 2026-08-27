# The engine is self-contained: it does not need the backend's source at all.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
# The `eval` extra (RAGAS) is included so the admin dashboard can run the RAG
# evaluation live. It is imported lazily, so normal serving never loads it.
RUN uv pip install --system --no-cache ".[eval]"

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
