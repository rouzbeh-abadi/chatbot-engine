# The engine is self-contained: it does not need the backend's source at all.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache .

ENV PYTHONUNBUFFERED=1
EXPOSE 8100

CMD ["uvicorn", "chatbot_engine.app:app", "--host", "0.0.0.0", "--port", "8100"]
