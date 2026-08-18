# Self-contained: the backend no longer needs the engine's source to build.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "support_agent.app:app", "--host", "0.0.0.0", "--port", "8000"]
