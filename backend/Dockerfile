# Self-contained: the backend no longer needs the engine's source to build.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache .

# Migrations and the demo seed have to be runnable in the deployed container, not
# just on a laptop.
COPY alembic.ini ./
COPY alembic ./alembic
COPY scripts ./scripts

# Drop root. A web process that never needs to write to its own image should not
# be able to: it turns a file-write bug into a failed syscall instead of a
# modified application. Added after the installs, which do need root.
RUN useradd --system --uid 10001 --create-home app \
    && chown -R app:app /app
USER app

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# `--proxy-headers` makes uvicorn read the client address from X-Forwarded-For
# rather than seeing every request as coming from the proxy -- which is what the
# rate limiter buckets on. It is only honoured from addresses named in
# FORWARDED_ALLOW_IPS, which must be set to your proxy: the header is trivially
# forged, so trusting it from anywhere would let one client spread its usage
# across as many buckets as it likes.
CMD ["uvicorn", "support_agent.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers"]
