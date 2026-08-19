"""Backend configuration loaded from `BACKEND_*` environment variables.

The backend owns application infrastructure such as the database and communicates
with the AI engine as a separate service over HTTP.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application backend configuration."""

    model_config = SettingsConfigDict(
        env_prefix="BACKEND_",
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: PostgreSQL connection URL. The `psycopg` (v3) dialect serves both the
    #: async application and sync Alembic migrations from one URL, so there is
    #: no second driver to keep in step.
    database_url: str = (
        "postgresql+psycopg://support_agent:support_agent@localhost:5432/support_agent"
    )

    #: Base URL of the AI engine service.
    engine_url: str = "http://localhost:8100"

    #: Shared secret used when the AI engine requires authentication.
    engine_api_key: str | None = None

    #: Maximum time, in seconds, to wait for an AI engine response.
    engine_timeout_s: float = 120.0

    #: Overrides the MCP server URL in `projects/*.yaml` when set. The engine is
    #: the one dialling it, so under Docker Compose this has to be the service
    #: name (`http://mcp-tools:8200/mcp`) rather than localhost. Which tools are
    #: allowed stays in the YAML -- only the address moves.
    mcp_tools_url: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the cached backend configuration."""
    return Settings()