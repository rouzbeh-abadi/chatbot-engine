"""Application settings, read from `BACKEND_*` environment variables."""

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

    # PostgreSQL connection URL.
    database_url: str = (
        "postgresql+psycopg://support_agent:support_agent@localhost:5432/support_agent"
    )

    # Base URL of the AI engine service.
    engine_url: str = "http://localhost:8100"

    # Shared secret sent to the engine when it requires authentication.
    engine_api_key: str | None = None

    # Request timeout for the engine, in seconds.
    engine_timeout_s: float = 120.0

    # Overrides the MCP server URL from projects/*.yaml. Set to the Compose
    # service name (http://mcp-tools:8200/mcp) when the engine runs in Docker.
    mcp_tools_url: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the cached backend configuration."""
    return Settings()