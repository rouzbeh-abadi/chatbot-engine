"""Application settings, read from `BACKEND_*` environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application backend configuration."""

    model_config = SettingsConfigDict(
        env_prefix="BACKEND_",
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Trust the caller's `X-User-Id`. Only ever true behind a proxy that
    # authenticates the user and overwrites the header itself; a browser can
    # send any value it likes, so trusting it on the open internet means every
    # caller can claim to be every user.
    trust_user_header: bool = False

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

    # --- rate limits --------------------------------------------------------
    # Per caller, in this process. Generous for a person, tight for a script:
    # every chat turn and every eval case is a model call on your provider key.
    # Zero disables a limit.
    chat_rate_limit_per_minute: int = 30
    eval_rate_limit_per_hour: int = 20

    # Shared secret guarding the /admin routes. When set, every admin request
    # must send a matching `X-Admin-Key`. Left unset the dashboard is open,
    # which is fine on localhost and not fine anywhere else.
    admin_key: str | None = None

    @field_validator("engine_api_key", "admin_key", mode="after")
    @classmethod
    def _blank_means_unset(cls, value: str | None) -> str | None:
        """`BACKEND_ADMIN_KEY=` in a .env file arrives as "", not as None.

        Without this the backend would demand `X-Admin-Key: ""` and reject every
        admin request -- a confusing failure for anyone who copied .env.example.
        """
        return value or None


@lru_cache
def get_settings() -> Settings:
    """Return the cached backend configuration."""
    return Settings()