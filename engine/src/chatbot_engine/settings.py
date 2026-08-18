"""Engine settings, read from `ENGINE_*` environment variables.

Never raises at import time. When you add provider credentials here, resolve
them where the client is constructed rather than on startup -- an engine that
dies because a provider it was not asked to use lacks a key is hard to deploy.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ENGINE_",
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: Optional shared secret. When set, every request except the /health routes
    #: must send a matching `X-API-Key`. Left unset the engine is open, which is
    #: fine on localhost and not fine anywhere else.
    api_key: str | None = None

    #: Seconds to wait on an MCP server before giving up.
    mcp_timeout_s: float = 30.0

    log_level: str = "INFO"

    @field_validator("api_key", mode="after")
    @classmethod
    def _blank_means_unset(cls, value: str | None) -> str | None:
        """`ENGINE_API_KEY=` in a .env file arrives as "", not as None.

        Without this the engine would demand `X-API-Key: ""` and reject every
        request -- a confusing failure for anyone who copied .env.example.
        """
        return value or None

    # Storage and provider settings belong here too, once you write those
    # adapters: a vector store location, an embedding model name, an API key.


@lru_cache
def get_settings() -> Settings:
    return Settings()
