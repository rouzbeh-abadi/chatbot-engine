"""Backend settings, read from `BACKEND_*` environment variables.

The engine is a remote service now, so its address is configuration rather than
an import.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BACKEND_",
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: Where the AI engine service lives.
    engine_url: str = "http://localhost:8100"
    #: Shared secret, if the engine requires one (`ENGINE_API_KEY` on its side).
    engine_api_key: str | None = None
    engine_timeout_s: float = 120.0



@lru_cache
def get_settings() -> Settings:
    return Settings()
