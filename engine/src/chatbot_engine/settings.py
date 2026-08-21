"""Every engine setting, read from `ENGINE_*` environment variables.

One place. Anything an operator might want to change lives here, with its default
written inline -- no separate constants module to keep in step.

Never raises at import time: a missing credential is checked where the client is
built, so an engine that was never asked to call a model still starts.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from chatbot_engine.errors import NotConfiguredError
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ENGINE_",
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- who may call us ----------------------------------------------------

    #: Optional shared secret. When set, every request except the /health routes
    #: must send a matching `X-API-Key`. Left unset the engine is open, which is
    #: fine on localhost and not fine anywhere else.
    api_key: str | None = None

    # --- the model provider -------------------------------------------------

    #: Checked where the client is built, so an engine that only ingests
    #: documents needs no model key.
    openrouter_api_key: str | None = None

    #: Point this elsewhere for a proxy or a locally hosted model.
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    #: Used when the backend sends no `model` in `AssistantConfig`.
    chat_model: str = "openai/gpt-5-mini"

    #: Changing this invalidates every vector already stored -- distances against
    #: a different model are nonsense, not an error. Treat it as a full re-index.
    embedding_model: str = "openai/text-embedding-3-small"

    # --- the vector store ---------------------------------------------------

    #: Where Chroma keeps its files. Relative paths resolve against the working
    #: directory; `var/` is already gitignored for exactly this.
    chroma_dir: Path = Path("var/chroma")

    #: One collection for every project. Chunks carry `project_id` in their
    #: metadata, so scoping a query is a filter, not a second collection.
    chroma_collection: str = "documents"

    #: Document metadata. As durable as the vectors, or a restart leaves chunks
    #: that nothing lists and nothing can delete.
    registry_db: Path = Path("var/documents.sqlite3")

    #: The uploaded files themselves, kept so a change of chunk size or embedding
    #: model is an internal re-index rather than a re-upload for every caller.
    blob_dir: Path = Path("var/blobs")

    # --- chunking -----------------------------------------------------------

    #: The overlap keeps a sentence that straddles a boundary findable from both
    #: sides. Changing either means re-indexing.
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- everything else ----------------------------------------------------

    #: Seconds to wait on an MCP server before giving up.
    mcp_timeout_s: float = 30.0

    log_level: str = "INFO"

    @field_validator("api_key", "openrouter_api_key", mode="after")
    @classmethod
    def _blank_means_unset(cls, value: str | None) -> str | None:
        """`ENGINE_API_KEY=` in a .env file arrives as "", not as None.

        Without this the engine would demand `X-API-Key: ""` and reject every
        request -- a confusing failure for anyone who copied .env.example.
        """
        return value or None

    def require_openrouter_key(self) -> str:
        """The provider credential, or a 501 naming the variable to set.

        A method rather than a check at startup: an engine that only ingests
        documents needs no model key, and must still start without one.
        """
        if self.openrouter_api_key is None:
            raise NotConfiguredError(
                "no ENGINE_OPENROUTER_API_KEY is set -- put your OpenRouter key "
                "in .env, or set it in the engine's environment"
            )

        return self.openrouter_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
