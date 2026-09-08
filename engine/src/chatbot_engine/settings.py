"""Every engine setting, read from `ENGINE_*` environment variables.

One place. Anything an operator might want to change lives here, with its default
written inline - no separate constants module to keep in step.

Never raises at import time: a missing credential is checked where the client is
built, so an engine that was never asked to call a model still starts.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from chatbot_engine.errors import NotConfiguredError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ENGINE_",
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- who may call us ----------------------------------------------------

    #: Which set of defaults to trust. `local` is permissive so the engine runs
    #: with no configuration; `production` refuses to start without a key, since
    #: an open engine is an open provider account.
    env: Literal["local", "production"] = "local"

    #: Optional shared secret. When set, every request except the /health routes
    #: must send a matching `X-API-Key`. Left unset the engine is open, which is
    #: fine on localhost and not fine anywhere else.
    api_key: str | None = None

    #: Named shared secrets, as `name:secret` separated by commas:
    #:
    #:     ENGINE_API_KEYS=web:s3cret,batch:0ther,web-next:r0tated
    #:
    #: Two things one key cannot do. **Rotation**: issue the new key, let both
    #: work while callers move over, then withdraw the old one -- with a single
    #: key every rotation is a synchronised restart of everything that calls the
    #: engine. **Attribution**: the name is logged and is what rate limits are
    #: counted against, so one misbehaving caller can be found and cut off
    #: without disturbing the others.
    #:
    #: Combines with `api_key`, which is the same thing named `default`.
    api_keys: str | None = None

    # --- rate limits --------------------------------------------------------
    #: Per caller, per process. The engine is where provider credits are
    #: actually spent, so the limit belongs here as well as in whatever calls
    #: it: a caller's own limiter is not a control, it is a courtesy. Generous
    #: by default -- these stop runaway loops, not normal use. Zero disables.
    chat_rate_limit_per_minute: int = 60
    eval_rate_limit_per_hour: int = 20
    ingest_rate_limit_per_minute: int = 20

    #: Where the rate-limit buckets live. Unset, they are in this process's
    #: memory, and every replica counts on its own. Set to a Redis URL
    #: (`redis://host:6379/0`) and all replicas share one exact bucket per
    #: caller. Needs the `redis` extra.
    redis_url: str | None = None

    # --- the model provider -------------------------------------------------

    #: Checked where the client is built, so an engine that only ingests
    #: documents needs no model key.
    openrouter_api_key: str | None = None

    #: Point this elsewhere for a proxy or a locally hosted model.
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    #: How many times a provider call is retried on a transient failure (a
    #: 429, a 5xx, a dropped connection), with exponential backoff between
    #: attempts. Applies to the request; a stream that fails after its first
    #: token is not retried, because the tokens already sent cannot be taken
    #: back.
    provider_max_retries: int = 3

    #: Seconds to wait for a provider before giving up on one attempt.
    provider_timeout_s: float = 60.0

    #: Used when the backend sends no `model` in `AssistantConfig`.
    chat_model: str = "openai/gpt-5-mini"

    #: The model for the small calls a turn makes besides the answer: the
    #: query rewrite and the rerank. Unset, the assistant's own model is used.
    #: A cheap, fast model here cuts latency and cost without touching the
    #: answer, which those calls never write.
    utility_model: str | None = None

    #: The model that scores answers in `/eval/rag`. Keep it in a different
    #: family than `chat_model`, so the metrics are not self-graded, and keep
    #: it cheap: scoring one case is twenty-odd judge calls, most of them
    #: carrying the retrieved contexts as input.
    rag_judge_model: str = "google/gemini-2.5-flash-lite"

    #: Prices per million tokens, `{model: [input, output]}` in USD, used to
    #: put a cost on the `usage` event. The engine ships none: prices belong
    #: to the provider and change without notice, so they are configuration.
    #: A model not listed reports no cost rather than a wrong one.
    #:
    #:     ENGINE_PRICING='{"openai/gpt-5-mini": [0.25, 2.00]}'
    #:
    #: Decoded by the validator below rather than by pydantic-settings, so that
    #: a blank value, which is what `ENGINE_PRICING=` in a .env file or a
    #: compose default becomes, means "no prices" instead of a startup crash.
    pricing: Annotated[dict[str, tuple[float, float]], NoDecode] = {}

    #: Changing this invalidates every vector already stored -- distances against
    #: a different model are nonsense, not an error. Treat it as a full re-index.
    embedding_model: str = "openai/text-embedding-3-small"

    # --- the vector store ---------------------------------------------------

    #: Where Chroma keeps its files, when it runs embedded in this process.
    #: Relative paths resolve against the working directory; `var/` is already
    #: gitignored for exactly this.
    chroma_dir: Path = Path("var/chroma")

    #: A Chroma server to use instead of the embedded one, as
    #: `http://host:8000`. Embedded Chroma is a set of files owned by one
    #: process, so two engine replicas cannot share it; a server can be shared
    #: by any number. Unset, the engine runs embedded.
    chroma_url: str | None = None

    #: The credential for a Chroma server that requires one. Sent on every
    #: request in `chroma_token_header`: as `Authorization: Bearer <token>` by
    #: default, or as `X-Chroma-Token: <token>` when the server is configured
    #: for that header. Unset, requests carry no credential.
    chroma_token: str | None = None
    chroma_token_header: str = "Authorization"

    #: One collection for every project. Chunks carry `project_id` in their
    #: metadata, so scoping a query is a filter, not a second collection.
    chroma_collection: str = "documents"

    #: Document metadata. As durable as the vectors, or a restart leaves chunks
    #: that nothing lists and nothing can delete.
    registry_db: Path = Path("var/documents.sqlite3")

    #: A Postgres URL (`postgresql://user:pass@host/db`) to keep the registry
    #: in instead of the SQLite file, so several replicas share one. Needs
    #: the `postgres` extra. Unset, the file is used.
    registry_url: str | None = None

    #: The uploaded files themselves, kept so a change of chunk size or embedding
    #: model is an internal re-index rather than a re-upload for every caller.
    blob_dir: Path = Path("var/blobs")

    #: An S3-compatible bucket to keep the uploads in instead of `blob_dir`,
    #: so several replicas share them. Credentials come from the standard AWS
    #: environment; `endpoint_url` points at MinIO or another compatible
    #: service. Needs the `s3` extra. Unset, the directory is used.
    blob_s3_bucket: str | None = None
    blob_s3_prefix: str = ""
    blob_s3_endpoint_url: str | None = None
    blob_s3_region: str | None = None

    # --- retrieval ----------------------------------------------------------

    #: How chunks are found when the assistant config does not say. `hybrid`
    #: fuses vector similarity with a BM25 keyword search; `vector` is
    #: similarity alone.
    retrieval: Literal["vector", "hybrid"] = "hybrid"

    #: Whether the assistant's model re-orders the candidates before the top
    #: `top_k` are kept. Off by default: it is one more model call per turn.
    rerank: bool = False

    #: Candidates per search before fusion and reranking.
    retrieval_candidates: int = 20

    # --- chunking -----------------------------------------------------------

    #: The overlap keeps a sentence that straddles a boundary findable from both
    #: sides. Changing either means re-indexing.
    chunk_size: int = 1000
    chunk_overlap: int = 200

    #: Default chunking strategy when the assistant config names none:
    #: `size`, `headings`, or `page`. See rag/splitter.py.
    chunk_strategy: Literal["size", "headings", "page"] = "size"

    # --- agents -------------------------------------------------------------

    #: Which agent runs a turn when the assistant config names none. `loop` is
    #: the one the engine ships; any other name must be an installed plugin.
    #: See agent/registry.py.
    agent: str = "loop"

    # --- everything else ----------------------------------------------------

    #: Seconds to wait on an MCP server before giving up.
    mcp_timeout_s: float = 30.0

    #: How long a server's tool list is reused before it is asked again. Every
    #: turn needs the list, and a graph agent asks for it at every step, so
    #: without a cache each answer opens a connection per server just to learn
    #: what has not changed. Zero disables the cache. A new tool on a server
    #: takes this long to appear; the allowlist still decides whether it may.
    mcp_tools_ttl_s: float = 60.0

    log_level: str = "INFO"

    #: `text` for a person reading a terminal; `json` for a collector that
    #: indexes fields. Either way every line carries the request id.
    log_format: Literal["text", "json"] = "text"

    #: Whether `GET /metrics` is served. On by default; it is unauthenticated
    #: and carries counts, not content.
    metrics_enabled: bool = True

    @field_validator("pricing", mode="before")
    @classmethod
    def _blank_pricing_means_none(cls, value: object) -> object:
        """`ENGINE_PRICING=` arrives as "", and "" is not JSON."""
        if isinstance(value, str):
            return json.loads(value) if value.strip() else {}
        return value

    @field_validator("pricing", mode="after")
    @classmethod
    def _prices_are_not_negative(
        cls, value: dict[str, tuple[float, float]]
    ) -> dict[str, tuple[float, float]]:
        for model, (input_price, output_price) in value.items():
            if input_price < 0 or output_price < 0:
                raise ValueError(f"ENGINE_PRICING for {model!r} is negative")
        return value

    @field_validator(
        "api_key",
        "api_keys",
        "openrouter_api_key",
        "redis_url",
        "chroma_url",
        "chroma_token",
        "utility_model",
        "registry_url",
        "blob_s3_bucket",
        "blob_s3_endpoint_url",
        "blob_s3_region",
        mode="after",
    )
    @classmethod
    def _blank_means_unset(cls, value: str | None) -> str | None:
        """`ENGINE_API_KEY=` in a .env file arrives as "", not as None.

        Without this the engine would demand `X-API-Key: ""` and reject every
        request -- a confusing failure for anyone who copied .env.example.
        """
        return value or None

    def credentials(self) -> dict[str, str]:
        """Every accepted key, by caller name. Empty means the engine is open.

        Malformed entries raise here rather than being skipped: a typo that
        silently dropped a key would leave a caller mysteriously unable to
        authenticate, or -- far worse -- drop the only key and open the engine.
        """
        keys: dict[str, str] = {}

        if self.api_key is not None:
            keys["default"] = self.api_key

        for entry in (self.api_keys or "").split(","):
            entry = entry.strip()
            if not entry:
                continue

            name, separator, secret = entry.partition(":")
            name, secret = name.strip(), secret.strip()
            if not separator or not name or not secret:
                raise ValueError(
                    f"ENGINE_API_KEYS entry {entry!r} is not `name:secret`"
                )
            if name in keys:
                raise ValueError(f"ENGINE_API_KEYS names {name!r} twice")

            keys[name] = secret

        return keys

    def unsafe_for_production(self) -> list[str]:
        """Defaults that are convenient locally and dangerous on the internet.

        Returned rather than raised so the caller decides: `app.py` refuses to
        start under `env=production`, and logs them as warnings otherwise.
        """
        problems: list[str] = []

        if not self.credentials():
            problems.append(
                "No API key is set (ENGINE_API_KEY or ENGINE_API_KEYS), so "
                "anyone who can reach this port can spend your provider "
                "credits and read anything indexed. The engine has no notion "
                "of end users -- it must only ever be reachable by your "
                "application backend."
            )

        return problems

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
