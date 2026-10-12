"""Turn chunks into vectors.

The embedding model is not fixed by the engine: it arrives as part of the
assistant's config, the same way the chat model does, so one engine can serve
projects that embed with different models. `resolve_embedding_model` supplies the
engine's configured default when a caller passes none.

"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from chatbot_engine.settings import Settings, get_settings


def resolve_embedding_model(model: str | None = None) -> str:
    """The caller's model, or the engine default when they gave none.

    Mirrors how the chat model falls back to `settings.chat_model`: the backend
    owns the choice, the engine only fills in a default so it still runs with no
    per-request config at all.
    """
    return model or get_settings().embedding_model


@lru_cache
def get_embeddings(model: str | None = None) -> OpenAIEmbeddings:
    """An embedder for `model`, one per distinct model in the process.

    Cached by model, so a project embedding with a different model gets its own
    client rather than sharing one bound to the wrong model. A missing provider
    key is a 501, not a crash.
    """
    return build_embeddings(get_settings(), model)


def build_embeddings(settings: Settings, model: str | None = None) -> OpenAIEmbeddings:
    """The embedder for `model` under `settings`. Uncached; see `get_embeddings`.

    Kept with the model's maker, as every chat call is (TURN-10).
    """
    from chatbot_engine.agent.routing import routing_for

    resolved = resolve_embedding_model(model)
    routing = routing_for(resolved)
    return OpenAIEmbeddings(
        model=resolved,
        api_key=settings.require_openrouter_key(),
        base_url=settings.openrouter_base_url,
        max_retries=settings.provider_max_retries,
        timeout=settings.provider_timeout_s,
        model_kwargs={"extra_body": {"provider": routing}} if routing else {},
    )
