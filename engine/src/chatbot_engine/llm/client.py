from __future__ import annotations

from chatbot_engine.errors import NotConfiguredError
from chatbot_engine.settings import Settings
from openai import AsyncOpenAI

_MISSING_KEY = (
    "no ENGINE_OPENROUTER_API_KEY is set - put your OpenRouter key in .env, or "
    "set it in the engine's environment"
)


def require_provider_key(settings: Settings) -> str:
    """The provider credential, or a 501 naming the variable to set.

    Not to be confused with `deps.require_api_key`, which checks the secret our
    own callers send us. This one is the key we send outward.
    """
    if not settings.openrouter_api_key:
        raise NotConfiguredError(_MISSING_KEY)

    return settings.openrouter_api_key


def build_model_client(settings: Settings) -> AsyncOpenAI:
    """Build the chat client, or raise a 501 naming the variable to set."""
    require_provider_key(settings)

    return AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
