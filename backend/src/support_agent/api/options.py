"""Options endpoint: the model choices the frontend may offer the user."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["options"])

# Allowed model ids, from https://openrouter.ai/models. The first is the default
# the picker selects before the user chooses, so keep it in sync with the
# `model:` in projects/support.yaml.
CHAT_MODELS = [
    "openai/gpt-5-mini",
    "openai/gpt-5",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-flash",
    "deepseek/deepseek-chat-v3.1",
]


@router.get("/models")
async def list_models() -> list[str]:
    """Return the allowed model ids, the default first."""
    return CHAT_MODELS
