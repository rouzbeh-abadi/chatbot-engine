from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["options"])

#: Models the UI may pick, verified against OpenRouter's catalogue. The first is
#: the default the picker shows before anyone chooses, so keep it in step with
#: the `model:` in `projects/support.yaml`. Add or remove freely -- ids come from
#: https://openrouter.ai/models.
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
    """Every model the UI may pick, the default first."""
    return CHAT_MODELS
