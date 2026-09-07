"""Options endpoints: the choices the frontend may offer the user."""

from __future__ import annotations

from fastapi import APIRouter

from support_agent.engine import EngineDep

router = APIRouter(tags=["options"])

# Allowed model ids, from https://openrouter.ai/models. The first is the default
# the picker selects before the user chooses, so keep it in sync with the
# `model:` in projects/support.yaml. A model shows a cost in the UI only when
# the engine's ENGINE_PRICING lists it; see .env.example.
CHAT_MODELS = [
    "openai/gpt-5-mini",
    "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-flash",
]


@router.get("/models")
async def list_models() -> list[str]:
    """Return the allowed model ids, the default first."""
    return CHAT_MODELS


@router.get("/agents")
async def list_agents(engine: EngineDep) -> list[str]:
    """The agents the engine can run, for the picker.

    Proxied rather than hardcoded: an adopter who installs their own agent in
    the engine should see it offered here without touching this backend.
    """
    return await engine.list_agents()
