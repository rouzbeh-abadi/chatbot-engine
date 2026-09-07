"""GET /agents - which agents this engine can run.

The set is not fixed: it is the built-ins plus whatever an adopter installed
under the entry-point group. Only the engine knows what that is, so anything
offering the choice to a user has to ask rather than assume.
"""

from __future__ import annotations

from fastapi import APIRouter

from chatbot_engine.agent.registry import available_agents

router = APIRouter(tags=["agents"])


@router.get("/agents")
async def list_agents() -> list[str]:
    """The names accepted in an assistant config's `agent` field.

    Sorted, so a caller rendering a list gets a stable order. The engine's own
    default is not marked here: a config that names nothing gets it anyway.
    """
    return sorted(available_agents())
