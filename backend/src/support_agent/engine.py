"""The engine dependency.

Kept as a FastAPI dependency rather than a module-level singleton so tests can
override it with `app.dependency_overrides` and run without a live engine.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from support_agent.engine_client import EngineClient
from support_agent.settings import get_settings


@lru_cache
def get_engine_client() -> EngineClient:
    settings = get_settings()
    return EngineClient(
        base_url=settings.engine_url,
        api_key=settings.engine_api_key,
        timeout_s=settings.engine_timeout_s,
    )


EngineDep = Annotated[EngineClient, Depends(get_engine_client)]
