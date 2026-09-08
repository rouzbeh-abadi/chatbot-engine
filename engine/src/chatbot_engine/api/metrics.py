"""GET /metrics, in Prometheus exposition format.

Unauthenticated, like /health: a scraper is not a caller, and the numbers here
are counts and durations, not content. Reachable only from wherever the engine
is reachable, which in a deployment is the private network.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["health"])


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
