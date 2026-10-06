"""GET /metrics, in Prometheus exposition format.

The numbers here are counts and durations, not content, but they name the
callers and the models. Locally the route is open, like /health; under
`ENGINE_ENV=production` it needs the API key like any other caller, unless
`ENGINE_METRICS_PUBLIC` opens it for a scraper that cannot send a header.
`app.py` decides which, where the router is mounted.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["health"])


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
