"""The project a request names, or 404, for every route that takes one."""

from __future__ import annotations

from fastapi import HTTPException, status

from support_agent.assistant import ProjectNotFoundError, load_project
from support_agent.engine_client.models import AssistantConfig


def project_or_404(name: str | None) -> AssistantConfig:
    """The assistant config `name` resolves to, or a 404.

    One helper for every route, so an unknown project answers the same way
    everywhere rather than as a 500 wherever a route forgot to catch it. The
    error's words never carry a server path (see `load_project`).
    `load_project` is cached, so asking twice costs nothing.
    """
    try:
        return load_project(name)
    except ProjectNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
