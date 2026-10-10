"""Forgetting a project, or one of its sessions, everywhere the engine keeps it.

What a caller deletes on its side (a chatbot, a conversation, everything about
a person) the engine keeps in more than one place: documents, their chunks and
stored originals, and the turns of a workflow that paused on a question, with
the answers given so far. Deleting documents one by one reaches only the
first three, and only those already recorded. These routes reach all of it.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Path

from chatbot_engine.api.dependencies import DocumentServiceDep, get_agent
from chatbot_engine.models.documents import ProjectPurged, SessionForgotten
from chatbot_engine.ports.agent import Agent

router = APIRouter(prefix="/projects", tags=["projects"])

AgentDep = Annotated[Agent, Depends(get_agent)]
_Id = Path(min_length=1, max_length=256)


async def _forget(agent: Agent, project_id: str, session_id: str | None = None) -> int:
    forget = getattr(agent, "forget", None)
    return await forget(project_id, session_id) if forget is not None else 0


@router.delete("/{project_id}")
async def purge_project(
    service: DocumentServiceDep,
    agent: AgentDep,
    project_id: Annotated[str, _Id],
) -> ProjectPurged:
    """Remove everything the engine keeps for a project: every document, with
    its chunks and stored original, chunks no record names, uploads under way
    (which keep nothing), and paused workflow turns. Uploads to the project
    are refused for ten minutes after, so a crawl still running cannot put
    pages back. For a caller deleting the chatbot the project is."""
    knowledge = await service.purge(project_id=project_id)
    return ProjectPurged(
        project_id=project_id,
        knowledge=knowledge,
        paused_turns=await _forget(agent, project_id),
    )


@router.delete("/{project_id}/sessions/{session_id}")
async def forget_session(
    agent: AgentDep,
    project_id: Annotated[str, _Id],
    session_id: Annotated[str, _Id],
) -> SessionForgotten:
    """Forget the workflow turns of one session that paused on a question,
    with the answers they held. For a caller deleting a conversation, or
    everything about the person in it."""
    return SessionForgotten(
        project_id=project_id,
        session_id=session_id,
        paused_turns=await _forget(agent, project_id, session_id),
    )
