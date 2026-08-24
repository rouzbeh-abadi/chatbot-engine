"""Evaluation: score a finished run of questions and answers.

A development endpoint, not part of a chat turn. It exists because the caller
cannot run a judge itself -- the model credentials live here.

No service layer: unlike chat and documents there is nothing to coordinate, so
the readiness check is these three lines rather than a class holding them.
"""

from __future__ import annotations

from fastapi import APIRouter

from chatbot_engine.api.dependencies import JudgeDep
from chatbot_engine.errors import NotConfiguredError
from chatbot_engine.models.evals import JudgeReport, JudgeRequest

router = APIRouter(prefix="/judge", tags=["evaluation"])


@router.post("", responses={501: {"description": "No Judge is registered yet."}})
async def score_run(request: JudgeRequest, judge: JudgeDep) -> JudgeReport:
    """Score every case in the transcript."""
    if judge is None:
        raise NotConfiguredError(
            "no Judge is registered -- implement one under chatbot_engine/agent/ "
            "and return it from chatbot_engine.api.dependencies.get_judge()"
        )

    return await judge(request)
