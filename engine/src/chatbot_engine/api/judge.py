"""Evaluation: score a finished run of questions and answers.

A development endpoint, not part of a chat turn. It exists because the caller
cannot run a judge itself -- the model credentials live here.

No service layer: unlike chat and documents there is nothing to coordinate.
"""

from __future__ import annotations

from fastapi import APIRouter

from chatbot_engine.api.dependencies import JudgeDep, SettingsDep
from chatbot_engine.models.evals import JudgeReport, JudgeRequest

router = APIRouter(prefix="/judge", tags=["evaluation"])


@router.post("", responses={501: {"description": "No model provider key is configured."}})
async def score_run(
    request: JudgeRequest, judge: JudgeDep, settings: SettingsDep
) -> JudgeReport:
    """Score every case in the transcript."""
    settings.require_openrouter_key()
    return await judge(request)
