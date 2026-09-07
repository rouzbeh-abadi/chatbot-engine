"""RAG evaluation: score a retrieval dataset with RAGAS.

A development endpoint, like `/judge`. It answers every case through the same
path a chat turn uses, then grades the retrieval. The model credentials and the
RAGAS dependency both live here, which is why the caller cannot run it itself.
"""

from __future__ import annotations

from fastapi import APIRouter

from chatbot_engine.api.dependencies import RagEvaluatorDep, SettingsDep
from chatbot_engine.models.evals import RagEvalRequest, RagReport

router = APIRouter(prefix="/eval/rag", tags=["evaluation"])


@router.post(
    "", responses={501: {"description": "No model provider key is configured."}}
)
async def score_retrieval(
    request: RagEvalRequest, evaluator: RagEvaluatorDep, settings: SettingsDep
) -> RagReport:
    """Answer and score every retrieval case."""
    settings.require_openrouter_key()
    return await evaluator(request)
