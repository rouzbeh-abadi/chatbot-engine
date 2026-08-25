"""RAG evaluation: score a retrieval dataset with RAGAS.

A development endpoint, like `/judge`. It answers every case through the same
path a chat turn uses, then grades the retrieval. The model credentials and the
RAGAS dependency both live here, which is why the caller cannot run it itself.
"""

from __future__ import annotations

from fastapi import APIRouter

from chatbot_engine.api.dependencies import RagEvaluatorDep
from chatbot_engine.errors import NotConfiguredError
from chatbot_engine.models.evals import RagEvalRequest, RagReport

router = APIRouter(prefix="/eval/rag", tags=["evaluation"])


@router.post(
    "", responses={501: {"description": "No RagEvaluator is registered yet."}}
)
async def score_retrieval(
    request: RagEvalRequest, evaluator: RagEvaluatorDep
) -> RagReport:
    """Answer and score every retrieval case."""
    if evaluator is None:
        raise NotConfiguredError(
            "no RagEvaluator is registered -- return one from "
            "chatbot_engine.api.dependencies.get_rag_evaluator()"
        )

    return await evaluator(request)
