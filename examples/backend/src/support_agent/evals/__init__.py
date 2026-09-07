"""Judging the assistant's own prompt.

The dataset and the rubric are the backend's: they describe how *this* assistant
should behave. Answering the questions and grading them both run in the engine,
which is the only side allowed to hold model credentials.
"""

from support_agent.evals.dataset import (
    load_judge_cases,
    load_judge_prompt,
    load_rag_cases,
)
from support_agent.evals.models import (
    JudgeReport,
    RagCaseResult,
    RagReport,
    Verdict,
)

__all__ = [
    "JudgeReport",
    "RagCaseResult",
    "RagReport",
    "Verdict",
    "load_judge_cases",
    "load_judge_prompt",
    "load_rag_cases",
]
