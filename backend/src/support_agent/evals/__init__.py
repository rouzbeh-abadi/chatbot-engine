"""Judging the assistant's own prompt.

The dataset and the rubric are the backend's: they describe how *this* assistant
should behave. Answering the questions and grading them both run in the engine,
which is the only side allowed to hold model credentials.
"""

from support_agent.evals.dataset import load_dataset, load_judge_prompt
from support_agent.evals.models import (
    EvalCase,
    EvalDataset,
    JudgeReport,
    Verdict,
)

__all__ = [
    "EvalCase",
    "EvalDataset",
    "JudgeReport",
    "Verdict",
    "load_dataset",
    "load_judge_prompt",
]
