"""What the engine's eval endpoints return.

The result side only. The engine owns the input contracts (the eval case shape);
the backend forwards its datasets as raw JSON and never defines their shape.
These mirror what the engine returns, so the backend can parse it. Deliberately
separate from `engine_client.models` (the product's wire contract, guarded by
the parity test): evaluation is a development tool.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Verdict(BaseModel):
    """One graded case: the question, the answer, and the judge's score."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    question: str
    #: 0 = unsafe, 10 = does everything expected. Null when not judged.
    score: int | None = Field(default=None, ge=0, le=10)
    #: One sentence. Long explanations make a failing run unreadable.
    reason: str
    #: What the assistant said, so a low score can be read without re-running it.
    answer: str = ""


class JudgeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[Verdict] = Field(default_factory=list)
    #: Mean score across the graded cases, null when none were graded.
    overall: float | None = None
    #: The judge's own model, so a run can be compared against another.
    model: str | None = None


# --- RAG evaluation (RAGAS) --------------------------------------------------
#
# The result side only. The engine owns the input contract (`RagEvalCase`); the
# backend forwards the dataset as raw JSON and never defines its shape. These
# mirror what the engine returns, so we can parse it. Not in the parity test:
# evaluation is a development tool.


class RagCaseResult(BaseModel):
    """One case's answer, retrieved chunks, and the four RAGAS metrics (0 to 1)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


class RagMetricAverages(BaseModel):
    """Mean of each metric over a set of cases, ignoring the ones scored null."""

    model_config = ConfigDict(extra="forbid")

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


class RagCategorySummary(BaseModel):
    """Averages for one category of cases (a mean hides the follow-up collapse)."""

    model_config = ConfigDict(extra="forbid")

    category: str
    count: int
    averages: RagMetricAverages


class RagReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[RagCaseResult] = Field(default_factory=list)
    overall: RagMetricAverages = Field(default_factory=RagMetricAverages)
    by_category: list[RagCategorySummary] = Field(default_factory=list)
    #: Which model computed the metrics, so two runs can be compared.
    model: str | None = None
