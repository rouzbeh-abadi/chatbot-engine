"""The evaluation contract.

The engine does not own the dataset, the questions or the rubric - all three
belong to whoever is being evaluated, and arrive with the request like every
other piece of configuration. What the engine contributes is the model.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from chatbot_engine.models.chat import AssistantConfig, Message


class EvalCase(BaseModel):
    """One question, and what a good answer to it does."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    question: str
    #: Prose, not an exact string: there are many good phrasings of one correct
    #: answer, and the judge compares behaviour rather than wording.
    expected: str


class JudgeRequest(BaseModel):
    """A dataset to answer and then grade."""

    model_config = ConfigDict(extra="forbid")

    #: Supplies the model and its settings, exactly as a chat turn does.
    project: AssistantConfig
    #: The rubric. The caller's, because only the caller knows what "correct"
    #: means for their assistant.
    judge_prompt: str = Field(min_length=1)
    cases: list[EvalCase] = Field(min_length=1)


class GradedVerdict(BaseModel):
    """One score straight from the judge model, before we enrich it.

    Kept minimal so the structured-output schema asks the judge only for what
    it decides: a score and why. The case text and the answer are added later.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    #: 0 = unsafe, 10 = does everything `expected` asks.
    score: int = Field(ge=0, le=10)
    reason: str


class JudgeVerdicts(BaseModel):
    """What the judging model returns: one graded verdict per case.

    The structured-output root has to be a model, not a bare list.
    """

    model_config = ConfigDict(extra="forbid")

    verdicts: list[GradedVerdict] = Field(default_factory=list)


class Verdict(BaseModel):
    """One graded case, self-contained: the question, the answer, and the score.

    Carries the case text so a report can be read without the dataset beside
    it. `score` is null when the judge returned nothing for the case.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    question: str
    score: int | None = Field(default=None, ge=0, le=10)
    reason: str
    #: What the assistant actually said, so a low score can be read without
    #: running the case again.
    answer: str = ""


class JudgeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[Verdict] = Field(default_factory=list)
    #: Mean score across the graded cases, null when none were graded.
    overall: float | None = None
    #: Which model judged, so two runs can be compared meaningfully.
    model: str | None = None


# --- RAG evaluation ---------------------------------------------------------
#
# A separate track from the system-prompt judge above. The judge grades the
# assistant's behaviour; this grades retrieval: did the search find the right
# context, and is the answer grounded in it. Metrics come from RAGAS.


class RagEvalCase(BaseModel):
    """One retrieval question, with the answer a correct search should support."""

    model_config = ConfigDict(extra="forbid")

    id: str
    #: `single_turn` (the question stands alone) or `follow_up` (the subject is
    #: in `history`, so the question alone is ambiguous).
    category: str
    question: str
    #: Earlier turns, for a follow-up whose subject is not in the question.
    history: list[Message] = Field(default_factory=list)
    #: The ground-truth answer, used by context recall to check the retrieved
    #: chunks actually contain what the question needs.
    reference: str = Field(min_length=1)


class RagEvalRequest(BaseModel):
    """A retrieval dataset to answer and then score."""

    model_config = ConfigDict(extra="forbid")

    #: Supplies the model and retrieval settings, exactly as a chat turn does.
    project: AssistantConfig
    cases: list[RagEvalCase] = Field(min_length=1)


class RagCaseResult(BaseModel):
    """The scored outcome for one case.

    Each metric runs 0 to 1, or is null when RAGAS could not compute it (an
    empty retrieval, or a model call that failed). The answer and contexts are
    kept so a low score can be read without running the case again.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    #: Answer supported by the retrieved context (not invented).
    faithfulness: float | None = None
    #: Answer addresses the question that was asked.
    answer_relevancy: float | None = None
    #: Retrieved chunks are relevant, and the relevant ones rank first.
    context_precision: float | None = None
    #: Retrieval found the context the reference answer needs.
    context_recall: float | None = None


class RagMetricAverages(BaseModel):
    """Mean of each metric over a set of cases, ignoring the ones RAGAS could
    not score. Null when nothing in the set had that metric."""

    model_config = ConfigDict(extra="forbid")

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


class RagCategorySummary(BaseModel):
    """Averages for one category of cases.

    Reported because a single overall mean hides the follow-up cases, which is
    exactly where retrieval on the question alone tends to fall down.
    """

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
