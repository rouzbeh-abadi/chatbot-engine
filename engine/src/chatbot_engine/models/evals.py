"""The evaluation contract.

The engine does not own the dataset, the questions or the rubric - all three
belong to whoever is being evaluated, and arrive with the request like every
other piece of configuration. What the engine contributes is the model.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from chatbot_engine.models.chat import AssistantConfig


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


class Verdict(BaseModel):
    """The judge's opinion of one answer."""

    model_config = ConfigDict(extra="forbid")

    #: The case id, so the caller can pair verdicts with their questions.
    id: str
    #: 0 = unsafe, 10 = does everything `expected` asks.
    score: int = Field(ge=0, le=10)
    reason: str
    #: What the assistant actually said. Filled in after judging, so a low score
    #: can be read without running the case again.
    answer: str = ""


class JudgeVerdicts(BaseModel):
    """What the judging model returns.

    The structured-output root has to be a model, not a bare list. `answer` on
    each verdict is overwritten afterwards from our own record, so whatever the
    model puts there is discarded.
    """

    model_config = ConfigDict(extra="forbid")

    verdicts: list[Verdict] = Field(default_factory=list)


class JudgeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[Verdict] = Field(default_factory=list)
    #: Which model judged, so two runs can be compared meaningfully.
    model: str | None = None
