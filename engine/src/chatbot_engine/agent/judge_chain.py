from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from chatbot_engine.models.evals import JudgeVerdicts


def build_judge_prompt(judge_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=judge_prompt),
            ("human", "{transcript}"),
        ]
    )


#: The verdicts' shape, as the judge is asked for it. A schema rather than
#: the model itself, so the reply comes back as it was written and is checked
#: a verdict at a time (`read_verdicts`): one the judge got wrong costs that
#: case its score, not every answer already paid for (docs/review-2026-10.md,
#: EVALTRACE-4).
VERDICTS_SCHEMA = JudgeVerdicts.model_json_schema()


def create_judge_chain(model: BaseChatModel, judge_prompt: str) -> Runnable:
    """Build the judging chain; it answers the judge's reply as a dict."""

    # Strict, as the model class made it: a provider that enforces the schema
    # still holds the judge to it, so a verdict outside it is a rare event.
    return build_judge_prompt(judge_prompt) | model.with_structured_output(
        VERDICTS_SCHEMA, strict=True
    )
