from __future__ import annotations

from chatbot_engine.models.evals import JudgeVerdicts
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


def build_judge_prompt(judge_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=judge_prompt),
            ("human", "{transcript}"),
        ]
    )


def create_judge_chain(model: BaseChatModel, judge_prompt: str) -> Runnable:
    """Build the judging chain."""
    
    return build_judge_prompt(judge_prompt) | model.with_structured_output(
        JudgeVerdicts
    )
