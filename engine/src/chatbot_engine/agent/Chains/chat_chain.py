from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable


def build_prompt(system_prompt: str) -> ChatPromptTemplate:
    """Build the prompt template: the system prompt"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("messages"),
        ]
    )


def build_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copy the tool schemas into plain dicts"""
    return [dict(tool) for tool in tools]


def create_chain(
    model: BaseChatModel,
    system_prompt: str,
    tools: Sequence[Mapping[str, Any]],
) -> Runnable:

    prompt = build_prompt(system_prompt)

    if not tools:
        return prompt | model

    return prompt | model.bind_tools(build_tools(tools))
