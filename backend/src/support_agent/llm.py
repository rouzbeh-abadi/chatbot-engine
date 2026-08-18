from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from support_agent.config import api_key
from support_agent.constants import BASE_URL, DEFAULT_CHAT_MODEL
from support_agent.models.llm import LLMResponse, TokenUsage


class StructuredOutputError(ValueError):
    """The model did not return a response matching the requested schema."""

def create_llm(
    model: str = DEFAULT_CHAT_MODEL,
) -> ChatOpenAI:

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=BASE_URL,
    )

def extract_token_usage(
    raw_response: AIMessage,
) -> TokenUsage | None:

    usage_metadata = raw_response.usage_metadata

    if usage_metadata is None:
        return None

    token_usage = raw_response.response_metadata.get(
        "token_usage",
        {},
    )

    return TokenUsage(
        input_tokens=usage_metadata["input_tokens"],
        output_tokens=usage_metadata["output_tokens"],
        total_tokens=usage_metadata["total_tokens"],
        cost_usd=token_usage.get("cost"),
    )

def get_completion[ResponseModel: BaseModel](
    chain: Runnable[dict[str, Any], dict[str, Any]],
    inputs: dict[str, Any],
) -> LLMResponse[ResponseModel]:

    response = chain.invoke(inputs)

    parsed_response = response["parsed"]
    parsing_error = response["parsing_error"]
    raw_response = response["raw"]

    if parsed_response is None:
        raise StructuredOutputError(
            "The model returned no structured response."
        ) from parsing_error

    return LLMResponse(
        data=parsed_response,
        usage=extract_token_usage(raw_response),
    )
