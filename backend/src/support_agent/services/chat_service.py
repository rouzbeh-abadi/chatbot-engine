from support_agent.chains.support_chain import create_support_chain
from support_agent.llm import get_completion
from support_agent.models.chat import ChatRequest, ChatResponse
from support_agent.models.support import SupportResponse

support_chain = create_support_chain()


async def handle_chat(
    request: ChatRequest,
) -> ChatResponse:
    """Generate a customer support response for the incoming chat request."""

    result = get_completion(
        chain=support_chain,
        inputs={
            "message": request.message,
        },
    )

    support_response: SupportResponse = result.data

    return ChatResponse(
        answer=support_response.answer,
    )
