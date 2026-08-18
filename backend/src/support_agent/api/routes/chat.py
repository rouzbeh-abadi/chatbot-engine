from fastapi import APIRouter

from support_agent.models.chat import ChatRequest, ChatResponse
from support_agent.services.chat_service import handle_chat

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.post("")
async def chat(request: ChatRequest) -> ChatResponse:
    """Handle a customer support chat request."""
    return await handle_chat(request)
