"""The service layer: a thin, stable boundary between HTTP and AI logic.

Each service checks that an implementation has been registered and delegates to
it. They exist so `api/` never has to know how the AI works, and so the AI logic
never has to know it is behind HTTP.
"""

from chatbot_engine.services.chat import ChatService
from chatbot_engine.services.documents import DocumentService

__all__ = ["ChatService", "DocumentService"]
