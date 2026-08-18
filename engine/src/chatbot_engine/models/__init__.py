"""Request and response contracts for the engine's HTTP API.

Pure pydantic: no framework, no provider, no storage client. The application
backend mirrors these shapes in its own `engine_client.models`, because the two
services must not share a Python package.
"""

from chatbot_engine.models.chat import (
    AssistantConfig,
    ChatRequest,
    McpServerConfig,
    Message,
)
from chatbot_engine.models.common import HealthResponse
from chatbot_engine.models.documents import (
    DeleteResult,
    DocumentRecord,
    IngestStatus,
)
from chatbot_engine.models.events import (
    DoneEvent,
    ErrorEvent,
    Event,
    RetrievalEvent,
    SourceRef,
    TokenEvent,
    UsageEvent,
)

__all__ = [
    "AssistantConfig",
    "ChatRequest",
    "DeleteResult",
    "DocumentRecord",
    "DoneEvent",
    "ErrorEvent",
    "Event",
    "HealthResponse",
    "IngestStatus",
    "McpServerConfig",
    "Message",
    "RetrievalEvent",
    "SourceRef",
    "TokenEvent",
    "UsageEvent",
]
