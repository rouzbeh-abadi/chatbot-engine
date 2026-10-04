"""Request and response contracts for the engine's HTTP API.

Pure pydantic: no framework, no provider, no storage client. The application
backend mirrors these shapes in its own `engine_client.models`, because the two
services must not share a Python package.
"""

from chatbot_engine.models.chat import (
    AssistantConfig,
    Attachment,
    ChatRequest,
    McpServerConfig,
    Message,
)
from chatbot_engine.models.common import HealthResponse
from chatbot_engine.models.documents import (
    DeleteResult,
    DocumentRecord,
    ExtractedText,
    ExtractUsage,
    IngestStatus,
)
from chatbot_engine.models.events import (
    DoneEvent,
    ErrorEvent,
    Event,
    RetrievalEvent,
    SourceRef,
    TokenEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
    UsageEvent,
)

__all__ = [
    "AssistantConfig",
    "Attachment",
    "ChatRequest",
    "DeleteResult",
    "DocumentRecord",
    "DoneEvent",
    "ErrorEvent",
    "Event",
    "ExtractUsage",
    "ExtractedText",
    "HealthResponse",
    "IngestStatus",
    "McpServerConfig",
    "Message",
    "RetrievalEvent",
    "SourceRef",
    "TokenEvent",
    "ToolCallFinishedEvent",
    "ToolCallStartedEvent",
    "UsageEvent",
]
