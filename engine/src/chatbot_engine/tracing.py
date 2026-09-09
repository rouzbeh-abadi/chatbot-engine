"""Tracing: every model call of a turn recorded where an operator can read it.

Off by default. `ENGINE_TRACING=langsmith` uses LangChain's own tracer, which
needs nothing from this module beyond the environment variables it reads;
`ENGINE_TRACING=langfuse` attaches Langfuse's callback handler, self-hostable
and the choice when traces must stay on a server the operator controls.

Either way every run carries the request id, the project id, the session id
and the user id as metadata, so a trace links back to the engine's own log
lines and to whatever conversation the caller keeps.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from chatbot_engine.errors import EngineError
from chatbot_engine.observability import request_id

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from chatbot_engine.models.chat import ChatRequest
    from chatbot_engine.settings import Settings

logger = logging.getLogger(__name__)

_handler: Any | None = None


def configure(settings: Settings) -> None:
    """Validate the tracing settings at startup and prepare the destination.

    Raises `EngineError` when a destination is selected but cannot work: a
    misconfigured tracer should stop the process, not silently record nothing.
    """
    global _handler
    _handler = None

    if settings.tracing == "off":
        return

    if settings.tracing == "langsmith":
        # LangChain's tracer is configured through the environment; the
        # engine's settings are a convenience that map onto it.
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        if settings.langsmith_api_key:
            os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
        if not os.environ.get("LANGCHAIN_API_KEY"):
            raise EngineError("ENGINE_TRACING=langsmith needs ENGINE_LANGSMITH_API_KEY")
        logger.info("tracing to LangSmith project %s", os.environ["LANGCHAIN_PROJECT"])
        return

    if settings.tracing == "langfuse":
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            raise EngineError(
                "ENGINE_TRACING=langfuse needs ENGINE_LANGFUSE_PUBLIC_KEY and "
                "ENGINE_LANGFUSE_SECRET_KEY"
            )
        try:
            from langfuse import Langfuse
            from langfuse.langchain import CallbackHandler
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise EngineError(
                "ENGINE_TRACING=langfuse needs the `tracing` extra: "
                "pip install 'chatbot-engine[tracing]'"
            ) from exc
        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        _handler = CallbackHandler()
        logger.info("tracing to Langfuse at %s", settings.langfuse_host)
        return

    raise EngineError(f"unknown ENGINE_TRACING value {settings.tracing!r}")


def run_config(request: ChatRequest, *, name: str) -> RunnableConfig:
    """The config every model call is made with: a name, the ids, the tracer.

    Cheap when tracing is off: metadata only, no callbacks. LangChain and
    LangGraph pass the config down to nested runs, so one call at the top of
    a turn covers the retrieval, the answer and each tool round.
    """
    project = request.project
    metadata: dict[str, Any] = {
        "request_id": request_id(),
        "project_id": project.project_id,
        "session_id": request.session_id,
        "user_id": request.user_id,
        "agent": project.agent or "loop",
        "model": project.model,
    }
    if _handler is not None:
        # Langfuse reads these to group traces by session and user.
        metadata["langfuse_session_id"] = request.session_id
        metadata["langfuse_user_id"] = request.user_id
        metadata["langfuse_tags"] = [project.project_id]

    config: RunnableConfig = {
        "run_name": name,
        "metadata": {k: v for k, v in metadata.items() if v is not None},
        "tags": [f"project:{project.project_id}"],
    }
    if _handler is not None:
        config["callbacks"] = [_handler]
    return config


def flush() -> None:
    """Send what is buffered. Called at shutdown so the last traces are not lost."""
    if _handler is None:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:  # pragma: no cover - best effort at shutdown
        logger.warning("tracing: flush failed: %s", exc)
