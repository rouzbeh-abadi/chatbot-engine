"""Tracing: every model call of a turn recorded where an operator can read it.

Off by default. `ENGINE_TRACING=langsmith` uses LangChain's own tracer, which
needs nothing from this module beyond the environment variables it reads;
`ENGINE_TRACING=langfuse` attaches Langfuse's callback handler, self-hostable
and the choice when traces must stay on a server the operator controls.

Either way every run carries the request id, the project id, the session id
and a pseudonym of the user id (`traced_user`) as metadata, so a trace links
back to the engine's own log lines and to whatever conversation the caller
keeps, without holding a phone number or an email address.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import TYPE_CHECKING, Any

from chatbot_engine.errors import EngineError
from chatbot_engine.observability import request_id

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from chatbot_engine.models.chat import ChatRequest, TracingConfig
    from chatbot_engine.settings import Settings

logger = logging.getLogger(__name__)

_handler: Any | None = None

#: Handlers for assistants that bring their own Langfuse, keyed by public key.
#: A Langfuse client is registered once per key and reused across turns.
_per_project: dict[str, Any] = {}
_PER_PROJECT_MAX = 256


def configure(settings: Settings) -> None:
    """Validate the tracing settings at startup and prepare the destination.

    Raises `EngineError` when a destination is selected but cannot work: a
    misconfigured tracer should stop the process, not silently record nothing.
    """
    global _handler
    _handler = None
    _per_project.clear()

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


def _handler_for(config: TracingConfig) -> Any:
    """The Langfuse handler for one assistant's own destination, cached by key."""
    handler = _per_project.get(config.public_key)
    if handler is not None:
        return handler
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise EngineError(
            "per-assistant tracing needs the `tracing` extra: "
            "pip install 'chatbot-engine[tracing]'"
        ) from exc
    if len(_per_project) >= _PER_PROJECT_MAX:
        _per_project.pop(next(iter(_per_project)))
    # Registering a client under its public key is how the handler finds it.
    Langfuse(
        public_key=config.public_key, secret_key=config.secret_key, host=config.host
    )
    handler = CallbackHandler(public_key=config.public_key)
    _per_project[config.public_key] = handler
    return handler


def traced_user(user_id: str | None) -> str | None:
    """The person as a trace names them: a pseudonym, never the id itself.

    A caller's user id can be a phone number or an email address (a chat
    app's person), and a trace store is no place for one. The pseudonym is
    the same for the same id, so traces still group by person, and it is
    keyed with the engine's API keys, so it cannot be undone by trying every
    phone number; it changes when the keys do. The id itself still reaches
    the tool servers, which need it.
    """
    if user_id is None:
        return None
    from chatbot_engine.settings import get_settings

    keys = get_settings().credentials()
    key = "\n".join(sorted(keys.values())).encode() or b"chatbot-engine"
    digest = hmac.new(key, user_id.encode(), hashlib.sha256).hexdigest()
    return f"user-{digest[:20]}"


def run_config(request: ChatRequest, *, name: str) -> RunnableConfig:
    """The config every model call is made with: a name, the ids, the tracer.

    Cheap when tracing is off: metadata only, no callbacks. LangChain and
    LangGraph pass the config down to nested runs, so one call at the top of
    a turn covers the retrieval, the answer and each tool round. An assistant
    with its own `tracing` block goes to its destination instead of the
    engine's.
    """
    project = request.project
    handler = _handler_for(project.tracing) if project.tracing else _handler
    user = traced_user(request.user_id)
    metadata: dict[str, Any] = {
        "request_id": request_id(),
        "project_id": project.project_id,
        "session_id": request.session_id,
        "user_id": user,
        "agent": project.agent or "loop",
        "model": project.model,
    }
    if handler is not None:
        # Langfuse reads these to group traces by session and user.
        metadata["langfuse_session_id"] = request.session_id
        metadata["langfuse_user_id"] = user
        metadata["langfuse_tags"] = [project.project_id]

    config: RunnableConfig = {
        "run_name": name,
        "metadata": {k: v for k, v in metadata.items() if v is not None},
        "tags": [f"project:{project.project_id}"],
    }
    if handler is not None:
        config["callbacks"] = [handler]
    return config


def flush() -> None:
    """Send what is buffered. Called at shutdown so the last traces are not lost."""
    if _handler is None and not _per_project:
        return
    try:
        from langfuse import get_client

        if _handler is not None:
            get_client().flush()
        for public_key in list(_per_project):
            get_client(public_key=public_key).flush()
    except Exception as exc:  # pragma: no cover - best effort at shutdown
        logger.warning("tracing: flush failed: %s", exc)
