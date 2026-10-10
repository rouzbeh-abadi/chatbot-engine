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
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from chatbot_engine.errors import EngineError
from chatbot_engine.observability import request_id

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from chatbot_engine.models.chat import ChatRequest, TracingConfig
    from chatbot_engine.settings import Settings

logger = logging.getLogger(__name__)

_handler: Any | None = None


@dataclass
class _Destination:
    """One assistant's own Langfuse: its handler, and what to flush and stop."""

    handler: Any
    resources: Any
    provider: Any


#: The Langfuse destinations of assistants that bring their own, by public
#: key, host and a hash of the secret together, the one used longest ago
#: first. The same public key with another host or secret is another
#: destination: whoever names a key first does not get the turns of whoever
#: names it next (docs/review-2026-10.md, EVALTRACE-1). Each one holds a few
#: threads, so only so many are kept, and one dropped is shut down
#: (EVALTRACE-2).
_per_project: OrderedDict[tuple[str, str, str], _Destination] = OrderedDict()
_per_project_lock = threading.Lock()
_PER_PROJECT_MAX = 32


def configure(settings: Settings) -> None:
    """Validate the tracing settings at startup and prepare the destination.

    Raises `EngineError` when a destination is selected but cannot work: a
    misconfigured tracer should stop the process, not silently record nothing.
    """
    global _handler
    _handler = None
    with _per_project_lock:
        dropped = list(_per_project.values())
        _per_project.clear()
    for destination in dropped:
        _close(destination)

    if settings.tracing == "off":
        return

    if settings.tracing == "langsmith":
        # LangChain's tracer is configured through the environment; the
        # engine's settings are a convenience that map onto it. Checked
        # first, so a refusal turns nothing on.
        if not (settings.langsmith_api_key or os.environ.get("LANGCHAIN_API_KEY")):
            raise EngineError("ENGINE_TRACING=langsmith needs ENGINE_LANGSMITH_API_KEY")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        if settings.langsmith_api_key:
            os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
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


def _handler_for(config: TracingConfig) -> Any | None:
    """The Langfuse handler for one assistant's own destination: made once
    for its public key, host and secret together, and kept while it is used.
    None when no destination of its own can be made, so the turn goes
    untraced rather than anywhere else."""
    secret = hashlib.sha256(config.secret_key.encode()).hexdigest()
    key = (config.public_key, config.host, secret)
    with _per_project_lock:
        found = _per_project.get(key)
        if found is not None:
            _per_project.move_to_end(key)
            return found.handler
        destination = _open(config)
        if destination is None:
            return None
        _per_project[key] = destination
        dropped = []
        while len(_per_project) > _PER_PROJECT_MAX:
            dropped.append(_per_project.popitem(last=False)[1])
    for old in dropped:
        # Off the request: shutting down sends what is still buffered.
        threading.Thread(target=_close, args=(old,), daemon=True).start()
    return destination.handler


def _open(config: TracingConfig) -> _Destination | None:
    """A Langfuse client for one assistant's destination, apart from every
    other: a tracer provider of its own, so its spans reach no other
    destination's exporter, and kept out of the SDK's registry, which holds
    one client per public key for the whole process."""
    try:
        from langfuse import Langfuse
        from langfuse._client.resource_manager import LangfuseResourceManager
        from langfuse.langchain import CallbackHandler
        from opentelemetry.sdk.trace import TracerProvider
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise EngineError(
            "per-assistant tracing needs the `tracing` extra: "
            "pip install 'chatbot-engine[tracing]'"
        ) from exc
    registry = getattr(LangfuseResourceManager, "_instances", None)
    lock = getattr(LangfuseResourceManager, "_lock", None)
    if not isinstance(registry, dict) or lock is None:
        # A Langfuse whose registry this code does not know: a client made
        # here could share another destination's, so none is made.
        logger.warning(
            "tracing: per-assistant tracing is off for this Langfuse version"
        )
        return None
    provider = TracerProvider()
    handler = None
    with lock:
        # Out of the registry while this client is made, and kept out after,
        # so the client cannot be another one under the same key, nor be
        # found by it. The engine's own client, if it has this key, goes back.
        shared = registry.pop(config.public_key, None)
        try:
            Langfuse(
                public_key=config.public_key,
                secret_key=config.secret_key,
                base_url=config.host,
                tracer_provider=provider,
            )
            handler = CallbackHandler(public_key=config.public_key)
        except Exception as exc:  # a turn is never failed by its tracing
            logger.warning(
                "tracing: an assistant's destination could not be made: %s",
                type(exc).__name__,
            )
        finally:
            resources = registry.pop(config.public_key, None)
            if shared is not None:
                registry[config.public_key] = shared
    destination = _Destination(handler=handler, resources=resources, provider=provider)
    if handler is None or resources is None:
        _close(destination)
        return None
    return destination


def _close(destination: _Destination) -> None:
    """Send what a destination still holds and stop its threads."""
    try:
        if destination.resources is not None:
            destination.resources.shutdown()
        destination.provider.shutdown()
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("tracing: closing a destination failed: %s", exc)


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
    with _per_project_lock:
        destinations = list(_per_project.values())
    if _handler is None and not destinations:
        return
    try:
        if _handler is not None:
            from langfuse import get_client

            get_client().flush()
        for destination in destinations:
            destination.resources.flush()
    except Exception as exc:  # pragma: no cover - best effort at shutdown
        logger.warning("tracing: flush failed: %s", exc)
