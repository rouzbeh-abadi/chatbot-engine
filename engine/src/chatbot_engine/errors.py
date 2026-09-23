"""Engine errors.

`EngineError` is the base so a caller can catch one thing. Subclasses exist for
the cases the API maps to a status code other than 500.
"""

from __future__ import annotations


class EngineError(Exception):
    """Base class for every error the engine raises deliberately."""


class NotConfiguredError(EngineError):
    """The engine lacks configuration this route needs.

    Today that is the model provider key: an engine without one can record and
    chunk documents but cannot embed, retrieve or answer. Mapped to 501, and the
    message names the variable to set.
    """


class DocumentRejectedError(EngineError):
    """A readable document with nothing worth indexing -- usually a scanned PDF.

    Mapped to 422, not 500: the engine worked, the answer is "not this file".
    """


#: The most of a provider's reason passed on; its body can be long.
PROVIDER_REASON_CHARS = 500


def provider_reason(exc: BaseException) -> str:
    """What the model provider said, without the SDK's wrapping.

    The SDK's own message is `Error code: 404 - {...}`, the whole body printed
    as a dict; the provider's sentence is inside it, under `message`. A call
    that never got an answer says so.
    """
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and isinstance(inner.get("message"), str):
            reason = inner["message"]
        else:
            reason = str(exc)
    else:
        reason = str(exc)
    where = (
        f"the model provider answered {status}"
        if status
        else "the model provider could not be reached"
    )
    return f"{where}: {reason.strip()}"[:PROVIDER_REASON_CHARS]
