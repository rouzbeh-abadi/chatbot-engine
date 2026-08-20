"""Engine errors.

`EngineError` is the base so a caller can catch one thing. `NotConfiguredError`
means "this capability has no implementation yet" and the API maps it to 501; a
bare `raise NotImplementedError` from half-written code gets the same treatment
through its own handler.
"""

from __future__ import annotations


class EngineError(Exception):
    """Base class for every error the engine raises deliberately."""


class NotConfiguredError(EngineError):
    """A capability was requested before an implementation was wired in.

    Surfaced as 501 with the name of the thing to implement and where to
    register it.
    """


class DocumentRejectedError(EngineError):
    """A readable document with nothing worth indexing -- usually a scanned PDF.

    Mapped to 422, not 500: the engine worked, the answer is "not this file".
    """
