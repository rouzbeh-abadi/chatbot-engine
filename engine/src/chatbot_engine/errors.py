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
