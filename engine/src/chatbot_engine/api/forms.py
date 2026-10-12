"""A form's fields checked against the ones its route takes.

A JSON body is a model with `extra="forbid"`, so a misspelled field is a 422
that names it. A form is read field by field, and FastAPI drops any it was
not asked for: `chunk_szie=200` was indexed at the defaults with a 201, and
a misspelled `provider_api_key` billed an image read to the engine's own key
(docs/review-2026-10.md, API-6). The form is parsed before this runs, so it
costs nothing more.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request


def only_fields(*names: str) -> Callable[[Request], Awaitable[None]]:
    """A dependency refusing, with 422, a form with a field not in `names`."""
    allowed = frozenset(names)

    async def check(request: Request) -> None:
        unknown = sorted({key for key in await request.form() if key not in allowed})
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=[
                    {
                        "type": "extra_forbidden",
                        "loc": ["body", key],
                        "msg": "This field is not taken here",
                    }
                    for key in unknown
                ],
            )

    return check
