"""A form's fields, and a query's parameters, checked against the ones its
route takes.

A JSON body is a model with `extra="forbid"`, so a misspelled field is a 422
that names it. A form is read field by field, and FastAPI drops any it was
not asked for: `chunk_szie=200` was indexed at the defaults with a 201, and
a misspelled `provider_api_key` billed an image read to the engine's own key
(docs/review-2026-10.md, API-6). The form is parsed before this runs, so it
costs nothing more. A query is read the same way: `/reindex?chunk_size=200`
re-chunked at the defaults with a 200 until its parameters were checked too.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable

from fastapi import HTTPException, Request


def only_fields(*names: str) -> Callable[[Request], Awaitable[None]]:
    """A dependency refusing, with 422, a form with a field not in `names`."""
    allowed = frozenset(names)

    async def check(request: Request) -> None:
        form = await request.form()
        _refuse("body", (key for key in form if key not in allowed))

    return check


def only_params(*names: str) -> Callable[[Request], Awaitable[None]]:
    """A dependency refusing, with 422, a query parameter not in `names`."""
    allowed = frozenset(names)

    async def check(request: Request) -> None:
        _refuse("query", (key for key in request.query_params if key not in allowed))

    return check


def _refuse(where: str, unknown: Iterable[str]) -> None:
    """A 422 naming each field or parameter that is not taken, if there is any."""
    names = sorted(set(unknown))
    if names:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "type": "extra_forbidden",
                    "loc": [where, name],
                    "msg": "This field is not taken here",
                }
                for name in names
            ],
        )
