"""Who may call the engine.

The engine holds provider credentials and has no notion of end users: it cannot
tell a legitimate question from an expensive one, because both look identical on
the wire. So the only meaningful boundary is *which service* is calling, and
that boundary has to hold on its own -- not because a backend in front is
expected to be well behaved.

Authentication is a shared secret in `X-API-Key`, matched against the named keys
in `ENGINE_API_KEYS` (see `Settings.credentials`). Naming them is what makes
rotation and attribution possible; see the `Caller` note below.

Deliberately not implemented here: end-user identity. An engine that tried to
decide *which person* may ask something would need your user model, and it has
no business having one. That check belongs in the service that calls it.
"""

from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from chatbot_engine.settings import Settings, get_settings

logger = logging.getLogger(__name__)

SettingsDep = Annotated[Settings, Depends(get_settings)]

#: The caller's name when no keys are configured at all -- an open engine, which
#: is a localhost-only arrangement (`ENGINE_ENV=production` refuses to start
#: that way). Named rather than left blank so logs and rate limits still have
#: something to group by.
OPEN_CALLER = "unauthenticated"


@dataclass(frozen=True)
class Caller:
    """Which credential authenticated this request.

    Carried rather than discarded because a bare yes/no cannot answer the two
    questions that matter once something goes wrong: *who* is doing this, and
    can they be stopped without stopping everyone else. The name is what appears
    in logs and what rate limits are counted against.
    """

    name: str

    @property
    def is_authenticated(self) -> bool:
        return self.name != OPEN_CALLER


def _match(offered: str, credentials: dict[str, str]) -> str | None:
    """The name of the key that matches, or None.

    Every candidate is compared with `compare_digest`, never `==`: a normal
    comparison stops at the first differing byte, so the time it takes reveals
    how much of the secret was right. That is enough to recover a key one byte
    at a time over a few thousand requests, and it is the single most common way
    a shared-secret check is written wrong.

    Compared as bytes, because `compare_digest` raises on a str holding any
    character above U+007F -- and headers are latin-1 decoded, so a caller can
    send one.
    """
    offered_bytes = offered.encode("utf-8")

    matched: str | None = None
    for name, secret in credentials.items():
        # No early exit: returning here would make the reply time depend on
        # which key matched, which leaks the order of the key list.
        if hmac.compare_digest(offered_bytes, secret.encode("utf-8")):
            matched = name

    return matched


async def require_api_key(
    request: Request,
    settings: SettingsDep,
    x_api_key: Annotated[str | None, Header()] = None,
) -> Caller:
    """Authenticate the caller, or refuse with 401.

    With no keys configured the engine is open and every caller is
    `unauthenticated`. That is fine on a laptop and refused outright under
    `ENGINE_ENV=production`, which is where the decision is enforced -- this
    function's job is to be honest about what it did, not to second-guess it.
    """
    credentials = settings.credentials()
    if not credentials:
        return Caller(name=OPEN_CALLER)

    name = _match(x_api_key, credentials) if x_api_key else None
    if name is None:
        # Logged because a failed key is the only externally visible sign of
        # someone guessing at one, and a silent 401 makes that invisible. The
        # key itself is never logged, only that one was offered.
        client = request.client
        logger.warning(
            "rejected a request to %s from %s: %s X-API-Key",
            request.url.path,
            client.host if client else "an unknown address",
            "invalid" if x_api_key else "missing",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid X-API-Key",
        )

    return Caller(name=name)


CallerDep = Annotated[Caller, Depends(require_api_key)]
