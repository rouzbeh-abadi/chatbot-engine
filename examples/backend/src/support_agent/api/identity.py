"""Who is calling: the one place the backend decides a caller's identity.

This is the seam to replace when putting the app behind real authentication.
Every route that needs a user takes `UserIdDep`, so swapping the implementation
here -- for a session cookie, a validated JWT, whatever you use -- reaches all
of them without touching a single endpoint.

What ships is deliberately not authentication. It is the smallest honest thing:
either nobody is identified, or a trusted proxy in front has already done the
identifying and said so in a header.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from support_agent.settings import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]

#: Who everyone is when no proxy is authenticating callers. An opaque id, the
#: same shape the engine and the MCP tool server already expect -- they never
#: needed it to mean a person, only to be stable within a conversation.
ANONYMOUS_USER_ID = "anonymous"


async def resolve_user_id(
    settings: SettingsDep,
    x_user_id: Annotated[str | None, Header()] = None,
) -> str:
    """The caller's user id, or `anonymous` when nothing has identified them.

    `X-User-Id` is only believed when `BACKEND_TRUST_USER_HEADER` says a proxy
    is setting it. That switch is the whole point: a header is something any
    browser can type, so trusting it by default would let every caller claim to
    be every user. Off, the header is discarded rather than rejected -- it may
    be a stale client, and there is nothing to protect when nobody is anybody.

    On, a request with no header did not come through the proxy, which means the
    deployment is misrouted and letting it through anonymously would quietly
    undo the authentication it is supposed to have. That is a 401.
    """
    if not settings.trust_user_header:
        return ANONYMOUS_USER_ID

    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing X-User-Id",
        )

    return x_user_id


UserIdDep = Annotated[str, Depends(resolve_user_id)]
