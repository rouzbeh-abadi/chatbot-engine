"""The operator key: one shared secret for the privileged routes.

Separate from `api/identity.py`, which answers "which user is this?". This
answers "is the caller an operator at all?" -- a different question with a
different answer shape, and the two must not be conflated: an authenticated
*user* is still not allowed to read every booking or delete the knowledge base.

One key rather than accounts is a deliberate limit of this showcase. It is the
right size for routes only an operator touches, and the wrong size for anything
where you need to know *which* operator acted.
"""

from __future__ import annotations

import hmac
import logging
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from support_agent.settings import Settings, get_settings

logger = logging.getLogger(__name__)

SettingsDep = Annotated[Settings, Depends(get_settings)]


async def require_admin_key(
    settings: SettingsDep,
    x_admin_key: Annotated[str | None, Header()] = None,
) -> None:
    """Shared operator secret for `/admin` and the document write routes.

    These read every booking and every ticket, spend real model credits, and can
    replace what the assistant knows, so they must not be open to whoever can
    reach the port. Unset `BACKEND_ADMIN_KEY` leaves them open, which is fine on
    localhost and nowhere else -- hence the warning rather than a silent pass,
    and the refusal to start at all under `BACKEND_ENV=production`.
    """
    if settings.admin_key is None:
        logger.warning(
            "a privileged route was served without authentication: set "
            "BACKEND_ADMIN_KEY to require an X-Admin-Key header"
        )
        return

    # Constant-time: a plain `!=` leaks the shared secret one character at a
    # time to anyone who can measure the reply. Compared as bytes, because
    # `compare_digest` raises on a str with any character above U+007F -- and a
    # header is decoded as latin-1, so a client can send one.
    if x_admin_key is None or not hmac.compare_digest(
        x_admin_key.encode("utf-8"), settings.admin_key.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid X-Admin-Key",
        )


#: For routers where every route is privileged.
AdminOnly = Depends(require_admin_key)
