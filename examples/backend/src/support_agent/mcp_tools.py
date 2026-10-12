"""Application-owned domain tools exposed to the AI engine through MCP.

These run in the application backend, not in the engine, because they read this
application's business data and must eventually enforce the calling user's
permissions -- something the engine cannot evaluate. The engine discovers them
over MCP and invokes them; it never owns their implementation.

    make tools     # streamable HTTP on http://localhost:8200/mcp

It listens on loopback unless `BACKEND_MCP_TOOLS_HOST` says otherwise, answers
only to a Host it knows, and asks for a bearer token when
`BACKEND_MCP_TOOLS_TOKEN` is set (see `http_app`). The token is what lets it
believe the `X-User-Id` a call carries: only the engine, sent the token by the
backend, can make one.

Two conventions the tools follow, both aimed at the model that reads them:

* every value comes back as a string, in the words a customer would recognise
  ("cabin baggage only" rather than `null`), because the model relays these to a
  person and a bare null invites it to invent an explanation;
* a missing booking or flight is a normal answer, returned as data. Only an
  unusable *request* raises. A tool that raises for "not found" teaches the model
  that the tool is broken, and it stops trying.
"""

from __future__ import annotations

import hmac
import logging
from datetime import date
from urllib.parse import urlsplit

import uvicorn
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from sqlalchemy import select
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from support_agent.database.connection import get_session_factory
from support_agent.database.models import Booking, Flight, Memory, SupportTicket
from support_agent.settings import Settings, get_settings

#: Where the server listens unless `BACKEND_MCP_TOOLS_HOST` says otherwise:
#: loopback, so nothing beyond this machine reaches it.
HOST = "127.0.0.1"
PORT = 8200

#: The Host and Origin values that mean this machine.
LOOPBACK_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
LOOPBACK_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]

logger = logging.getLogger(__name__)

TICKET_CATEGORIES = ("refund", "baggage", "schedule_change", "complaint", "other")

#: Which assistant's memory these tools read and write. One project today; a
#: multi-project backend would take it from the request the same way the
#: conversation id is taken.
DEFAULT_PROJECT_ID = "support"


def _trace(ctx: Context, tool: str) -> None:
    """One log line per call, carrying the engine's request id.

    The engine forwards the id of the turn it is serving as `X-Request-Id`,
    so a tool call can be found in this server's log by the same id as in
    the backend's and the engine's. Nothing else about the request is logged
    here; the arguments are the customer's data.
    """
    headers = ctx.headers or {}
    rid = headers.get("x-request-id") or headers.get("X-Request-Id") or "-"
    logger.info("tool %s [%s]", tool, rid)


mcp = MCPServer(
    name="support-tools",
    instructions=(
        "Domain tools for a travel support assistant. Use them to look up real "
        "booking and flight data instead of guessing, and to escalate to a human "
        "when the knowledge base cannot resolve the customer's problem."
    ),
)


def _is_last_name(passenger_name: str, given: str) -> bool:
    """Whether `given` is how the booking's passenger name ends, in any case.

    "Miller" and "Daniel Miller" both match "Daniel Miller"; "Mill" does not.
    """
    name, said = passenger_name.casefold().split(), given.casefold().split()
    return bool(said) and name[-len(said) :] == said


@mcp.tool()
async def get_booking_status(
    ctx: Context,
    booking_reference: str,
    last_name: str = "",
) -> dict[str, str]:
    """Look up one booking by its reference and the passenger's last name.

    Returns the passenger, route, travel date, fare type, checked baggage
    allowance, flight number, and current status. Use this whenever a customer
    mentions a booking reference; ask for the passenger's last name too if
    they have not given it, and pass both. Use the flight number it returns to
    check that flight's status.

    Args:
        booking_reference: Six-character booking reference, e.g. "AB12CD".
        last_name: The passenger's last name, as the customer gave it,
            e.g. "Miller".

    Returns:
        Booking details and its current status, a request for the last name,
        or a not-found result.
    """
    _trace(ctx, "get_booking_status")
    reference = booking_reference.strip().upper()

    # The reference alone is not enough. It is six characters that travel on
    # confirmations, boarding passes and forwarded emails, and with nothing
    # else it would hand a passenger's name, route and date to whoever typed
    # it. The last name is checked here, against the booking, where the model
    # cannot talk its way past it.
    if not last_name.strip():
        return {
            "booking_reference": reference,
            "status": "last_name_required",
            "message": (
                "Ask the customer for the passenger's last name as it is on the "
                "booking, then look the booking up again with both."
            ),
        }

    async with get_session_factory()() as session:
        booking = await session.scalar(
            select(Booking).where(Booking.booking_reference == reference)
        )

    # A wrong name is answered exactly as an unknown reference, so the tool
    # cannot be used to learn which references exist.
    if booking is None or not _is_last_name(booking.passenger_name, last_name):
        return {
            "booking_reference": reference,
            "status": "not_found",
            "message": "No booking was found with this reference and last name.",
        }

    result = {
        "booking_reference": booking.booking_reference,
        "passenger_name": booking.passenger_name,
        "route": f"{booking.origin} → {booking.destination}",
        "travel_date": booking.travel_date.isoformat(),
        "fare_type": booking.fare_type,
        "status": booking.status,
        "flight_number": booking.flight_number,
        "checked_baggage": booking.checked_baggage or "none (cabin baggage only)",
    }

    if booking.connecting_flight_number:
        result["connecting_flight_number"] = booking.connecting_flight_number
        result["itinerary"] = "two flights; refund rules may differ per segment"

    return result


@mcp.tool()
async def get_flight_status(
    ctx: Context,
    flight_number: str,
    departure_date: str,
) -> dict[str, str]:
    """Check whether a flight is on time, delayed, or cancelled.

    Returns the scheduled departure, the expected departure when a flight is
    delayed, the gate once assigned, and a status. Use this for a specific flight
    on a specific date; it is not a schedule search. The flight number for a
    customer's booking comes from get_booking_status.

    Args:
        flight_number: Airline code and flight number, e.g. "SD204".
        departure_date: Departure date in ISO format, e.g. "2026-08-24".

    Returns:
        Current flight status and available timing information.
    """
    _trace(ctx, "get_flight_status")
    number = flight_number.strip().upper()

    try:
        parsed_date = date.fromisoformat(departure_date.strip())
    except ValueError:
        # An unusable argument, not a missing record: say what was wrong so the
        # model can retry with a corrected date rather than giving up.
        return {
            "flight_number": number,
            "status": "invalid_request",
            "message": (
                f"{departure_date!r} is not an ISO date. Use YYYY-MM-DD, "
                "e.g. 2026-08-24."
            ),
        }

    async with get_session_factory()() as session:
        flight = await session.scalar(
            select(Flight).where(
                Flight.flight_number == number,
                Flight.departure_date == parsed_date,
            )
        )

    if flight is None:
        return {
            "flight_number": number,
            "departure_date": parsed_date.isoformat(),
            "status": "not_found",
            "message": "No flight was found with this number on this date.",
        }

    result = {
        "flight_number": flight.flight_number,
        "departure_date": flight.departure_date.isoformat(),
        "route": f"{flight.origin} → {flight.destination}",
        "status": flight.status,
        "scheduled_departure": flight.scheduled_departure.isoformat(),
        "gate": flight.gate or "not assigned yet",
    }

    if flight.expected_departure is not None:
        result["expected_departure"] = flight.expected_departure.isoformat()

    return result


@mcp.tool()
async def create_support_ticket(
    ctx: Context,
    booking_reference: str,
    summary: str,
    category: str,
    last_name: str = "",
) -> dict[str, str]:
    """Escalate to a human agent and return the new ticket reference.

    Use this only when the customer's problem cannot be resolved from the
    knowledge base or the other tools -- a refund dispute, a medical request, or a
    complaint. Summarise the issue in your own words; do not paste the whole
    conversation.

    Args:
        booking_reference: The booking the ticket relates to.
        summary: One or two sentences describing what the customer needs.
        category: One of "refund", "baggage", "schedule_change", "complaint",
            "other".
        last_name: The passenger's last name, as the customer gave it,
            checked against the booking as for `get_booking_status`.

    Returns:
        The created ticket, or a rejection explaining what to fix.
    """
    _trace(ctx, "create_support_ticket")
    reference = booking_reference.strip().upper()
    chosen = category.strip().lower()

    if chosen not in TICKET_CATEGORIES:
        return {
            "status": "invalid_request",
            "message": (
                f"{category!r} is not a valid category. Choose one of: "
                + ", ".join(TICKET_CATEGORIES)
            ),
        }

    if not summary.strip():
        return {
            "status": "invalid_request",
            "message": "A summary is required so the agent knows what to act on.",
        }

    if not last_name.strip():
        return {
            "booking_reference": reference,
            "status": "last_name_required",
            "message": (
                "Ask the customer for the passenger's last name as it is on the "
                "booking, then create the ticket again with both."
            ),
        }

    async with get_session_factory()() as session:
        # Tickets hang off a real booking, the customer's own: a reference on
        # its own would let anyone file against someone else's booking, and
        # learn which references exist. A wrong name is answered exactly as
        # an unknown reference.
        booking = await session.scalar(
            select(Booking).where(Booking.booking_reference == reference)
        )
        if booking is None or not _is_last_name(booking.passenger_name, last_name):
            return {
                "booking_reference": reference,
                "status": "not_found",
                "message": (
                    "No booking was found with this reference and last name, so "
                    "no ticket was created. Confirm both with the customer."
                ),
            }

        ticket = SupportTicket(
            booking_reference=reference,
            summary=summary.strip(),
            category=chosen,
            status="open",
        )
        session.add(ticket)
        await session.commit()
        await session.refresh(ticket)

        return {
            "ticket_id": str(ticket.id),
            "booking_reference": ticket.booking_reference,
            "category": ticket.category,
            "status": ticket.status,
            "created_at": ticket.created_at.isoformat(),
            "message": "A support agent will follow up on this ticket.",
        }


# --- memory ------------------------------------------------------------------
#
# One tool, and one rule that matters more than it: the owner is taken from the
# request headers the engine forwards, never from an argument the model
# supplies. A model that could name the owner could write into anyone's memory.


def _owner(ctx: Context) -> tuple[str, str | None, str]:
    """Who this note belongs to, where it was noted, and which project.

    The owner comes from the header the engine forwards, never from a tool
    argument: a model that could name the owner could write into someone else's
    memory, or read it back. The header is only as good as the caller, which is
    why the server asks for a token wherever more than this machine can reach
    it (see `http_app`).

    Raises rather than guessing. A note written to the wrong person, or read
    from everyone, is worse than a tool that says it cannot run.
    """
    headers = ctx.headers or {}
    user_id = headers.get("x-user-id") or headers.get("X-User-Id")
    session_id = headers.get("x-session-id") or headers.get("X-Session-Id")

    if not user_id:
        raise ValueError(
            "no user id on this request, so memory cannot be scoped; "
            "the caller must send a user_id"
        )

    return user_id, session_id, DEFAULT_PROJECT_ID


@mcp.tool()
async def remember(
    ctx: Context,
    subject: str,
    content: str,
) -> dict[str, str]:
    """Store one fact about this customer, for this and later conversations.

    Use it when the customer states a lasting preference or detail worth having
    later ("I always fly economy", "my company pays for changes"). Do not use it
    for anything they would not expect a support agent to write down, and never
    for payment details, passwords, or identity documents.

    Storing the same `subject` twice replaces the earlier value, so a changed
    preference corrects the old one rather than sitting beside it.

    Do not announce that you stored something. Note it and carry on answering.

    Args:
        subject: A short label for what this is about, e.g. "seat preference".
        content: The fact itself, in one sentence.

    Returns:
        What was stored. Not for relaying to the customer.
    """
    _trace(ctx, "remember")
    user_id, session_id, project_id = _owner(ctx)
    subject = subject.strip()[:120]
    content = content.strip()[:2000]

    if not subject or not content:
        return {"status": "rejected", "message": "A subject and content are required."}

    async with get_session_factory()() as session:
        existing = await session.scalar(
            select(Memory).where(
                Memory.user_id == user_id,
                Memory.project_id == project_id,
                Memory.subject == subject,
            )
        )
        if existing is None:
            session.add(
                Memory(
                    user_id=user_id,
                    # Where it was noted, for the debug view only: cut to the
                    # column, since a chat's session id may run to 256.
                    session_id=session_id[:64] if session_id else None,
                    project_id=project_id,
                    subject=subject,
                    content=content,
                )
            )
        else:
            existing.content = content
        await session.commit()

    return {"status": "stored", "subject": subject, "content": content}


# --- serving -------------------------------------------------------------------


def transport_security(settings: Settings) -> TransportSecuritySettings:
    """The MCP SDK's DNS-rebinding guard, on whatever address the server listens.

    The SDK turns it on by itself only for a loopback address, and listening on
    0.0.0.0 turned it off. Here it is always on: a request is answered only when
    its Host is this machine or the host the engine is told to dial
    (`BACKEND_MCP_TOOLS_URL`, `mcp-tools` under Compose), so a web page that
    points its own name at this server is refused.
    """
    hosts = list(LOOPBACK_HOSTS)
    name = urlsplit(settings.mcp_tools_url or "").hostname
    if name:
        name = f"[{name}]" if ":" in name else name
        hosts += [name, f"{name}:*"]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=list(LOOPBACK_ORIGINS),
    )


class RequireToken:
    """Answers 401 to an HTTP request without `Authorization: Bearer <token>`.

    Compared in constant time. The engine sends the header with discovery and
    with every call, because the backend puts it in this server's entry
    (`assistant._apply_deployment_overrides`).
    """

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            given = Headers(scope=scope).get("authorization", "").encode()
            if not hmac.compare_digest(given, self.expected):
                refused = PlainTextResponse(
                    "missing or wrong bearer token",
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await refused(scope, receive, send)
                return
        await self.app(scope, receive, send)


def http_app(settings: Settings | None = None) -> Starlette:
    """The server as `main` serves it.

    The rebinding guard is always on, and a bearer token is required when
    `BACKEND_MCP_TOOLS_TOKEN` is set.
    """
    settings = settings or get_settings()
    app = mcp.streamable_http_app(transport_security=transport_security(settings))
    if settings.mcp_tools_token:
        app.add_middleware(RequireToken, token=settings.mcp_tools_token)
    return app


def main() -> None:
    """Start the backend MCP tool server over streamable HTTP."""
    settings = get_settings()
    host = settings.mcp_tools_host or HOST
    if not settings.mcp_tools_token and host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "listening on %s with no BACKEND_MCP_TOOLS_TOKEN: whatever reaches "
            "port %d can call every tool and write into any person's memory",
            host,
            PORT,
        )
    uvicorn.run(http_app(settings), host=host, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
