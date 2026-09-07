from datetime import date

from sqlalchemy import Date, String
from sqlalchemy.orm import Mapped, mapped_column
from support_agent.database.base import Base


class Booking(Base):
    """Database model representing one customer flight booking."""

    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    booking_reference: Mapped[str] = mapped_column(
        String(6),
        unique=True,
        index=True,
        nullable=False,
    )

    passenger_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    origin: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    destination: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    travel_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    #: The flight this booking travels on. Without it the assistant cannot get
    #: from a booking reference to a flight status, so the two tools could never
    #: be chained in one conversation.
    flight_number: Mapped[str] = mapped_column(
        String(16),
        index=True,
        nullable=False,
    )

    #: Second leg, for connecting itineraries. Two legs is enough to exercise the
    #: multi-segment refund rules in the knowledge base; a booking with more legs
    #: would want its own segments table.
    connecting_flight_number: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
    )

    #: Checked allowance as the customer would read it, e.g. "1 x 23 kg". NULL
    #: means cabin baggage only, which is what the cheaper fares include.
    checked_baggage: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    fare_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )