from datetime import date, datetime

from sqlalchemy import Date, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column
from support_agent.database.base import Base


class Flight(Base):
    """Represent one scheduled flight in the application database."""

    __tablename__ = "flights"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    flight_number: Mapped[str] = mapped_column(
        String(16),
        index=True,
    )

    departure_date: Mapped[date] = mapped_column(
        Date,
        index=True,
    )

    origin: Mapped[str] = mapped_column(
        String(8),
    )

    destination: Mapped[str] = mapped_column(
        String(8),
    )

    status: Mapped[str] = mapped_column(
        String(32),
    )

    scheduled_departure: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
    )

    expected_departure: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    gate: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
    )