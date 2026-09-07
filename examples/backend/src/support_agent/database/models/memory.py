from datetime import datetime

from sqlalchemy import DateTime, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from support_agent.database.base import Base


class Memory(Base):
    """One thing the assistant was asked to remember.

    Scoped to a person, not to a conversation. That is what makes it long-term:
    a new chat is a new thread, so keying on the thread would erase everything
    the moment someone started one.

    Who that person is comes from `api/identity.py`. Behind a proxy it is the
    authenticated user; without one it is the id the browser keeps, which
    partitions notes without pretending to enforce anything. See docs/memory.md.

    The content is written by a model from what a customer typed, so it is
    untrusted text in both directions: it is stored as data and fed back to the
    model as data, never as instructions.
    """

    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    #: Who this belongs to. Every read filters on it; without that filter one
    #: person would recall another's notes, which is the whole risk of storing
    #: anything at all.
    user_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    #: Which conversation it was first noted in. Recorded for the debug view, so
    #: a note can be traced back to where it came from. Nothing filters on it:
    #: the whole point is that memory outlives the thread that created it.
    session_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    #: Which assistant stored it, so two projects on one backend stay separate.
    project_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    #: Short label for what this is about ("seat preference"), for the debug
    #: view and for replacing a fact rather than accumulating near-duplicates.
    subject: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        # Every query is "this person, this project", so index the pair.
        Index("ix_memories_user_project", "user_id", "project_id"),
        # One row per subject per person: remembering the same thing twice
        # updates it instead of leaving two answers to the same question.
        Index(
            "uq_memories_user_project_subject",
            "user_id",
            "project_id",
            "subject",
            unique=True,
        ),
    )
