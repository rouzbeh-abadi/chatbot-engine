"""Alembic environment.

Two things this wires that the generated template does not:

* `target_metadata` points at the application's models, so `--autogenerate` can
  actually see them. Importing `support_agent.database.models` is what registers
  the tables -- without that import the metadata is empty and autogenerate
  cheerfully produces a migration that drops everything.
* the URL comes from `BACKEND_DATABASE_URL` rather than `alembic.ini`, so there is
  one source of truth and no credentials in a tracked file.

Migrations run on a *synchronous* connection even though the application is
async. The `psycopg` (v3) dialect supports both, so the same URL string serves
each without a second driver.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool
from support_agent.database.base import Base
from support_agent.database.models import Booking, Flight, SupportTicket  # noqa: F401
from support_agent.settings import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """Prefer `-x url=...`, then the application settings."""
    return context.get_x_argument(as_dictionary=True).get(
        "url", get_settings().database_url
    )


def run_migrations_offline() -> None:
    """Render SQL without connecting -- `alembic upgrade head --sql`."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(_database_url(), poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
