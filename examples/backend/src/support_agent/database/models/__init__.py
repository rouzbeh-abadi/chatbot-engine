"""SQLAlchemy models.

Importing this package registers every table on `Base.metadata`, which is what
Alembic's `env.py` needs in order to see them. Add new models to the imports
below or autogenerate will silently miss them.
"""

from support_agent.database.models.booking import Booking
from support_agent.database.models.flight import Flight
from support_agent.database.models.support_ticket import SupportTicket

__all__ = ["Booking", "Flight", "SupportTicket"]
