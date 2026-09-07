"""long-term memory

Revision ID: c1205407a441
Revises: 0b1da044c12c
Create Date: 2026-09-07 19:34:33.610562

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c1205407a441'
down_revision: Union[str, Sequence[str], None] = '0b1da044c12c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'memories',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        # Who the note belongs to. Memory follows a person across
        # conversations, so this, not session_id, is what every read filters on.
        sa.Column('user_id', sa.String(length=64), nullable=False),
        sa.Column('project_id', sa.String(length=64), nullable=False),
        # Where it was first noted. Recorded for the debug view only; nothing
        # filters on it, and it is nullable because a note may outlive the
        # thread that created it.
        sa.Column('session_id', sa.String(length=64), nullable=True),
        sa.Column('subject', sa.String(length=120), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_memories_user_project', 'memories', ['user_id', 'project_id'], unique=False)
    # One row per subject per person, so remembering the same thing twice
    # corrects it instead of leaving two answers to the same question.
    op.create_index('uq_memories_user_project_subject', 'memories', ['user_id', 'project_id', 'subject'], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('uq_memories_user_project_subject', table_name='memories')
    op.drop_index('ix_memories_user_project', table_name='memories')
    op.drop_table('memories')
