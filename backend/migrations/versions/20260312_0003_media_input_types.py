"""add audio and video job input types"""

from alembic import op


revision = "20260312_0003"
down_revision = "20260312_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE jobinputtype ADD VALUE IF NOT EXISTS 'audio'")
    op.execute("ALTER TYPE jobinputtype ADD VALUE IF NOT EXISTS 'video'")


def downgrade() -> None:
    # PostgreSQL enum value removal is intentionally omitted to avoid destructive rewrites.
    pass
