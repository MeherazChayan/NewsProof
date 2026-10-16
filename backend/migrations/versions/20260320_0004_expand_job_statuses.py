"""expand job statuses for likely-verified and unverified"""

from alembic import op


revision = "20260320_0004"
down_revision = "20260312_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE jobstatus ADD VALUE IF NOT EXISTS 'Likely Verified'")
    op.execute("ALTER TYPE jobstatus ADD VALUE IF NOT EXISTS 'Unverified'")


def downgrade() -> None:
    pass
