"""add video verification enum value

Revision ID: 20260420_0014
Revises: 20260420_0013
Create Date: 2026-04-20 10:45:00.000000
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "20260420_0014"
down_revision = "20260420_0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_enum e
                JOIN pg_type t ON e.enumtypid = t.oid
                WHERE t.typname = 'cryptoobjecttype'
                  AND e.enumlabel = 'video_verification_record'
            ) THEN
                ALTER TYPE cryptoobjecttype ADD VALUE 'video_verification_record';
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    # PostgreSQL enum values are not safely removable in-place.
    pass
