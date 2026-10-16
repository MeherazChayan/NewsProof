"""widen article snapshot author field

Revision ID: 20260419_0010
Revises: 20260419_0009
Create Date: 2026-04-19 20:20:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260419_0010"
down_revision = "20260419_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "article_snapshots",
        "author_name",
        existing_type=sa.String(length=255),
        type_=sa.Text(),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "article_snapshots",
        "author_name",
        existing_type=sa.Text(),
        type_=sa.String(length=255),
        existing_nullable=True,
    )
