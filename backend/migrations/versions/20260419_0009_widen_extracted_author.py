"""widen extracted content author column"""

from alembic import op
import sqlalchemy as sa


revision = "20260419_0009"
down_revision = "20260419_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "extracted_content",
        "author",
        existing_type=sa.String(length=255),
        type_=sa.Text(),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "extracted_content",
        "author",
        existing_type=sa.Text(),
        type_=sa.String(length=255),
        existing_nullable=True,
    )
