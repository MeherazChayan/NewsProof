"""attach consensus votes to content identity"""

from alembic import op
import sqlalchemy as sa


revision = "20260321_0006"
down_revision = "20260321_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("consensus_votes", sa.Column("canonical_url", sa.String(length=2048), nullable=True))
    op.add_column("consensus_votes", sa.Column("content_hash", sa.String(length=64), nullable=True))
    op.create_index("ix_consensus_votes_content_hash", "consensus_votes", ["content_hash"], unique=False)

    op.execute(
        """
        UPDATE consensus_votes AS votes
        SET
            canonical_url = extracted_content.canonical_url,
            content_hash = extracted_content.content_hash
        FROM extracted_content
        WHERE votes.job_id = extracted_content.job_id
        """
    )


def downgrade() -> None:
    op.drop_index("ix_consensus_votes_content_hash", table_name="consensus_votes")
    op.drop_column("consensus_votes", "content_hash")
    op.drop_column("consensus_votes", "canonical_url")
