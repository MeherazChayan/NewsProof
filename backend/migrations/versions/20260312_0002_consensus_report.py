"""add consensus report fields"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260312_0002"
down_revision = "20260311_0001"
branch_labels = None
depends_on = None


consensus_role = postgresql.ENUM("user", "fact_checker", "moderator", name="consensusrole")


def upgrade() -> None:
    bind = op.get_bind()
    consensus_role.create(bind, checkfirst=True)

    op.add_column("users", sa.Column("display_name", sa.String(length=255), nullable=True))
    op.add_column("users", sa.Column("role", consensus_role, nullable=False, server_default="user"))

    op.add_column("verification_jobs", sa.Column("analysis_report", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")))

    op.add_column("consensus_votes", sa.Column("reporter_name", sa.String(length=255), nullable=True))
    op.add_column("consensus_votes", sa.Column("role", consensus_role, nullable=False, server_default="user"))
    op.add_column("consensus_votes", sa.Column("confidence", sa.Float(), nullable=False, server_default="0.5"))
    op.add_column("consensus_votes", sa.Column("report_reason", sa.Text(), nullable=True))
    op.add_column("consensus_votes", sa.Column("evidence_url", sa.String(length=2048), nullable=True))


def downgrade() -> None:
    op.drop_column("consensus_votes", "evidence_url")
    op.drop_column("consensus_votes", "report_reason")
    op.drop_column("consensus_votes", "confidence")
    op.drop_column("consensus_votes", "role")
    op.drop_column("consensus_votes", "reporter_name")
    op.drop_column("verification_jobs", "analysis_report")
    op.drop_column("users", "role")
    op.drop_column("users", "display_name")

    bind = op.get_bind()
    consensus_role.drop(bind, checkfirst=True)
