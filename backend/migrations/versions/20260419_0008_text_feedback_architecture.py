"""add text verification records and feedback review layer"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260419_0008"
down_revision = "20260418_0007"
branch_labels = None
depends_on = None


feedback_user_role = postgresql.ENUM(
    "user",
    "expert",
    "moderator",
    name="feedbackuserrole",
    create_type=False,
)
feedback_type = postgresql.ENUM(
    "correction",
    "dispute",
    "source_suggestion",
    "clarification",
    "review_note",
    name="feedbacktype",
    create_type=False,
)
feedback_status = postgresql.ENUM(
    "open",
    "reviewed",
    "resolved",
    name="feedbackstatus",
    create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    feedback_user_role.create(bind, checkfirst=True)
    feedback_type.create(bind, checkfirst=True)
    feedback_status.create(bind, checkfirst=True)

    op.execute("ALTER TYPE cryptoobjecttype ADD VALUE IF NOT EXISTS 'verification_record'")

    op.create_table(
        "verification_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("input_text", sa.Text(), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("input_text_hash", sa.String(length=64), nullable=False),
        sa.Column("extracted_claims", sa.JSON(), nullable=False),
        sa.Column("evidence_sources", sa.JSON(), nullable=False),
        sa.Column("nlp_result", sa.JSON(), nullable=False),
        sa.Column("nlp_confidence_score", sa.Float(), nullable=False),
        sa.Column("nlp_label", sa.String(length=255), nullable=False),
        sa.Column("current_manifest_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("provenance_manifests.id"), nullable=True),
        sa.Column("current_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("previous_record_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("verification_records.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_verification_records_input_text_hash", "verification_records", ["input_text_hash"], unique=False)

    op.create_table(
        "feedback_entries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("verification_record_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("verification_records.id", ondelete="CASCADE"), nullable=False),
        sa.Column("manifest_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("provenance_manifests.id"), nullable=True),
        sa.Column("feedback_type", feedback_type, nullable=False),
        sa.Column("feedback_text", sa.Text(), nullable=False),
        sa.Column("user_role", feedback_user_role, nullable=False),
        sa.Column("submitted_by", sa.String(length=255), nullable=False),
        sa.Column("linked_source_url", sa.String(length=2048), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("status", feedback_status, nullable=False),
        sa.Column("moderator_note", sa.Text(), nullable=True),
    )

    op.add_column("verification_jobs", sa.Column("verification_record_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        "fk_verification_jobs_verification_record_id",
        "verification_jobs",
        "verification_records",
        ["verification_record_id"],
        ["id"],
    )

    op.add_column("provenance_manifests", sa.Column("version_number", sa.Integer(), nullable=False, server_default="1"))


def downgrade() -> None:
    op.drop_column("provenance_manifests", "version_number")
    op.drop_constraint("fk_verification_jobs_verification_record_id", "verification_jobs", type_="foreignkey")
    op.drop_column("verification_jobs", "verification_record_id")
    op.drop_table("feedback_entries")
    op.drop_index("ix_verification_records_input_text_hash", table_name="verification_records")
    op.drop_table("verification_records")

    bind = op.get_bind()
    feedback_status.drop(bind, checkfirst=True)
    feedback_type.drop(bind, checkfirst=True)
    feedback_user_role.drop(bind, checkfirst=True)
