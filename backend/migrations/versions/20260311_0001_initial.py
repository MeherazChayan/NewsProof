"""initial schema"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260311_0001"
down_revision = None
branch_labels = None
depends_on = None


job_input_type = postgresql.ENUM("url", "text", "image", name="jobinputtype", create_type=False)
job_status = postgresql.ENUM("Verified", "Not Verifiable", "Verification Failed", "Timeout", name="jobstatus", create_type=False)


def upgrade() -> None:
    bind = op.get_bind()
    job_input_type.create(bind, checkfirst=True)
    job_status.create(bind, checkfirst=True)

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False, unique=True),
        sa.Column("reputation_score", sa.Float(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "trusted_publishers",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=255), nullable=False, unique=True),
        sa.Column("domain", sa.String(length=255), nullable=False, unique=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "verification_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("input_type", job_input_type, nullable=False),
        sa.Column("input_value", sa.Text(), nullable=True),
        sa.Column("status", job_status, nullable=False),
        sa.Column("headline_message", sa.String(length=255), nullable=False),
        sa.Column("short_reason", sa.Text(), nullable=False),
        sa.Column("content_preview", sa.JSON(), nullable=False),
        sa.Column("proof_summary", sa.JSON(), nullable=False),
        sa.Column("layer_results", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "public_keys",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("publisher_id", sa.Integer(), sa.ForeignKey("trusted_publishers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("key_id", sa.String(length=255), nullable=False),
        sa.Column("algorithm", sa.String(length=50), nullable=False),
        sa.Column("public_key", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("publisher_id", "key_id", name="uq_public_key_publisher_key"),
    )
    op.create_table(
        "content_records",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("publisher_id", sa.Integer(), sa.ForeignKey("trusted_publishers.id"), nullable=True),
        sa.Column("canonical_url", sa.String(length=2048), nullable=True),
        sa.Column("title", sa.String(length=512), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("signature", sa.Text(), nullable=True),
        sa.Column("key_id", sa.String(length=255), nullable=True),
        sa.Column("source_metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_content_records_content_hash", "content_records", ["content_hash"])
    op.create_table(
        "extracted_content",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("verification_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=True),
        sa.Column("author", sa.String(length=255), nullable=True),
        sa.Column("publish_time", sa.DateTime(), nullable=True),
        sa.Column("canonical_url", sa.String(length=2048), nullable=True),
        sa.Column("main_text", sa.Text(), nullable=False),
        sa.Column("hero_image_url", sa.String(length=2048), nullable=True),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
    )
    op.create_table(
        "uploaded_files",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("verification_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("stored_path", sa.String(length=1024), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "similarity_matches",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("verification_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("candidate_url", sa.String(length=2048), nullable=True),
        sa.Column("candidate_title", sa.String(length=512), nullable=True),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
    )
    op.create_table(
        "consensus_votes",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("verification_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("voter_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("vote_value", sa.Integer(), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("moderation_flag", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "audit_log_entries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("verification_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("event_payload", sa.JSON(), nullable=False),
        sa.Column("previous_hash", sa.String(length=64), nullable=True),
        sa.Column("entry_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("audit_log_entries")
    op.drop_table("consensus_votes")
    op.drop_table("similarity_matches")
    op.drop_table("uploaded_files")
    op.drop_table("extracted_content")
    op.drop_index("ix_content_records_content_hash", table_name="content_records")
    op.drop_table("content_records")
    op.drop_table("public_keys")
    op.drop_table("verification_jobs")
    op.drop_table("trusted_publishers")
    op.drop_table("users")

    bind = op.get_bind()
    job_status.drop(bind, checkfirst=True)
    job_input_type.drop(bind, checkfirst=True)
