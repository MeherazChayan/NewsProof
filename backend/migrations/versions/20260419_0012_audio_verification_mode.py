"""add audio verification mode

Revision ID: 20260419_0012
Revises: 20260419_0011
Create Date: 2026-04-19 17:10:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260419_0012"
down_revision = "20260419_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE cryptoobjecttype ADD VALUE IF NOT EXISTS 'audio_verification_record'")
    feedback_type_enum = postgresql.ENUM(
        "correction",
        "dispute",
        "source_suggestion",
        "clarification",
        "review_note",
        name="feedbacktype",
        create_type=False,
    )
    feedback_user_role_enum = postgresql.ENUM(
        "user",
        "expert",
        "moderator",
        name="feedbackuserrole",
        create_type=False,
    )
    feedback_status_enum = postgresql.ENUM(
        "open",
        "reviewed",
        "resolved",
        name="feedbackstatus",
        create_type=False,
    )

    op.create_table(
        "audio_verification_records",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("audio_hash", sa.String(length=64), nullable=False),
        sa.Column("audio_fingerprint", sa.String(length=255), nullable=True),
        sa.Column("uploaded_filename", sa.String(length=255), nullable=False),
        sa.Column("caption_or_claim", sa.Text(), nullable=True),
        sa.Column("claimed_speaker", sa.String(length=255), nullable=True),
        sa.Column("transcript_text", sa.Text(), nullable=True),
        sa.Column("transcript_summary", sa.Text(), nullable=True),
        sa.Column("audio_analysis_result", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column("crypto_result", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column("final_score", sa.Float(), nullable=False),
        sa.Column("final_label", sa.String(length=255), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("current_manifest_id", sa.Uuid(), nullable=True),
        sa.Column("current_version", sa.Integer(), nullable=False),
        sa.Column("previous_record_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["current_manifest_id"], ["provenance_manifests.id"]),
        sa.ForeignKeyConstraint(["previous_record_id"], ["audio_verification_records.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_audio_verification_records_audio_hash"), "audio_verification_records", ["audio_hash"], unique=False)
    op.create_index(op.f("ix_audio_verification_records_audio_fingerprint"), "audio_verification_records", ["audio_fingerprint"], unique=False)

    op.create_table(
        "audio_feedback_entries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("audio_verification_record_id", sa.Uuid(), nullable=False),
        sa.Column("manifest_id", sa.Uuid(), nullable=True),
        sa.Column("feedback_type", feedback_type_enum, nullable=False),
        sa.Column("feedback_text", sa.Text(), nullable=False),
        sa.Column("user_role", feedback_user_role_enum, nullable=False),
        sa.Column("submitted_by", sa.String(length=255), nullable=False),
        sa.Column("linked_source_url", sa.String(length=2048), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("status", feedback_status_enum, nullable=False),
        sa.Column("moderator_note", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["audio_verification_record_id"], ["audio_verification_records.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["manifest_id"], ["provenance_manifests.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.add_column("verification_jobs", sa.Column("audio_verification_record_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_verification_jobs_audio_verification_record_id",
        "verification_jobs",
        "audio_verification_records",
        ["audio_verification_record_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_verification_jobs_audio_verification_record_id", "verification_jobs", type_="foreignkey")
    op.drop_column("verification_jobs", "audio_verification_record_id")
    op.drop_table("audio_feedback_entries")
    op.drop_index(op.f("ix_audio_verification_records_audio_fingerprint"), table_name="audio_verification_records")
    op.drop_index(op.f("ix_audio_verification_records_audio_hash"), table_name="audio_verification_records")
    op.drop_table("audio_verification_records")
