"""add video verification mode

Revision ID: 20260420_0013
Revises: 20260419_0012
Create Date: 2026-04-20 10:15:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260420_0013"
down_revision = "20260419_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    object_type_enum = postgresql.ENUM(
        "article_snapshot",
        "evidence_bundle",
        "review_event",
        "verification_record",
        "image_verification_record",
        "audio_verification_record",
        "video_verification_record",
        name="cryptoobjecttype",
        create_type=False,
    )
    object_type_enum.create(bind, checkfirst=True)

    op.create_table(
        "video_verification_records",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("video_hash", sa.String(length=64), nullable=False),
        sa.Column("video_fingerprint", sa.String(length=255), nullable=True),
        sa.Column("uploaded_filename", sa.String(length=255), nullable=False),
        sa.Column("caption_or_claim", sa.Text(), nullable=True),
        sa.Column("ocr_result", sa.JSON(), nullable=False),
        sa.Column("audio_result", sa.JSON(), nullable=False),
        sa.Column("video_analysis_result", sa.JSON(), nullable=False),
        sa.Column("crypto_result", sa.JSON(), nullable=False),
        sa.Column("final_score", sa.Float(), nullable=False),
        sa.Column("final_label", sa.String(length=255), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("current_manifest_id", sa.Uuid(), nullable=True),
        sa.Column("current_version", sa.Integer(), nullable=False),
        sa.Column("previous_record_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["current_manifest_id"], ["provenance_manifests.id"]),
        sa.ForeignKeyConstraint(["previous_record_id"], ["video_verification_records.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_video_verification_records_video_hash"), "video_verification_records", ["video_hash"], unique=False)
    op.create_index(op.f("ix_video_verification_records_video_fingerprint"), "video_verification_records", ["video_fingerprint"], unique=False)

    op.create_table(
        "video_feedback_entries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("video_verification_record_id", sa.Uuid(), nullable=False),
        sa.Column("manifest_id", sa.Uuid(), nullable=True),
        sa.Column(
            "feedback_type",
            postgresql.ENUM(
                "correction",
                "dispute",
                "source_suggestion",
                "clarification",
                "review_note",
                name="feedbacktype",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("feedback_text", sa.Text(), nullable=False),
        sa.Column(
            "user_role",
            postgresql.ENUM("user", "expert", "moderator", name="feedbackuserrole", create_type=False),
            nullable=False,
        ),
        sa.Column("submitted_by", sa.String(length=255), nullable=False),
        sa.Column("linked_source_url", sa.String(length=2048), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM("open", "reviewed", "resolved", name="feedbackstatus", create_type=False),
            nullable=False,
        ),
        sa.Column("moderator_note", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["manifest_id"], ["provenance_manifests.id"]),
        sa.ForeignKeyConstraint(["video_verification_record_id"], ["video_verification_records.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.add_column("verification_jobs", sa.Column("video_verification_record_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_verification_jobs_video_verification_record_id",
        "verification_jobs",
        "video_verification_records",
        ["video_verification_record_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_verification_jobs_video_verification_record_id", "verification_jobs", type_="foreignkey")
    op.drop_column("verification_jobs", "video_verification_record_id")
    op.drop_table("video_feedback_entries")
    op.drop_index(op.f("ix_video_verification_records_video_fingerprint"), table_name="video_verification_records")
    op.drop_index(op.f("ix_video_verification_records_video_hash"), table_name="video_verification_records")
    op.drop_table("video_verification_records")
