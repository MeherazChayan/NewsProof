"""add image verification mode records and feedback

Revision ID: 20260419_0011
Revises: 20260419_0010
Create Date: 2026-04-19 10:15:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260419_0011"
down_revision = "20260419_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE cryptoobjecttype ADD VALUE IF NOT EXISTS 'image_verification_record'")

    op.create_table(
        "image_verification_records",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("image_hash", sa.String(length=64), nullable=False),
        sa.Column("perceptual_hash", sa.String(length=255), nullable=True),
        sa.Column("uploaded_filename", sa.String(length=255), nullable=False),
        sa.Column("caption_or_claim", sa.Text(), nullable=True),
        sa.Column("ocr_result", sa.JSON(), nullable=False),
        sa.Column("image_analysis_result", sa.JSON(), nullable=False),
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
        sa.ForeignKeyConstraint(["previous_record_id"], ["image_verification_records.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_image_verification_records_image_hash"), "image_verification_records", ["image_hash"], unique=False)
    op.create_index(op.f("ix_image_verification_records_perceptual_hash"), "image_verification_records", ["perceptual_hash"], unique=False)

    op.create_table(
        "image_feedback_entries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("image_verification_record_id", sa.Uuid(), nullable=False),
        sa.Column("manifest_id", sa.Uuid(), nullable=True),
        sa.Column("feedback_type", sa.Enum("correction", "dispute", "source_suggestion", "clarification", "review_note", name="feedbacktype", native_enum=False), nullable=False),
        sa.Column("feedback_text", sa.Text(), nullable=False),
        sa.Column("user_role", sa.Enum("user", "expert", "moderator", name="feedbackuserrole", native_enum=False), nullable=False),
        sa.Column("submitted_by", sa.String(length=255), nullable=False),
        sa.Column("linked_source_url", sa.String(length=2048), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("status", sa.Enum("open", "reviewed", "resolved", name="feedbackstatus", native_enum=False), nullable=False),
        sa.Column("moderator_note", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["image_verification_record_id"], ["image_verification_records.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["manifest_id"], ["provenance_manifests.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.add_column("verification_jobs", sa.Column("image_verification_record_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_verification_jobs_image_verification_record_id",
        "verification_jobs",
        "image_verification_records",
        ["image_verification_record_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_verification_jobs_image_verification_record_id", "verification_jobs", type_="foreignkey")
    op.drop_column("verification_jobs", "image_verification_record_id")
    op.drop_table("image_feedback_entries")
    op.drop_index(op.f("ix_image_verification_records_perceptual_hash"), table_name="image_verification_records")
    op.drop_index(op.f("ix_image_verification_records_image_hash"), table_name="image_verification_records")
    op.drop_table("image_verification_records")
