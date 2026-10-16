"""add crypto provenance layer tables"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260321_0005"
down_revision = "20260320_0004"
branch_labels = None
depends_on = None


issuer_type = postgresql.ENUM(
    "platform",
    "publisher",
    "journalist",
    "fact_checker",
    "moderator",
    name="issuertype",
    create_type=False,
)
crypto_object_type = postgresql.ENUM(
    "article_snapshot",
    "evidence_bundle",
    "review_event",
    name="cryptoobjecttype",
    create_type=False,
)
review_event_type = postgresql.ENUM(
    "fact_check",
    "correction",
    "retraction",
    "update",
    "moderation_note",
    name="revieweventtype",
    create_type=False,
)
provenance_verification_status = postgresql.ENUM(
    "valid",
    "hash_mismatch",
    "signature_invalid",
    "issuer_revoked",
    "timestamp_invalid",
    "version_chain_broken",
    name="provenanceverificationstatus",
    create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    issuer_type.create(bind, checkfirst=True)
    crypto_object_type.create(bind, checkfirst=True)
    review_event_type.create(bind, checkfirst=True)
    provenance_verification_status.create(bind, checkfirst=True)

    op.create_table(
        "issuers",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("issuer_type", issuer_type, nullable=False),
        sa.Column("issuer_name", sa.String(length=255), nullable=False, unique=True),
        sa.Column("public_key", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("issued_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "article_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source_url", sa.String(length=2048), nullable=False),
        sa.Column("publisher_name", sa.String(length=255), nullable=False),
        sa.Column("author_name", sa.String(length=255), nullable=True),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("published_at", sa.DateTime(), nullable=True),
        sa.Column("captured_at", sa.DateTime(), nullable=False),
        sa.Column("normalized_article_text", sa.Text(), nullable=False),
        sa.Column("media_urls", sa.JSON(), nullable=False),
        sa.Column("media_hashes", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("previous_version_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("article_snapshots.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "evidence_bundles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("linked_article_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("article_snapshots.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_links", sa.JSON(), nullable=False),
        sa.Column("extracted_claims", sa.JSON(), nullable=False),
        sa.Column("supporting_notes", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("previous_version_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("evidence_bundles.id"), nullable=True),
    )

    op.create_table(
        "review_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("linked_article_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("article_snapshots.id", ondelete="CASCADE"), nullable=False),
        sa.Column("event_type", review_event_type, nullable=False),
        sa.Column("event_payload", sa.JSON(), nullable=False),
        sa.Column("created_by", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("previous_version_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("review_events.id"), nullable=True),
    )

    op.create_table(
        "provenance_manifests",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("object_type", crypto_object_type, nullable=False),
        sa.Column("object_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("canonical_json", sa.Text(), nullable=False),
        sa.Column("sha256_hash", sa.String(length=64), nullable=False),
        sa.Column("signature", sa.Text(), nullable=False),
        sa.Column("signing_algorithm", sa.String(length=50), nullable=False),
        sa.Column("signer_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("issuers.id"), nullable=False),
        sa.Column("signer_certificate_or_public_key_reference", sa.Text(), nullable=False),
        sa.Column("timestamp_token", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("previous_manifest_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("provenance_manifests.id"), nullable=True),
        sa.Column("verification_status", provenance_verification_status, nullable=False),
    )
    op.create_index("ix_provenance_manifests_object_id", "provenance_manifests", ["object_id"])
    op.create_index("ix_provenance_manifests_sha256_hash", "provenance_manifests", ["sha256_hash"])

    op.create_table(
        "timestamp_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("manifest_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("provenance_manifests.id", ondelete="CASCADE"), nullable=False),
        sa.Column("provider_type", sa.String(length=50), nullable=False),
        sa.Column("token", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "verification_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("manifest_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("provenance_manifests.id", ondelete="CASCADE"), nullable=True),
        sa.Column("verified_by", sa.String(length=255), nullable=True),
        sa.Column("verification_status", provenance_verification_status, nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("verification_logs")
    op.drop_table("timestamp_logs")
    op.drop_index("ix_provenance_manifests_sha256_hash", table_name="provenance_manifests")
    op.drop_index("ix_provenance_manifests_object_id", table_name="provenance_manifests")
    op.drop_table("provenance_manifests")
    op.drop_table("review_events")
    op.drop_table("evidence_bundles")
    op.drop_table("article_snapshots")
    op.drop_table("issuers")

    bind = op.get_bind()
    provenance_verification_status.drop(bind, checkfirst=True)
    review_event_type.drop(bind, checkfirst=True)
    crypto_object_type.drop(bind, checkfirst=True)
    issuer_type.drop(bind, checkfirst=True)
