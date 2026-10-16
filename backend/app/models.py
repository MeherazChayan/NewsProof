from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class JobInputType(str, enum.Enum):
    url = "url"
    text = "text"
    image = "image"
    audio = "audio"
    video = "video"


class JobStatus(str, enum.Enum):
    verified = "Verified"
    likely_verified = "Likely Verified"
    not_verifiable = "Not Verifiable"
    unverified = "Unverified"
    timeout = "Timeout"


class ConsensusRole(str, enum.Enum):
    user = "user"
    fact_checker = "fact_checker"
    moderator = "moderator"


class FeedbackUserRole(str, enum.Enum):
    user = "user"
    expert = "expert"
    moderator = "moderator"


class FeedbackType(str, enum.Enum):
    correction = "correction"
    dispute = "dispute"
    source_suggestion = "source_suggestion"
    clarification = "clarification"
    review_note = "review_note"


class FeedbackStatus(str, enum.Enum):
    open = "open"
    reviewed = "reviewed"
    resolved = "resolved"


class IssuerType(str, enum.Enum):
    platform = "platform"
    publisher = "publisher"
    journalist = "journalist"
    fact_checker = "fact_checker"
    moderator = "moderator"


class CryptoObjectType(str, enum.Enum):
    article_snapshot = "article_snapshot"
    evidence_bundle = "evidence_bundle"
    review_event = "review_event"
    verification_record = "verification_record"
    image_verification_record = "image_verification_record"
    audio_verification_record = "audio_verification_record"
    video_verification_record = "video_verification_record"


class ReviewEventType(str, enum.Enum):
    fact_check = "fact_check"
    correction = "correction"
    retraction = "retraction"
    update = "update"
    moderation_note = "moderation_note"


class ProvenanceVerificationStatus(str, enum.Enum):
    valid = "valid"
    partial_provenance = "partial_provenance"
    no_provenance = "no_provenance"
    hash_mismatch = "hash_mismatch"
    signature_invalid = "signature_invalid"
    issuer_unknown = "issuer_unknown"
    issuer_revoked = "issuer_revoked"
    timestamp_invalid = "timestamp_invalid"
    version_chain_broken = "version_chain_broken"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    display_name: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[ConsensusRole] = mapped_column(
        Enum(ConsensusRole, values_callable=lambda enum_cls: [item.value for item in enum_cls]),
        default=ConsensusRole.user,
    )
    reputation_score: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TrustedPublisher(Base):
    __tablename__ = "trusted_publishers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    domain: Mapped[str] = mapped_column(String(255), unique=True)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    public_keys: Mapped[list["PublicKey"]] = relationship(back_populates="publisher", cascade="all, delete-orphan")
    content_records: Mapped[list["ContentRecord"]] = relationship(back_populates="publisher")


class PublicKey(Base):
    __tablename__ = "public_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    publisher_id: Mapped[int] = mapped_column(ForeignKey("trusted_publishers.id", ondelete="CASCADE"))
    key_id: Mapped[str] = mapped_column(String(255))
    algorithm: Mapped[str] = mapped_column(String(50), default="Ed25519")
    public_key: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    publisher: Mapped["TrustedPublisher"] = relationship(back_populates="public_keys")

    __table_args__ = (UniqueConstraint("publisher_id", "key_id", name="uq_public_key_publisher_key"),)


class ContentRecord(Base):
    __tablename__ = "content_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    publisher_id: Mapped[int | None] = mapped_column(ForeignKey("trusted_publishers.id"))
    canonical_url: Mapped[str | None] = mapped_column(String(2048))
    title: Mapped[str | None] = mapped_column(String(512))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    signature: Mapped[str | None] = mapped_column(Text)
    key_id: Mapped[str | None] = mapped_column(String(255))
    source_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    publisher: Mapped["TrustedPublisher"] = relationship(back_populates="content_records")


class VerificationJob(Base):
    __tablename__ = "verification_jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    input_type: Mapped[JobInputType] = mapped_column(
        Enum(JobInputType, values_callable=lambda enum_cls: [item.value for item in enum_cls])
    )
    input_value: Mapped[str | None] = mapped_column(Text)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, values_callable=lambda enum_cls: [item.value for item in enum_cls])
    )
    headline_message: Mapped[str] = mapped_column(String(255))
    short_reason: Mapped[str] = mapped_column(Text)
    content_preview: Mapped[dict] = mapped_column(JSON, default=dict)
    proof_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    analysis_report: Mapped[dict] = mapped_column(JSON, default=dict)
    layer_results: Mapped[dict] = mapped_column(JSON, default=dict)
    verification_record_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("verification_records.id"))
    image_verification_record_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("image_verification_records.id"))
    audio_verification_record_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("audio_verification_records.id"))
    video_verification_record_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("video_verification_records.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    extracted_content: Mapped["ExtractedContent"] = relationship(back_populates="job", uselist=False, cascade="all, delete-orphan")
    uploaded_file: Mapped["UploadedFile"] = relationship(back_populates="job", uselist=False, cascade="all, delete-orphan")
    similarity_matches: Mapped[list["SimilarityMatch"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    consensus_votes: Mapped[list["ConsensusVote"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    audit_entries: Mapped[list["AuditLogEntry"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    verification_record: Mapped["VerificationRecord | None"] = relationship(foreign_keys=[verification_record_id])
    image_verification_record: Mapped["ImageVerificationRecord | None"] = relationship(foreign_keys=[image_verification_record_id])
    audio_verification_record: Mapped["AudioVerificationRecord | None"] = relationship(foreign_keys=[audio_verification_record_id])
    video_verification_record: Mapped["VideoVerificationRecord | None"] = relationship(foreign_keys=[video_verification_record_id])


class ExtractedContent(Base):
    __tablename__ = "extracted_content"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("verification_jobs.id", ondelete="CASCADE"))
    title: Mapped[str | None] = mapped_column(String(512))
    # Some live-news pages emit extremely long multi-author bylines; keep this as Text so
    # article extraction cannot fail just because the publisher includes a long credit list.
    author: Mapped[str | None] = mapped_column(Text)
    publish_time: Mapped[datetime | None] = mapped_column(DateTime)
    canonical_url: Mapped[str | None] = mapped_column(String(2048))
    main_text: Mapped[str] = mapped_column(Text)
    hero_image_url: Mapped[str | None] = mapped_column(String(2048))
    normalized_text: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    job: Mapped["VerificationJob"] = relationship(back_populates="extracted_content")


class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("verification_jobs.id", ondelete="CASCADE"))
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(1024))
    mime_type: Mapped[str] = mapped_column(String(255))
    file_size: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    job: Mapped["VerificationJob"] = relationship(back_populates="uploaded_file")


class SimilarityMatch(Base):
    __tablename__ = "similarity_matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("verification_jobs.id", ondelete="CASCADE"))
    candidate_url: Mapped[str | None] = mapped_column(String(2048))
    candidate_title: Mapped[str | None] = mapped_column(String(512))
    score: Mapped[float] = mapped_column(Float)
    explanation: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    job: Mapped["VerificationJob"] = relationship(back_populates="similarity_matches")


class ConsensusVote(Base):
    __tablename__ = "consensus_votes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("verification_jobs.id", ondelete="CASCADE"))
    canonical_url: Mapped[str | None] = mapped_column(String(2048))
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    voter_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    reporter_name: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[ConsensusRole] = mapped_column(
        Enum(ConsensusRole, values_callable=lambda enum_cls: [item.value for item in enum_cls]),
        default=ConsensusRole.user,
    )
    vote_value: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    report_reason: Mapped[str | None] = mapped_column(Text)
    evidence_url: Mapped[str | None] = mapped_column(String(2048))
    moderation_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    job: Mapped["VerificationJob"] = relationship(back_populates="consensus_votes")


class AuditLogEntry(Base):
    __tablename__ = "audit_log_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("verification_jobs.id", ondelete="CASCADE"))
    event_type: Mapped[str] = mapped_column(String(100))
    event_payload: Mapped[dict] = mapped_column(JSON, default=dict)
    previous_hash: Mapped[str | None] = mapped_column(String(64))
    entry_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    job: Mapped["VerificationJob"] = relationship(back_populates="audit_entries")


class IssuerCredential(Base):
    __tablename__ = "issuers"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    issuer_type: Mapped[IssuerType] = mapped_column(
        Enum(IssuerType, values_callable=lambda enum_cls: [item.value for item in enum_cls])
    )
    issuer_name: Mapped[str] = mapped_column(String(255), unique=True)
    public_key: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(50), default="active")
    issued_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    manifests: Mapped[list["ProvenanceManifest"]] = relationship(back_populates="signer")


class ArticleSnapshot(Base):
    __tablename__ = "article_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_url: Mapped[str] = mapped_column(String(2048))
    publisher_name: Mapped[str] = mapped_column(String(255))
    # Live coverage and syndicated stories can contain very long multi-author bylines.
    author_name: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str] = mapped_column(String(512))
    published_at: Mapped[datetime | None] = mapped_column(DateTime)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    normalized_article_text: Mapped[str] = mapped_column(Text)
    media_urls: Mapped[list] = mapped_column(JSON, default=list)
    media_hashes: Mapped[list] = mapped_column(JSON, default=list)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    previous_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("article_snapshots.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    previous_version: Mapped[ArticleSnapshot | None] = relationship(remote_side=[id])


class EvidenceBundle(Base):
    __tablename__ = "evidence_bundles"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    linked_article_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("article_snapshots.id", ondelete="CASCADE"))
    source_links: Mapped[list] = mapped_column(JSON, default=list)
    extracted_claims: Mapped[list] = mapped_column(JSON, default=list)
    supporting_notes: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    previous_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("evidence_bundles.id"))

    linked_article: Mapped["ArticleSnapshot"] = relationship()
    previous_version: Mapped[EvidenceBundle | None] = relationship(remote_side=[id])


class ReviewEvent(Base):
    __tablename__ = "review_events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    linked_article_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("article_snapshots.id", ondelete="CASCADE"))
    event_type: Mapped[ReviewEventType] = mapped_column(
        Enum(ReviewEventType, values_callable=lambda enum_cls: [item.value for item in enum_cls])
    )
    event_payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_by: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    previous_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("review_events.id"))

    linked_article: Mapped["ArticleSnapshot"] = relationship()
    previous_version: Mapped[ReviewEvent | None] = relationship(remote_side=[id])


class VerificationRecord(Base):
    __tablename__ = "verification_records"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    input_text: Mapped[str] = mapped_column(Text)
    normalized_text: Mapped[str] = mapped_column(Text)
    input_text_hash: Mapped[str] = mapped_column(String(64), index=True)
    extracted_claims: Mapped[list] = mapped_column(JSON, default=list)
    evidence_sources: Mapped[list] = mapped_column(JSON, default=list)
    nlp_result: Mapped[dict] = mapped_column(JSON, default=dict)
    nlp_confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    nlp_label: Mapped[str] = mapped_column(String(255))
    current_manifest_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("provenance_manifests.id"))
    current_version: Mapped[int] = mapped_column(Integer, default=1)
    previous_record_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("verification_records.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    current_manifest: Mapped["ProvenanceManifest | None"] = relationship(foreign_keys=[current_manifest_id])
    previous_record: Mapped["VerificationRecord | None"] = relationship(remote_side=[id])
    feedback_entries: Mapped[list["FeedbackEntry"]] = relationship(back_populates="verification_record", cascade="all, delete-orphan")


class ImageVerificationRecord(Base):
    __tablename__ = "image_verification_records"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_hash: Mapped[str] = mapped_column(String(64), index=True)
    perceptual_hash: Mapped[str | None] = mapped_column(String(255), index=True)
    uploaded_filename: Mapped[str] = mapped_column(String(255))
    caption_or_claim: Mapped[str | None] = mapped_column(Text)
    ocr_result: Mapped[dict] = mapped_column(JSON, default=dict)
    image_analysis_result: Mapped[dict] = mapped_column(JSON, default=dict)
    crypto_result: Mapped[dict] = mapped_column(JSON, default=dict)
    final_score: Mapped[float] = mapped_column(Float, default=0.0)
    final_label: Mapped[str] = mapped_column(String(255))
    explanation: Mapped[str] = mapped_column(Text)
    current_manifest_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("provenance_manifests.id"))
    current_version: Mapped[int] = mapped_column(Integer, default=1)
    previous_record_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("image_verification_records.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    current_manifest: Mapped["ProvenanceManifest | None"] = relationship(foreign_keys=[current_manifest_id])
    previous_record: Mapped["ImageVerificationRecord | None"] = relationship(remote_side=[id])
    feedback_entries: Mapped[list["ImageFeedbackEntry"]] = relationship(back_populates="image_verification_record", cascade="all, delete-orphan")


class AudioVerificationRecord(Base):
    __tablename__ = "audio_verification_records"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audio_hash: Mapped[str] = mapped_column(String(64), index=True)
    audio_fingerprint: Mapped[str | None] = mapped_column(String(255), index=True)
    uploaded_filename: Mapped[str] = mapped_column(String(255))
    caption_or_claim: Mapped[str | None] = mapped_column(Text)
    claimed_speaker: Mapped[str | None] = mapped_column(String(255))
    transcript_text: Mapped[str | None] = mapped_column(Text)
    transcript_summary: Mapped[str | None] = mapped_column(Text)
    audio_analysis_result: Mapped[dict] = mapped_column(JSON, default=dict)
    crypto_result: Mapped[dict] = mapped_column(JSON, default=dict)
    final_score: Mapped[float] = mapped_column(Float, default=0.0)
    final_label: Mapped[str] = mapped_column(String(255))
    explanation: Mapped[str] = mapped_column(Text)
    current_manifest_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("provenance_manifests.id"))
    current_version: Mapped[int] = mapped_column(Integer, default=1)
    previous_record_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("audio_verification_records.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    current_manifest: Mapped["ProvenanceManifest | None"] = relationship(foreign_keys=[current_manifest_id])
    previous_record: Mapped["AudioVerificationRecord | None"] = relationship(remote_side=[id])
    feedback_entries: Mapped[list["AudioFeedbackEntry"]] = relationship(back_populates="audio_verification_record", cascade="all, delete-orphan")


class VideoVerificationRecord(Base):
    __tablename__ = "video_verification_records"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_hash: Mapped[str] = mapped_column(String(64), index=True)
    video_fingerprint: Mapped[str | None] = mapped_column(String(255), index=True)
    uploaded_filename: Mapped[str] = mapped_column(String(255))
    caption_or_claim: Mapped[str | None] = mapped_column(Text)
    ocr_result: Mapped[dict] = mapped_column(JSON, default=dict)
    audio_result: Mapped[dict] = mapped_column(JSON, default=dict)
    video_analysis_result: Mapped[dict] = mapped_column(JSON, default=dict)
    crypto_result: Mapped[dict] = mapped_column(JSON, default=dict)
    final_score: Mapped[float] = mapped_column(Float, default=0.0)
    final_label: Mapped[str] = mapped_column(String(255))
    explanation: Mapped[str] = mapped_column(Text)
    current_manifest_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("provenance_manifests.id"))
    current_version: Mapped[int] = mapped_column(Integer, default=1)
    previous_record_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("video_verification_records.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    current_manifest: Mapped["ProvenanceManifest | None"] = relationship(foreign_keys=[current_manifest_id])
    previous_record: Mapped["VideoVerificationRecord | None"] = relationship(remote_side=[id])
    feedback_entries: Mapped[list["VideoFeedbackEntry"]] = relationship(back_populates="video_verification_record", cascade="all, delete-orphan")


class FeedbackEntry(Base):
    __tablename__ = "feedback_entries"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    verification_record_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("verification_records.id", ondelete="CASCADE"))
    manifest_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("provenance_manifests.id"))
    feedback_type: Mapped[FeedbackType] = mapped_column(
        Enum(FeedbackType, values_callable=lambda enum_cls: [item.value for item in enum_cls])
    )
    feedback_text: Mapped[str] = mapped_column(Text)
    user_role: Mapped[FeedbackUserRole] = mapped_column(
        Enum(FeedbackUserRole, values_callable=lambda enum_cls: [item.value for item in enum_cls]),
        default=FeedbackUserRole.user,
    )
    submitted_by: Mapped[str] = mapped_column(String(255))
    linked_source_url: Mapped[str | None] = mapped_column(String(2048))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    status: Mapped[FeedbackStatus] = mapped_column(
        Enum(FeedbackStatus, values_callable=lambda enum_cls: [item.value for item in enum_cls]),
        default=FeedbackStatus.open,
    )
    moderator_note: Mapped[str | None] = mapped_column(Text)

    verification_record: Mapped["VerificationRecord"] = relationship(back_populates="feedback_entries")
    manifest: Mapped["ProvenanceManifest | None"] = relationship()


class ImageFeedbackEntry(Base):
    __tablename__ = "image_feedback_entries"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_verification_record_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("image_verification_records.id", ondelete="CASCADE"))
    manifest_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("provenance_manifests.id"))
    feedback_type: Mapped[FeedbackType] = mapped_column(
        Enum(FeedbackType, values_callable=lambda enum_cls: [item.value for item in enum_cls])
    )
    feedback_text: Mapped[str] = mapped_column(Text)
    user_role: Mapped[FeedbackUserRole] = mapped_column(
        Enum(FeedbackUserRole, values_callable=lambda enum_cls: [item.value for item in enum_cls]),
        default=FeedbackUserRole.user,
    )
    submitted_by: Mapped[str] = mapped_column(String(255))
    linked_source_url: Mapped[str | None] = mapped_column(String(2048))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    status: Mapped[FeedbackStatus] = mapped_column(
        Enum(FeedbackStatus, values_callable=lambda enum_cls: [item.value for item in enum_cls]),
        default=FeedbackStatus.open,
    )
    moderator_note: Mapped[str | None] = mapped_column(Text)

    image_verification_record: Mapped["ImageVerificationRecord"] = relationship(back_populates="feedback_entries")
    manifest: Mapped["ProvenanceManifest | None"] = relationship()


class AudioFeedbackEntry(Base):
    __tablename__ = "audio_feedback_entries"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audio_verification_record_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("audio_verification_records.id", ondelete="CASCADE"))
    manifest_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("provenance_manifests.id"))
    feedback_type: Mapped[FeedbackType] = mapped_column(
        Enum(FeedbackType, values_callable=lambda enum_cls: [item.value for item in enum_cls])
    )
    feedback_text: Mapped[str] = mapped_column(Text)
    user_role: Mapped[FeedbackUserRole] = mapped_column(
        Enum(FeedbackUserRole, values_callable=lambda enum_cls: [item.value for item in enum_cls]),
        default=FeedbackUserRole.user,
    )
    submitted_by: Mapped[str] = mapped_column(String(255))
    linked_source_url: Mapped[str | None] = mapped_column(String(2048))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    status: Mapped[FeedbackStatus] = mapped_column(
        Enum(FeedbackStatus, values_callable=lambda enum_cls: [item.value for item in enum_cls]),
        default=FeedbackStatus.open,
    )
    moderator_note: Mapped[str | None] = mapped_column(Text)

    audio_verification_record: Mapped["AudioVerificationRecord"] = relationship(back_populates="feedback_entries")
    manifest: Mapped["ProvenanceManifest | None"] = relationship()


class VideoFeedbackEntry(Base):
    __tablename__ = "video_feedback_entries"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_verification_record_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("video_verification_records.id", ondelete="CASCADE"))
    manifest_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("provenance_manifests.id"))
    feedback_type: Mapped[FeedbackType] = mapped_column(
        Enum(FeedbackType, values_callable=lambda enum_cls: [item.value for item in enum_cls])
    )
    feedback_text: Mapped[str] = mapped_column(Text)
    user_role: Mapped[FeedbackUserRole] = mapped_column(
        Enum(FeedbackUserRole, values_callable=lambda enum_cls: [item.value for item in enum_cls]),
        default=FeedbackUserRole.user,
    )
    submitted_by: Mapped[str] = mapped_column(String(255))
    linked_source_url: Mapped[str | None] = mapped_column(String(2048))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    status: Mapped[FeedbackStatus] = mapped_column(
        Enum(FeedbackStatus, values_callable=lambda enum_cls: [item.value for item in enum_cls]),
        default=FeedbackStatus.open,
    )
    moderator_note: Mapped[str | None] = mapped_column(Text)

    video_verification_record: Mapped["VideoVerificationRecord"] = relationship(back_populates="feedback_entries")
    manifest: Mapped["ProvenanceManifest | None"] = relationship()


class ProvenanceManifest(Base):
    __tablename__ = "provenance_manifests"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    object_type: Mapped[CryptoObjectType] = mapped_column(
        Enum(CryptoObjectType, values_callable=lambda enum_cls: [item.value for item in enum_cls])
    )
    object_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), index=True)
    canonical_json: Mapped[str] = mapped_column(Text)
    sha256_hash: Mapped[str] = mapped_column(String(64), index=True)
    signature: Mapped[str] = mapped_column(Text)
    signing_algorithm: Mapped[str] = mapped_column(String(50), default="Ed25519")
    signer_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("issuers.id"))
    signer_certificate_or_public_key_reference: Mapped[str] = mapped_column(Text)
    timestamp_token: Mapped[dict] = mapped_column(JSON, default=dict)
    version_number: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    previous_manifest_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("provenance_manifests.id"))
    verification_status: Mapped[ProvenanceVerificationStatus] = mapped_column(
        Enum(ProvenanceVerificationStatus, values_callable=lambda enum_cls: [item.value for item in enum_cls]),
        default=ProvenanceVerificationStatus.valid,
    )

    signer: Mapped["IssuerCredential"] = relationship(back_populates="manifests")
    previous_manifest: Mapped[ProvenanceManifest | None] = relationship(remote_side=[id])
    timestamp_logs: Mapped[list["TimestampLog"]] = relationship(back_populates="manifest", cascade="all, delete-orphan")
    verification_logs: Mapped[list["VerificationLog"]] = relationship(back_populates="manifest", cascade="all, delete-orphan")


class TimestampLog(Base):
    __tablename__ = "timestamp_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    manifest_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("provenance_manifests.id", ondelete="CASCADE"))
    provider_type: Mapped[str] = mapped_column(String(50))
    token: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    manifest: Mapped["ProvenanceManifest"] = relationship(back_populates="timestamp_logs")


class VerificationLog(Base):
    __tablename__ = "verification_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    manifest_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("provenance_manifests.id", ondelete="CASCADE"))
    verified_by: Mapped[str | None] = mapped_column(String(255))
    verification_status: Mapped[ProvenanceVerificationStatus] = mapped_column(
        Enum(ProvenanceVerificationStatus, values_callable=lambda enum_cls: [item.value for item in enum_cls])
    )
    result_json: Mapped[dict] = mapped_column("result", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    manifest: Mapped[ProvenanceManifest | None] = relationship(back_populates="verification_logs")
