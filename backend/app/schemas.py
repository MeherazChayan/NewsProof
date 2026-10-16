from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.models import (
    ConsensusRole,
    CryptoObjectType,
    FeedbackStatus,
    FeedbackType,
    FeedbackUserRole,
    IssuerType,
    JobStatus,
    ProvenanceVerificationStatus,
    ReviewEventType,
)


class UrlVerificationRequest(BaseModel):
    url: HttpUrl


class TextVerificationRequest(BaseModel):
    text: str = Field(min_length=10, max_length=50000)


class PublisherKeyPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    key_id: str
    algorithm: str = "Ed25519"
    public_key: str
    is_active: bool = True


class PublisherBase(BaseModel):
    name: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PublisherCreate(PublisherBase):
    public_keys: list[PublisherKeyPayload] = Field(default_factory=list)


class PublisherUpdate(PublisherBase):
    public_keys: list[PublisherKeyPayload] = Field(default_factory=list)


class PublisherResponse(PublisherBase):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int
    created_at: datetime
    updated_at: datetime
    public_keys: list[PublisherKeyPayload]
    metadata: dict[str, Any] = Field(default_factory=dict, validation_alias="metadata_json")


class ContentRecordBase(BaseModel):
    canonical_url: str | None = None
    title: str | None = None
    author: str | None = None
    publish_time: datetime | None = None
    normalized_text: str = Field(min_length=1)
    content_hash: str
    signature: str | None = None
    key_id: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class ContentRecordCreate(ContentRecordBase):
    publisher_id: int


class ContentRecordUpdate(ContentRecordBase):
    pass


class ContentRecordResponse(ContentRecordBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    publisher_id: int | None
    created_at: datetime


class AuditEntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_type: str
    event_payload: dict[str, Any]
    previous_hash: str | None
    entry_hash: str
    created_at: datetime


class LayerResult(BaseModel):
    status: str
    confidence: float | None = None
    explanation: str
    details: dict[str, Any] = Field(default_factory=dict)


class VerificationResultResponse(BaseModel):
    job_id: UUID
    input_type: str
    status: JobStatus
    headline_message: str
    short_reason: str
    content_preview: dict[str, Any]
    proof_summary: dict[str, Any]
    analysis_report: dict[str, Any]
    layer_results: dict[str, LayerResult | dict[str, Any]]
    audit_trail: list[AuditEntryResponse]
    created_at: datetime


class SimilarityMatchResponse(BaseModel):
    title: str | None = None
    url: str | None = None
    score: float
    explanation: str
    source: str = "trusted_corpus"
    snippet: str | None = None
    domain: str | None = None
    relation: str | None = None
    evidence_relation: str | None = None
    aboutness: str | None = None
    stance: str | None = None
    exact_claim_support: bool | None = None
    evidence_quote: str | None = None
    relation_confidence: float | None = None
    rerank_score: float | None = None
    embedding_similarity: float | None = None
    entity_match_score: float | None = None
    location_match_score: float | None = None
    location_match_pass: bool | None = None
    event_match_score: float | None = None
    context_match_pass: bool | None = None
    mismatch_reason: str | None = None
    reason_accepted_or_rejected: str | None = None


class SimilarityMatchesPageResponse(BaseModel):
    items: list[SimilarityMatchResponse]
    total: int
    offset: int
    limit: int
    has_more: bool


class ConsensusVoteRequest(BaseModel):
    reporter_name: str | None = None
    vote_value: int = Field(ge=-1, le=1)
    report_reason: str | None = Field(default=None, max_length=1000)


class ConsensusVoteResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    reporter_name: str | None
    role: ConsensusRole
    vote_value: int
    confidence: float
    weight: float
    report_reason: str | None
    evidence_url: str | None
    moderation_flag: bool
    created_at: datetime


class ConsensusSummaryResponse(BaseModel):
    job_id: UUID
    totals: dict[str, Any]
    votes: list[ConsensusVoteResponse]


class FeedbackEntryRequest(BaseModel):
    feedback_type: FeedbackType
    feedback_text: str = Field(min_length=3, max_length=5000)
    submitted_by: str = Field(min_length=1, max_length=255)
    user_role: FeedbackUserRole = FeedbackUserRole.user
    linked_source_url: str | None = Field(default=None, max_length=2048)
    moderator_note: str | None = Field(default=None, max_length=2000)


class FeedbackEntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    verification_record_id: UUID
    manifest_id: UUID | None
    feedback_type: FeedbackType
    feedback_text: str
    user_role: FeedbackUserRole
    submitted_by: str
    linked_source_url: str | None
    created_at: datetime
    status: FeedbackStatus
    moderator_note: str | None


class FeedbackSummaryResponse(BaseModel):
    job_id: UUID
    verification_record_id: UUID
    current_manifest_id: UUID | None
    totals: dict[str, Any]
    feedback_entries: list[FeedbackEntryResponse]


class VerificationRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    verification_record_id: UUID = Field(validation_alias="id")
    input_text: str
    normalized_text: str
    input_text_hash: str
    extracted_claims: list[Any]
    evidence_sources: list[Any]
    nlp_result: dict[str, Any]
    nlp_confidence_score: float
    nlp_label: str
    created_at: datetime
    updated_at: datetime
    current_manifest_id: UUID | None
    current_version: int
    previous_record_id: UUID | None


class HistoryItemResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    input_type: str
    headline_message: str
    created_at: datetime


class HistoryResponse(BaseModel):
    items: list[HistoryItemResponse]


class IssuerCredentialCreate(BaseModel):
    issuer_type: IssuerType
    issuer_name: str = Field(min_length=1, max_length=255)
    public_key: str = Field(min_length=1)
    issued_at: datetime | None = None
    expires_at: datetime | None = None


class IssuerCredentialResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    issuer_type: IssuerType
    issuer_name: str
    public_key: str
    status: str
    issued_at: datetime
    expires_at: datetime | None
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict, validation_alias="metadata_json")


class ArticleSnapshotCreate(BaseModel):
    source_url: HttpUrl
    publisher_name: str = Field(min_length=1, max_length=255)
    author_name: str | None = None
    title: str = Field(min_length=1, max_length=512)
    published_at: datetime | None = None
    captured_at: datetime | None = None
    normalized_article_text: str = Field(min_length=1)
    media_urls: list[HttpUrl] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    previous_version_id: UUID | None = None


class ArticleSnapshotResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    source_url: str
    publisher_name: str
    author_name: str | None
    title: str
    published_at: datetime | None
    captured_at: datetime
    normalized_article_text: str
    media_urls: list[Any]
    media_hashes: list[Any]
    metadata: dict[str, Any] = Field(default_factory=dict, validation_alias="metadata_json")
    previous_version_id: UUID | None
    created_at: datetime


class EvidenceBundleCreate(BaseModel):
    linked_article_id: UUID
    source_links: list[HttpUrl] = Field(default_factory=list)
    extracted_claims: list[str] = Field(default_factory=list)
    supporting_notes: str | None = None
    created_by: str = Field(min_length=1, max_length=255)
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    previous_version_id: UUID | None = None


class EvidenceBundleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    linked_article_id: UUID
    source_links: list[Any]
    extracted_claims: list[Any]
    supporting_notes: str | None
    created_by: str
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict, validation_alias="metadata_json")
    previous_version_id: UUID | None


class ReviewEventCreate(BaseModel):
    linked_article_id: UUID
    event_type: ReviewEventType
    event_payload: dict[str, Any] = Field(default_factory=dict)
    created_by: str = Field(min_length=1, max_length=255)
    created_at: datetime | None = None
    previous_version_id: UUID | None = None


class ReviewEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    linked_article_id: UUID
    event_type: ReviewEventType
    event_payload: dict[str, Any]
    created_by: str
    created_at: datetime
    previous_version_id: UUID | None


class ManifestSignRequest(BaseModel):
    object_type: CryptoObjectType
    object_id: UUID
    signer_id: UUID | None = None


class ManifestVerifyRequest(BaseModel):
    manifest_id: UUID
    verified_by: str | None = None


class ProvenanceManifestResponse(BaseModel):
    manifest_id: UUID
    object_type: CryptoObjectType
    object_id: UUID
    article: dict[str, Any]
    source: dict[str, Any]
    assets: list[dict[str, Any]]
    references: list[dict[str, Any]]
    transport_evidence: dict[str, Any]
    version_chain: dict[str, Any]
    canonicalization: dict[str, Any]
    crypto: dict[str, Any]
    verification: dict[str, Any]
    confidence: dict[str, Any]
    explanation: dict[str, Any]
    canonical_json: str
    created_at: datetime
    previous_manifest_id: UUID | None
    verification_status: ProvenanceVerificationStatus


class ManifestVerificationResponse(BaseModel):
    manifest_id: UUID
    object_type: CryptoObjectType
    object_id: UUID
    status: ProvenanceVerificationStatus
    valid: bool
    crypto: dict[str, Any]
    timestamps: dict[str, Any]
    version_chain: dict[str, Any]
    checks: dict[str, Any]
    factors: dict[str, Any]
    confidence: dict[str, Any]
    explanation: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)


class ProvenanceHistoryResponse(BaseModel):
    object_type: CryptoObjectType
    record_id: UUID
    versions: list[dict[str, Any]]
    manifests: list[dict[str, Any]]


class VersionCompareResponse(BaseModel):
    object_type: CryptoObjectType
    left_version_id: UUID
    right_version_id: UUID
    differences: list[dict[str, Any]]
    changed_fields: list[str]
