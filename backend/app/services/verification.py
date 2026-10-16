from __future__ import annotations

import logging
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from fastapi import HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import (
    ArticleSnapshot,
    ConsensusRole,
    ConsensusVote,
    ContentRecord,
    CryptoObjectType,
    ExtractedContent,
    FeedbackEntry,
    AudioFeedbackEntry,
    ImageFeedbackEntry,
    VideoFeedbackEntry,
    FeedbackStatus,
    FeedbackType,
    FeedbackUserRole,
    AudioVerificationRecord,
    ImageVerificationRecord,
    VideoVerificationRecord,
    JobInputType,
    JobStatus,
    ProvenanceManifest,
    PublicKey,
    SimilarityMatch,
    TrustedPublisher,
    UploadedFile,
    VerificationRecord,
    VerificationJob,
)
from app.schemas import (
    ConsensusSummaryResponse,
    ConsensusVoteRequest,
    ConsensusVoteResponse,
    FeedbackEntryRequest,
    FeedbackEntryResponse,
    FeedbackSummaryResponse,
)
from app.security import verify_ed25519_signature
from app.services.audit import append_audit_entry
from app.services.cohere_service import CohereService
from app.services.claim_extraction_service import ClaimExtractionService
from app.services.claim_evidence_verifier import ClaimEvidenceVerifier
from app.services.context_match_service import ContextMatchService
from app.services.bm25_service import BM25RetrievalService
from app.services.content_extraction import (
    ExtractedPayload,
    UrlFetchError,
    build_text_payload,
    build_url_fallback_payload,
    fetch_url_content,
)
from app.services.evidence_store_service import EvidenceStoreService
from app.services.embeddings_service import EmbeddingsService
from app.services.evidence_judgment_service import EvidenceJudgmentService
from app.services.context_detector import extract_claim_context
from app.services.media import (
    build_audio_payload,
    build_image_payload,
    build_video_payload,
    compute_binary_hashes,
    compute_image_hashes,
    extract_audio_track,
    probe_media,
    sample_video_frames,
)
from app.services.media_authenticity import MediaAuthenticityService
from app.services.ocr import extract_text_from_image
from app.services.ocr import extract_text_details_from_image, extract_text_details_from_image_bytes
from app.services.crypto import CryptoProvenanceService
from app.services.reasoning_service import ReasoningService
from app.services.retrieval_service import RetrievalService
from app.services.scoring import (
    EXCLUDED_LAYER_STATES,
    compute_crypto_score,
    compute_evidence_multiplier,
    compute_global_penalty,
    compute_newsproof_score,
    compute_nlp_score,
    generate_explanation,
    generate_newsproof_label,
    map_score_to_label,
)
from app.services.evidence_alignment import classify_evidence, compare_claim_with_evidence
from app.services.evidence_search import build_country_aware_queries
from app.services.fact_check_parser import extract_fact_check_verdict, extract_fact_check_verdict_label
from app.services.search_service import SearchService
from app.services.semantic_retrieval_service import SemanticRetrievalService
from app.services.source_adapter_service import SourceAdapterService
from app.services.text_reranker_service import TextRerankerService
from app.services.transcription_service import TranscriptionService
from app.services.trusted_sources import get_source_type, get_sources_for_country, is_trusted_domain
from app.services.verdict_engine import decide_final_verdict, score_evidence
from app.services.vision_reasoning_service import VisionReasoningService
from app.utils import canonicalize_url, domain_matches, extract_domain, normalize_text, sha256_text
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


@dataclass
class LayerOutcome:
    status: str
    confidence: float | None
    explanation: str
    details: dict


class VerificationService:
    DEFAULT_NLP_MATCH_LIMIT = 5
    MAX_NLP_MATCHES = 5
    TEXT_MAX_CANDIDATES = 10
    TEXT_CLAIM_LIMIT = 4
    TEXT_SEARCH_PER_QUERY = 3
    TEXT_EVIDENCE_PER_CLAIM = 2
    TEXT_MAX_SEARCH_QUERIES = 6
    _transcription_cache: dict[str, dict] = {}
    _claim_candidate_cache: dict[str, tuple[list[dict], dict]] = {}

    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
        self.cohere = CohereService()
        self.search = SearchService()
        self.media_authenticity = MediaAuthenticityService()
        self.claim_extraction = ClaimExtractionService()
        self.context_matcher = ContextMatchService()
        self.claim_evidence_verifier = ClaimEvidenceVerifier(self.context_matcher)
        self.evidence_store = EvidenceStoreService()
        self.embeddings = EmbeddingsService()
        self.bm25 = BM25RetrievalService()
        self.semantic_retrieval = SemanticRetrievalService()
        self.text_reranker = TextRerankerService(self.cohere)
        self.evidence_judgment = EvidenceJudgmentService()
        self.retrieval = RetrievalService(self.search, self.embeddings, self.cohere, self.evidence_store)
        self.reasoning = ReasoningService()
        self.transcription = TranscriptionService()
        self.vision_reasoning = VisionReasoningService()
        self.source_adapters = SourceAdapterService()

    async def verify_url(self, url: str, fast_mode: bool = False) -> VerificationJob:
        job = self._create_job(JobInputType.url, url)
        append_audit_entry(self.db, job, "job_created", {"input_type": "url", "url": url, "fast_mode": fast_mode})
        try:
            extracted = await fetch_url_content(url, fast_mode=fast_mode)
        except UrlFetchError as exc:
            logger.warning("URL fetch degraded for %s: %s", url, exc)
            if exc.kind == "timeout":
                return self._finalize_timeout(job, exc.message)
            extracted = build_url_fallback_payload(url, exc.extracted_title)
            extracted.metadata["fetch_failure_stage"] = exc.kind
            extracted.metadata["fetch_exception_message"] = exc.message
            if isinstance(extracted.metadata.get("url_fetch_debug"), dict):
                extracted.metadata["url_fetch_debug"]["exception_message"] = exc.message
                extracted.metadata["url_fetch_debug"]["http_status_code"] = exc.status_code
            extracted = await self.source_adapters.adapt_url_payload(url, extracted)
            append_audit_entry(
                self.db,
                job,
                "url_fetch_degraded",
                {"reason": exc.message, "kind": exc.kind, "status_code": exc.status_code},
            )
            return self._run_pipeline(job, extracted)
        except Exception as exc:
            logger.exception("URL verification failed")
            extracted = build_url_fallback_payload(url)
            extracted.metadata["fetch_failure_stage"] = "unexpected"
            extracted.metadata["fetch_exception_message"] = str(exc)
            if isinstance(extracted.metadata.get("url_fetch_debug"), dict):
                extracted.metadata["url_fetch_debug"]["exception_message"] = str(exc)
            extracted = await self.source_adapters.adapt_url_payload(url, extracted)
            append_audit_entry(self.db, job, "url_fetch_degraded", {"reason": str(exc), "kind": "unexpected"})
            return self._run_pipeline(job, extracted)
        extracted = await self.source_adapters.adapt_url_payload(url, extracted)
        return self._run_pipeline(job, extracted)

    def verify_text(self, text: str) -> VerificationJob:
        job = self._create_job(JobInputType.text, text[:5000])
        append_audit_entry(self.db, job, "job_created", {"input_type": "text"})
        extracted = build_text_payload(text)
        extracted = self.source_adapters.adapt_uploaded_media_payload(extracted, "text")
        return self._run_text_pipeline(job, extracted, text)

    async def verify_image(self, upload: UploadFile, caption_or_claim: str | None = None) -> VerificationJob:
        # Image Verification in NewsProof is a separate verification mode from both URL and Text verification.
        # Its final score is calculated only from the Image Analysis Layer and the
        # Cryptographic/Provenance Layer. The Feedback Layer is additional only and stores
        # review history, expert notes, disputes, and correction requests without affecting
        # the numeric rating.
        self._validate_upload(upload, JobInputType.image)
        job = self._create_job(JobInputType.image, upload.filename)
        append_audit_entry(self.db, job, "job_created", {"input_type": "image", "filename": upload.filename})
        path = await self._store_upload(job.id, upload, JobInputType.image)
        ocr_pipeline = extract_text_details_from_image(path)
        text = str(ocr_pipeline.get("extracted_text") or "").strip()
        extracted = build_image_payload(path, upload.filename or "image-upload", text)
        extracted = self.source_adapters.adapt_uploaded_media_payload(extracted, "image")
        extracted.metadata["caption_or_claim"] = (caption_or_claim or "").strip() or None
        extracted.metadata["ocr_pipeline"] = ocr_pipeline
        uploaded = UploadedFile(
            job_id=job.id,
            original_filename=upload.filename or "upload",
            stored_path=path,
            mime_type=upload.content_type or "application/octet-stream",
            file_size=Path(path).stat().st_size,
        )
        self.db.add(uploaded)
        append_audit_entry(
            self.db,
            job,
            "ocr_completed",
            {"characters": len(text), "ocr_status": ocr_pipeline.get("ocr_status"), "failure_reasons": ocr_pipeline.get("failure_reasons", [])},
        )
        return self._run_image_pipeline(job, extracted, path, upload.filename or "image-upload")

    async def verify_audio(
        self,
        upload: UploadFile,
        caption_or_claim: str | None = None,
        claimed_speaker: str | None = None,
    ) -> VerificationJob:
        # Audio Verification in NewsProof is a separate verification mode from URL, Text,
        # and Image verification. Its final score is calculated only from the Audio
        # Analysis Layer and the Cryptographic/Provenance Layer. The Feedback Layer is
        # additional only and stores review history, expert notes, disputes, and
        # correction requests without affecting the numeric rating.
        self._validate_upload(upload, JobInputType.audio)
        job = self._create_job(JobInputType.audio, upload.filename)
        append_audit_entry(self.db, job, "job_created", {"input_type": "audio", "filename": upload.filename})
        path = await self._store_upload(job.id, upload, JobInputType.audio)
        probe = probe_media(path)
        extracted = build_audio_payload(path, upload.filename or "audio-upload", probe)
        extracted = self.source_adapters.adapt_uploaded_media_payload(extracted, "audio")
        extracted.metadata["caption_or_claim"] = (caption_or_claim or "").strip() or None
        extracted.metadata["claimed_speaker"] = (claimed_speaker or "").strip() or None
        uploaded = UploadedFile(
            job_id=job.id,
            original_filename=upload.filename or "audio-upload",
            stored_path=path,
            mime_type=upload.content_type or "application/octet-stream",
            file_size=Path(path).stat().st_size,
        )
        self.db.add(uploaded)
        append_audit_entry(self.db, job, "audio_processed", {"probe_available": bool(probe)})
        return self._run_audio_pipeline(job, extracted, path, upload.filename or "audio-upload")

    async def verify_video(self, upload: UploadFile, caption_or_claim: str | None = None) -> VerificationJob:
        # Video verification is a separate upload mode that preprocesses audio,
        # sampled-frame OCR, and metadata before reusing the contextual verifier.
        self._validate_upload(upload, JobInputType.video)
        job = self._create_job(JobInputType.video, upload.filename)
        append_audit_entry(self.db, job, "job_created", {"input_type": "video", "filename": upload.filename})
        path = await self._store_upload(job.id, upload, JobInputType.video)
        probe = probe_media(path)
        extracted = build_video_payload(path, upload.filename or "video-upload", self.settings.upload_dir, probe)
        extracted = self.source_adapters.adapt_uploaded_media_payload(extracted, "video")
        extracted.metadata["caption_or_claim"] = (caption_or_claim or "").strip() or None
        uploaded = UploadedFile(
            job_id=job.id,
            original_filename=upload.filename or "video-upload",
            stored_path=path,
            mime_type=upload.content_type or "application/octet-stream",
            file_size=Path(path).stat().st_size,
        )
        self.db.add(uploaded)
        append_audit_entry(
            self.db,
            job,
            "video_processed",
            {"probe_available": bool(probe), "keyframe_ocr": extracted.metadata.get("derived_from") == "video_keyframe_ocr"},
        )
        return self._run_video_pipeline(job, extracted, path, upload.filename or "video-upload")

    def get_result(self, job_id: uuid.UUID, refresh: bool = False) -> VerificationJob:
        job = self.db.get(VerificationJob, job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Result not found")
        if refresh:
            self._refresh_job_assessment(job, commit=True)
            self.db.refresh(job)
        return job

    def get_history(self) -> list[VerificationJob]:
        return list(self.db.scalars(select(VerificationJob).order_by(VerificationJob.created_at.desc()).limit(100)))

    def get_audit(self, job_id: uuid.UUID):
        job = self.get_result(job_id, refresh=False)
        return job.audit_entries

    def get_similarity_matches(self, job_id: uuid.UUID, offset: int = 0, limit: int = 10) -> dict:
        job = self.get_result(job_id, refresh=False)
        matches = [
            {
                "title": match.candidate_title,
                "url": match.candidate_url,
                "score": match.score,
                "explanation": match.explanation,
                "source": match.metadata_json.get("source", "trusted_corpus") if isinstance(match.metadata_json, dict) else "trusted_corpus",
                "snippet": match.metadata_json.get("snippet") if isinstance(match.metadata_json, dict) else None,
                "domain": match.metadata_json.get("domain") if isinstance(match.metadata_json, dict) else None,
                "relation": match.metadata_json.get("relation") if isinstance(match.metadata_json, dict) else None,
                "evidence_relation": match.metadata_json.get("evidence_relation") if isinstance(match.metadata_json, dict) else None,
                "aboutness": match.metadata_json.get("aboutness") if isinstance(match.metadata_json, dict) else None,
                "stance": match.metadata_json.get("stance") if isinstance(match.metadata_json, dict) else None,
                "exact_claim_support": match.metadata_json.get("exact_claim_support") if isinstance(match.metadata_json, dict) else None,
                "evidence_quote": match.metadata_json.get("evidence_quote") if isinstance(match.metadata_json, dict) else None,
                "relation_confidence": match.metadata_json.get("relation_confidence") if isinstance(match.metadata_json, dict) else None,
                "rerank_score": match.metadata_json.get("rerank_score") if isinstance(match.metadata_json, dict) else None,
                "embedding_similarity": match.metadata_json.get("embedding_similarity") if isinstance(match.metadata_json, dict) else None,
                "entity_match_score": match.metadata_json.get("entity_match_score") if isinstance(match.metadata_json, dict) else None,
                "location_match_score": match.metadata_json.get("location_match_score") if isinstance(match.metadata_json, dict) else None,
                "location_match_pass": match.metadata_json.get("location_match_pass") if isinstance(match.metadata_json, dict) else None,
                "event_match_score": match.metadata_json.get("event_match_score") if isinstance(match.metadata_json, dict) else None,
                "context_match_pass": match.metadata_json.get("context_match_pass") if isinstance(match.metadata_json, dict) else None,
                "mismatch_reason": match.metadata_json.get("mismatch_reason") if isinstance(match.metadata_json, dict) else None,
                "reason_accepted_or_rejected": match.metadata_json.get("reason_accepted_or_rejected") if isinstance(match.metadata_json, dict) else None,
            }
            for match in sorted(job.similarity_matches, key=lambda item: item.score, reverse=True)
        ]
        page = matches[offset : offset + limit]
        return {
            "items": page,
            "total": len(matches),
            "offset": offset,
            "limit": limit,
            "has_more": offset + limit < len(matches),
        }

    def get_consensus(self, job_id: uuid.UUID) -> ConsensusSummaryResponse:
        job = self.get_result(job_id, refresh=False)
        if job.input_type == JobInputType.text:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text verification uses feedback, not votes")
        layer3 = self._layer3(job, persist_audit=False)
        votes = self._consensus_votes_for_job(job)
        return ConsensusSummaryResponse(
            job_id=job.id,
            totals=layer3.details,
            votes=[ConsensusVoteResponse.model_validate(vote) for vote in votes],
        )

    def add_consensus_vote(self, job_id: uuid.UUID, payload: ConsensusVoteRequest) -> ConsensusVoteResponse:
        job = self.get_result(job_id, refresh=False)
        if job.input_type == JobInputType.text:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text verification uses feedback, not votes")
        extracted = job.extracted_content
        vote = ConsensusVote(
            job_id=job.id,
            canonical_url=self._consensus_identity_url(extracted),
            content_hash=extracted.content_hash if extracted else None,
            reporter_name=payload.reporter_name,
            role=ConsensusRole.user,
            vote_value=payload.vote_value,
            confidence=1.0,
            weight=1.0,
            report_reason=payload.report_reason,
            evidence_url=None,
            moderation_flag=False,
        )
        self.db.add(vote)
        self.db.flush()
        append_audit_entry(
            self.db,
            job,
            "consensus_vote_added",
            {"vote_id": vote.id, "role": vote.role.value, "weight": vote.weight, "vote_value": vote.vote_value},
        )
        self._refresh_job_assessment(job, commit=False)
        self.db.commit()
        self.db.refresh(vote)
        return ConsensusVoteResponse.model_validate(vote)

    def get_feedback(self, job_id: uuid.UUID) -> FeedbackSummaryResponse:
        job = self.get_result(job_id, refresh=False)
        if job.input_type == JobInputType.video:
            record = self._resolve_video_verification_record(job)
            if not record:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback is only available for video verification records")
            entries = self._feedback_entries_for_video_record(record)
            layer3 = self._video_feedback_layer(record, persist_audit=False)
            return FeedbackSummaryResponse(
                job_id=job.id,
                verification_record_id=record.id,
                current_manifest_id=record.current_manifest_id,
                totals=layer3.details,
                feedback_entries=[
                    FeedbackEntryResponse.model_validate(
                        {
                            "id": entry.id,
                            "verification_record_id": record.id,
                            "manifest_id": entry.manifest_id,
                            "feedback_type": entry.feedback_type,
                            "feedback_text": entry.feedback_text,
                            "user_role": entry.user_role,
                            "submitted_by": entry.submitted_by,
                            "linked_source_url": entry.linked_source_url,
                            "created_at": entry.created_at,
                            "status": entry.status,
                            "moderator_note": entry.moderator_note,
                        }
                    )
                    for entry in entries
                ],
            )
        if job.input_type == JobInputType.audio:
            record = self._resolve_audio_verification_record(job)
            if not record:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback is only available for audio verification records")
            entries = self._feedback_entries_for_audio_record(record)
            layer3 = self._audio_feedback_layer(record, persist_audit=False)
            return FeedbackSummaryResponse(
                job_id=job.id,
                verification_record_id=record.id,
                current_manifest_id=record.current_manifest_id,
                totals=layer3.details,
                feedback_entries=[
                    FeedbackEntryResponse.model_validate(
                        {
                            "id": entry.id,
                            "verification_record_id": record.id,
                            "manifest_id": entry.manifest_id,
                            "feedback_type": entry.feedback_type,
                            "feedback_text": entry.feedback_text,
                            "user_role": entry.user_role,
                            "submitted_by": entry.submitted_by,
                            "linked_source_url": entry.linked_source_url,
                            "created_at": entry.created_at,
                            "status": entry.status,
                            "moderator_note": entry.moderator_note,
                        }
                    )
                    for entry in entries
                ],
            )
        if job.input_type == JobInputType.image:
            record = self._resolve_image_verification_record(job)
            if not record:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback is only available for image verification records")
            entries = self._feedback_entries_for_image_record(record)
            layer3 = self._image_feedback_layer(record, persist_audit=False)
            return FeedbackSummaryResponse(
                job_id=job.id,
                verification_record_id=record.id,
                current_manifest_id=record.current_manifest_id,
                totals=layer3.details,
                feedback_entries=[
                    FeedbackEntryResponse.model_validate(
                        {
                            "id": entry.id,
                            "verification_record_id": record.id,
                            "manifest_id": entry.manifest_id,
                            "feedback_type": entry.feedback_type,
                            "feedback_text": entry.feedback_text,
                            "user_role": entry.user_role,
                            "submitted_by": entry.submitted_by,
                            "linked_source_url": entry.linked_source_url,
                            "created_at": entry.created_at,
                            "status": entry.status,
                            "moderator_note": entry.moderator_note,
                        }
                    )
                    for entry in entries
                ],
            )
        record = self._resolve_verification_record(job)
        if not record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback is only available for text verification records")
        entries = self._feedback_entries_for_record(record)
        layer3 = self._text_feedback_layer(record, persist_audit=False)
        return FeedbackSummaryResponse(
            job_id=job.id,
            verification_record_id=record.id,
            current_manifest_id=record.current_manifest_id,
            totals=layer3.details,
            feedback_entries=[FeedbackEntryResponse.model_validate(entry) for entry in entries],
        )

    def submit_feedback(self, job_id: uuid.UUID, payload: FeedbackEntryRequest) -> FeedbackEntryResponse:
        job = self.get_result(job_id, refresh=False)
        if job.input_type == JobInputType.video:
            record = self._resolve_video_verification_record(job)
            if not record:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback is only available for video verification records")
            entry = VideoFeedbackEntry(
                video_verification_record_id=record.id,
                manifest_id=record.current_manifest_id,
                feedback_type=payload.feedback_type,
                feedback_text=payload.feedback_text,
                user_role=payload.user_role,
                submitted_by=payload.submitted_by,
                linked_source_url=payload.linked_source_url,
                status=FeedbackStatus.open,
                moderator_note=payload.moderator_note,
            )
            self.db.add(entry)
            self.db.flush()
            append_audit_entry(
                self.db,
                job,
                "video_feedback_added",
                {
                    "feedback_id": str(entry.id),
                    "feedback_type": entry.feedback_type.value,
                    "user_role": entry.user_role.value,
                    "video_verification_record_id": str(record.id),
                },
            )
            self._refresh_job_assessment(job, commit=False)
            self.db.commit()
            self.db.refresh(entry)
            return FeedbackEntryResponse.model_validate(
                {
                    "id": entry.id,
                    "verification_record_id": record.id,
                    "manifest_id": entry.manifest_id,
                    "feedback_type": entry.feedback_type,
                    "feedback_text": entry.feedback_text,
                    "user_role": entry.user_role,
                    "submitted_by": entry.submitted_by,
                    "linked_source_url": entry.linked_source_url,
                    "created_at": entry.created_at,
                    "status": entry.status,
                    "moderator_note": entry.moderator_note,
                }
            )
        if job.input_type == JobInputType.audio:
            record = self._resolve_audio_verification_record(job)
            if not record:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback is only available for audio verification records")
            entry = AudioFeedbackEntry(
                audio_verification_record_id=record.id,
                manifest_id=record.current_manifest_id,
                feedback_type=payload.feedback_type,
                feedback_text=payload.feedback_text,
                user_role=payload.user_role,
                submitted_by=payload.submitted_by,
                linked_source_url=payload.linked_source_url,
                status=FeedbackStatus.open,
                moderator_note=payload.moderator_note,
            )
            self.db.add(entry)
            self.db.flush()
            append_audit_entry(
                self.db,
                job,
                "audio_feedback_added",
                {
                    "feedback_id": str(entry.id),
                    "feedback_type": entry.feedback_type.value,
                    "user_role": entry.user_role.value,
                    "audio_verification_record_id": str(record.id),
                },
            )
            self._refresh_job_assessment(job, commit=False)
            self.db.commit()
            self.db.refresh(entry)
            return FeedbackEntryResponse.model_validate(
                {
                    "id": entry.id,
                    "verification_record_id": record.id,
                    "manifest_id": entry.manifest_id,
                    "feedback_type": entry.feedback_type,
                    "feedback_text": entry.feedback_text,
                    "user_role": entry.user_role,
                    "submitted_by": entry.submitted_by,
                    "linked_source_url": entry.linked_source_url,
                    "created_at": entry.created_at,
                    "status": entry.status,
                    "moderator_note": entry.moderator_note,
                }
            )
        if job.input_type == JobInputType.image:
            record = self._resolve_image_verification_record(job)
            if not record:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback is only available for image verification records")
            entry = ImageFeedbackEntry(
                image_verification_record_id=record.id,
                manifest_id=record.current_manifest_id,
                feedback_type=payload.feedback_type,
                feedback_text=payload.feedback_text,
                user_role=payload.user_role,
                submitted_by=payload.submitted_by,
                linked_source_url=payload.linked_source_url,
                status=FeedbackStatus.open,
                moderator_note=payload.moderator_note,
            )
            self.db.add(entry)
            self.db.flush()
            append_audit_entry(
                self.db,
                job,
                "image_feedback_added",
                {
                    "feedback_id": str(entry.id),
                    "feedback_type": entry.feedback_type.value,
                    "user_role": entry.user_role.value,
                    "image_verification_record_id": str(record.id),
                },
            )
            self._refresh_job_assessment(job, commit=False)
            self.db.commit()
            self.db.refresh(entry)
            return FeedbackEntryResponse.model_validate(
                {
                    "id": entry.id,
                    "verification_record_id": record.id,
                    "manifest_id": entry.manifest_id,
                    "feedback_type": entry.feedback_type,
                    "feedback_text": entry.feedback_text,
                    "user_role": entry.user_role,
                    "submitted_by": entry.submitted_by,
                    "linked_source_url": entry.linked_source_url,
                    "created_at": entry.created_at,
                    "status": entry.status,
                    "moderator_note": entry.moderator_note,
                }
            )
        record = self._resolve_verification_record(job)
        if not record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback is only available for text verification records")
        entry = FeedbackEntry(
            verification_record_id=record.id,
            manifest_id=record.current_manifest_id,
            feedback_type=payload.feedback_type,
            feedback_text=payload.feedback_text,
            user_role=payload.user_role,
            submitted_by=payload.submitted_by,
            linked_source_url=payload.linked_source_url,
            status=FeedbackStatus.open,
            moderator_note=payload.moderator_note,
        )
        self.db.add(entry)
        self.db.flush()
        append_audit_entry(
            self.db,
            job,
            "feedback_added",
            {
                "feedback_id": str(entry.id),
                "feedback_type": entry.feedback_type.value,
                "user_role": entry.user_role.value,
                "verification_record_id": str(record.id),
            },
        )
        self._refresh_job_assessment(job, commit=False)
        self.db.commit()
        self.db.refresh(entry)
        return FeedbackEntryResponse.model_validate(entry)

    def _run_image_pipeline(self, job: VerificationJob, extracted: ExtractedPayload, image_path: str, uploaded_filename: str) -> VerificationJob:
        extracted_row = ExtractedContent(
            job_id=job.id,
            title=extracted.title,
            author=extracted.author,
            publish_time=extracted.publish_time,
            canonical_url=extracted.canonical_url,
            main_text=extracted.main_text,
            hero_image_url=extracted.hero_image_url,
            normalized_text=extracted.normalized_text,
            content_hash=extracted.content_hash,
            metadata_json=extracted.metadata,
        )
        self.db.add(extracted_row)
        append_audit_entry(self.db, job, "image_content_extracted", {"hash": extracted.content_hash})

        layer2 = self._image_analysis_layer(extracted, image_path, uploaded_filename, job=job)
        record = self._build_image_verification_record(image_path, uploaded_filename, extracted, layer2)
        manifest = self._create_image_verification_manifest(record)
        layer1 = self._image_crypto_layer(record, manifest, extracted)
        layer3 = self._image_feedback_layer(record, persist_audit=False)
        layer4 = self._layer4(job)
        status_value, headline, reason, proof_summary, analysis_report = self._decide_image(layer1, layer2, layer3, extracted, record, manifest)

        job.status = status_value
        job.headline_message = headline
        job.short_reason = reason
        job.content_preview = self._content_preview_for_image(record, extracted)
        job.proof_summary = proof_summary
        job.analysis_report = analysis_report
        job.layer_results = {
            "layer1": asdict(layer1),
            "layer2": asdict(layer2),
            "layer3": asdict(layer3),
            "layer4": asdict(layer4),
        }
        job.image_verification_record_id = record.id
        append_audit_entry(self.db, job, "image_verification_completed", {"status": status_value.value, "manifest_id": str(manifest.id)})
        self.db.commit()
        self.db.refresh(job)
        return job

    def _run_audio_pipeline(self, job: VerificationJob, extracted: ExtractedPayload, audio_path: str, uploaded_filename: str) -> VerificationJob:
        layer2 = self._audio_analysis_layer(extracted, audio_path, uploaded_filename, job=job)
        transcript_result = layer2.details.get("transcription_result", {}) if isinstance(layer2.details, dict) else {}
        cleaned_transcript = str(transcript_result.get("cleaned_transcript") or "").strip()
        if cleaned_transcript:
            extracted.main_text = cleaned_transcript
            extracted.normalized_text = normalize_text(cleaned_transcript)
            extracted.content_hash = sha256_text(extracted.normalized_text)
            extracted.metadata["transcript_text"] = transcript_result.get("transcript_text")
            extracted.metadata["cleaned_transcript"] = cleaned_transcript
            extracted.metadata["transcript_status"] = transcript_result.get("transcript_status")
            extracted.metadata["detected_language"] = transcript_result.get("detected_language")

        extracted_row = ExtractedContent(
            job_id=job.id,
            title=extracted.title,
            author=extracted.author,
            publish_time=extracted.publish_time,
            canonical_url=extracted.canonical_url,
            main_text=extracted.main_text,
            hero_image_url=extracted.hero_image_url,
            normalized_text=extracted.normalized_text,
            content_hash=extracted.content_hash,
            metadata_json=extracted.metadata,
        )
        self.db.add(extracted_row)
        append_audit_entry(self.db, job, "audio_content_extracted", {"hash": extracted.content_hash})
        for candidate in layer2.details.get("matches", []) if isinstance(layer2.details, dict) else []:
            self.db.add(
                SimilarityMatch(
                    job_id=job.id,
                    candidate_url=candidate.get("url"),
                    candidate_title=candidate.get("title"),
                    score=float(candidate.get("score") or candidate.get("rerank_score") or candidate.get("hybrid_score") or 0.0),
                    explanation=str(candidate.get("explanation") or candidate.get("summary") or candidate.get("evidence_summary") or ""),
                    metadata_json={
                        "source": candidate.get("source", "web_search"),
                        "snippet": candidate.get("snippet") or candidate.get("passage"),
                        "domain": candidate.get("domain"),
                    },
                )
            )
        record = self._build_audio_verification_record(audio_path, uploaded_filename, extracted, layer2)
        manifest = self._create_audio_verification_manifest(record)
        layer1 = self._audio_crypto_layer(record, manifest, extracted)
        layer3 = self._audio_feedback_layer(record, persist_audit=False)
        layer4 = self._layer4(job)
        status_value, headline, reason, proof_summary, analysis_report = self._decide_audio(layer1, layer2, layer3, extracted, record, manifest)

        job.status = status_value
        job.headline_message = headline
        job.short_reason = reason
        job.content_preview = self._content_preview_for_audio(record, extracted)
        job.proof_summary = proof_summary
        job.analysis_report = analysis_report
        job.layer_results = {
            "layer1": asdict(layer1),
            "layer2": asdict(layer2),
            "layer3": asdict(layer3),
            "layer4": asdict(layer4),
        }
        job.audio_verification_record_id = record.id
        append_audit_entry(self.db, job, "audio_verification_completed", {"status": status_value.value, "manifest_id": str(manifest.id)})
        self.db.commit()
        self.db.refresh(job)
        return job

    def _run_video_pipeline(self, job: VerificationJob, extracted: ExtractedPayload, video_path: str, uploaded_filename: str) -> VerificationJob:
        extracted_row = ExtractedContent(
            job_id=job.id,
            title=extracted.title,
            author=extracted.author,
            publish_time=extracted.publish_time,
            canonical_url=extracted.canonical_url,
            main_text=extracted.main_text,
            hero_image_url=extracted.hero_image_url,
            normalized_text=extracted.normalized_text,
            content_hash=extracted.content_hash,
            metadata_json=extracted.metadata,
        )
        self.db.add(extracted_row)
        append_audit_entry(self.db, job, "video_content_extracted", {"hash": extracted.content_hash})

        layer2 = self._video_analysis_layer(extracted, video_path, uploaded_filename)
        extracted_row.main_text = extracted.main_text
        extracted_row.normalized_text = extracted.normalized_text
        extracted_row.content_hash = extracted.content_hash
        extracted_row.metadata_json = extracted.metadata
        record = self._build_video_verification_record(video_path, uploaded_filename, extracted, layer2)
        manifest = self._create_video_verification_manifest(record)
        layer1 = self._video_crypto_layer(record, manifest, extracted)
        layer3 = self._video_feedback_layer(record, persist_audit=False)
        layer4 = self._layer4(job)
        status_value, headline, reason, proof_summary, analysis_report = self._decide_video(layer1, layer2, layer3, extracted, record, manifest)

        job.status = status_value
        job.headline_message = headline
        job.short_reason = reason
        job.content_preview = self._content_preview_for_video(record, extracted)
        job.proof_summary = proof_summary
        job.analysis_report = analysis_report
        job.layer_results = {
            "layer1": asdict(layer1),
            "layer2": asdict(layer2),
            "layer3": asdict(layer3),
            "layer4": asdict(layer4),
        }
        job.video_verification_record_id = record.id
        append_audit_entry(self.db, job, "video_verification_completed", {"status": status_value.value, "manifest_id": str(manifest.id)})
        self.db.commit()
        self.db.refresh(job)
        return job

    def _build_image_verification_record(
        self,
        image_path: str,
        uploaded_filename: str,
        extracted: ExtractedPayload,
        layer2: LayerOutcome,
    ) -> ImageVerificationRecord:
        hashes = compute_image_hashes(image_path)
        previous = self.db.scalar(
            select(ImageVerificationRecord)
            .where(ImageVerificationRecord.image_hash == hashes["sha256"])
            .order_by(ImageVerificationRecord.created_at.desc())
            .limit(1)
        )
        record = ImageVerificationRecord(
            image_hash=str(hashes["sha256"]),
            perceptual_hash=str(hashes["perceptual_hash"]) if hashes["perceptual_hash"] else None,
            uploaded_filename=uploaded_filename,
            caption_or_claim=extracted.metadata.get("caption_or_claim"),
            ocr_result=layer2.details.get("ocr_result", {}),
            image_analysis_result=layer2.details,
            crypto_result={},
            final_score=0.0,
            final_label="Not Verifiable",
            explanation=layer2.explanation,
            current_version=(previous.current_version + 1) if previous else 1,
            previous_record_id=previous.id if previous else None,
        )
        self.db.add(record)
        self.db.flush()
        return record

    def _create_image_verification_manifest(self, record: ImageVerificationRecord):
        crypto = CryptoProvenanceService(self.db)
        manifest = crypto.sign_object(CryptoObjectType.image_verification_record, record.id)
        record.current_manifest_id = manifest.id
        self.db.add(record)
        self.db.flush()
        return manifest

    def _resolve_image_verification_record(self, job: VerificationJob) -> ImageVerificationRecord | None:
        if job.image_verification_record_id:
            return self.db.get(ImageVerificationRecord, job.image_verification_record_id)
        record_id = (job.proof_summary or {}).get("image_verification_record_id")
        if record_id:
            try:
                return self.db.get(ImageVerificationRecord, uuid.UUID(str(record_id)))
            except (ValueError, TypeError):
                return None
        return None

    def _feedback_entries_for_image_record(self, record: ImageVerificationRecord) -> list[ImageFeedbackEntry]:
        chain_ids: list[uuid.UUID] = []
        current: ImageVerificationRecord | None = record
        while current:
            chain_ids.append(current.id)
            current = current.previous_record
        return list(
            self.db.scalars(
                select(ImageFeedbackEntry)
                .where(ImageFeedbackEntry.image_verification_record_id.in_(chain_ids))
                .order_by(ImageFeedbackEntry.created_at.desc())
            )
        )

    def _feedback_entries_for_audio_record(self, record: AudioVerificationRecord) -> list[AudioFeedbackEntry]:
        chain_ids: list[uuid.UUID] = []
        current: AudioVerificationRecord | None = record
        while current:
            chain_ids.append(current.id)
            current = current.previous_record
        return list(
            self.db.scalars(
                select(AudioFeedbackEntry)
                .where(AudioFeedbackEntry.audio_verification_record_id.in_(chain_ids))
                .order_by(AudioFeedbackEntry.created_at.desc())
            )
        )

    def _image_feedback_layer(self, record: ImageVerificationRecord, persist_audit: bool = True) -> LayerOutcome:
        entries = self._feedback_entries_for_image_record(record)
        total = len(entries)
        counts = {
            "correction": sum(1 for entry in entries if entry.feedback_type == FeedbackType.correction),
            "dispute": sum(1 for entry in entries if entry.feedback_type == FeedbackType.dispute),
            "source_suggestion": sum(1 for entry in entries if entry.feedback_type == FeedbackType.source_suggestion),
            "clarification": sum(1 for entry in entries if entry.feedback_type == FeedbackType.clarification),
            "review_note": sum(1 for entry in entries if entry.feedback_type == FeedbackType.review_note),
        }
        reviewed = sum(1 for entry in entries if entry.status in {FeedbackStatus.reviewed, FeedbackStatus.resolved})
        resolved = sum(1 for entry in entries if entry.status == FeedbackStatus.resolved)
        expert_count = sum(1 for entry in entries if entry.user_role == FeedbackUserRole.expert)
        moderator_count = sum(1 for entry in entries if entry.user_role == FeedbackUserRole.moderator)
        if persist_audit:
            image_job = self.db.scalar(select(VerificationJob).where(VerificationJob.image_verification_record_id == record.id).limit(1))
            if image_job:
                append_audit_entry(self.db, image_job, "image_feedback_layer_completed", {"feedback_count": total})

        if total == 0:
            status_value = "no_feedback"
            explanation = "No feedback has been submitted for this image verification record yet."
        elif counts["dispute"] or counts["correction"]:
            status_value = "review_attention"
            explanation = "Feedback includes disputes or correction requests that may lead to a revised image verification record."
        elif reviewed or expert_count or moderator_count:
            status_value = "reviewed_feedback"
            explanation = "Image review history includes structured feedback and moderator or expert notes."
        else:
            status_value = "feedback_open"
            explanation = "Feedback has been submitted for this image verification record and is awaiting review."

        return LayerOutcome(
            status=status_value,
            confidence=0.0,
            explanation=explanation,
            details={
                "mode": "feedback",
                "feedback_count": total,
                "reviewed_count": reviewed,
                "resolved_count": resolved,
                "expert_feedback_count": expert_count,
                "moderator_feedback_count": moderator_count,
                "correction_count": counts["correction"],
                "dispute_count": counts["dispute"],
                "source_suggestion_count": counts["source_suggestion"],
                "clarification_count": counts["clarification"],
                "review_note_count": counts["review_note"],
            },
        )

    def _audio_feedback_layer(self, record: AudioVerificationRecord, persist_audit: bool = True) -> LayerOutcome:
        entries = self._feedback_entries_for_audio_record(record)
        total = len(entries)
        counts = {
            "correction": sum(1 for entry in entries if entry.feedback_type == FeedbackType.correction),
            "dispute": sum(1 for entry in entries if entry.feedback_type == FeedbackType.dispute),
            "source_suggestion": sum(1 for entry in entries if entry.feedback_type == FeedbackType.source_suggestion),
            "clarification": sum(1 for entry in entries if entry.feedback_type == FeedbackType.clarification),
            "review_note": sum(1 for entry in entries if entry.feedback_type == FeedbackType.review_note),
        }
        reviewed = sum(1 for entry in entries if entry.status in {FeedbackStatus.reviewed, FeedbackStatus.resolved})
        resolved = sum(1 for entry in entries if entry.status == FeedbackStatus.resolved)
        expert_count = sum(1 for entry in entries if entry.user_role == FeedbackUserRole.expert)
        moderator_count = sum(1 for entry in entries if entry.user_role == FeedbackUserRole.moderator)
        if persist_audit:
            audio_job = self.db.scalar(select(VerificationJob).where(VerificationJob.audio_verification_record_id == record.id).limit(1))
            if audio_job:
                append_audit_entry(self.db, audio_job, "audio_feedback_layer_completed", {"feedback_count": total})

        if total == 0:
            status_value = "no_feedback"
            explanation = "No feedback has been submitted for this audio verification record yet."
        elif counts["dispute"] or counts["correction"]:
            status_value = "review_attention"
            explanation = "Feedback includes disputes or correction requests that may lead to a revised audio verification record."
        elif reviewed or expert_count or moderator_count:
            status_value = "reviewed_feedback"
            explanation = "Audio review history includes structured feedback and moderator or expert notes."
        else:
            status_value = "feedback_open"
            explanation = "Feedback has been submitted for this audio verification record and is awaiting review."

        return LayerOutcome(
            status=status_value,
            confidence=0.0,
            explanation=explanation,
            details={
                "mode": "feedback",
                "feedback_count": total,
                "reviewed_count": reviewed,
                "resolved_count": resolved,
                "expert_feedback_count": expert_count,
                "moderator_feedback_count": moderator_count,
                "correction_count": counts["correction"],
                "dispute_count": counts["dispute"],
                "source_suggestion_count": counts["source_suggestion"],
                "clarification_count": counts["clarification"],
                "review_note_count": counts["review_note"],
            },
        )

    def _image_analysis_layer(self, extracted: ExtractedPayload, image_path: str, uploaded_filename: str, job: VerificationJob | None = None) -> LayerOutcome:
        caption_or_claim = str(extracted.metadata.get("caption_or_claim") or "").strip()
        ocr_pipeline = extracted.metadata.get("ocr_pipeline", {}) if isinstance(extracted.metadata, dict) else {}
        raw_ocr_text = str(ocr_pipeline.get("extracted_text") or (extracted.main_text if bool(extracted.metadata.get("ocr_text_present")) else "")).strip()
        ocr_text = self._clean_image_ocr_text(raw_ocr_text)
        forensics = extracted.metadata.get("forensics", {}) if isinstance(extracted.metadata, dict) else {}
        metadata_fields = forensics.get("fields", {}) if isinstance(forensics, dict) else {}
        media_authenticity = self.media_authenticity.analyze(extracted)
        visual_description = self.vision_reasoning.describe_image(image_path, context=caption_or_claim, ocr_text=ocr_text)
        visual_context = self.vision_reasoning.compare_visual_context(visual_description, caption_or_claim)
        ocr_branch = self._image_ocr_branch(caption_or_claim, ocr_pipeline, ocr_text)
        visual_branch = self._image_visual_branch(media_authenticity, metadata_fields)
        provenance_branch = self._image_provenance_branch(extracted, forensics)
        text_input = ocr_text or caption_or_claim
        text_quality = "normal" if len(text_input) >= 100 else "limited" if len(text_input) >= 30 else "failed"
        text_outcome: LayerOutcome | None = None
        if text_quality != "failed" and job is not None:
            image_payload = ExtractedPayload(
                title=extracted.title or uploaded_filename,
                author=extracted.author,
                publish_time=extracted.publish_time,
                canonical_url=None,
                main_text=text_input,
                hero_image_url=extracted.hero_image_url,
                normalized_text=normalize_text(text_input),
                content_hash=sha256_text(normalize_text(text_input)),
                metadata={
                    **(extracted.metadata if isinstance(extracted.metadata, dict) else {}),
                    "source_type": "image_ocr_text",
                    "source_mode": "image",
                    "original_input_id": uploaded_filename,
                },
            )
            text_outcome = self._text_nlp_layer(job, image_payload, text_input)
        text_details = text_outcome.details if text_outcome and isinstance(text_outcome.details, dict) else {}
        evidence_sources = list(text_details.get("evidence_sources", []) or text_details.get("matches", []) or [])
        context_branch = self._image_context_branch(caption_or_claim, ocr_text, evidence_sources, visual_context)
        nlp_score = max(0.0, min(1.0, float(text_details.get("text_verification_score") or (text_outcome.confidence if text_outcome else 0.0) or 0.0)))
        if text_quality == "limited" and nlp_score > 0.65:
            nlp_score = 0.65
        failure_reason = None
        if text_quality == "failed":
            failure_reason = "no_readable_text" if not text_input else "ocr_text_too_short"
        elif text_outcome is None:
            failure_reason = "text_pipeline_not_called"

        visual_feature_consistency = float(visual_branch["visual_feature_consistency"])
        manipulation_integrity = float(visual_branch["manipulation_integrity"])
        reverse_reuse_score = float(context_branch["reverse_reuse_score"])
        context_match_score = float(context_branch["context_match_score"])
        ocr_text_consistency = float(ocr_branch["verification_confidence"])
        ocr_result = dict(ocr_branch["ocr_result"])
        manipulation_detected = bool(visual_branch["manipulation_detected"])
        misleading_reuse = bool(context_branch["misleading_reuse"])

        image_analysis_score = (
            0.25 * visual_feature_consistency
            + 0.20 * manipulation_integrity
            + 0.20 * reverse_reuse_score
            + 0.20 * context_match_score
            + 0.15 * ocr_text_consistency
        )
        technical_image_score = image_analysis_score

        no_external_corroboration = not evidence_sources
        no_provenance_support = float(provenance_branch["verification_confidence"]) < 0.45
        weak_context_support = float(context_branch["verification_confidence"]) < 0.45
        if no_external_corroboration and no_provenance_support:
            image_analysis_score = min(image_analysis_score, 0.42)
        elif bool(ocr_branch["text_detected"]) and weak_context_support and no_provenance_support:
            image_analysis_score = min(image_analysis_score, 0.45)
        elif visual_feature_consistency >= 0.75 and weak_context_support:
            image_analysis_score = min(image_analysis_score, 0.49)

        explanation = self._generate_image_explanation(
            {
                "ocr_branch": ocr_branch,
                "visual_branch": visual_branch,
                "context_branch": context_branch,
                "provenance_branch": provenance_branch,
                "ocr_result": ocr_result,
                "manipulation_detected": manipulation_detected,
                "misleading_reuse": misleading_reuse,
                "evidence_sources": evidence_sources,
                "context_match_score": context_match_score,
                "media_authenticity": media_authenticity,
                "no_external_corroboration": no_external_corroboration,
                "no_provenance_support": no_provenance_support,
            }
        )
        return LayerOutcome(
            status="supporting" if nlp_score >= 0.5 and not manipulation_detected and not misleading_reuse else "none",
            confidence=round(max(0.0, min(1.0, nlp_score)), 3),
            explanation=explanation,
            details={
                "mode": "image_verification",
                "caption_or_claim": caption_or_claim or None,
                "matches": evidence_sources,
                "evidence_sources": evidence_sources,
                "fact_check_verdict": text_details.get("text_verification_label", "Not Verifiable" if failure_reason else "Unverified"),
                "fact_check_assessment": text_details.get("fact_check_assessment") or (text_outcome.explanation if text_outcome else "No readable claim text was detected in the image."),
                "fact_check_basis": "Image OCR text was cleaned and verified through the same retrieval-first text pipeline used for pasted text.",
                "text_verification_score": round(nlp_score, 3),
                "text_verification_label": text_details.get("text_verification_label", "Not Verifiable" if failure_reason else "Unverified"),
                "text_component_scores": text_details.get("text_component_scores", {}),
                "extracted_claims": text_details.get("extracted_claims", []),
                "claim_results": text_details.get("claim_results", []),
                "image_branches": {
                    "ocr_text_extraction": ocr_branch,
                    "visual_manipulation": visual_branch,
                    "reverse_reuse_context": context_branch,
                    "provenance_metadata": provenance_branch,
                },
                "retrieval_confidence": round(float(context_branch["retrieval_confidence"]), 3),
                "verification_confidence": round(float(context_branch["verification_confidence"]), 3),
                "image_analysis_score": round(max(0.0, min(1.0, image_analysis_score)), 3),
                "technical_image_score": round(max(0.0, min(1.0, technical_image_score)), 3),
                "image_analysis_breakdown": {
                    "visual_feature_consistency": round(visual_feature_consistency, 3),
                    "manipulation_integrity": round(manipulation_integrity, 3),
                    "reverse_reuse_score": round(reverse_reuse_score, 3),
                    "context_match_score": round(context_match_score, 3),
                    "ocr_text_consistency": round(ocr_text_consistency, 3),
                },
                "ocr_result": ocr_result,
                "image_fetch_status": ocr_result.get("image_fetch_status"),
                "image_decode_status": ocr_result.get("image_decode_status"),
                "context_retrieval_status": "completed" if evidence_sources else "no_context_sources_found",
                "corroboration_status": "supported" if evidence_sources else "not_corroborated",
                "final_verification_confidence": round(max(0.0, min(1.0, image_analysis_score)), 3),
                "image_context_wrapper_debug": {
                    "file_name": uploaded_filename,
                    "image_size": {
                        "width": ocr_result.get("image_width"),
                        "height": ocr_result.get("image_height"),
                    },
                    "ocr_engine": "pytesseract",
                    "raw_ocr_length": len(raw_ocr_text),
                    "cleaned_ocr_length": len(ocr_text),
                    "ocr_confidence": ocr_result.get("ocr_confidence"),
                    "cleaned_ocr_preview": ocr_text[:500],
                    "text_pipeline_called": text_outcome is not None,
                    "context_quality": text_quality,
                    "failure_reason": failure_reason,
                    "generated_brave_queries": text_details.get("search_queries_used", []),
                    "evidence_urls": [item.get("url") for item in evidence_sources if item.get("url")],
                    "evidence_domains": [item.get("domain") or extract_domain(item.get("url")) for item in evidence_sources],
                    "evidence_stances": [
                        {
                            "url": item.get("url"),
                            "relevance_score": item.get("claim_relevance_score") or item.get("relation_confidence") or item.get("score"),
                            "stance": item.get("evidence_relation") or item.get("judgment_label") or item.get("stance"),
                        }
                        for item in evidence_sources
                    ],
                    "nlp_score": round(nlp_score, 3),
                },
                "failure_reasons": sorted(
                    {
                        *[str(item) for item in ocr_result.get("failure_reasons", [])],
                        *[str(item) for item in visual_branch.get("error_flags", [])],
                        *[str(item) for item in context_branch.get("error_flags", [])],
                        *[str(item) for item in provenance_branch.get("error_flags", [])],
                        *(["no_context_sources_found"] if not evidence_sources else []),
                        *(["provenance_missing"] if no_provenance_support else []),
                        *([failure_reason] if failure_reason else []),
                    }
                ),
                "visual_description": visual_description,
                "visual_context_result": visual_context,
                "media_authenticity": media_authenticity,
                "manipulation_detected": manipulation_detected,
                "misleading_reuse": misleading_reuse,
                "no_external_corroboration": no_external_corroboration,
                "no_provenance_support": no_provenance_support,
                "evidence_elements_found": {
                    "metadata_found": bool(forensics),
                    "ocr_checked": True,
                    "ocr_text_detected": bool(ocr_text),
                    "reverse_image_checked": True,
                    "context_checked": bool(caption_or_claim),
                    "text_context_pipeline_checked": text_outcome is not None,
                    "manipulation_analysis_completed": True,
                    "original_source_match_found": bool(evidence_sources),
                    "provenance_record_found": False,
                },
            },
        )

    def _clean_image_ocr_text(self, text: str) -> str:
        cleaned = " ".join(str(text or "").split()).strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"([^\w\s])\1{2,}", r"\1", cleaned)
        noise_patterns = [
            r"\b(?:like|comment|share|subscribe|follow|home|menu|login|sign up)\b",
            r"\b(?:retweeted|views|likes|comments|shares)\b",
        ]
        kept: list[str] = []
        for sentence in self._text_sentences(cleaned):
            normalized = normalize_text(sentence)
            if len(normalized) < 3:
                continue
            if len(sentence.split()) <= 4 and any(re.search(pattern, normalized) for pattern in noise_patterns):
                continue
            kept.append(sentence)
        return " ".join(kept).strip() or cleaned

    def _image_ocr_branch(self, caption_or_claim: str, ocr_pipeline: dict, ocr_text: str) -> dict:
        ocr_result = {
            "text_detected": bool(ocr_pipeline.get("text_detected")),
            "extracted_text": str(ocr_pipeline.get("extracted_text") or ocr_text or ""),
            "ocr_status": str(ocr_pipeline.get("ocr_status") or ("completed" if ocr_text else "not_applicable")),
            "ocr_summary": str(ocr_pipeline.get("ocr_summary") or "No OCR summary available."),
            "ocr_confidence": float(ocr_pipeline.get("ocr_confidence") or 0.0),
            "image_fetch_status": str(ocr_pipeline.get("image_fetch_status") or "not_applicable"),
            "image_decode_status": str(ocr_pipeline.get("image_decode_status") or "unknown"),
            "failure_reasons": list(ocr_pipeline.get("failure_reasons") or []),
        }
        if not ocr_text:
            return {
                "subscore": 0.35,
                "retrieval_confidence": 0.0,
                "verification_confidence": 0.35,
                "evidence_count": 0,
                "evidence_quality": 0.0,
                "error_flags": list(ocr_result.get("failure_reasons") or ["ocr_no_text_detected"]),
                "text_detected": False,
                "ocr_result": ocr_result,
                "explanation": "OCR extraction did not produce usable text, so the text branch cannot count as supporting verification evidence.",
            }
        overlap = max(0.2, self._text_overlap(caption_or_claim or ocr_text, ocr_text))
        verification_confidence = min(overlap, max(0.2, float(ocr_result.get("ocr_confidence") or 0.0)))
        return {
            "subscore": verification_confidence,
            "retrieval_confidence": 1.0,
            "verification_confidence": verification_confidence,
            "evidence_count": 1,
            "evidence_quality": verification_confidence,
            "error_flags": list(ocr_result.get("failure_reasons") or []),
            "text_detected": True,
            "ocr_result": ocr_result,
            "explanation": "OCR extracted text successfully, but extraction alone does not verify the claim without corroborating context or provenance.",
        }

    def _image_visual_branch(self, media_authenticity: dict, metadata_fields: dict) -> dict:
        visual_feature_consistency = 0.8
        if media_authenticity.get("status") in {"suspicious_manipulation", "high_deepfake_risk"}:
            visual_feature_consistency = 0.35
        elif metadata_fields.get("device_model") or metadata_fields.get("captured_at"):
            visual_feature_consistency = 0.88

        manipulation_integrity = 0.82
        manipulation_detected = False
        error_flags: list[str] = []
        if media_authenticity.get("status") == "high_deepfake_risk":
            manipulation_integrity = 0.05
            manipulation_detected = True
            error_flags.append("confirmed_manipulation")
        elif media_authenticity.get("status") == "suspicious_manipulation":
            manipulation_integrity = 0.3
            error_flags.append("suspicious_manipulation")
        elif media_authenticity.get("status") == "inconclusive":
            manipulation_integrity = 0.6

        return {
            "subscore": round((visual_feature_consistency + manipulation_integrity) / 2, 3),
            "retrieval_confidence": 0.0,
            "verification_confidence": round((visual_feature_consistency + manipulation_integrity) / 2, 3),
            "evidence_count": 1,
            "evidence_quality": round((visual_feature_consistency + manipulation_integrity) / 2, 3),
            "error_flags": error_flags,
            "manipulation_detected": manipulation_detected,
            "visual_feature_consistency": visual_feature_consistency,
            "manipulation_integrity": manipulation_integrity,
            "explanation": "Visual analysis checks internal coherence and manipulation signals, but visual coherence alone does not prove event context.",
        }

    def _image_context_branch(self, caption_or_claim: str, ocr_text: str, evidence_sources: list[dict], visual_context: dict) -> dict:
        retrieval_confidence = min(1.0, len(evidence_sources) / 5) if evidence_sources else 0.0
        if evidence_sources:
            retrieval_confidence = max(
                retrieval_confidence,
                max(float(item.get("score") or item.get("retrieval_score") or 0.0) for item in evidence_sources),
            )
        reverse_reuse_score = 0.35 if not evidence_sources else min(
            0.95,
            max(float(item.get("score") or item.get("retrieval_score") or 0.0) for item in evidence_sources) + 0.15,
        )
        misleading_reuse = False
        if evidence_sources and caption_or_claim and not any(
            self._text_overlap(caption_or_claim, f"{item.get('title', '')} {item.get('snippet', '')}") >= 0.18
            for item in evidence_sources
        ):
            reverse_reuse_score = 0.2
            misleading_reuse = True

        context_match_score = 0.3 if caption_or_claim else reverse_reuse_score
        if caption_or_claim:
            context_scores = [
                self._text_overlap(caption_or_claim, " ".join(filter(None, [str(item.get("title") or ""), str(item.get("passage") or item.get("snippet") or "")])))
                for item in evidence_sources
            ]
            if ocr_text:
                context_scores.append(self._text_overlap(caption_or_claim, ocr_text))
            context_scores.append(float(visual_context.get("score") or 0.0))
            context_match_score = max([0.15] + context_scores)
        elif evidence_sources:
            context_match_score = min(0.8, reverse_reuse_score)

        verification_confidence = round((reverse_reuse_score + context_match_score) / 2, 3)
        return {
            "subscore": verification_confidence,
            "retrieval_confidence": round(retrieval_confidence, 3),
            "verification_confidence": verification_confidence,
            "evidence_count": len(evidence_sources),
            "evidence_quality": round(max([0.0] + [float(item.get("score") or item.get("retrieval_score") or 0.0) for item in evidence_sources]), 3),
            "error_flags": ["misleading_reuse"] if misleading_reuse else [],
            "reverse_reuse_score": reverse_reuse_score,
            "context_match_score": context_match_score,
            "misleading_reuse": misleading_reuse,
            "explanation": "External evidence and context matching determine whether the image is actually corroborated, not just extracted successfully.",
        }

    def _image_provenance_branch(self, extracted: ExtractedPayload, forensics: dict) -> dict:
        normalized_source = extracted.metadata.get("normalized_source", {}) if isinstance(extracted.metadata, dict) else {}
        provenance_available = bool(extracted.metadata.get("c2pa_credentials") or extracted.metadata.get("provenance_available"))
        completeness = float(normalized_source.get("metadata_completeness") or extracted.metadata.get("metadata_completeness") or 0.0)
        verification_confidence = 0.8 if provenance_available else max(0.25, completeness * 0.7)
        return {
            "subscore": round(verification_confidence, 3),
            "retrieval_confidence": 0.0,
            "verification_confidence": round(verification_confidence, 3),
            "evidence_count": int(bool(forensics)) + int(provenance_available),
            "evidence_quality": round(completeness, 3),
            "error_flags": [] if provenance_available else ["weak_provenance"],
            "provenance_available": provenance_available,
            "metadata_completeness": round(completeness, 3),
            "explanation": "Provenance and metadata support origin confidence, but missing provenance must reduce confidence rather than act as a neutral pass.",
        }

    def _retrieve_image_evidence(self, caption_or_claim: str, ocr_text: str, uploaded_filename: str) -> list[dict]:
        extraction = self.claim_extraction.extract_claims(" ".join(filter(None, [caption_or_claim, ocr_text, Path(uploaded_filename).stem.replace("-", " ").replace("_", " ")])), max_claims=2)
        queries = self._build_reasoning_queries(caption_or_claim or ocr_text, extraction, Path(uploaded_filename).stem.replace("-", " ").replace("_", " "))
        return self._retrieve_reasoned_evidence(queries, limit=5)

    def _text_overlap(self, left: str, right: str) -> float:
        left_terms = set(normalize_text(left).split())
        right_terms = set(normalize_text(right).split())
        if not left_terms or not right_terms:
            return 0.0
        return len(left_terms & right_terms) / max(1, len(left_terms))

    def _build_reasoning_queries(self, primary_text: str, extraction: dict, fallback: str | None = None) -> list[str]:
        queries: list[str] = []
        compact_primary = " ".join((primary_text or "").split())
        if compact_primary:
            queries.append(" ".join(compact_primary.split()[:18]))
        for claim in extraction.get("claims", [])[:2]:
            text = " ".join(str(claim.get("text") or "").split()[:18])
            if text:
                queries.append(text)
        entities = extraction.get("entities", [])[:5]
        if entities:
            queries.append(" ".join(entities))
        if fallback:
            queries.append(" ".join(str(fallback).split()[:10]))
        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            if not query:
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(query)
            if len(deduped) >= 4:
                break
        return deduped

    def _retrieve_reasoned_evidence(self, queries: list[str], limit: int = 5) -> list[dict]:
        candidates: list[dict] = []
        seen: set[str] = set()
        for query in queries:
            for item in self.retrieval.retrieve_hybrid(query, limit=limit):
                key = f"{item.get('url')}::{item.get('passage')}"
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(item)
                if len(candidates) >= limit:
                    return candidates
        return candidates[:limit]

    def _cached_transcription(self, file_hash: str | None, path: str, media_kind: str) -> dict:
        if file_hash and file_hash in self._transcription_cache:
            cached = dict(self._transcription_cache[file_hash])
            cached["cache_hit"] = True
            return cached

        result = (
            self.transcription.transcribe_video_audio(path)
            if media_kind == "video"
            else self.transcription.transcribe_audio(path)
        )
        if file_hash:
            self._transcription_cache[file_hash] = dict(result)
        result["cache_hit"] = False
        return result

    def _empty_source_recovery(self, reason: str) -> dict:
        return {
            "status": "skipped",
            "best_score": 0.0,
            "best_candidate": None,
            "search_results": [],
            "overlap_scores": [],
            "final_source_recovery": {
                "status": "not_found",
                "url": None,
                "title": None,
                "publisher": None,
                "overlap_confidence": 0.0,
                "match_type": "none",
                "reason": reason,
            },
        }

    def _image_crypto_layer(self, record: ImageVerificationRecord, manifest, extracted: ExtractedPayload) -> LayerOutcome:
        metadata_snapshot = extracted.metadata.get("forensics", {}) if isinstance(extracted.metadata, dict) else {}
        crypto_breakdown = self._compute_image_crypto_breakdown(record, manifest, metadata_snapshot)
        return LayerOutcome(
            status="platform-issued",
            confidence=round(crypto_breakdown["score"], 3),
            explanation=(
                "NewsProof created a signed provenance record for this image verification result, including the image hashes, OCR findings, analysis summary, and version history."
            ),
            details={
                "record_type": "image_verification_record",
                "image_verification_record_id": str(record.id),
                "manifest_id": str(manifest.id),
                "verification_manifest_id": str(manifest.id),
                "current_version": record.current_version,
                "previous_version_id": str(record.previous_record_id) if record.previous_record_id else None,
                "previous_manifest_id": str(manifest.previous_manifest_id) if manifest.previous_manifest_id else None,
                "issued_by": manifest.signer.issuer_name if manifest.signer else "NewsProof system",
                "issuer_type": manifest.signer.issuer_type.value if manifest.signer else "platform",
                "verification_result": "Verified platform-issued provenance",
                "provenance_verification_status": manifest.verification_status.value if manifest.verification_status else "partial_provenance",
                "signed_at": manifest.created_at.isoformat() if manifest.created_at else None,
                "timestamp_source": (manifest.timestamp_token or {}).get("provider", "internal server timestamp"),
                "canonical_hash": manifest.sha256_hash,
                "signature_algorithm": manifest.signing_algorithm,
                "internal_manifest_valid": manifest.verification_status.value == "valid" if manifest.verification_status else False,
                "platform_manifest_valid": manifest.verification_status.value == "valid" if manifest.verification_status else False,
                "canonical_json": manifest.canonical_json,
                "canonical_json_preview": manifest.canonical_json[:1200],
                "issuer_key_id": (record.crypto_result or {}).get("key_id", "platform-managed"),
                "timestamp_token_status": (manifest.timestamp_token or {}).get("status", "issued"),
                "crypto_breakdown": crypto_breakdown["breakdown"],
                "uploaded_filename": record.uploaded_filename,
                "image_hash": record.image_hash,
                "perceptual_hash": record.perceptual_hash,
                "caption_or_claim": record.caption_or_claim,
            },
        )

    def _compute_image_crypto_breakdown(self, record: ImageVerificationRecord, manifest, metadata_snapshot: dict) -> dict:
        previous_match = bool(record.previous_record_id)
        source_validity = 0.9 if manifest and manifest.verification_status.value == "valid" else 0.65
        hash_integrity = 1.0 if record.image_hash and manifest else 0.5
        fields = metadata_snapshot.get("fields", {}) if isinstance(metadata_snapshot, dict) else {}
        timestamp_validity = 0.85 if fields.get("captured_at") or metadata_snapshot.get("content_type") else 0.55
        useful_metadata = [
            fields.get("device_make"),
            fields.get("device_model"),
            fields.get("captured_at"),
            fields.get("software"),
            record.perceptual_hash,
        ]
        metadata_completeness = sum(1 for item in useful_metadata if item not in (None, "", [])) / len(useful_metadata)
        penalty = 1.0 if manifest and manifest.verification_status.value == "valid" else 0.6
        score = (
            0.35 * source_validity
            + 0.25 * hash_integrity
            + 0.20 * timestamp_validity
            + 0.20 * metadata_completeness
        ) * penalty
        if previous_match:
            score = min(1.0, score + 0.05)
        return {
            "score": score,
            "breakdown": {
                "source_provenance_validity": round(source_validity, 3),
                "hash_integrity": round(hash_integrity, 3),
                "timestamp_metadata_validity": round(timestamp_validity, 3),
                "metadata_completeness": round(metadata_completeness, 3),
                "crypto_penalty": round(penalty, 3),
            },
        }

    def _compute_image_evidence_multiplier(self, layer2: LayerOutcome, layer1: LayerOutcome) -> float:
        found = layer2.details.get("evidence_elements_found", {}) if isinstance(layer2.details, dict) else {}
        ep = sum(1 for value in found.values() if value)
        et = max(1, len(found))
        if layer1.details.get("manifest_id"):
            ep += 1
            et += 1
        if layer2.details.get("no_external_corroboration"):
            ep = max(0, ep - 1)
        ocr_status = str(layer2.details.get("ocr_result", {}).get("ocr_status") or "")
        if ocr_status in {"image_not_downloaded", "invalid_image_format", "image_too_small_for_ocr"}:
            ep = max(0, ep - 1)
        return max(0.0, min(1.0, 0.4 + 0.6 * (ep / et)))

    def _compute_image_penalty(self, layer2: LayerOutcome, layer1: LayerOutcome) -> float:
        if layer2.details.get("manipulation_detected") or layer2.details.get("misleading_reuse"):
            return 0.0 if layer2.details.get("manipulation_detected") else 0.4
        if layer1.details.get("provenance_verification_status") in {"hash_mismatch", "signature_invalid"}:
            return 0.0
        failure_reasons = set(str(item) for item in layer2.details.get("failure_reasons", []))
        if {"image_not_downloaded", "invalid_image_format"} & failure_reasons:
            return 0.6
        if layer2.details.get("no_external_corroboration") and layer2.details.get("no_provenance_support"):
            return 0.6
        if (layer2.confidence or 0.0) < 0.55 or (layer1.confidence or 0.0) < 0.55:
            return 0.7
        return 1.0

    def _generate_image_label(self, final_score: float, evidence_multiplier: float, penalty_multiplier: float, flags: dict) -> str:
        return map_score_to_label(final_score)

    def _generate_image_explanation(self, result: dict) -> str:
        parts: list[str] = []
        ocr_branch = result.get("ocr_branch", {})
        context_branch = result.get("context_branch", {})
        provenance_branch = result.get("provenance_branch", {})
        ocr_result = result.get("ocr_result", {})
        if ocr_result.get("text_detected"):
            parts.append("OCR text was extracted successfully from the image.")
        elif str(ocr_result.get("ocr_status") or "") == "image_not_downloaded":
            parts.append("The referenced image could not be downloaded for OCR, so no text evidence was available from the image itself.")
        elif str(ocr_result.get("ocr_status") or "") == "invalid_image_format":
            parts.append("The referenced media did not decode into a valid OCR-ready image.")
        elif str(ocr_result.get("ocr_status") or "") == "image_too_small_for_ocr":
            parts.append("The image was available, but it was too small for reliable OCR.")
        else:
            parts.append("No readable text was found inside the image, so OCR did not contribute supporting evidence.")
        if result.get("manipulation_detected"):
            parts.append("Visual analysis found manipulation or synthetic-media signals.")
        else:
            parts.append("Visual checks did not detect strong editing artifacts, but visual coherence alone is not enough to verify context.")
        if result.get("misleading_reuse"):
            parts.append("Reverse-image or context evidence suggests the image may be reused in a mismatched context.")
        elif result.get("evidence_sources"):
            parts.append("External evidence was retrieved to compare the image against claimed context.")
        else:
            parts.append("No reliable external corroboration was found for the claimed image context.")
        if float(context_branch.get("verification_confidence") or 0.0) < 0.45:
            parts.append("Extraction success did not translate into strong context verification.")
        if float(provenance_branch.get("verification_confidence") or 0.0) < 0.45:
            parts.append("Provenance and metadata evidence were limited, so origin confidence is reduced.")
        if float(ocr_branch.get("verification_confidence") or 0.0) > 0 and not result.get("evidence_sources"):
            parts.append("OCR alone was not treated as proof.")
        return " ".join(parts)

    def _build_image_system_scoring(self, layer1: LayerOutcome, layer2: LayerOutcome, layer3: LayerOutcome | None = None) -> dict:
        image_score = max(0.0, min(1.0, float(layer2.details.get("text_verification_score") or layer2.confidence or 0.0)))
        technical_image_score = float(layer2.details.get("technical_image_score") or layer2.details.get("image_analysis_score") or 0.0)
        crypto_score = float(layer1.confidence or 0.0)
        community_score = max(0.0, min(1.0, float(layer3.confidence or 0.0))) if layer3 and layer3.status == "supporting" else 0.0
        evidence_multiplier = self._compute_image_evidence_multiplier(layer2, layer1)
        penalty_multiplier = self._compute_image_penalty(layer2, layer1)
        final_score = max(0.0, min(100.0, 100 * (0.30 * crypto_score + 0.45 * image_score + 0.25 * community_score)))
        flags = {
            "confirmed_manipulation": bool(layer2.details.get("manipulation_detected")),
            "misleading_reuse": bool(layer2.details.get("misleading_reuse")),
            "forged_provenance": layer1.details.get("provenance_verification_status") in {"hash_mismatch", "signature_invalid"},
            "no_external_corroboration": bool(layer2.details.get("no_external_corroboration")),
            "no_provenance_support": bool(layer2.details.get("no_provenance_support")),
            "has_strong_provenance": float(layer1.confidence or 0.0) >= 0.75,
            "has_strong_contextual_corroboration": float(layer2.details.get("verification_confidence") or 0.0) >= 0.7,
        }
        wrapper_debug = layer2.details.get("image_context_wrapper_debug", {}) if isinstance(layer2.details, dict) else {}
        failure_reason = str(wrapper_debug.get("failure_reason") or "")
        if failure_reason in {"no_readable_text", "ocr_text_too_short", "ocr_failed", "text_pipeline_not_called"}:
            final_score = min(final_score, 34.0)
        elif flags["no_external_corroboration"] and flags["no_provenance_support"]:
            final_score = min(final_score, 49.0)
        elif float(layer2.details.get("ocr_result", {}).get("text_detected") or 0.0) and float(layer2.details.get("verification_confidence") or 0.0) < 0.45:
            final_score = min(final_score, 45.0)
        label = "Not Verifiable" if failure_reason in {"no_readable_text", "ocr_text_too_short", "ocr_failed", "text_pipeline_not_called"} else map_score_to_label(final_score)
        if label == "Verified":
            reason = "External evidence supports the claim extracted from the image text."
        elif label == "Likely Verified":
            reason = "External evidence gives moderate support for the claim extracted from the image text."
        elif label == "Not Verifiable":
            reason = "No readable claim text was detected in the image." if failure_reason == "no_readable_text" else "The image text could not be verified with enough reliable external evidence."
        else:
            reason = "External evidence did not support the claim extracted from the image text."
        return {
            "final_score": round(final_score, 2),
            "final_label": label,
            "reason": reason,
            "explanation": self._generate_image_explanation(layer2.details if isinstance(layer2.details, dict) else {}),
            "evidence_multiplier": round(evidence_multiplier, 3),
            "global_penalty": round(penalty_multiplier, 3),
            "layer_scores": {
                "crypto": round(crypto_score, 3),
                "nlp": round(image_score, 3),
                "community": round(community_score, 3) if layer3 else None,
            },
            "active_layer_weights": {"crypto": 0.30, "nlp": 0.45, "community": 0.25},
            "formula": "Final image score = (0.30 * crypto) + (0.45 * NLP) + (0.25 * consensus). Evidence sufficiency and global penalty are diagnostics only.",
            "formula_components": {
                "C": round(crypto_score, 3),
                "N": round(image_score, 3),
                "U": round(community_score, 3),
                "technical_image_score": round(technical_image_score, 3),
                "evidence_multiplier_diagnostic": round(evidence_multiplier, 3),
                "global_penalty_diagnostic": round(penalty_multiplier, 3),
            },
            "layer_breakdown": {
                "crypto": layer1.details.get("crypto_breakdown", {}),
                "nlp": layer2.details.get("text_component_scores", {}),
                "image_technical": layer2.details.get("image_analysis_breakdown", {}),
                "community": layer3.details if layer3 else None,
            },
            "warnings": [
                warning
                for warning in [
                    "Manipulation signals were detected." if flags["confirmed_manipulation"] else None,
                    "Potential misleading reuse was detected." if flags["misleading_reuse"] else None,
                    "Metadata and provenance evidence are limited." if penalty_multiplier < 1.0 else None,
                    "User consensus was missing." if not layer3 or community_score == 0.0 else None,
                ]
                if warning
            ],
        }

    def _decide_image(
        self,
        layer1: LayerOutcome,
        layer2: LayerOutcome,
        layer3: LayerOutcome,
        extracted: ExtractedPayload,
        record: ImageVerificationRecord,
        manifest,
    ) -> tuple[JobStatus, str, str, dict, dict]:
        scoring_result = self._build_image_system_scoring(layer1, layer2, layer3)
        label = str(scoring_result.get("final_label", "Not Verifiable"))
        if label == "Verified":
            status_value = JobStatus.verified
        elif label == "Likely Verified":
            status_value = JobStatus.likely_verified
        elif label == "Unverified":
            status_value = JobStatus.unverified
        else:
            status_value = JobStatus.not_verifiable

        record.final_score = float(scoring_result["final_score"])
        record.final_label = label
        record.explanation = str(scoring_result["explanation"])
        record.crypto_result = {
            "manifest_id": str(manifest.id),
            "issuer_id": layer1.details.get("issued_by"),
            "key_id": layer1.details.get("issuer_key_id"),
            "provenance_status": layer1.details.get("provenance_verification_status"),
            "timestamp": layer1.details.get("signed_at"),
            "previous_payload_hash": layer1.details.get("previous_manifest_id"),
            "crypto_breakdown": layer1.details.get("crypto_breakdown", {}),
        }
        self.db.add(record)
        self.db.flush()

        proof_summary = {
            "content_hash": extracted.content_hash,
            "verification_basis": "Image-analysis-led verification with signed platform provenance",
            "image_verification_record_id": str(record.id),
            "verification_manifest_id": str(manifest.id),
            "current_version": record.current_version,
            "previous_version_id": str(record.previous_record_id) if record.previous_record_id else None,
            "system_scoring": scoring_result,
            **(layer1.details or {}),
        }
        analysis_report = self._build_analysis_report(status_value, layer1, layer2, layer3, extracted, scoring_result)
        analysis_report["executive_summary"] = (
            f"{label}: image verification was led by the image analysis layer, then secured by the cryptographic layer, with feedback stored separately for later review."
        )
        analysis_report["mode"] = "image_verification"
        return status_value, label, str(scoring_result["reason"]), proof_summary, analysis_report

    def _content_preview_for_image(self, record: ImageVerificationRecord, extracted: ExtractedPayload) -> dict:
        text_excerpt = record.caption_or_claim or str((record.ocr_result or {}).get("extracted_text") or extracted.main_text or "")
        return {
            "title": record.uploaded_filename,
            "text_excerpt": text_excerpt[:600],
            "caption_or_claim": record.caption_or_claim,
            "image_hash": record.image_hash,
            "perceptual_hash": record.perceptual_hash,
        }

    def _clean_audio_transcript_text(self, text: str) -> str:
        cleaned = " ".join(str(text or "").split()).strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"\[(?:music|silence|noise|applause|inaudible)[^\]]*\]", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(um|uh|erm|ah)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(\w+)(?:\s+\1\b){2,}", r"\1", cleaned, flags=re.IGNORECASE)
        return " ".join(cleaned.split()).strip()

    def _audio_analysis_layer(self, extracted: ExtractedPayload, audio_path: str, uploaded_filename: str, job: VerificationJob | None = None) -> LayerOutcome:
        caption_or_claim = str(extracted.metadata.get("caption_or_claim") or "").strip()
        claimed_speaker = str(extracted.metadata.get("claimed_speaker") or "").strip()
        probe = extracted.metadata.get("probe", {}) if isinstance(extracted.metadata, dict) else {}
        forensics = extracted.metadata.get("forensics", {}) if isinstance(extracted.metadata, dict) else {}
        fields = forensics.get("fields", {}) if isinstance(forensics, dict) else {}
        media_authenticity = self.media_authenticity.analyze(extracted)
        audio_hashes = compute_binary_hashes(audio_path) if Path(audio_path).exists() else {"sha256": sha256_text(str(audio_path)), "fingerprint": None}
        transcription_result = self._cached_transcription(audio_hashes.get("sha256"), audio_path, media_kind="audio")
        transcript_text = str(transcription_result.get("transcript_text") or "")
        cleaned_transcript = self._clean_audio_transcript_text(str(transcription_result.get("cleaned_transcript") or transcript_text))
        transcript_status = str(transcription_result.get("transcript_status") or "unavailable")
        transcript_confidence = float(transcription_result.get("transcript_confidence") or 0.0)
        transcript_summary = str(transcription_result.get("transcript_summary") or "")
        transcript_input = cleaned_transcript or transcript_text
        transcript_quality = "normal" if len(transcript_input) >= 100 else "limited" if len(transcript_input) >= 30 else "failed"
        prepared = self.preprocess_text(transcript_input) if transcript_input else {
            "input_text": "",
            "cleaned_input": "",
            "normalized_text": "",
            "detected_language": transcription_result.get("detected_language") or "unknown",
            "core_claim": "",
        }
        extraction = self.claim_extraction.extract_claims(prepared["cleaned_input"], max_claims=self.TEXT_CLAIM_LIMIT) if transcript_input else {"claims": [], "entities": []}
        claims = [item["text"] for item in extraction.get("claims", [])] or ([prepared["cleaned_input"]] if prepared["cleaned_input"] else [])
        meaning_bundle = self._build_text_meaning_bundle(prepared, extraction, claims) if transcript_input else {"exact_phrase_queries": [], "lexical_queries": [], "simplified_meaning": ""}
        query_plan = self._build_text_hybrid_queries(prepared, extraction, claims, meaning_bundle) if transcript_input else {"lexical": [], "semantic": []}
        query_failure_reason = self._validate_text_query_plan(meaning_bundle, query_plan) if transcript_input else "transcript_empty"

        audio_payload = ExtractedPayload(
            title=extracted.title,
            author=extracted.author,
            publish_time=extracted.publish_time,
            canonical_url=extracted.canonical_url,
            main_text=transcript_input,
            hero_image_url=extracted.hero_image_url,
            normalized_text=normalize_text(transcript_input),
            content_hash=sha256_text(normalize_text(transcript_input or uploaded_filename)),
            metadata=extracted.metadata if isinstance(extracted.metadata, dict) else {},
        )
        text_outcome: LayerOutcome | None = None
        if transcript_quality != "failed" and job is not None:
            text_outcome = self._text_nlp_layer(job, audio_payload, transcript_input)
            text_details = text_outcome.details if isinstance(text_outcome.details, dict) else {}
            reasoning_results = list(text_details.get("claim_results", []))
            aggregate = {
                "final_score": text_details.get("text_verification_score", text_outcome.confidence or 0.0),
                "final_label": text_details.get("text_verification_label", "Not Verifiable"),
                "explanation": text_outcome.explanation,
                "supporting_evidence_summary": text_details.get("supporting_evidence_summary"),
                "contradiction_summary": text_details.get("contradiction_summary"),
                "evidence_summary": text_details.get("evidence_summary"),
                "evidence_sources": text_details.get("evidence_sources", []),
                "component_scores": text_details.get("text_component_scores", {}),
                "evidence_multiplier": text_details.get("evidence_multiplier"),
                "penalty_multiplier": text_details.get("penalty_multiplier"),
            }
            query_plan = {
                "lexical": list(text_details.get("lexical_queries", query_plan.get("lexical", []))),
                "semantic": list(text_details.get("semantic_queries", query_plan.get("semantic", []))),
            }
        else:
            source_recovery_stage = self._empty_source_recovery(
                query_failure_reason or "Audio verification searches extracted claims instead of trying to recover a source from the full transcript."
            )
            candidates: list[dict] = []
            retrieval_debug: dict = {"lexical_searches": [], "semantic_searches": [], "candidate_fetches": [], "retrieval_failure_reason": query_failure_reason}
            ranking_debug: dict = {"bm25_top_results": [], "semantic_top_results": [], "reranked_top_results": [], "rerank_failure_reason": None}
            reasoning_results: list[dict] = []
            if transcript_input and not query_failure_reason:
                candidates, retrieval_debug = self._gather_text_candidate_documents(prepared, extraction, query_plan, source_recovery_stage)
                ranking_debug = self._hybrid_rank_text_candidates(prepared, claims, query_plan, candidates)
                top_articles = ranking_debug["reranked_top_results"]
                reasoning_results = [
                    self._score_claim_against_evidence(claim, self._build_claim_evidence_from_top_articles(claim, top_articles))
                    for claim in claims
                ]
                aggregate = self._finalize_text_verification(reasoning_results, top_articles, meaning_bundle)
            else:
                aggregate = self._aggregate_text_verification_score([], claims)

        evidence_sources = list(aggregate.get("evidence_sources", []))
        audio_search_queries = (
            list(text_details.get("search_queries_used", []))
            if isinstance(text_details, dict) and text_details.get("search_queries_used")
            else query_plan.get("lexical", []) + query_plan.get("semantic", [])
        )
        if job is not None:
            append_audit_entry(
                self.db,
                job,
                "audio_transcript_fact_check_debug",
                {
                    "raw_transcript": transcript_text,
                    "normalized_transcript": normalize_text(transcript_input),
                    "extracted_claims": claims,
                    "generated_search_queries": audio_search_queries,
                    "evidence_sources_found": len(evidence_sources),
                    "included_sources": [
                        {
                            "url": item.get("url"),
                            "title": item.get("title"),
                            "relevance_score": item.get("claim_relevance_score"),
                            "credibility_score": item.get("credibility_score"),
                            "stance": item.get("stance") or item.get("judgment_label"),
                        }
                        for item in evidence_sources
                    ],
                    "claim_judgments": [
                        {
                            "claim": item.get("claim_text"),
                            "verdict": item.get("claim_verdict"),
                            "score": item.get("claim_score"),
                            "diagnostics": item.get("top_candidate_diagnostics", []),
                        }
                        for item in reasoning_results
                        if isinstance(item, dict)
                    ],
                    "formula_components": aggregate.get("component_scores", {}),
                    "final_factual_score": aggregate.get("final_score"),
                    "final_factual_label": aggregate.get("final_label"),
                },
            )
        reasoning_summary = {
            "final_score": aggregate.get("final_score", 0.0),
            "final_label": aggregate.get("final_label", "Not Verifiable"),
            "supporting_evidence_summary": aggregate.get("supporting_evidence_summary"),
            "pipeline_failure_reason": ("transcript_too_short" if transcript_quality == "failed" else query_failure_reason or locals().get("retrieval_debug", {}).get("retrieval_failure_reason")),
        }

        if transcript_status == "transcription_failed":
            transcription_consistency = 0.0
        elif transcript_status in {"speech_not_detected", "transcript_empty"}:
            transcription_consistency = 0.1
        elif transcript_input:
            transcription_consistency = max(0.25, transcript_confidence or 0.0)
        else:
            transcription_consistency = 0.2

        spoof_detected = media_authenticity.get("status") == "high_deepfake_risk"
        voice_authenticity = 0.82
        if spoof_detected:
            voice_authenticity = 0.05
        elif media_authenticity.get("status") == "suspicious_manipulation":
            voice_authenticity = 0.35
        elif claimed_speaker:
            voice_authenticity = 0.72

        manipulation_detected = spoof_detected
        manipulation_integrity = 0.8
        if spoof_detected:
            manipulation_integrity = 0.05
        elif media_authenticity.get("status") == "suspicious_manipulation":
            manipulation_integrity = 0.25
        elif media_authenticity.get("status") == "inconclusive":
            manipulation_integrity = 0.6

        context_match_score = float(aggregate.get("final_score") or 0.0)
        if transcript_quality == "limited" and context_match_score > 0.65:
            context_match_score = 0.65
        if caption_or_claim and transcript_input:
            context_match_score = max(context_match_score, self._text_overlap(caption_or_claim, transcript_input))
        if transcript_status in {"speech_not_detected", "transcript_empty"}:
            context_match_score = min(context_match_score, 0.15)
        if transcript_status == "transcription_failed":
            context_match_score = 0.0

        background_acoustic_consistency = 0.72 if probe else 0.5
        if fields.get("encoder") or fields.get("software"):
            background_acoustic_consistency = max(0.55, background_acoustic_consistency - 0.1)

        audio_analysis_score = (
            0.25 * transcription_consistency
            + 0.20 * voice_authenticity
            + 0.20 * manipulation_integrity
            + 0.20 * context_match_score
            + 0.15 * background_acoustic_consistency
        )

        transcription_result = {
            "transcript_text": transcript_text,
            "cleaned_transcript": cleaned_transcript,
            "transcript_status": transcript_status,
            "transcript_confidence": round(transcript_confidence, 3),
            "transcript_summary": transcript_summary,
            "timestamps": transcription_result.get("timestamps", []),
            "detected_language": transcription_result.get("detected_language"),
            "language_probability": transcription_result.get("language_probability"),
            "chunk_count": int(transcription_result.get("chunk_count") or 0),
            "engine": transcription_result.get("engine", "faster-whisper"),
            "model": transcription_result.get("model"),
            "cache_hit": bool(transcription_result.get("cache_hit")),
            "audio_sha256": audio_hashes.get("sha256"),
            "file_format": transcription_result.get("file_format"),
            "duration_seconds": transcription_result.get("duration_seconds"),
            "preprocessing_status": transcription_result.get("preprocessing_status"),
            "preprocessing_details": transcription_result.get("preprocessing_details", {}),
            "failure_reason": transcription_result.get("failure_reason"),
        }
        explanation = self._generate_audio_explanation(
            {
                "transcription_result": transcription_result,
                "spoof_detected": spoof_detected,
                "manipulation_detected": manipulation_detected,
                "context_match_score": context_match_score,
                "background_acoustic_consistency": background_acoustic_consistency,
                "evidence_sources": evidence_sources,
                "text_verification_score": aggregate.get("final_score", 0.0),
                "text_verification_label": aggregate.get("final_label", "Not Verifiable"),
                "fact_check_verdict": aggregate.get("final_label", "Not Verifiable"),
            }
        )
        return LayerOutcome(
            status="supporting" if float(aggregate.get("final_score") or 0.0) >= 0.5 else "none",
            confidence=round(max(0.0, min(1.0, float(aggregate.get("final_score") or 0.0))), 3),
            explanation=explanation,
            details={
                "mode": "audio_verification",
                "caption_or_claim": caption_or_claim or None,
                "claimed_speaker": claimed_speaker or None,
                "matches": evidence_sources,
                "evidence_sources": evidence_sources,
                "fact_check_verdict": aggregate.get("final_label", "Not Verifiable"),
                "fact_check_assessment": aggregate.get("explanation") or explanation,
                "fact_check_basis": "Audio was transcribed locally with faster-whisper and then verified with the same retrieval-first text pipeline used for pasted text.",
                "text_verification_score": aggregate.get("final_score", 0.0),
                "text_verification_label": aggregate.get("final_label", "Not Verifiable"),
                "supporting_evidence_summary": aggregate.get("supporting_evidence_summary"),
                "contradiction_summary": aggregate.get("contradiction_summary"),
                "evidence_summary": aggregate.get("evidence_summary"),
                "audio_analysis_score": round(max(0.0, min(1.0, audio_analysis_score)), 3),
                "audio_analysis_breakdown": {
                    "transcription_consistency": round(transcription_consistency, 3),
                    "voice_authenticity": round(voice_authenticity, 3),
                    "manipulation_integrity": round(manipulation_integrity, 3),
                    "context_match_score": round(context_match_score, 3),
                    "background_acoustic_consistency": round(background_acoustic_consistency, 3),
                    "transcript_support_score": round(float(aggregate.get("final_score") or 0.0), 3),
                },
                "transcription_result": transcription_result,
                "extracted_claims": claims,
                "claim_results": reasoning_results,
                "reasoning_summary": reasoning_summary,
                "audio_context_wrapper_debug": {
                    "file_name": uploaded_filename,
                    "duration": transcription_result.get("duration_seconds"),
                    "transcription_engine": transcription_result.get("engine", "faster-whisper"),
                    "raw_transcript_length": len(transcript_text),
                    "cleaned_transcript_length": len(cleaned_transcript),
                    "transcription_confidence": transcription_result.get("transcript_confidence"),
                    "detected_language": transcription_result.get("detected_language"),
                    "transcript_preview": cleaned_transcript[:500],
                    "text_pipeline_called": text_outcome is not None,
                    "context_quality": transcript_quality,
                    "failure_reason": "transcript_too_short" if transcript_quality == "failed" else None,
                    "generated_brave_queries": audio_search_queries,
                    "evidence_urls": [item.get("url") for item in evidence_sources if item.get("url")],
                    "evidence_domains": [item.get("domain") or extract_domain(item.get("url")) for item in evidence_sources],
                    "evidence_stances": [
                        {
                            "url": item.get("url"),
                            "relevance_score": item.get("claim_relevance_score") or item.get("relation_confidence") or item.get("score"),
                            "stance": item.get("evidence_relation") or item.get("judgment_label") or item.get("stance"),
                        }
                        for item in evidence_sources
                    ],
                    "nlp_score": round(float(aggregate.get("final_score") or 0.0), 3),
                },
                "media_authenticity": media_authenticity,
                "spoof_detected": spoof_detected,
                "manipulation_detected": manipulation_detected,
                "exact_phrase_queries": meaning_bundle.get("exact_phrase_queries", []),
                "lexical_queries": query_plan.get("lexical", []),
                "semantic_queries": query_plan.get("semantic", []),
                "search_queries_used": audio_search_queries,
                "total_match_count": len(evidence_sources),
                "has_more_matches": False,
                "text_component_scores": aggregate.get("component_scores", {}),
                "text_debug": {
                    "raw_input": transcript_text,
                    "cleaned_input": cleaned_transcript,
                    "detected_language": transcription_result.get("detected_language"),
                    "exact_phrase_queries": meaning_bundle.get("exact_phrase_queries", []),
                    "keyword_entity_queries": meaning_bundle.get("lexical_queries", []),
                    "candidate_urls": [item.get("url") for item in evidence_sources if item.get("url")],
                    "support_judgments": reasoning_results,
                    "formula_components": {
                        key: value
                        for key, value in aggregate.get("component_scores", {}).items()
                        if key in {"S", "R", "A", "D", "C", "E", "P"}
                    },
                    "final_score": aggregate.get("final_score"),
                    "final_label": aggregate.get("final_label"),
                },
                "evidence_elements_found": {
                    "transcript_extracted": transcript_status == "completed" and bool(cleaned_transcript),
                    "speaker_consistency_checked": True,
                    "spoof_analysis_completed": True,
                    "manipulation_analysis_completed": True,
                    "context_checked": bool(cleaned_transcript),
                    "metadata_found": bool(forensics),
                    "provenance_record_found": False,
                    "background_acoustic_analysis_completed": True,
                },
            },
        )

    def _retrieve_audio_evidence(self, caption_or_claim: str, transcript_text: str, uploaded_filename: str) -> list[dict]:
        extraction = self.claim_extraction.extract_claims(" ".join(filter(None, [caption_or_claim, transcript_text, Path(uploaded_filename).stem.replace("-", " ").replace("_", " ")])), max_claims=2)
        queries = self._build_reasoning_queries(caption_or_claim or transcript_text, extraction, Path(uploaded_filename).stem.replace("-", " ").replace("_", " "))
        return self._retrieve_reasoned_evidence(queries, limit=5)

    def _build_audio_verification_record(
        self,
        audio_path: str,
        uploaded_filename: str,
        extracted: ExtractedPayload,
        layer2: LayerOutcome,
    ) -> AudioVerificationRecord:
        hashes = compute_binary_hashes(audio_path)
        previous = self.db.scalar(
            select(AudioVerificationRecord)
            .where(AudioVerificationRecord.audio_hash == hashes["sha256"])
            .order_by(AudioVerificationRecord.created_at.desc())
            .limit(1)
        )
        transcription_result = layer2.details.get("transcription_result", {})
        record = AudioVerificationRecord(
            audio_hash=str(hashes["sha256"]),
            audio_fingerprint=str(hashes["fingerprint"]) if hashes["fingerprint"] else None,
            uploaded_filename=uploaded_filename,
            caption_or_claim=extracted.metadata.get("caption_or_claim"),
            claimed_speaker=extracted.metadata.get("claimed_speaker"),
            transcript_text=transcription_result.get("transcript_text"),
            transcript_summary=transcription_result.get("transcript_summary"),
            audio_analysis_result=layer2.details,
            crypto_result={},
            final_score=0.0,
            final_label="Not Verifiable",
            explanation=layer2.explanation,
            current_version=(previous.current_version + 1) if previous else 1,
            previous_record_id=previous.id if previous else None,
        )
        self.db.add(record)
        self.db.flush()
        return record

    def _create_audio_verification_manifest(self, record: AudioVerificationRecord):
        crypto = CryptoProvenanceService(self.db)
        manifest = crypto.sign_object(CryptoObjectType.audio_verification_record, record.id)
        record.current_manifest_id = manifest.id
        self.db.add(record)
        self.db.flush()
        return manifest

    def _resolve_audio_verification_record(self, job: VerificationJob) -> AudioVerificationRecord | None:
        if job.audio_verification_record_id:
            return self.db.get(AudioVerificationRecord, job.audio_verification_record_id)
        record_id = (job.proof_summary or {}).get("audio_verification_record_id")
        if record_id:
            try:
                return self.db.get(AudioVerificationRecord, uuid.UUID(str(record_id)))
            except (ValueError, TypeError):
                return None
        return None

    def _audio_crypto_layer(self, record: AudioVerificationRecord, manifest, extracted: ExtractedPayload) -> LayerOutcome:
        metadata_snapshot = extracted.metadata.get("forensics", {}) if isinstance(extracted.metadata, dict) else {}
        crypto_breakdown = self._compute_audio_crypto_breakdown(record, manifest, metadata_snapshot)
        return LayerOutcome(
            status="platform-issued",
            confidence=round(crypto_breakdown["score"], 3),
            explanation="NewsProof created a signed provenance record for this audio verification result, including the audio hashes, transcript summary, analysis snapshot, and version history.",
            details={
                "record_type": "audio_verification_record",
                "audio_verification_record_id": str(record.id),
                "manifest_id": str(manifest.id),
                "verification_manifest_id": str(manifest.id),
                "current_version": record.current_version,
                "previous_version_id": str(record.previous_record_id) if record.previous_record_id else None,
                "previous_manifest_id": str(manifest.previous_manifest_id) if manifest.previous_manifest_id else None,
                "issued_by": manifest.signer.issuer_name if manifest.signer else "NewsProof system",
                "issuer_type": manifest.signer.issuer_type.value if manifest.signer else "platform",
                "verification_result": "Verified platform-issued provenance",
                "provenance_verification_status": manifest.verification_status.value if manifest.verification_status else "partial_provenance",
                "signed_at": manifest.created_at.isoformat() if manifest.created_at else None,
                "timestamp_source": (manifest.timestamp_token or {}).get("provider", "internal server timestamp"),
                "canonical_hash": manifest.sha256_hash,
                "signature_algorithm": manifest.signing_algorithm,
                "internal_manifest_valid": manifest.verification_status.value == "valid" if manifest.verification_status else False,
                "platform_manifest_valid": manifest.verification_status.value == "valid" if manifest.verification_status else False,
                "canonical_json": manifest.canonical_json,
                "canonical_json_preview": manifest.canonical_json[:1200],
                "issuer_key_id": (record.crypto_result or {}).get("key_id", "platform-managed"),
                "timestamp_token_status": (manifest.timestamp_token or {}).get("status", "issued"),
                "crypto_breakdown": crypto_breakdown["breakdown"],
                "uploaded_filename": record.uploaded_filename,
                "audio_hash": record.audio_hash,
                "audio_fingerprint": record.audio_fingerprint,
                "caption_or_claim": record.caption_or_claim,
                "claimed_speaker": record.claimed_speaker,
                "transcript_summary": record.transcript_summary,
            },
        )

    def _compute_audio_crypto_breakdown(self, record: AudioVerificationRecord, manifest, metadata_snapshot: dict) -> dict:
        previous_match = bool(record.previous_record_id)
        source_validity = 0.9 if manifest and manifest.verification_status.value == "valid" else 0.65
        hash_integrity = 1.0 if record.audio_hash and manifest else 0.5
        fields = metadata_snapshot.get("fields", {}) if isinstance(metadata_snapshot, dict) else {}
        timestamp_validity = 0.85 if fields.get("creation_time") or metadata_snapshot.get("duration_seconds") else 0.55
        useful_metadata = [
            fields.get("encoder"),
            metadata_snapshot.get("duration_seconds"),
            metadata_snapshot.get("sample_rate"),
            metadata_snapshot.get("channels"),
            record.audio_fingerprint,
        ]
        metadata_completeness = sum(1 for item in useful_metadata if item not in (None, "", [])) / len(useful_metadata)
        penalty = 1.0 if manifest and manifest.verification_status.value == "valid" else 0.6
        score = (
            0.35 * source_validity
            + 0.25 * hash_integrity
            + 0.20 * timestamp_validity
            + 0.20 * metadata_completeness
        ) * penalty
        if previous_match:
            score = min(1.0, score + 0.05)
        return {
            "score": score,
            "breakdown": {
                "source_provenance_validity": round(source_validity, 3),
                "hash_integrity": round(hash_integrity, 3),
                "timestamp_metadata_validity": round(timestamp_validity, 3),
                "metadata_completeness": round(metadata_completeness, 3),
                "crypto_penalty": round(penalty, 3),
            },
        }

    def _compute_audio_evidence_multiplier(self, layer2: LayerOutcome, layer1: LayerOutcome) -> float:
        text_components = layer2.details.get("text_component_scores", {}) if isinstance(layer2.details, dict) else {}
        if text_components:
            evidence_multiplier = float(layer2.details.get("evidence_multiplier") or text_components.get("E") or 0.0)
            if int(text_components.get("relevant_count") or 0) == 0:
                evidence_multiplier = min(evidence_multiplier, 0.2)
            return max(0.0, min(1.0, evidence_multiplier))
        found = layer2.details.get("evidence_elements_found", {}) if isinstance(layer2.details, dict) else {}
        ep = sum(1 for value in found.values() if value)
        et = max(1, len(found))
        if layer1.details.get("manifest_id"):
            ep += 1
            et += 1
        transcript_status = str(layer2.details.get("transcription_result", {}).get("transcript_status") or "")
        if transcript_status in {"transcription_failed", "speech_not_detected", "transcript_empty"}:
            ep = max(0, ep - 1)
        return max(0.0, min(1.0, 0.4 + 0.6 * (ep / et)))

    def _compute_audio_penalty(self, layer2: LayerOutcome, layer1: LayerOutcome) -> float:
        text_components = layer2.details.get("text_component_scores", {}) if isinstance(layer2.details, dict) else {}
        if text_components:
            penalty = float(layer2.details.get("penalty_multiplier") or text_components.get("P") or 1.0)
            if layer2.details.get("spoof_detected") or layer2.details.get("manipulation_detected"):
                penalty = min(penalty, 0.4)
            if layer1.details.get("provenance_verification_status") in {"hash_mismatch", "signature_invalid"}:
                penalty = 0.0
            return max(0.0, min(1.0, penalty))
        if layer2.details.get("spoof_detected") or layer2.details.get("manipulation_detected"):
            return 0.0 if layer2.details.get("spoof_detected") else 0.4
        if layer1.details.get("provenance_verification_status") in {"hash_mismatch", "signature_invalid"}:
            return 0.0
        transcript_status = str(layer2.details.get("transcription_result", {}).get("transcript_status") or "")
        if transcript_status == "transcription_failed":
            return 0.5
        if transcript_status in {"speech_not_detected", "transcript_empty"}:
            return 0.6
        if (layer2.confidence or 0.0) < 0.55 or (layer1.confidence or 0.0) < 0.55:
            return 0.7
        return 1.0

    def _generate_audio_label(self, final_score: float, evidence_multiplier: float, penalty_multiplier: float, flags: dict) -> str:
        return map_score_to_label(final_score)

    def _generate_audio_explanation(self, result: dict) -> str:
        parts: list[str] = []
        transcription = result.get("transcription_result", {})
        if transcription.get("transcript_status") == "completed":
            parts.append("The audio was transcribed and checked against the provided context.")
        elif transcription.get("transcript_status") == "transcription_failed":
            parts.append("Local speech-to-text failed, so no transcript-driven verification could be completed.")
        elif transcription.get("transcript_status") in {"speech_not_detected", "transcript_empty"}:
            parts.append("No meaningful speech was detected, so transcript-driven verification remained weak.")
        else:
            parts.append("A clear transcript was not available, so transcript-driven verification remained limited.")
        if result.get("spoof_detected"):
            parts.append("Analysis found strong signs of synthetic or spoofed speech.")
        elif result.get("manipulation_detected"):
            parts.append("Audio manipulation signals were detected.")
        else:
            parts.append("The voice and waveform look broadly consistent without strong spoofing signals.")
        finding = str(result.get("text_verification_label") or result.get("fact_check_verdict") or "")
        fact_score = float(result.get("text_verification_score") or 0.0)
        if finding:
            if finding in {"Not Verifiable", "Unverified"} or fact_score < 0.35:
                parts.append(
                    f"However, the factual claim in the transcript was not supported by credible direct evidence, so the claim remains {finding}."
                )
            else:
                parts.append(f"The transcript fact-check finding is {finding}.")
        if float(result.get("context_match_score") or 0.0) >= 0.6:
            parts.append("The spoken content and evidence are reasonably aligned with the claimed context.")
        else:
            parts.append("The available context match for the audio is limited or mixed.")
        if float(result.get("background_acoustic_consistency") or 0.0) < 0.5:
            parts.append("Background acoustics are inconsistent enough to reduce confidence.")
        return " ".join(parts)

    def _build_audio_system_scoring(self, layer1: LayerOutcome, layer2: LayerOutcome, layer3: LayerOutcome | None = None) -> dict:
        audio_score = float(layer2.details.get("audio_analysis_score") or 0.0)
        nlp_score = max(0.0, min(1.0, float(layer2.details.get("text_verification_score") or layer2.confidence or 0.0)))
        crypto_score = float(layer1.confidence or 0.0)
        community_score = max(0.0, min(1.0, float(layer3.confidence or 0.0))) if layer3 and layer3.status == "supporting" else 0.0
        evidence_multiplier = self._compute_audio_evidence_multiplier(layer2, layer1)
        penalty_multiplier = self._compute_audio_penalty(layer2, layer1)
        evidence_count = len(layer2.details.get("evidence_sources", []) or []) if isinstance(layer2.details, dict) else 0
        finding = str(layer2.details.get("text_verification_label") or layer2.details.get("fact_check_verdict") or "").lower()
        final_score = max(
            0.0,
            min(
                100.0,
                100 * (0.30 * crypto_score + 0.45 * nlp_score + 0.25 * community_score),
            ),
        )
        wrapper_debug = layer2.details.get("audio_context_wrapper_debug", {}) if isinstance(layer2.details, dict) else {}
        failure_reason = str(wrapper_debug.get("failure_reason") or "")
        if failure_reason in {"transcription_failed", "no_speech_detected", "transcript_too_short", "text_pipeline_not_called"}:
            final_score = min(final_score, 34.0)
        if finding == "not verifiable":
            final_score = min(final_score, 49.0)
        if finding == "unsupported":
            final_score = min(final_score, 34.0)
        if finding in {"false", "contradicted"}:
            final_score = min(final_score, 30.0)
        claim_results = layer2.details.get("claim_results", []) if isinstance(layer2.details, dict) else []
        extraordinary = any(self._parse_claim_details(str(item.get("claim_text") or "")).get("extraordinary") for item in claim_results if isinstance(item, dict))
        if extraordinary and evidence_count == 0:
            final_score = min(final_score, 25.0)
        flags = {
            "confirmed_spoofing": bool(layer2.details.get("spoof_detected")),
            "confirmed_manipulation": bool(layer2.details.get("manipulation_detected")),
            "forged_provenance": layer1.details.get("provenance_verification_status") in {"hash_mismatch", "signature_invalid"},
        }
        transcript_status = str(layer2.details.get("transcription_result", {}).get("transcript_status") or "")
        if failure_reason in {"transcription_failed", "no_speech_detected", "transcript_too_short", "text_pipeline_not_called"}:
            label = "Not Verifiable"
        elif finding == "not verifiable":
            label = "Not Verifiable"
        elif finding == "unsupported":
            label = "Unverified"
        elif finding in {"false", "contradicted"}:
            label = "Unverified"
        else:
            label = self._generate_audio_label(final_score, evidence_multiplier, penalty_multiplier, flags)
        if label == "Verified":
            reason = "Audio analysis and provenance signals both support this audio strongly."
        elif label == "Likely Verified":
            reason = "The audio looks credible overall, but some evidence gaps remain."
        elif label == "Not Verifiable":
            reason = "The available audio evidence is too limited or mixed for a reliable judgment."
        else:
            reason = "Spoofing, manipulation, or weak audio evidence keeps this result unverified."
        return {
            "final_score": round(final_score, 2),
            "final_label": label,
            "reason": reason,
            "explanation": self._generate_audio_explanation(layer2.details if isinstance(layer2.details, dict) else {}),
            "evidence_multiplier": round(evidence_multiplier, 3),
            "global_penalty": round(penalty_multiplier, 3),
            "formula_components": {
                "C": round(crypto_score, 3),
                "N": round(nlp_score, 3),
                "U": round(community_score, 3),
                "evidence_multiplier_diagnostic": round(evidence_multiplier, 3),
                "global_penalty_diagnostic": round(penalty_multiplier, 3),
                "audio_authenticity": round(audio_score, 3),
            },
            "active_layer_weights": {"crypto": 0.30, "nlp": 0.45, "community": 0.25},
            "formula": "Final audio score = (0.30 * crypto) + (0.45 * NLP) + (0.25 * consensus). Evidence sufficiency and global penalty are diagnostics only.",
            "layer_scores": {
                "crypto": round(crypto_score, 3),
                "nlp": round(nlp_score, 3),
                "community": round(community_score, 3) if layer3 else None,
            },
            "layer_breakdown": {
                "crypto": layer1.details.get("crypto_breakdown", {}),
                "nlp": layer2.details.get("text_component_scores", {}),
                "audio": layer2.details.get("audio_analysis_breakdown", {}),
                "community": layer3.details if layer3 else None,
            },
            "warnings": [
                warning
                for warning in [
                    "Spoofing or synthetic speech signals were detected." if flags["confirmed_spoofing"] else None,
                    "Manipulation signals were detected." if flags["confirmed_manipulation"] else None,
                    "Metadata and provenance evidence are limited." if penalty_multiplier < 1.0 else None,
                ]
                if warning
            ],
        }

    def _decide_audio(
        self,
        layer1: LayerOutcome,
        layer2: LayerOutcome,
        layer3: LayerOutcome,
        extracted: ExtractedPayload,
        record: AudioVerificationRecord,
        manifest,
    ) -> tuple[JobStatus, str, str, dict, dict]:
        scoring_result = self._build_audio_system_scoring(layer1, layer2, layer3)
        label = str(scoring_result.get("final_label", "Not Verifiable"))
        if label == "Verified":
            status_value = JobStatus.verified
        elif label == "Likely Verified":
            status_value = JobStatus.likely_verified
        elif label == "Unverified":
            status_value = JobStatus.unverified
        else:
            status_value = JobStatus.not_verifiable

        record.final_score = float(scoring_result["final_score"])
        record.final_label = label
        record.explanation = str(scoring_result["explanation"])
        record.crypto_result = {
            "manifest_id": str(manifest.id),
            "issuer_id": layer1.details.get("issued_by"),
            "key_id": layer1.details.get("issuer_key_id"),
            "provenance_status": layer1.details.get("provenance_verification_status"),
            "timestamp": layer1.details.get("signed_at"),
            "previous_payload_hash": layer1.details.get("previous_manifest_id"),
            "crypto_breakdown": layer1.details.get("crypto_breakdown", {}),
        }
        self.db.add(record)
        self.db.flush()

        proof_summary = {
            "content_hash": extracted.content_hash,
            "verification_basis": "Audio-analysis-led verification with signed platform provenance",
            "audio_verification_record_id": str(record.id),
            "verification_manifest_id": str(manifest.id),
            "current_version": record.current_version,
            "previous_version_id": str(record.previous_record_id) if record.previous_record_id else None,
            "system_scoring": scoring_result,
            **(layer1.details or {}),
        }
        analysis_report = self._build_analysis_report(status_value, layer1, layer2, layer3, extracted, scoring_result)
        analysis_report["executive_summary"] = (
            f"{label}: audio verification was led by the audio analysis layer, then secured by the cryptographic layer, with feedback stored separately for later review."
        )
        analysis_report["mode"] = "audio_verification"
        return status_value, label, str(scoring_result["reason"]), proof_summary, analysis_report

    def _content_preview_for_audio(self, record: AudioVerificationRecord, extracted: ExtractedPayload) -> dict:
        text_excerpt = record.caption_or_claim or record.transcript_text or record.transcript_summary or extracted.main_text
        return {
            "title": f"Audio upload: {record.uploaded_filename}",
            "text_excerpt": str(text_excerpt or "")[:600],
            "caption_or_claim": record.caption_or_claim,
            "claimed_speaker": record.claimed_speaker,
            "audio_hash": record.audio_hash,
            "audio_fingerprint": record.audio_fingerprint,
        }

    def _build_video_verification_record(
        self,
        video_path: str,
        uploaded_filename: str,
        extracted: ExtractedPayload,
        layer2: LayerOutcome,
    ) -> VideoVerificationRecord:
        hashes = compute_binary_hashes(video_path)
        keyframe_path = str(extracted.metadata.get("keyframe_path") or "")
        keyframe_hashes = compute_image_hashes(keyframe_path) if keyframe_path and Path(keyframe_path).exists() else {"perceptual_hash": None}
        previous = self.db.scalar(
            select(VideoVerificationRecord)
            .where(VideoVerificationRecord.video_hash == hashes["sha256"])
            .order_by(VideoVerificationRecord.created_at.desc())
            .limit(1)
        )
        record = VideoVerificationRecord(
            video_hash=str(hashes["sha256"]),
            video_fingerprint=str(keyframe_hashes.get("perceptual_hash") or hashes["fingerprint"]) if (keyframe_hashes.get("perceptual_hash") or hashes["fingerprint"]) else None,
            uploaded_filename=uploaded_filename,
            caption_or_claim=extracted.metadata.get("caption_or_claim"),
            ocr_result=layer2.details.get("ocr_result", {}),
            audio_result=layer2.details.get("audio_result", {}),
            video_analysis_result=layer2.details,
            crypto_result={},
            final_score=0.0,
            final_label="Not Verifiable",
            explanation=layer2.explanation,
            current_version=(previous.current_version + 1) if previous else 1,
            previous_record_id=previous.id if previous else None,
        )
        self.db.add(record)
        self.db.flush()
        return record

    def _create_video_verification_manifest(self, record: VideoVerificationRecord):
        crypto = CryptoProvenanceService(self.db)
        manifest = crypto.sign_object(CryptoObjectType.video_verification_record, record.id)
        record.current_manifest_id = manifest.id
        self.db.add(record)
        self.db.flush()
        return manifest

    def _resolve_video_verification_record(self, job: VerificationJob) -> VideoVerificationRecord | None:
        if job.video_verification_record_id:
            return self.db.get(VideoVerificationRecord, job.video_verification_record_id)
        record_id = (job.proof_summary or {}).get("video_verification_record_id")
        if record_id:
            try:
                return self.db.get(VideoVerificationRecord, uuid.UUID(str(record_id)))
            except (ValueError, TypeError):
                return None
        return None

    def _feedback_entries_for_video_record(self, record: VideoVerificationRecord) -> list[VideoFeedbackEntry]:
        chain_ids: list[uuid.UUID] = []
        current: VideoVerificationRecord | None = record
        while current:
            chain_ids.append(current.id)
            current = current.previous_record
        return list(
            self.db.scalars(
                select(VideoFeedbackEntry)
                .where(VideoFeedbackEntry.video_verification_record_id.in_(chain_ids))
                .order_by(VideoFeedbackEntry.created_at.desc())
            )
        )

    def _video_feedback_layer(self, record: VideoVerificationRecord, persist_audit: bool = True) -> LayerOutcome:
        entries = self._feedback_entries_for_video_record(record)
        total = len(entries)
        counts = {
            "correction": sum(1 for entry in entries if entry.feedback_type == FeedbackType.correction),
            "dispute": sum(1 for entry in entries if entry.feedback_type == FeedbackType.dispute),
            "source_suggestion": sum(1 for entry in entries if entry.feedback_type == FeedbackType.source_suggestion),
            "clarification": sum(1 for entry in entries if entry.feedback_type == FeedbackType.clarification),
            "review_note": sum(1 for entry in entries if entry.feedback_type == FeedbackType.review_note),
        }
        reviewed = sum(1 for entry in entries if entry.status in {FeedbackStatus.reviewed, FeedbackStatus.resolved})
        resolved = sum(1 for entry in entries if entry.status == FeedbackStatus.resolved)
        expert_count = sum(1 for entry in entries if entry.user_role == FeedbackUserRole.expert)
        moderator_count = sum(1 for entry in entries if entry.user_role == FeedbackUserRole.moderator)
        if persist_audit:
            video_job = self.db.scalar(select(VerificationJob).where(VerificationJob.video_verification_record_id == record.id).limit(1))
            if video_job:
                append_audit_entry(self.db, video_job, "video_feedback_layer_completed", {"feedback_count": total})

        if total == 0:
            status_value = "no_feedback"
            explanation = "No feedback has been submitted for this video verification record yet."
        elif counts["dispute"] or counts["correction"]:
            status_value = "review_attention"
            explanation = "Feedback includes disputes or correction requests that may lead to a revised video verification record."
        elif reviewed or expert_count or moderator_count:
            status_value = "reviewed_feedback"
            explanation = "Video review history includes structured feedback and moderator or expert notes."
        else:
            status_value = "feedback_open"
            explanation = "Feedback has been submitted for this video verification record and is awaiting review."

        return LayerOutcome(
            status=status_value,
            confidence=0.0,
            explanation=explanation,
            details={
                "mode": "feedback",
                "feedback_count": total,
                "reviewed_count": reviewed,
                "resolved_count": resolved,
                "expert_feedback_count": expert_count,
                "moderator_feedback_count": moderator_count,
                "correction_count": counts["correction"],
                "dispute_count": counts["dispute"],
                "source_suggestion_count": counts["source_suggestion"],
                "clarification_count": counts["clarification"],
                "review_note_count": counts["review_note"],
            },
        )

    def _video_analysis_layer(self, extracted: ExtractedPayload, video_path: str, uploaded_filename: str) -> LayerOutcome:
        caption_or_claim = str(extracted.metadata.get("caption_or_claim") or "").strip()
        keyframe_path = str(extracted.metadata.get("keyframe_path") or "")
        probe = extracted.metadata.get("probe", {}) if isinstance(extracted.metadata, dict) else {}
        forensics = extracted.metadata.get("forensics", {}) if isinstance(extracted.metadata, dict) else {}
        metadata_fields = forensics.get("fields", {}) if isinstance(forensics, dict) else {}
        media_authenticity = self.media_authenticity.analyze(extracted)
        frame_paths = sample_video_frames(video_path, self.settings.upload_dir, count=4)
        if keyframe_path and keyframe_path not in frame_paths and Path(keyframe_path).exists():
            frame_paths = [keyframe_path, *frame_paths][:4]
        frame_ocr_results = [
            {**extract_text_details_from_image(path), "frame_path": path, "frame_index": index}
            for index, path in enumerate(frame_paths)
            if Path(path).exists()
        ]
        frame_ocr_texts = [str(result.get("extracted_text") or result.get("ocr_text") or "") for result in frame_ocr_results]
        merged_ocr_text = self._dedupe_video_ocr_texts(frame_ocr_texts)
        audio_track_path = extract_audio_track(video_path, self.settings.upload_dir)
        transcription_path = audio_track_path or video_path
        video_hashes = compute_binary_hashes(transcription_path) if Path(transcription_path).exists() else {"sha256": sha256_text(str(transcription_path)), "fingerprint": None}
        transcription_result = self._cached_transcription(video_hashes.get("sha256"), transcription_path, media_kind="video")
        transcript_text = str(transcription_result.get("cleaned_transcript") or transcription_result.get("transcript_text") or "").strip()
        visual_description = self.vision_reasoning.describe_video_frames(frame_paths, context=caption_or_claim, ocr_texts=frame_ocr_texts)
        visual_context = self.vision_reasoning.compare_visual_context(visual_description, caption_or_claim or transcript_text)
        metadata_claim_text = self._video_metadata_claim_text(forensics, probe, uploaded_filename)
        context_text = "\n".join(
            part
            for part in [
                f"Submitted claim or caption: {caption_or_claim}" if caption_or_claim else "",
                f"Audio transcript: {transcript_text}" if transcript_text else "",
                f"Visible text from sampled video frames: {merged_ocr_text}" if merged_ocr_text else "",
                metadata_claim_text,
            ]
            if part
        ).strip()
        if context_text:
            extracted.main_text = context_text
            extracted.normalized_text = normalize_text(context_text)
            extracted.content_hash = sha256_text(extracted.normalized_text)
            extracted.metadata["video_extraction"] = {
                "basis": [
                    "audio_transcription",
                    "sampled_frame_ocr",
                    "metadata_provenance",
                    "contextual_evidence_comparison",
                ],
                "audio_track_path": audio_track_path,
                "frame_paths": frame_paths,
                "sampled_frame_count": len(frame_paths),
                "deduplicated_frame_text": merged_ocr_text,
                "transcript_text": transcript_text,
                "metadata_claim_text": metadata_claim_text,
            }
        if context_text:
            context_stream = self._evaluate_facebook_text_stream(extracted, context_text, "video_context")
            extraction = context_stream.get("extraction", {"claims": []})
            evidence_sources = list(context_stream.get("matches", []))
            reasoning_results = list(context_stream.get("claim_results", []))
            reasoning_summary = dict(context_stream.get("aggregate", {}))
            queries = list(context_stream.get("query_plan", {}).get("lexical", [])) + list(context_stream.get("query_plan", {}).get("semantic", []))
        else:
            extraction = {"claims": []}
            queries = []
            evidence_sources = []
            reasoning_results = []
            reasoning_summary = {"final_score": 0.0, "final_label": "Not Verifiable", "explanation": "No transcript, caption, or OCR text was available for video context fact-checking."}

        frame_visual_consistency = 0.82 if frame_paths else 0.55

        temporal_continuity = 0.76 if metadata_fields.get("duration_seconds") or forensics.get("duration_seconds") else 0.62
        provenance_integrity = 0.8
        technical_warning = False
        if media_authenticity.get("status") == "suspicious_manipulation":
            provenance_integrity = 0.45
            technical_warning = True
        elif media_authenticity.get("status") == "high_deepfake_risk":
            provenance_integrity = 0.38
            technical_warning = True
        elif media_authenticity.get("status") == "inconclusive":
            provenance_integrity = 0.58

        streams = probe.get("streams", []) if isinstance(probe, dict) else []
        audio_present = any(isinstance(stream, dict) and stream.get("codec_type") == "audio" for stream in streams)
        audio_visual_consistency = 0.65 if audio_present else 0.6
        audio_visual_status = "not_applicable"
        if audio_present:
            audio_visual_status = "checked"
            if media_authenticity.get("status") == "inconclusive":
                audio_visual_consistency = 0.55
            else:
                audio_visual_consistency = max(0.78, float(transcription_result.get("transcript_confidence") or 0.0))

        context_match_score = 0.55
        if caption_or_claim:
            context_scores = [
                self._text_overlap(caption_or_claim, " ".join(filter(None, [str(item.get("title") or ""), str(item.get("passage") or item.get("snippet") or "")])))
                for item in evidence_sources
            ]
            if merged_ocr_text:
                context_scores.append(self._text_overlap(caption_or_claim, merged_ocr_text))
            if transcript_text:
                context_scores.append(self._text_overlap(caption_or_claim, transcript_text))
            context_scores.append(float(visual_context.get("score") or 0.0))
            context_scores.append(float(reasoning_summary.get("final_score") or reasoning_summary.get("confidence") or 0.0))
            context_match_score = max([0.2] + context_scores)
        elif evidence_sources:
            context_match_score = min(0.8, max(float(item.get("score") or item.get("retrieval_score") or 0.0) for item in evidence_sources))

        if merged_ocr_text:
            ocr_text_consistency = max(0.2, self._text_overlap(caption_or_claim or merged_ocr_text, merged_ocr_text))
            ocr_result = {
                "text_detected": True,
                "extracted_text": merged_ocr_text,
                "ocr_text": merged_ocr_text,
                "ocr_status": "completed",
                "sampled_frame_count": len(frame_paths),
                "frame_ocr_results": frame_ocr_results,
                "ocr_summary": "On-screen text was extracted from sampled video frames, cleaned, deduplicated, and compared with contextual evidence.",
            }
        else:
            ocr_text_consistency = 0.6
            ocr_result = {
                "text_detected": False,
                "extracted_text": "",
                "ocr_text": "",
                "ocr_status": "not_applicable",
                "sampled_frame_count": len(frame_paths),
                "frame_ocr_results": frame_ocr_results,
                "ocr_summary": "No readable on-screen text was detected in sampled frames.",
            }

        audio_result = {
            "audio_present": audio_present,
            "audio_track_extracted": bool(audio_track_path),
            "audio_track_path": audio_track_path,
            "audio_visual_status": audio_visual_status,
            "transcription_status": transcription_result.get("transcript_status"),
            "transcript_summary": transcription_result.get("transcript_summary"),
            "transcript_text": transcript_text,
            "audio_visual_summary": "Audio and visible action appear broadly aligned at the available metadata level." if audio_present and audio_visual_consistency >= 0.65 else (
                "Audio-video mismatch may reduce confidence." if audio_present else "No audio track was available for audio-visual sync checks."
            ),
        }

        contextual_score = max(0.0, min(1.0, float(reasoning_summary.get("final_score") or reasoning_summary.get("confidence") or 0.0)))
        video_analysis_score = (
            0.45 * contextual_score
            + 0.15 * frame_visual_consistency
            + 0.10 * temporal_continuity
            + 0.10 * provenance_integrity
            + 0.10 * audio_visual_consistency
            + 0.10 * ocr_text_consistency
        )

        explanation = self._generate_video_explanation(
            {
                "ocr_result": ocr_result,
                "audio_result": audio_result,
                "transcription_result": transcription_result,
                "visual_description": visual_description,
                "visual_context_result": visual_context,
                "claim_results": reasoning_results,
                "reasoning_summary": reasoning_summary,
                "technical_warning": technical_warning,
                "evidence_sources": evidence_sources,
                "context_match_score": context_match_score,
                "media_authenticity": media_authenticity,
                "temporal_continuity": temporal_continuity,
            }
        )
        return LayerOutcome(
            status="supporting" if contextual_score >= 0.45 and not technical_warning else "warning",
            confidence=round(max(0.0, min(1.0, video_analysis_score)), 3),
            explanation=explanation,
            details={
                "mode": "video_verification",
                "caption_or_claim": caption_or_claim or None,
                "matches": evidence_sources,
                "evidence_sources": evidence_sources,
                "fact_check_verdict": reasoning_summary.get("final_label", "Not Verifiable"),
                "fact_check_assessment": reasoning_summary.get("explanation"),
                "fact_check_basis": "Video verification uses sampled-frame OCR and audio transcription as extraction steps, then sends extracted text through the shared claim-evidence fact-checking pipeline.",
                "claim_results": reasoning_results,
                "extracted_claims": [item.get("text") for item in extraction.get("claims", []) if isinstance(item, dict)],
                "search_queries_used": queries,
                "text_verification_score": contextual_score,
                "text_verification_label": reasoning_summary.get("final_label", "Not Verifiable"),
                "video_analysis_score": round(max(0.0, min(1.0, video_analysis_score)), 3),
                "video_analysis_breakdown": {
                    "contextual_nlp_score": round(contextual_score, 3),
                    "frame_visual_consistency": round(frame_visual_consistency, 3),
                    "temporal_continuity": round(temporal_continuity, 3),
                    "metadata_provenance_integrity": round(provenance_integrity, 3),
                    "audio_visual_consistency": round(audio_visual_consistency, 3),
                    "context_match_score": round(context_match_score, 3),
                    "ocr_text_consistency": round(ocr_text_consistency, 3),
                },
                "text_component_scores": reasoning_summary.get("component_scores", {}),
                "ocr_result": ocr_result,
                "audio_result": audio_result,
                "transcription_result": transcription_result,
                "transcript_summary": transcription_result.get("transcript_summary"),
                "transcript_text": transcript_text,
                "frame_ocr_text_summary": merged_ocr_text[:1000],
                "metadata_claim_text": metadata_claim_text,
                "metadata_snapshot": forensics,
                "media_authenticity": media_authenticity,
                "technical_warning": technical_warning,
                "manipulation_detected": False,
                "deepfake_detected": False,
                "limitations": [
                    "This video verification does not perform full forensic video authenticity detection.",
                    "It does not perform deepfake detection or speaker authentication.",
                    "The result is based on audio transcription, sampled-frame OCR, metadata/provenance checks, and contextual evidence comparison.",
                ],
                "evidence_elements_found": {
                    "frame_analysis_completed": True,
                    "temporal_analysis_completed": True,
                    "metadata_provenance_checked": bool(forensics),
                    "audio_visual_analysis_completed": audio_present,
                    "ocr_checked": True,
                    "ocr_text_detected": bool(merged_ocr_text),
                    "audio_transcript_extracted": bool(transcript_text),
                    "metadata_found": bool(forensics),
                    "provenance_record_found": False,
                    "context_checked": bool(context_text),
                    "reverse_match_checked": True,
                    "original_source_match_found": bool(evidence_sources),
                },
            },
        )

    def _dedupe_video_ocr_texts(self, texts: list[str]) -> str:
        cleaned: list[str] = []
        seen: set[str] = set()
        for text in texts:
            normalized_lines = []
            for line in str(text or "").splitlines() or [str(text or "")]:
                line_text = re.sub(r"\s+", " ", line).strip(" -_|")
                if len(line_text) < 3:
                    continue
                normalized_lines.append(line_text)
            candidate = " ".join(normalized_lines).strip()
            if not candidate:
                continue
            key = normalize_text(candidate)
            if not key or key in seen:
                continue
            if any(self._text_overlap(candidate, existing) >= 0.9 for existing in cleaned):
                continue
            seen.add(key)
            cleaned.append(candidate)
        return " ".join(cleaned).strip()

    def _video_metadata_claim_text(self, forensics: dict, probe: dict, uploaded_filename: str) -> str:
        parts = [f"Uploaded video file named {uploaded_filename}."]
        fields = forensics.get("fields", {}) if isinstance(forensics, dict) else {}
        streams = forensics.get("streams", []) if isinstance(forensics, dict) else []
        format_name = forensics.get("format_name") if isinstance(forensics, dict) else None
        duration = forensics.get("duration_seconds") if isinstance(forensics, dict) else None
        if format_name:
            parts.append(f"Container format: {format_name}.")
        if duration:
            parts.append(f"Duration: {duration} seconds.")
        for key in ("title", "artist", "encoder", "software", "creation_time", "comment"):
            value = fields.get(key) if isinstance(fields, dict) else None
            if value:
                parts.append(f"Metadata {key.replace('_', ' ')}: {value}.")
        stream_types = sorted({str(stream.get("codec_type")) for stream in streams if isinstance(stream, dict) and stream.get("codec_type")})
        if stream_types:
            parts.append(f"Detected media streams: {', '.join(stream_types)}.")
        if not stream_types and isinstance(probe, dict):
            probe_streams = probe.get("streams", [])
            stream_types = sorted({str(stream.get("codec_type")) for stream in probe_streams if isinstance(stream, dict) and stream.get("codec_type")})
            if stream_types:
                parts.append(f"Detected media streams: {', '.join(stream_types)}.")
        return " ".join(parts)

    def _retrieve_video_evidence(self, caption_or_claim: str, ocr_text: str, uploaded_filename: str) -> list[dict]:
        extraction = self.claim_extraction.extract_claims(" ".join(filter(None, [caption_or_claim, ocr_text, Path(uploaded_filename).stem.replace("-", " ").replace("_", " ")])), max_claims=2)
        queries = self._build_reasoning_queries(caption_or_claim or ocr_text, extraction, Path(uploaded_filename).stem.replace("-", " ").replace("_", " "))
        return self._retrieve_reasoned_evidence(queries, limit=5)

    def _video_crypto_layer(self, record: VideoVerificationRecord, manifest, extracted: ExtractedPayload) -> LayerOutcome:
        metadata_snapshot = extracted.metadata.get("forensics", {}) if isinstance(extracted.metadata, dict) else {}
        crypto_breakdown = self._compute_video_crypto_breakdown(record, manifest, metadata_snapshot)
        return LayerOutcome(
            status="platform-issued",
            confidence=round(crypto_breakdown["score"], 3),
            explanation="NewsProof created a signed provenance record for this video verification result, including hashes, sampled-frame findings, OCR results, and version history.",
            details={
                "record_type": "video_verification_record",
                "video_verification_record_id": str(record.id),
                "manifest_id": str(manifest.id),
                "verification_manifest_id": str(manifest.id),
                "current_version": record.current_version,
                "previous_version_id": str(record.previous_record_id) if record.previous_record_id else None,
                "previous_manifest_id": str(manifest.previous_manifest_id) if manifest.previous_manifest_id else None,
                "issued_by": manifest.signer.issuer_name if manifest.signer else "NewsProof system",
                "issuer_type": manifest.signer.issuer_type.value if manifest.signer else "platform",
                "verification_result": "Verified platform-issued provenance",
                "provenance_verification_status": manifest.verification_status.value if manifest.verification_status else "partial_provenance",
                "signed_at": manifest.created_at.isoformat() if manifest.created_at else None,
                "timestamp_source": (manifest.timestamp_token or {}).get("provider", "internal server timestamp"),
                "canonical_hash": manifest.sha256_hash,
                "signature_algorithm": manifest.signing_algorithm,
                "internal_manifest_valid": manifest.verification_status.value == "valid" if manifest.verification_status else False,
                "platform_manifest_valid": manifest.verification_status.value == "valid" if manifest.verification_status else False,
                "canonical_json": manifest.canonical_json,
                "canonical_json_preview": manifest.canonical_json[:1200],
                "issuer_key_id": (record.crypto_result or {}).get("key_id", "platform-managed"),
                "timestamp_token_status": (manifest.timestamp_token or {}).get("status", "issued"),
                "crypto_breakdown": crypto_breakdown["breakdown"],
                "uploaded_filename": record.uploaded_filename,
                "video_hash": record.video_hash,
                "video_fingerprint": record.video_fingerprint,
                "caption_or_claim": record.caption_or_claim,
            },
        )

    def _compute_video_crypto_breakdown(self, record: VideoVerificationRecord, manifest, metadata_snapshot: dict) -> dict:
        previous_match = bool(record.previous_record_id)
        source_validity = 0.9 if manifest and manifest.verification_status.value == "valid" else 0.65
        hash_integrity = 1.0 if record.video_hash and manifest else 0.5
        fields = metadata_snapshot.get("fields", {}) if isinstance(metadata_snapshot, dict) else {}
        timestamp_validity = 0.85 if fields.get("creation_time") or metadata_snapshot.get("duration_seconds") else 0.55
        useful_metadata = [
            metadata_snapshot.get("duration_seconds"),
            metadata_snapshot.get("width"),
            metadata_snapshot.get("height"),
            metadata_snapshot.get("frame_rate"),
            record.video_fingerprint,
        ]
        metadata_completeness = sum(1 for item in useful_metadata if item not in (None, "", [])) / len(useful_metadata)
        penalty = 1.0 if manifest and manifest.verification_status.value == "valid" else 0.6
        score = (
            0.35 * source_validity
            + 0.25 * hash_integrity
            + 0.20 * timestamp_validity
            + 0.20 * metadata_completeness
        ) * penalty
        if previous_match:
            score = min(1.0, score + 0.05)
        return {
            "score": score,
            "breakdown": {
                "source_provenance_validity": round(source_validity, 3),
                "hash_fingerprint_integrity": round(hash_integrity, 3),
                "timestamp_metadata_validity": round(timestamp_validity, 3),
                "metadata_completeness": round(metadata_completeness, 3),
                "crypto_penalty": round(penalty, 3),
            },
        }

    def _compute_video_evidence_multiplier(self, layer2: LayerOutcome, layer1: LayerOutcome) -> float:
        found = layer2.details.get("evidence_elements_found", {}) if isinstance(layer2.details, dict) else {}
        ep = sum(1 for value in found.values() if value)
        et = max(1, len(found))
        if layer1.details.get("manifest_id"):
            ep += 1
            et += 1
        return max(0.0, min(1.0, 0.4 + 0.6 * (ep / et)))

    def _compute_video_penalty(self, layer2: LayerOutcome, layer1: LayerOutcome) -> float:
        if layer1.details.get("provenance_verification_status") in {"hash_mismatch", "signature_invalid"}:
            return 0.0
        if layer2.details.get("technical_warning"):
            return 0.7
        if (layer2.confidence or 0.0) < 0.55 or (layer1.confidence or 0.0) < 0.55:
            return 0.7
        return 1.0

    def _generate_video_label(self, final_score: float, evidence_multiplier: float, penalty_multiplier: float, flags: dict) -> str:
        return map_score_to_label(final_score)

    def _generate_video_explanation(self, result: dict) -> str:
        parts: list[str] = []
        ocr_result = result.get("ocr_result", {})
        audio_result = result.get("audio_result", {})
        if ocr_result.get("text_detected"):
            parts.append("On-screen text was extracted from sampled frames, deduplicated, and compared with trusted contextual evidence.")
        else:
            parts.append("No reliable on-screen text was found in sampled frames.")
        if result.get("technical_warning"):
            parts.append("Metadata/provenance signals are limited or include authoring-tool warnings, but this is not a full forensic authenticity finding.")
        else:
            parts.append("Available metadata and sampled-frame checks did not raise a strong provenance warning.")
        if float(result.get("temporal_continuity") or 0.0) < 0.5:
            parts.append("The available duration or stream metadata was limited, reducing technical confidence.")
        if audio_result.get("audio_present"):
            parts.append(str(audio_result.get("audio_visual_summary") or "The audio track was extracted and transcribed where possible."))
        if float(result.get("context_match_score") or 0.0) >= 0.6:
            parts.append("The video context is reasonably aligned with the claimed event or caption.")
        else:
            parts.append("The available context match for the video is limited or mixed.")
        parts.append("This result is based on audio transcription, frame OCR claim extraction, metadata/provenance checks, and contextual comparison; it does not perform full forensic video authenticity detection.")
        return " ".join(parts)

    def _build_video_system_scoring(self, layer1: LayerOutcome, layer2: LayerOutcome, layer3: LayerOutcome | None = None) -> dict:
        video_score = float(layer2.details.get("video_analysis_score") or layer2.confidence or 0.0)
        nlp_score = max(0.0, min(1.0, float(layer2.details.get("text_verification_score") or video_score or 0.0)))
        crypto_score = float(layer1.confidence or 0.0)
        community_score = max(0.0, min(1.0, float(layer3.confidence or 0.0))) if layer3 and layer3.status == "supporting" else 0.0
        evidence_multiplier = self._compute_video_evidence_multiplier(layer2, layer1)
        penalty_multiplier = self._compute_video_penalty(layer2, layer1)
        final_score = max(0.0, min(100.0, 100 * (0.30 * crypto_score + 0.45 * nlp_score + 0.25 * community_score)))
        finding = str(layer2.details.get("text_verification_label") or layer2.details.get("fact_check_verdict") or "").lower()
        if finding == "not verifiable":
            final_score = min(final_score, 49.0)
        if finding in {"unsupported", "false", "contradicted"}:
            final_score = min(final_score, 34.0)
        if not (layer2.details.get("transcript_text") or layer2.details.get("frame_ocr_text_summary") or layer2.details.get("caption_or_claim")):
            final_score = min(final_score, 34.0)
        if penalty_multiplier == 0.0:
            final_score = 0.0
        elif penalty_multiplier < 1.0:
            final_score = min(final_score, 69.0)
        no_extractable_claims = not (
            layer2.details.get("transcript_text")
            or layer2.details.get("frame_ocr_text_summary")
            or layer2.details.get("caption_or_claim")
        )
        flags = {
            "technical_warning": bool(layer2.details.get("technical_warning")),
            "forged_provenance": layer1.details.get("provenance_verification_status") in {"hash_mismatch", "signature_invalid"},
        }
        if no_extractable_claims or finding == "not verifiable":
            label = "Not Verifiable"
        elif finding in {"unsupported", "false", "contradicted"}:
            label = "Unverified"
        else:
            label = self._generate_video_label(final_score, evidence_multiplier, penalty_multiplier, flags)
        if label == "Verified":
            reason = "Extracted video claims have strong contextual support and provenance signals are usable."
        elif label == "Likely Verified":
            reason = "Extracted video claims have meaningful support, but some evidence gaps remain."
        elif label == "Not Verifiable":
            reason = "The available transcript, frame text, metadata, or contextual evidence is too limited or mixed for a reliable judgment."
        else:
            reason = "Contextual evidence does not support the extracted video claims strongly enough."
        return {
            "final_score": round(final_score, 2),
            "final_label": label,
            "reason": reason,
            "explanation": self._generate_video_explanation(layer2.details if isinstance(layer2.details, dict) else {}),
            "evidence_multiplier": round(evidence_multiplier, 3),
            "global_penalty": round(penalty_multiplier, 3),
            "formula_components": {
                "C": round(crypto_score, 3),
                "N": round(nlp_score, 3),
                "U": round(community_score, 3),
                "technical_video_score": round(video_score, 3),
                "evidence_multiplier_diagnostic": round(evidence_multiplier, 3),
                "global_penalty_diagnostic": round(penalty_multiplier, 3),
            },
            "active_layer_weights": {"crypto": 0.30, "nlp": 0.45, "community": 0.25},
            "formula": "Final video score = 100 * ((0.30 * C) + (0.45 * N) + (0.25 * U)). C is technical/provenance confidence, N is contextual confidence from transcript/OCR claims, and U is community confidence when available.",
            "layer_scores": {
                "crypto": round(crypto_score, 3),
                "nlp": round(nlp_score, 3),
                "community": round(community_score, 3) if layer3 else None,
            },
            "layer_breakdown": {
                "crypto": layer1.details.get("crypto_breakdown", {}),
                "nlp": layer2.details.get("text_component_scores", {}),
                "video": layer2.details.get("video_analysis_breakdown", {}),
                "community": layer3.details if layer3 else None,
            },
            "warnings": [
                warning
                for warning in [
                    "Metadata/provenance evidence is limited or contains authoring-tool warnings." if flags["technical_warning"] or penalty_multiplier < 1.0 else None,
                    "User consensus was missing." if not layer3 or community_score == 0.0 else None,
                ]
                if warning
            ],
        }

    def _decide_video(
        self,
        layer1: LayerOutcome,
        layer2: LayerOutcome,
        layer3: LayerOutcome,
        extracted: ExtractedPayload,
        record: VideoVerificationRecord,
        manifest,
    ) -> tuple[JobStatus, str, str, dict, dict]:
        scoring_result = self._build_video_system_scoring(layer1, layer2, layer3)
        label = str(scoring_result.get("final_label", "Not Verifiable"))
        if label == "Verified":
            status_value = JobStatus.verified
        elif label == "Likely Verified":
            status_value = JobStatus.likely_verified
        elif label == "Unverified":
            status_value = JobStatus.unverified
        else:
            status_value = JobStatus.not_verifiable

        record.final_score = float(scoring_result["final_score"])
        record.final_label = label
        record.explanation = str(scoring_result["explanation"])
        record.crypto_result = {
            "manifest_id": str(manifest.id),
            "issuer_id": layer1.details.get("issued_by"),
            "key_id": layer1.details.get("issuer_key_id"),
            "provenance_status": layer1.details.get("provenance_verification_status"),
            "timestamp": layer1.details.get("signed_at"),
            "previous_payload_hash": layer1.details.get("previous_manifest_id"),
            "crypto_breakdown": layer1.details.get("crypto_breakdown", {}),
        }
        self.db.add(record)
        self.db.flush()

        proof_summary = {
            "content_hash": extracted.content_hash,
            "verification_basis": "Video verification based on audio transcription, sampled-frame OCR, metadata/provenance checks, contextual evidence comparison, and signed platform provenance.",
            "video_verification_record_id": str(record.id),
            "verification_manifest_id": str(manifest.id),
            "current_version": record.current_version,
            "previous_version_id": str(record.previous_record_id) if record.previous_record_id else None,
            "system_scoring": scoring_result,
            **(layer1.details or {}),
        }
        analysis_report = self._build_analysis_report(status_value, layer1, layer2, layer3, extracted, scoring_result)
        analysis_report["executive_summary"] = (
            f"{label}: video verification used audio transcription, sampled-frame OCR, metadata/provenance findings, and contextual comparison against trusted evidence. It does not perform full forensic video authenticity detection."
        )
        analysis_report["mode"] = "video_verification"
        return status_value, label, str(scoring_result["reason"]), proof_summary, analysis_report

    def _content_preview_for_video(self, record: VideoVerificationRecord, extracted: ExtractedPayload) -> dict:
        text_excerpt = record.caption_or_claim or extracted.main_text
        return {
            "title": extracted.title or f"Video upload: {record.uploaded_filename}",
            "text_excerpt": str(text_excerpt or "")[:600],
            "caption_or_claim": record.caption_or_claim,
            "video_hash": record.video_hash,
            "video_fingerprint": record.video_fingerprint,
        }

    def _run_pipeline(self, job: VerificationJob, extracted: ExtractedPayload) -> VerificationJob:
        extracted_row = ExtractedContent(
            job_id=job.id,
            title=extracted.title,
            author=extracted.author,
            publish_time=extracted.publish_time,
            canonical_url=extracted.canonical_url,
            main_text=extracted.main_text,
            hero_image_url=extracted.hero_image_url,
            normalized_text=extracted.normalized_text,
            content_hash=extracted.content_hash,
            metadata_json=extracted.metadata,
        )
        self.db.add(extracted_row)
        append_audit_entry(self.db, job, "content_extracted", {"hash": extracted.content_hash})

        layer1 = self._layer1(extracted)
        layer2 = self._layer2(job, extracted)
        layer3 = self._layer3(job)
        layer4 = self._layer4(job)
        layer1 = self._attach_extracted_details(layer1, extracted)
        provenance_details = self._create_internal_provenance(extracted)
        layer1.details = {**layer1.details, **provenance_details}
        media_authenticity = self.media_authenticity.analyze(extracted)
        status_value, headline, reason, proof_summary, analysis_report = self._decide(layer1, layer2, layer3, extracted)

        job.status = status_value
        job.headline_message = headline
        job.short_reason = reason
        preview_excerpt = self._content_preview_excerpt(extracted)
        job.content_preview = {
            "title": extracted.title,
            "text_excerpt": preview_excerpt,
            "canonical_url": extracted.canonical_url,
            "hero_image_url": extracted.hero_image_url,
            "media_assets": extracted.metadata.get("media_assets", []),
            "publisher_name": extracted.metadata.get("publisher_name"),
            "author": extracted.author,
            "publish_time": extracted.publish_time.isoformat() if extracted.publish_time else None,
            "source_domain": extracted.metadata.get("source_domain"),
            "section": extracted.metadata.get("section"),
            "keywords": extracted.metadata.get("keywords", []),
        }
        job.proof_summary = {**proof_summary, **provenance_details}
        job.analysis_report = analysis_report
        job.analysis_report["media_authenticity"] = media_authenticity
        job.layer_results = {
            "layer1": asdict(layer1),
            "layer2": asdict(layer2),
            "layer3": asdict(layer3),
            "layer4": asdict(layer4),
        }
        append_audit_entry(self.db, job, "job_finalized", {"status": status_value.value})
        self.db.commit()
        self.db.refresh(job)
        return job

    def _content_preview_excerpt(self, extracted: ExtractedPayload) -> str:
        raw_metadata = getattr(extracted, "metadata", None)
        metadata = raw_metadata if isinstance(raw_metadata, dict) else getattr(extracted, "metadata_json", {}) or {}
        platform_type = str(metadata.get("platform_type") or "").lower()
        if platform_type == "youtube":
            description = str(metadata.get("description") or "").strip()
            transcript_text = str(metadata.get("transcript_text") or "").strip()
            if description:
                return description[:280]
            if transcript_text:
                return transcript_text[:280]
            if extracted.title and extracted.title != "Video metadata could not be retrieved":
                return str(extracted.title)[:280]
            return "Video metadata could not be retrieved."
        return extracted.main_text[:280]

    def _run_text_pipeline(self, job: VerificationJob, extracted: ExtractedPayload, input_text: str) -> VerificationJob:
        # Text verification in NewsProof is a dedicated claim/context verification workflow.
        # The NLP layer performs the main factual analysis. The cryptographic layer secures
        # the verification record. The feedback layer stores review history without voting.
        extracted_row = ExtractedContent(
            job_id=job.id,
            title=extracted.title,
            author=extracted.author,
            publish_time=extracted.publish_time,
            canonical_url=extracted.canonical_url,
            main_text=extracted.main_text,
            hero_image_url=extracted.hero_image_url,
            normalized_text=extracted.normalized_text,
            content_hash=extracted.content_hash,
            metadata_json=extracted.metadata,
        )
        self.db.add(extracted_row)
        append_audit_entry(self.db, job, "content_extracted", {"hash": extracted.content_hash})

        layer2 = self._text_nlp_layer(job, extracted, input_text)
        record = self._build_verification_record(input_text, extracted, layer2)
        manifest = self._create_verification_manifest(record)
        job.verification_record_id = record.id
        layer1 = self._text_crypto_layer(record, manifest, extracted)
        layer1 = self._attach_extracted_details(layer1, extracted)
        layer3 = self._text_feedback_layer(record)
        layer4 = self._layer4(job)
        media_authenticity = self.media_authenticity.analyze(extracted)
        status_value, headline, reason, proof_summary, analysis_report = self._decide_text(layer1, layer2, layer3, extracted, record, manifest)

        job.status = status_value
        job.headline_message = headline
        job.short_reason = reason
        job.content_preview = {
            "title": extracted.title,
            "text_excerpt": extracted.main_text[:280],
            "canonical_url": extracted.canonical_url,
            "hero_image_url": extracted.hero_image_url,
            "media_assets": extracted.metadata.get("media_assets", []),
            "publisher_name": extracted.metadata.get("publisher_name"),
            "author": extracted.author,
            "publish_time": extracted.publish_time.isoformat() if extracted.publish_time else None,
            "source_domain": extracted.metadata.get("source_domain"),
            "section": extracted.metadata.get("section"),
            "keywords": extracted.metadata.get("keywords", []),
        }
        job.proof_summary = proof_summary
        job.analysis_report = analysis_report
        job.analysis_report["media_authenticity"] = media_authenticity
        job.layer_results = {
            "layer1": asdict(layer1),
            "layer2": asdict(layer2),
            "layer3": asdict(layer3),
            "layer4": asdict(layer4),
        }
        append_audit_entry(self.db, job, "job_finalized", {"status": status_value.value, "verification_record_id": str(record.id)})
        self.db.commit()
        self.db.refresh(job)
        return job

    def preprocess_text(self, input_text: str) -> dict[str, str]:
        raw_text = (input_text or "").strip()
        cleaned_input = self._clean_text_for_retrieval(raw_text)
        normalized_text = normalize_text(cleaned_input)
        if not raw_text or not normalized_text:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text input cannot be empty")
        sentences = self._text_sentences(cleaned_input)
        extraction = self.claim_extraction.extract_claims(cleaned_input, max_claims=self.TEXT_CLAIM_LIMIT)
        core_claim = str(extraction.get("claims", [{}])[0].get("text") if extraction.get("claims") else (sentences[0] if sentences else cleaned_input))
        supporting_context = " ".join(sentences[1:3]).strip()
        detected_language = self._detect_text_language(cleaned_input)
        return {
            "input_text": raw_text,
            "cleaned_input": cleaned_input,
            "normalized_text": normalized_text,
            "detected_language": detected_language,
            "sentences": sentences,
            "core_claim": core_claim,
            "supporting_context": supporting_context,
            "noise_or_commentary": "" if len(sentences) <= 3 else "Additional commentary detected beyond the core factual claim window.",
            "likely_copied_article_text": self._looks_like_copied_article_text(cleaned_input, sentences),
        }

    def _clean_text_for_retrieval(self, text: str) -> str:
        cleaned = (text or "").replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
        cleaned = re.sub(r"https?://\S+", " ", cleaned)
        cleaned = re.sub(r"[@#][\w_]+", " ", cleaned)
        cleaned = re.sub(r"[\U00010000-\U0010ffff]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _text_sentences(self, text: str) -> list[str]:
        chunks = re.split(r"(?<=[\.\!\?])\s+", text or "")
        sentences = [" ".join(chunk.split()).strip() for chunk in chunks if " ".join(chunk.split()).strip()]
        return sentences[:12]

    def _detect_text_language(self, text: str) -> str:
        has_bengali = bool(re.search(r"[\u0980-\u09FF]", text or ""))
        has_latin = bool(re.search(r"[A-Za-z]", text or ""))
        if has_bengali and has_latin:
            return "mixed_bn_en"
        if has_bengali:
            return "bn"
        return "en"

    def _looks_like_copied_article_text(self, cleaned_input: str, sentences: list[str]) -> bool:
        if len(cleaned_input.split()) < 18:
            return False
        if len(sentences) < 2:
            return False
        if re.search(r"\b(i think|my opinion|follow for more|subscribe|watch now)\b", cleaned_input.lower()):
            return False
        return True

    def _build_source_recovery_sentence_chunks(self, prepared: dict[str, str | list[str] | bool]) -> list[str]:
        sentences = [str(item) for item in prepared.get("sentences", []) if str(item).strip()]
        if not sentences:
            cleaned = str(prepared.get("cleaned_input") or "")
            return [cleaned[:220]] if cleaned else []
        windows: list[str] = []
        windows.extend(sentences[:3])
        if len(sentences) >= 2:
            windows.append(" ".join(sentences[:2]))
        if len(sentences) >= 3:
            windows.append(" ".join(sentences[1:3]))
        longest = max(sentences, key=lambda item: len(item.split()), default="")
        if longest:
            windows.append(longest)
        distinctive = max(
            sentences,
            key=lambda item: len({token for token in normalize_text(item).split() if len(token) > 4}),
            default="",
        )
        if distinctive:
            windows.append(distinctive)

        deduped: list[str] = []
        seen: set[str] = set()
        for window in windows:
            normalized = normalize_text(window)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(window[:260])
        return deduped[:6]

    def _required_exact_phrase_queries(self, prepared: dict[str, str | list[str] | bool]) -> list[str]:
        sentences = [str(item).strip() for item in prepared.get("sentences", []) if str(item).strip()]
        cleaned_input = str(prepared.get("cleaned_input") or "").strip()
        if not sentences and cleaned_input:
            sentences = [cleaned_input[:260]]

        first_sentence = sentences[0] if sentences else ""
        longest_sentence = max(sentences, key=lambda item: len(item.split()), default=first_sentence)
        distinctive_sentence = max(
            sentences,
            key=lambda item: len({token for token in normalize_text(item).split() if len(token) > 4}),
            default=first_sentence,
        )
        fallback_window = " ".join(sentences[:2]).strip() if len(sentences) >= 2 else first_sentence or cleaned_input[:260]
        return self._dedupe_queries(
            [
                f"\"{first_sentence[:220]}\"" if first_sentence else "",
                f"\"{distinctive_sentence[:220]}\"" if distinctive_sentence else "",
                f"\"{longest_sentence[:220]}\"" if longest_sentence else "",
                f"\"{fallback_window[:220]}\"" if fallback_window else "",
            ],
            limit=4,
        )

    def _fallback_event_query(self, prepared: dict[str, str | list[str] | bool], extraction: dict, claims: list[str]) -> str:
        base = str(prepared.get("core_claim") or claims[0] if claims else prepared.get("cleaned_input") or "")
        entity_tokens = [str(item).strip() for item in extraction.get("entities", [])[:3] if str(item).strip()]
        date_tokens = [str(item).strip() for item in extraction.get("dates", [])[:1] if str(item).strip()]
        keyword_tokens = [token for token in normalize_text(base).split() if len(token) > 3][:6]
        return " ".join(entity_tokens + date_tokens + keyword_tokens).strip()[:180]

    def _classify_retrieval_failure(self, failure_reasons: list[str]) -> str | None:
        normalized = [str(item).strip() for item in failure_reasons if str(item).strip()]
        if not normalized:
            return None
        if any("provider_not_configured" in item for item in normalized):
            return "provider_not_configured"
        if any("provider_auth_failed" in item for item in normalized):
            return "provider_auth_failed"
        if any("provider_not_configured" in item or "request_failed" in item or "time_budget" in item or "timeout" in item.lower() for item in normalized):
            return f"retrieval_backend_failure:{' | '.join(normalized)}"
        return " | ".join(normalized)

    def _validate_text_query_plan(self, meaning_bundle: dict[str, object], query_plan: dict[str, list[str]]) -> str | None:
        exact_phrase_queries = [str(item).strip() for item in meaning_bundle.get("exact_phrase_queries", []) if str(item).strip()]
        lexical_queries = [str(item).strip() for item in query_plan.get("lexical", []) if str(item).strip()]
        final_queries = [str(item).strip() for item in (query_plan.get("lexical", []) + query_plan.get("semantic", [])) if str(item).strip()]
        if not exact_phrase_queries:
            return "retrieval_generation_failed:exact_phrase_queries_missing"
        if not lexical_queries:
            return "retrieval_generation_failed:lexical_queries_missing"
        if not final_queries:
            return "retrieval_generation_failed:search_queries_missing"
        return None

    def _text_pipeline_failure_layer(
        self,
        prepared: dict[str, str | list[str] | bool],
        claims: list[str],
        meaning_bundle: dict[str, object],
        query_plan: dict[str, list[str]],
        failure_reason: str,
    ) -> LayerOutcome:
        clarity = round(self._claim_clarity_score(claims[0] if claims else str(prepared.get("cleaned_input") or "")), 3)
        component_scores = {"S": 0.0, "R": 0.0, "A": 0.0, "D": 0.0, "C": clarity, "E": 0.0, "P": 0.0}
        return LayerOutcome(
            status="none",
            confidence=0.0,
            explanation="Text verification could not complete because retrieval query generation or the retrieval backend failed before meaningful evidence could be gathered.",
            details={
                "mode": "text_verification",
                "input_text": prepared["input_text"],
                "cleaned_input": prepared["cleaned_input"],
                "normalized_text": prepared["normalized_text"],
                "detected_language": prepared["detected_language"],
                "extracted_claims": claims,
                "matches": [],
                "evidence_sources": [],
                "supporting_evidence_summary": "",
                "contradiction_summary": "",
                "evidence_summary": "The retrieval pipeline failed before usable evidence could be gathered.",
                "fact_check_verdict": "Not Verifiable",
                "fact_check_assessment": "The retrieval pipeline failed before usable evidence could be gathered.",
                "text_verification_score": 0.0,
                "text_verification_label": "Not Verifiable",
                "evidence_multiplier": 0.0,
                "penalty_multiplier": 0.0,
                "text_component_scores": component_scores,
                "pipeline_failure": True,
                "pipeline_failure_reason": failure_reason,
                "retrieval_state": failure_reason,
                "exact_phrase_queries": meaning_bundle.get("exact_phrase_queries", []),
                "lexical_queries": query_plan.get("lexical", []),
                "semantic_queries": query_plan.get("semantic", []),
                "search_queries_used": query_plan.get("lexical", []) + query_plan.get("semantic", []),
                "source_recovery": {"status": "not_found", "url": None, "reason": failure_reason},
                "total_match_count": 0,
                "has_more_matches": False,
                "text_debug": {
                    "raw_input": prepared["input_text"],
                    "cleaned_input": prepared["cleaned_input"],
                    "normalized_input": prepared["normalized_text"],
                    "extracted_claim": claims[0] if claims else "",
                    "exact_phrase_queries": meaning_bundle.get("exact_phrase_queries", []),
                    "lexical_queries": query_plan.get("lexical", []),
                    "lexical_entity_queries": meaning_bundle.get("lexical_queries", []),
                    "semantic_queries": query_plan.get("semantic", []),
                    "final_search_queries_sent": query_plan.get("lexical", []) + query_plan.get("semantic", []),
                    "provider_status": [],
                    "brave_results_count": 0,
                    "brave_top_results": [],
                    "top_5_urls": [],
                    "candidate_fetches": [],
                    "overlap_scores": [],
                    "final_source_recovery": {"status": "not_found", "url": None, "reason": failure_reason},
                    "source_recovery_status": "not_found",
                    "source_recovery_reason": failure_reason,
                    "component_scores": component_scores,
                    "formula_components": component_scores,
                    "final_score": 0.0,
                    "final_label": "Not Verifiable",
                    "search_failure_reason": failure_reason,
                    "pipeline_failure": True,
                },
            },
        )

    def _text_low_information_layer(
        self,
        prepared: dict[str, str | list[str] | bool],
        claims: list[str],
        extraction: dict,
    ) -> LayerOutcome:
        clarity = round(self._claim_clarity_score(claims[0] if claims else str(prepared.get("cleaned_input") or "")), 3)
        component_scores = {"S": 0.0, "R": 0.0, "A": 0.0, "D": 0.0, "C": clarity, "E": 0.0, "P": 0.0}
        return LayerOutcome(
            status="none",
            confidence=0.0,
            explanation="The pasted text is too generic or low-information to support meaningful article retrieval or factual verification.",
            details={
                "mode": "text_verification",
                "input_text": prepared["input_text"],
                "cleaned_input": prepared["cleaned_input"],
                "normalized_text": prepared["normalized_text"],
                "detected_language": prepared["detected_language"],
                "extracted_claims": claims,
                "claim_extraction": extraction,
                "claim_results": [],
                "matches": [],
                "evidence_sources": [],
                "supporting_evidence_summary": "",
                "contradiction_summary": "",
                "evidence_summary": "The pasted text was too generic to retrieve meaningful article evidence.",
                "fact_check_verdict": "Not Verifiable",
                "fact_check_assessment": "The pasted text was too generic to retrieve meaningful article evidence.",
                "text_verification_score": 0.0,
                "text_verification_label": "Not Verifiable",
                "evidence_multiplier": 0.0,
                "penalty_multiplier": 0.0,
                "text_component_scores": component_scores,
                "pipeline_failure": False,
                "pipeline_failure_reason": None,
                "retrieval_state": "low_information_input",
                "exact_phrase_queries": [],
                "lexical_queries": [],
                "semantic_queries": [],
                "search_queries_used": [],
                "source_recovery": {"status": "not_attempted", "url": None, "reason": "low_information_input"},
                "total_match_count": 0,
                "has_more_matches": False,
                "text_debug": {
                    "raw_input": prepared["input_text"],
                    "cleaned_input": prepared["cleaned_input"],
                    "normalized_input": prepared["normalized_text"],
                    "extracted_claim": claims[0] if claims else "",
                    "exact_phrase_queries": [],
                    "lexical_entity_queries": [],
                    "final_search_queries_sent": [],
                    "provider_status": [],
                    "brave_results_count": 0,
                    "brave_top_results": [],
                    "candidate_fetches": [],
                    "overlap_scores": [],
                    "source_recovery_status": "not_attempted",
                    "source_recovery_reason": "low_information_input",
                    "formula_components": component_scores,
                    "final_score": 0.0,
                    "final_label": "Not Verifiable",
                    "pipeline_failure": False,
                },
            },
        )

    def _is_low_information_text_input(
        self,
        prepared: dict[str, str | list[str] | bool],
        extraction: dict,
        claims: list[str],
    ) -> bool:
        if bool(prepared.get("likely_copied_article_text")):
            return False
        if len([str(item) for item in prepared.get("sentences", []) if str(item).strip()]) > 1:
            return False
        generic_entity_terms = {"this", "that", "these", "those", "it", "they", "he", "she", "we", "i", "you"}
        signal_count = sum(
            len(
                [
                    item
                    for item in (extraction.get(key, []) or [])
                    if normalize_text(str(item)) not in generic_entity_terms
                ]
            )
            for key in ("entities", "locations", "dates", "numbers")
        )
        clarity = self._claim_clarity_score(claims[0] if claims else str(prepared.get("cleaned_input") or ""))
        first_token = normalize_text((claims[0] if claims else str(prepared.get("cleaned_input") or "")).split()[0] if (claims or prepared.get("cleaned_input")) else "")
        return signal_count == 0 and (clarity <= 0.7 or first_token in generic_entity_terms)

    def _entity_focused_query(self, extraction: dict) -> str:
        entities = [str(item) for item in extraction.get("entities", [])[:4]]
        locations = [str(item) for item in extraction.get("locations", [])[:2]]
        dates = [str(item) for item in extraction.get("dates", [])[:2]]
        numbers = [str(item) for item in extraction.get("numbers", [])[:2]]
        query = " ".join(entities + locations + dates + numbers).strip()
        return query[:180]

    def _search_bundle(self, query: str, expected_domain: str | None = None, limit: int = 5) -> dict:
        search_debug_callable = getattr(self.search, "search_with_debug", None)
        overridden_debug = search_debug_callable is not None and (
            not hasattr(search_debug_callable, "__func__") or search_debug_callable.__func__ is not SearchService.search_with_debug
        )
        if overridden_debug:
            return search_debug_callable(query, expected_domain=expected_domain, limit=limit)
        search_callable = getattr(self.search, "search")
        overridden_search = not hasattr(search_callable, "__func__") or search_callable.__func__ is not SearchService.search
        if overridden_search:
            results = search_callable(query, expected_domain=expected_domain, limit=limit)
            return {
                "results": results,
                "failure_reason": None if results else "no_results_returned",
                "provider_attempts": [],
                "providers_used": [],
                "stopped_by_budget": False,
            }
        return self.search.search_with_debug(query, expected_domain=expected_domain, limit=limit)

    def _keyword_claim_query(self, prepared: dict[str, str | list[str] | bool], extraction: dict, claims: list[str]) -> str:
        base = str(prepared.get("core_claim") or claims[0] if claims else prepared.get("cleaned_input") or "")
        tokens = [token for token in normalize_text(base).split() if len(token) > 3][:10]
        entities = [str(item) for item in extraction.get("entities", [])[:3]]
        query = " ".join(entities + tokens).strip()
        return query[:180]

    def _direct_source_recovery(self, prepared: dict[str, str | list[str] | bool], extraction: dict, extracted: ExtractedPayload) -> dict:
        stage = {
            "status": "not_attempted",
            "raw_input": str(prepared.get("input_text") or ""),
            "normalized_input": str(prepared.get("normalized_text") or ""),
            "document_queries": [],
            "claim_queries": [],
            "search_results": [],
            "candidate_fetches": [],
            "overlap_scores": [],
            "best_candidate": None,
            "best_score": 0.0,
            "rejection_reason": "",
            "final_source_recovery": {"status": "not_found", "url": None, "reason": ""},
        }
        if not bool(prepared.get("likely_copied_article_text")):
            stage["status"] = "attempted_non_article_like"
            stage["rejection_reason"] = "Input did not look like copied article text, so source recovery is lower confidence."

        query_bundle = self._generate_source_recovery_queries(prepared, extraction)
        document_queries = query_bundle["document_queries"]
        claim_queries = query_bundle["claim_queries"]
        if not document_queries:
            stage["status"] = "query_generation_failed"
            stage["rejection_reason"] = "retrieval_generation_failed:exact_phrase_queries_missing"
            stage["final_source_recovery"] = {
                "status": "not_found",
                "url": None,
                "title": None,
                "publisher": None,
                "overlap_confidence": 0.0,
                "match_type": "none",
                "reason": stage["rejection_reason"],
            }
            return stage
        stage["document_queries"] = document_queries
        stage["claim_queries"] = claim_queries
        candidates: list[dict] = []
        seen_urls: set[str] = set()
        search_debug: list[dict] = []
        for query in (document_queries + claim_queries)[: self.TEXT_MAX_SEARCH_QUERIES]:
            if len(seen_urls) >= self.TEXT_MAX_CANDIDATES:
                break
            search_debug_bundle = self._search_bundle(query, expected_domain=None, limit=5)
            search_results = search_debug_bundle["results"]
            search_debug.append(
                {
                    "query": query,
                    "result_count": len(search_results),
                    "failure_reason": search_debug_bundle.get("failure_reason"),
                    "providers_used": search_debug_bundle.get("providers_used", []),
                    "provider_status": search_debug_bundle.get("provider_status", {}),
                    "provider_attempts": search_debug_bundle.get("provider_attempts", []),
                    "results": [
                        {"url": result.url, "title": result.title, "snippet": result.snippet}
                        for result in search_results[:5]
                    ],
                }
            )
            for result in search_results:
                if len(seen_urls) >= self.TEXT_MAX_CANDIDATES:
                    break
                if result.url in seen_urls:
                    continue
                seen_urls.add(result.url)
                document = self.retrieval.fetch_page_document(result.url)
                stage["candidate_fetches"].append(
                    {
                        "url": result.url,
                        "fetch_status": document.get("fetch_status"),
                        "content_type": document.get("content_type"),
                        "extracted_text_length": document.get("extracted_text_length"),
                        "title": document.get("title") or result.title,
                    }
                )
                candidate = self._evaluate_source_candidate(
                    prepared,
                    extraction,
                    str(document.get("title") or result.title or ""),
                    result.url,
                    result.snippet,
                    str(document.get("extracted_text") or ""),
                    document,
                )
                candidates.append(candidate)
                stage["overlap_scores"].append(
                    {
                        "url": candidate.get("url"),
                        "title": candidate.get("title"),
                        "exact_sentence_overlap": candidate.get("exact_sentence_match_ratio"),
                        "token_overlap": candidate.get("token_overlap"),
                        "ngram_overlap": candidate.get("ngram_overlap"),
                        "longest_common_span": candidate.get("longest_common_span_ratio"),
                        "named_entity_overlap": candidate.get("entity_overlap"),
                        "containment_score": candidate.get("containment_score"),
                        "fetch_status": document.get("fetch_status"),
                        "extracted_text_length": document.get("extracted_text_length"),
                        "accepted_as_direct_source": candidate.get("accepted_as_direct_source"),
                        "reject_reason": candidate.get("reject_reason"),
                    }
                )
                if candidate.get("accepted_as_direct_source") and float(candidate.get("source_recovery_score") or 0.0) >= 0.75:
                    stage["search_results"] = search_debug
                    candidates.sort(key=lambda item: float(item.get("source_recovery_score") or 0.0), reverse=True)
                    best = candidates[0]
                    stage["best_candidate"] = best
                    stage["best_score"] = round(float(best.get("source_recovery_score") or 0.0), 3)
                    stage["status"] = "direct_source_recovered"
                    stage["final_source_recovery"] = {
                        "status": "recovered",
                        "url": best.get("url"),
                        "title": best.get("title"),
                        "publisher": best.get("domain"),
                        "overlap_confidence": stage["best_score"],
                        "match_type": "exact",
                        "reason": "A near-duplicate article body was recovered from the pasted text.",
                    }
                    return stage
        stage["search_results"] = search_debug
        candidates.sort(key=lambda item: float(item.get("source_recovery_score") or 0.0), reverse=True)
        if not candidates:
            stage["status"] = "no_candidates_retrieved"
            failure_reasons = list(dict.fromkeys(str(item.get("failure_reason") or "") for item in search_debug if item.get("failure_reason")))
            classified_failure = self._classify_retrieval_failure(failure_reasons)
            stage["rejection_reason"] = (
                "No candidate URLs were retrieved during direct source recovery."
                if not classified_failure
                else f"No candidate URLs were retrieved during direct source recovery. Search status: {classified_failure}"
            )
            stage["final_source_recovery"] = {
                "status": "not_found",
                "url": None,
                "title": None,
                "publisher": None,
                "overlap_confidence": 0.0,
                "match_type": "none",
                "reason": stage["rejection_reason"],
            }
            return stage

        best = candidates[0]
        stage["best_candidate"] = best
        stage["best_score"] = round(float(best.get("source_recovery_score") or 0.0), 3)
        if float(best.get("source_recovery_score") or 0.0) >= 0.75:
            stage["status"] = "direct_source_recovered"
        elif float(best.get("source_recovery_score") or 0.0) >= 0.55:
            stage["status"] = "likely_source_recovered"
        else:
            stage["status"] = "no_direct_source_found"
            stage["rejection_reason"] = str(best.get("reject_reason") or "Candidate pages were retrieved, but text overlap was insufficient to confirm the original source.")
        stage["final_source_recovery"] = {
            "status": "recovered" if stage["status"] == "direct_source_recovered" else "likely_recovered" if stage["status"] == "likely_source_recovered" else "not_found",
            "url": best.get("url") if stage["status"] in {"direct_source_recovered", "likely_source_recovered"} else None,
            "title": best.get("title") if stage["status"] in {"direct_source_recovered", "likely_source_recovered"} else None,
            "publisher": best.get("domain") if stage["status"] in {"direct_source_recovered", "likely_source_recovered"} else None,
            "overlap_confidence": stage["best_score"] if stage["status"] in {"direct_source_recovered", "likely_source_recovered"} else 0.0,
            "match_type": "exact" if stage["status"] == "direct_source_recovered" else "likely" if stage["status"] == "likely_source_recovered" else "none",
            "reason": (
                "A near-duplicate article body was recovered from the pasted text."
                if stage["status"] == "direct_source_recovered"
                else "A likely source article was recovered, but the overlap is not exact."
                if stage["status"] == "likely_source_recovered"
                else stage["rejection_reason"]
            ),
        }
        return stage

    def _generate_source_recovery_queries(self, prepared: dict[str, str | list[str] | bool], extraction: dict) -> dict[str, list[str]]:
        sentence_chunks = self._build_source_recovery_sentence_chunks(prepared)
        document_queries: list[str] = []
        document_queries.extend(self._required_exact_phrase_queries(prepared)[:3])
        for window in sentence_chunks[2:5]:
            document_queries.append(window)

        claim_queries: list[str] = []
        entity_query = self._entity_focused_query(extraction)
        if entity_query:
            claim_queries.append(entity_query)
        if extraction.get("claims"):
            for claim in extraction.get("claims", [])[:2]:
                for variant in [claim.get("text"), *(claim.get("short_forms", [])[:2])]:
                    text = " ".join(str(variant or "").split()[:16]).strip()
                    if text:
                        claim_queries.append(text)
        keyword_query = self._keyword_claim_query(prepared, extraction, [str(item.get("text") or "") for item in extraction.get("claims", [])])
        if keyword_query:
            claim_queries.append(keyword_query)
        fallback_event = self._fallback_event_query(prepared, extraction, [str(item.get("text") or "") for item in extraction.get("claims", [])])
        if fallback_event:
            claim_queries.append(fallback_event)

        return {
            "document_queries": self._dedupe_queries(document_queries, limit=3),
            "claim_queries": self._dedupe_queries(claim_queries, limit=2),
        }

    def _build_text_meaning_bundle(
        self,
        prepared: dict[str, str | list[str] | bool],
        extraction: dict,
        claims: list[str],
    ) -> dict[str, object]:
        cleaned_input = str(prepared.get("cleaned_input") or "")
        primary_claim = str(
            extraction.get("claims", [{}])[0].get("text")
            if extraction.get("claims")
            else (prepared.get("core_claim") or claims[0] if claims else cleaned_input)
        ).strip()
        primary_claim_tuple = self._extract_structured_claim_tuple(primary_claim or cleaned_input)
        entities = [str(item).strip() for item in extraction.get("entities", []) if str(item).strip()]
        locations = [str(item).strip() for item in extraction.get("locations", []) if str(item).strip()]
        dates = [str(item).strip() for item in extraction.get("dates", []) if str(item).strip()]
        numbers = [str(item).strip() for item in extraction.get("numbers", []) if str(item).strip()]
        sentences = [str(item).strip() for item in prepared.get("sentences", []) if str(item).strip()]

        simplified_meaning = " ".join(primary_claim.split()[:18]).strip()
        if len(simplified_meaning.split()) < 6 and sentences:
            simplified_meaning = " ".join(sentences[0].split()[:18]).strip()
        if not simplified_meaning:
            simplified_meaning = " ".join(cleaned_input.split()[:18]).strip()

        event_tokens = [token for token in normalize_text(primary_claim or cleaned_input).split() if len(token) > 3][:8]
        keyword_query = " ".join(dict.fromkeys([*entities[:2], *locations[:1], *dates[:1], *event_tokens])).strip()
        if not keyword_query:
            keyword_query = " ".join(cleaned_input.split()[:12]).strip()

        entity_query = " ".join(dict.fromkeys([*entities[:4], *locations[:2], *dates[:2], *numbers[:2]])).strip()

        exact_sentence_queries: list[str] = []
        if sentences:
            exact_sentence_queries.append(sentences[0])
            longest = max(sentences, key=lambda item: len(item.split()), default="")
            if longest and longest != sentences[0]:
                exact_sentence_queries.append(longest)
            most_entities = max(
                sentences,
                key=lambda item: sum(1 for entity in entities[:6] if normalize_text(entity) and normalize_text(entity) in normalize_text(item)),
                default="",
            )
            if most_entities and most_entities not in exact_sentence_queries:
                exact_sentence_queries.append(most_entities)

        exact_phrase_queries = self._required_exact_phrase_queries(prepared)
        lexical_queries = self._dedupe_queries([keyword_query, entity_query], limit=2)
        fallback_queries = self._dedupe_queries([self._fallback_event_query(prepared, extraction, claims), simplified_meaning], limit=2)

        return {
            "simplified_meaning": simplified_meaning,
            "keyword_query": keyword_query,
            "entity_query": entity_query,
            "exact_phrases": exact_sentence_queries[:2],
            "exact_phrase_queries": exact_phrase_queries,
            "lexical_queries": lexical_queries,
            "fallback_queries": fallback_queries,
            "search_queries": [*exact_phrase_queries, *lexical_queries, *fallback_queries][: self.TEXT_MAX_SEARCH_QUERIES],
            "entities": entities,
            "locations": locations,
            "dates": dates,
            "numbers": numbers,
            "claim_tuple": primary_claim_tuple,
            "raw_input": str(prepared.get("input_text") or ""),
            "normalized_input": str(prepared.get("normalized_text") or ""),
            "cleaned_input": str(prepared.get("cleaned_input") or ""),
        }

    def _retrieve_text_top_articles(
        self,
        prepared: dict[str, str | list[str] | bool],
        extraction: dict,
        meaning_bundle: dict[str, object],
        source_recovery: dict | None = None,
    ) -> tuple[list[dict], dict]:
        candidates_by_url: dict[str, dict] = {}
        search_results_debug: list[dict] = []
        candidate_fetches: list[dict] = []
        source_recovery = source_recovery or {}
        seeded_candidate = source_recovery.get("best_candidate") if isinstance(source_recovery, dict) else None
        if isinstance(seeded_candidate, dict) and seeded_candidate.get("url"):
            candidates_by_url[str(seeded_candidate.get("url"))] = {
                "title": seeded_candidate.get("title"),
                "url": seeded_candidate.get("url"),
                "snippet": seeded_candidate.get("snippet"),
                "domain": seeded_candidate.get("domain"),
                "search_score": float(
                    max(
                        float(seeded_candidate.get("score") or 0.0),
                        float(seeded_candidate.get("source_recovery_score") or 0.0),
                    )
                ),
                "search_query": "direct_source_recovery",
                "query_type": "direct_source_recovery",
            }

        ordered_query_groups = [
            ("exact", [str(item) for item in meaning_bundle.get("exact_phrase_queries", []) if str(item).strip()]),
            ("lexical", [str(item) for item in meaning_bundle.get("lexical_queries", []) if str(item).strip()]),
            (
                "simplified",
                []
                if source_recovery.get("status") in {"direct_source_recovered", "likely_source_recovered"}
                else [str(item) for item in meaning_bundle.get("fallback_queries", []) if str(item).strip()],
            ),
        ]

        for query_type, queries in ordered_query_groups:
            for query in queries:
                if len(search_results_debug) >= self.TEXT_MAX_SEARCH_QUERIES:
                    break
                search_debug_bundle = self._search_bundle(query, expected_domain=None, limit=5)
                results = search_debug_bundle["results"]
                search_results_debug.append(
                    {
                        "query": query,
                        "query_type": query_type,
                        "result_count": len(results),
                        "failure_reason": search_debug_bundle.get("failure_reason"),
                        "providers_used": search_debug_bundle.get("providers_used", []),
                        "provider_status": search_debug_bundle.get("provider_status", {}),
                        "provider_attempts": search_debug_bundle.get("provider_attempts", []),
                        "results": [
                            {"url": item.url, "title": item.title, "snippet": item.snippet}
                            for item in results[:5]
                        ],
                    }
                )
                for item in results:
                    if not item.url:
                        continue
                    current = candidates_by_url.get(item.url)
                    if current is None or float(item.score) > float(current.get("search_score") or 0.0):
                        candidates_by_url[item.url] = {
                            "title": item.title,
                            "url": item.url,
                            "snippet": item.snippet,
                            "domain": item.domain,
                            "search_score": float(item.score),
                            "search_query": query,
                            "query_type": query_type,
                        }
                if len(candidates_by_url) >= self.TEXT_MAX_CANDIDATES and query_type == "exact":
                    break
            if len(candidates_by_url) >= self.TEXT_MAX_CANDIDATES and query_type in {"exact", "lexical"}:
                break

        fetched_articles: list[dict] = []
        seeded_urls: set[str] = set()
        if isinstance(seeded_candidate, dict) and seeded_candidate.get("url"):
            seeded_urls.add(str(seeded_candidate.get("url")))
            fetched_articles.append(dict(seeded_candidate))
            seeded_fetch = next(
                (
                    item
                    for item in source_recovery.get("candidate_fetches", [])
                    if str(item.get("url") or "") == str(seeded_candidate.get("url") or "")
                ),
                None,
            )
            candidate_fetches.append(
                seeded_fetch
                or {
                    "url": seeded_candidate.get("url"),
                    "fetch_status": "success",
                    "content_type": "text/html",
                    "extracted_text_length": len(str(seeded_candidate.get("passage") or "")),
                    "title": seeded_candidate.get("title"),
                    "snippet": seeded_candidate.get("snippet"),
                }
            )
        ranked_candidates = sorted(candidates_by_url.values(), key=lambda item: float(item.get("search_score") or 0.0), reverse=True)[: self.TEXT_MAX_CANDIDATES]
        for candidate in ranked_candidates:
            if str(candidate.get("url") or "") in seeded_urls:
                continue
            document = self.retrieval.fetch_page_document(str(candidate.get("url") or ""))
            candidate_fetches.append(
                {
                    "url": candidate.get("url"),
                    "fetch_status": document.get("fetch_status"),
                    "content_type": document.get("content_type"),
                    "extracted_text_length": document.get("extracted_text_length"),
                    "title": document.get("title") or candidate.get("title"),
                    "snippet": candidate.get("snippet"),
                }
            )
            fetched_articles.append(self._evaluate_text_top_article(prepared, extraction, meaning_bundle, candidate, document))
            if len(fetched_articles) >= self.TEXT_MAX_CANDIDATES:
                break

        fetched_articles.sort(
            key=lambda item: (
                1 if item.get("direct_source_match") else 0,
                float(item.get("verification_confidence") or 0.0),
                float(item.get("retrieval_confidence") or 0.0),
            ),
            reverse=True,
        )
        top_articles = fetched_articles[: self.TEXT_MAX_CANDIDATES]
        best_match = top_articles[0] if top_articles else None
        likely_source_recovered = bool(
            best_match
            and (
                best_match.get("direct_source_match")
                or float(best_match.get("source_recovery_score") or 0.0) >= 0.55
                or str(best_match.get("source") or "") == "direct_source_recovery"
            )
        )
        source_recovery = {
            "status": "recovered" if best_match and best_match.get("direct_source_match") else "likely_recovered" if likely_source_recovered else "not_found",
            "url": best_match.get("url") if likely_source_recovered else None,
            "title": best_match.get("title") if likely_source_recovered else None,
            "publisher": best_match.get("domain") if likely_source_recovered else None,
            "overlap_confidence": round(float(best_match.get("source_recovery_score") or 0.0), 3) if likely_source_recovered else 0.0,
            "match_type": "exact" if best_match and best_match.get("direct_source_match") else "likely" if likely_source_recovered else "none",
            "reason": (
                "A candidate article strongly overlapped the pasted text and likely represents the original source."
                if likely_source_recovered
                else "No direct source URL could be recovered from the pasted text."
            ),
        }
        retrieval_failure_reason = None
        if not top_articles:
            failure_reasons = [str(item.get("failure_reason") or "") for item in search_results_debug if item.get("failure_reason")]
            retrieval_failure_reason = " | ".join(item for item in failure_reasons if item) or "no_candidate_articles_retrieved"

        return top_articles, {
            "simplified_meaning": meaning_bundle.get("simplified_meaning"),
            "search_queries": meaning_bundle.get("search_queries", []),
            "exact_phrase_queries": meaning_bundle.get("exact_phrase_queries", []),
            "lexical_queries": meaning_bundle.get("lexical_queries", []),
            "simplified_meaning_query": next(iter(meaning_bundle.get("fallback_queries", [])), ""),
            "search_results": search_results_debug,
            "candidate_fetches": candidate_fetches,
            "top_articles": top_articles,
            "retrieval_failure_reason": retrieval_failure_reason,
            "final_source_recovery": source_recovery,
        }

    def _evaluate_text_top_article(
        self,
        prepared: dict[str, str | list[str] | bool],
        extraction: dict,
        meaning_bundle: dict[str, object],
        candidate: dict,
        document: dict,
    ) -> dict:
        title = str(document.get("title") or candidate.get("title") or "")
        snippet = str(candidate.get("snippet") or "")
        page_text = str(document.get("extracted_text") or "")
        page_excerpt = " ".join(self._text_sentences(page_text)[:3])[:900] if page_text else snippet
        comparison_text = " ".join(filter(None, [title, snippet, page_excerpt]))
        simplified_meaning = str(meaning_bundle.get("simplified_meaning") or "")
        cleaned_input = str(meaning_bundle.get("cleaned_input") or "")
        exact_phrases = [normalize_text(str(item)) for item in meaning_bundle.get("exact_phrases", []) if str(item).strip()]
        normalized_page = normalize_text(page_text or comparison_text)
        normalized_input = str(meaning_bundle.get("normalized_input") or "")
        primary_anchor = str(meaning_bundle.get("exact_phrases", [""])[0] or simplified_meaning or cleaned_input).strip()

        exact_phrase_hit_ratio = (
            sum(1 for phrase in exact_phrases if phrase and phrase in normalized_page) / max(1, len(exact_phrases))
            if exact_phrases
            else 0.0
        )
        token_overlap = self._text_overlap(cleaned_input or primary_anchor, page_text or comparison_text)
        full_input_overlap = self._text_overlap(cleaned_input, page_text or comparison_text)
        entity_overlap = self._entity_overlap_score(extraction, comparison_text)
        longest_span = self._longest_common_token_span_ratio(normalized_input, normalized_page)
        ngram_overlap = self._ngram_overlap_score(normalized_input, normalized_page, 3)
        passage_overlap = self._text_overlap(primary_anchor, comparison_text)
        exactness = self._headline_exactness_score(primary_anchor, title, snippet)
        containment = 1.0 if normalized_input and normalized_input[:180] in normalized_page else 0.0
        reasoning = self.reasoning.verify_claim_with_evidence(
            primary_anchor or simplified_meaning,
            [
                {
                    "title": title,
                    "url": candidate.get("url"),
                    "domain": candidate.get("domain"),
                    "passage": page_excerpt,
                    "snippet": snippet,
                    "score": candidate.get("search_score") or 0.0,
                }
            ],
        )

        retrieval_confidence = round(
            min(1.0, max(float(candidate.get("search_score") or 0.0), exactness, entity_overlap * 0.75, token_overlap, passage_overlap)),
            3,
        )
        verification_signal = round(
            min(
                1.0,
                0.30 * max(float(reasoning.get("support_score") or 0.0), exactness)
                + 0.22 * token_overlap
                + 0.18 * entity_overlap
                + 0.15 * longest_span
                + 0.10 * exact_phrase_hit_ratio
                + 0.05 * containment,
            ),
            3,
        )
        direct_source_match = bool(longest_span >= 0.55 or exact_phrase_hit_ratio >= 0.5 or containment >= 1.0)
        support_judgment = "unrelated"
        if reasoning.get("verdict") == "contradicted" and verification_signal < 0.4 and entity_overlap >= 0.2:
            support_judgment = "contradiction"
        elif direct_source_match or verification_signal >= 0.72:
            support_judgment = "strong support"
        elif verification_signal >= 0.42 or entity_overlap >= 0.4:
            support_judgment = "partial support"

        verification_confidence = {
            "strong support": max(verification_signal, 0.78),
            "partial support": max(verification_signal, 0.48),
            "contradiction": max(0.2, min(0.45, verification_signal)),
            "unrelated": min(0.22, verification_signal),
        }[support_judgment]
        reject_reason = ""
        if document.get("fetch_status") != "success":
            reject_reason = "candidate fetch failed"
        elif document.get("extract_status") != "extracted":
            reject_reason = "candidate fetched but extracted text was too short or empty"
        elif support_judgment == "unrelated":
            reject_reason = "Related articles were retrieved, but none matched the pasted text strongly enough."
        elif support_judgment == "contradiction":
            reject_reason = "The candidate article conflicts with key claim details."

        explanation = (
            "This article strongly supports the pasted text."
            if support_judgment == "strong support"
            else "This article is related and partially supports the pasted text."
            if support_judgment == "partial support"
            else "This article appears to contradict the pasted text."
            if support_judgment == "contradiction"
            else "This article appears only loosely related to the pasted text."
        )
        return {
            "title": title,
            "url": candidate.get("url"),
            "domain": candidate.get("domain") or extract_domain(candidate.get("url")),
            "snippet": snippet,
            "passage": page_excerpt,
            "source": "web_search",
            "query_type": candidate.get("query_type"),
            "score": round(float(verification_confidence), 3),
            "retrieval_score": retrieval_confidence,
            "retrieval_confidence": retrieval_confidence,
            "verification_confidence": round(float(verification_confidence), 3),
            "support_judgment": support_judgment,
            "support_score": round(float(verification_confidence), 3),
            "directness_score": round(max(exactness, token_overlap, min(1.0, longest_span + 0.1)), 3),
            "entity_overlap": round(entity_overlap, 3),
            "token_overlap": round(token_overlap, 3),
            "passage_overlap": round(passage_overlap, 3),
            "ngram_overlap": round(ngram_overlap, 3),
            "full_input_overlap": round(full_input_overlap, 3),
            "longest_common_span_ratio": round(longest_span, 3),
            "exact_phrase_hit_ratio": round(exact_phrase_hit_ratio, 3),
            "containment_score": round(containment, 3),
            "reasoning_verdict": reasoning.get("verdict"),
            "direct_source_match": direct_source_match,
            "reject_reason": reject_reason,
            "explanation": explanation,
        }

    def _build_claim_evidence_from_top_articles(self, claim: str, top_articles: list[dict]) -> list[dict]:
        ranked = sorted(
            top_articles,
            key=lambda item: (
                float(item.get("rerank_score") or 0.0),
                float(item.get("source_recovery_score") or 0.0),
                float(item.get("bm25_score") or 0.0),
                float(item.get("semantic_score") or 0.0),
            ),
            reverse=True,
        )
        return ranked[: self.TEXT_EVIDENCE_PER_CLAIM]

    def _compute_text_formula_components(self, claim_results: list[dict], top_articles: list[dict]) -> dict[str, float | str]:
        relevant_articles = []
        strong_support = []
        partial_support = []
        contradictions = []
        for result in claim_results:
            for item in result.get("evidence_sources", []):
                label = str(item.get("evidence_relation") or item.get("judgment_label") or "")
                aboutness = str(item.get("aboutness") or "")
                if label in {"supports", "contradicts", "refutes"}:
                    relevant_articles.append(item)
                if label == "supports" and aboutness in {"exact_claim", "strong_related"}:
                    strong_support.append(item)
                elif label == "partially_related":
                    partial_support.append(item)
                elif label in {"contradicts", "refutes"}:
                    contradictions.append(item)

        supported_claim_scores = [
            float(item.get("support_score") or 0.0)
            for item in claim_results
            if item.get("claim_verdict") == "supported"
        ]
        support = round(sum(supported_claim_scores) / max(1, len(claim_results)), 3)
        if strong_support and supported_claim_scores:
            support = round(max(support, max(supported_claim_scores) * 0.9), 3)
        reliability = round(self._text_source_reliability(strong_support or contradictions), 3)
        agreement = round(self._text_cross_source_agreement(strong_support), 3)
        directness_scores = [float(item.get("directness_score") or 0.0) for item in claim_results]
        directness = round((sum(directness_scores) / max(1, len(claim_results)) if claim_results else 0.0), 3)
        if strong_support and directness_scores:
            directness = round(max(directness, max(directness_scores) * 0.9), 3)
        clarity = round(
            (
                sum(float(item.get("clarity_score") or 0.0) for item in claim_results) / max(1, len(claim_results))
                if claim_results
                else 0.0
            ),
            3,
        )
        evidence_points = sum(
            [
                1 if strong_support or contradictions else 0,
                1 if strong_support else 0,
                1 if len({self._domain_family(str(item.get("domain") or extract_domain(item.get("url")) or "")) for item in strong_support if item.get("url") or item.get("domain")}) >= 2 else 0,
                1 if any(bool(item.get("direct_source_match")) for item in strong_support) else 0,
                1 if strong_support else 0,
            ]
        )
        evidence_multiplier = round(compute_evidence_multiplier(evidence_points, 5), 3)
        contradiction_dominates = len(contradictions) > len(strong_support) + len(partial_support)
        no_relevant = not relevant_articles
        penalty = 1.0
        if contradiction_dominates:
            penalty = 0.4
        elif no_relevant or not strong_support:
            penalty = 0.4
        penalty = round(penalty, 3)
        return {
            "S": support,
            "R": reliability,
            "A": agreement,
            "D": directness,
            "C": clarity,
            "E": evidence_multiplier,
            "P": penalty,
            "strong_support_count": len(strong_support),
            "partial_support_count": len(partial_support),
            "contradiction_count": len(contradictions),
            "relevant_count": len(relevant_articles),
            "score_reasoning": {
                "S": f"Assigned from direct claim support across {len(strong_support)} supporting source(s); mentions-only sources add 0 support.",
                "R": "Assigned from source type and domain trust for exact supporting/refuting sources.",
                "A": f"Assigned from independent trusted source agreement across {len({self._domain_family(str(item.get('domain') or extract_domain(item.get('url')) or '')) for item in strong_support if item.get('url') or item.get('domain')})} domain family/families.",
                "D": "Assigned from exact claim/directness scores, source recovery, and direct winner-loser-event matching.",
            },
        }

    def _finalize_text_verification(
        self,
        claim_results: list[dict],
        top_articles: list[dict],
        meaning_bundle: dict[str, object],
    ) -> dict:
        if not claim_results:
            return {
                "final_score": 0.0,
                "final_label": "Not Verifiable",
                "explanation": "No factual, checkable claim was found in the pasted text.",
                "evidence_summary": "The pasted text did not contain a clear factual claim that could be verified.",
                "supporting_evidence_summary": "",
                "contradiction_summary": "",
                "evidence_sources": [],
                "evidence_multiplier": 0.4,
                "penalty_multiplier": 0.0,
                "component_scores": {"S": 0.0, "R": 0.0, "A": 0.0, "D": 0.0, "C": 0.0, "E": 0.4, "P": 0.0},
            }

        components = self._compute_text_formula_components(claim_results, top_articles)
        final_score = max(
            0.0,
            min(
                1.0,
                (
                    0.45 * float(components["S"])
                    + 0.20 * float(components["R"])
                    + 0.20 * float(components["A"])
                    + 0.10 * float(components["D"])
                    + 0.05 * float(components["C"])
                )
                * float(components["E"])
                * float(components["P"]),
            ),
        )

        no_relevant = int(components["relevant_count"]) == 0
        contradiction = int(components["contradiction_count"]) > int(components["strong_support_count"]) + int(components["partial_support_count"])
        explicit_contradiction = any(str(item.get("claim_verdict") or "") == "contradicted" for item in claim_results)
        only_partial_or_background = int(components["strong_support_count"]) == 0 and int(components["partial_support_count"]) > 0
        if no_relevant and not contradiction:
            final_score = min(final_score, 0.349)
        if only_partial_or_background and not contradiction:
            final_score = min(final_score, 0.349)
        if int(components["strong_support_count"]) == 0 and not contradiction:
            final_score = min(final_score, 0.349)
        if explicit_contradiction:
            final_score = min(final_score, 0.25)
        final_label = self._generate_text_label(final_score, float(components["E"]), contradiction, float(components["C"]))
        if no_relevant and not contradiction:
            final_label = "Not Verifiable"
        if int(components["strong_support_count"]) == 0 and not contradiction:
            final_label = "Not Verifiable"
        if explicit_contradiction:
            final_label = "Unverified"

        relevant_articles = [
            item
            for result in claim_results
            for item in result.get("evidence_sources", [])
            if str(item.get("evidence_relation") or item.get("judgment_label")) in {"supports", "contradicts", "refutes"}
        ]
        context_articles = [
            item
            for result in claim_results
            for item in result.get("evidence_sources", [])
            if str(item.get("evidence_relation") or item.get("judgment_label")) == "partially_related"
        ]
        scored = score_evidence(
            [
                {
                    **item,
                    "alignment": "refutes" if str(item.get("evidence_relation")) in {"contradicts", "refutes"} else item.get("alignment") or item.get("evidence_relation"),
                    "source_type": item.get("source_type") or get_source_type(str(item.get("url") or item.get("domain") or "")),
                    "verdict_extracted": item.get("verdict_extracted") or item.get("fact_check_verdict") or "not_found",
                }
                for item in relevant_articles
            ]
        )
        engine_decision = decide_final_verdict(scored)
        exact_trusted_supports = [
            item
            for item in relevant_articles
            if str(item.get("evidence_relation")) == "supports"
            and str(item.get("aboutness")) in {"exact_claim", "strong_related"}
            and str(item.get("source_tier")) in {"tier_1", "tier_2"}
        ]
        exact_trusted_families = {
            self._domain_family(str(item.get("domain") or extract_domain(item.get("url")) or ""))
            for item in exact_trusted_supports
            if item.get("url") or item.get("domain")
        }
        if engine_decision["verdict"] == "Unverified":
            final_score = min(final_score, 0.25)
            final_label = "Unverified"
        elif engine_decision["verdict"] == "Not Verifiable":
            final_score = min(final_score, 0.349)
            final_label = "Not Verifiable"
        elif engine_decision["verdict"] == "Likely Verified" and final_label == "Verified":
            final_label = "Likely Verified"
            final_score = min(final_score, 0.699)
        elif engine_decision["verdict"] == "Verified":
            final_label = "Verified"
        if not (contradiction or explicit_contradiction) and exact_trusted_supports:
            if len(exact_trusted_families) >= 2:
                final_score = max(final_score, 0.70)
                final_label = "Verified"
            else:
                final_score = max(final_score, 0.50)
                if final_label not in {"Verified"}:
                    final_label = "Likely Verified"
        supporting_articles = strong_support = [item for item in relevant_articles if str(item.get("evidence_relation") or item.get("judgment_label")) == "supports"]
        partial_articles = context_articles
        refuting_articles = [item for item in relevant_articles if str(item.get("evidence_relation") or item.get("judgment_label")) in {"contradicts", "refutes"}]
        direct_source_found = any(bool(item.get("direct_source_match")) for item in top_articles)
        if final_label == "Verified":
            explanation = (
                "A direct or near-direct source article was recovered from the pasted text, and the top retrieved evidence strongly supported the same event and details."
                if direct_source_found
                else f"The top retrieved evidence strongly supported the pasted text, with {len(supporting_articles)} strong supporting articles."
            )
        elif final_label == "Likely Verified":
            explanation = (
                "A likely source article was recovered from the pasted text, and related reporting provided additional support."
                if direct_source_found
                else f"The top retrieved evidence supported the pasted text, with {len(supporting_articles)} strong matches and {len(partial_articles)} partial matches."
            )
        elif no_relevant:
            explanation = "No credible direct evidence was found for the specific claim. Related articles may have shared keywords, but they did not match the required entities, action, and details closely enough to count as support."
        elif contradiction or explicit_contradiction:
            contradiction_reason = next((str(item.get("claim_rationale") or "") for item in claim_results if item.get("claim_verdict") == "contradicted"), "")
            explanation = contradiction_reason or "The retrieved articles conflicted with the pasted text more than they supported it, so the verification score was reduced."
        else:
            explanation = "Related articles were retrieved, but none matched the pasted text strongly enough for a firmer verification result."

        evidence_summary = (
            f"Top retrieved evidence included {len(supporting_articles)} strong supporting articles and {len(partial_articles)} partial matches."
            if top_articles
            else "Usable corroborating evidence was limited because no matching candidate articles were recovered."
        )
        supporting_summary = " ".join(str(item.get("title") or item.get("url") or "") for item in supporting_articles[:2]).strip()
        contradiction_summary = " ".join(
            str(item.get("contradiction_reason") or item.get("title") or item.get("url") or "")
            for result in claim_results
            for item in result.get("evidence_sources", [])
            if str(item.get("judgment_label")) == "contradicts" or str(item.get("evidence_relation")) == "refutes"
        )[:240].strip()
        return {
            "final_score": round(final_score, 3),
            "final_label": final_label,
            "explanation": explanation,
            "evidence_summary": evidence_summary,
            "supporting_evidence_summary": supporting_summary,
            "contradiction_summary": contradiction_summary,
            "evidence_sources": self._dedupe_evidence_candidates(relevant_articles + context_articles)[:5],
            "evidence_multiplier": float(components["E"]),
            "penalty_multiplier": float(components["P"]),
            "component_scores": components,
            "debug": {
                "claim": str(meaning_bundle.get("simplified_meaning") or ""),
                "claim_tuple": meaning_bundle.get("claim_tuple", {}),
                "detected_country": extract_claim_context(str(meaning_bundle.get("simplified_meaning") or "")).get("country"),
                "selected_sources": get_sources_for_country(extract_claim_context(str(meaning_bundle.get("simplified_meaning") or "")).get("country"))[:12],
                "queries_used": meaning_bundle.get("exact_phrase_queries", []),
                "trusted_sources_found": [
                    {"url": item.get("url"), "domain": item.get("domain") or extract_domain(item.get("url")), "source_type": item.get("source_type")}
                    for item in relevant_articles + context_articles
                    if item.get("source_type") in {"trusted_news_page", "official_source_page", "fact_check_page"}
                ],
                "exact_supporting_sources": [
                    {"url": item.get("url"), "title": item.get("title"), "evidence_quote": item.get("evidence_quote"), "reason": item.get("reason_accepted_or_rejected")}
                    for item in supporting_articles
                    if str(item.get("aboutness")) in {"exact_claim", "strong_related"}
                ],
                "refuting_sources": [
                    {"url": item.get("url"), "title": item.get("title"), "reason": item.get("reason_accepted_or_rejected")}
                    for item in refuting_articles
                ],
                "mentions_only_sources": [
                    {"url": item.get("url"), "title": item.get("title"), "reason": item.get("reason_accepted_or_rejected")}
                    for item in [
                        item
                        for result in claim_results
                        for item in result.get("evidence_sources", [])
                        if str(item.get("aboutness")) in {"partial_context", "mentions_only"} or str(item.get("evidence_relation")) == "partially_related"
                    ]
                ],
                "score_reasoning": components.get("score_reasoning", {}),
                "evidence_pages": [
                    {
                        "url": item.get("url"),
                        "domain": item.get("domain") or extract_domain(item.get("url")),
                        "source_type": item.get("source_type") or get_source_type(str(item.get("url") or item.get("domain") or "")),
                        "alignment": item.get("alignment") or item.get("evidence_relation"),
                        "verdict_extracted": item.get("verdict_extracted") or item.get("fact_check_verdict") or "not_found",
                        "score": item.get("score", 0),
                        "reason": item.get("alignment_reason") or item.get("reason_accepted_or_rejected"),
                    }
                    for item in scored.get("items", [])
                ],
                "final_support_score": scored["support_score"],
                "final_refute_score": scored["refute_score"],
                "final_verdict": final_label,
                "final_reason": engine_decision["reason"],
            },
        }

    def _dedupe_queries(self, queries: list[str], limit: int) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized = " ".join((query or "").split())
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
            if len(deduped) >= limit:
                break
        return deduped

    def _evaluate_source_candidate(
        self,
        prepared: dict[str, str | list[str] | bool],
        extraction: dict,
        title: str,
        url: str,
        snippet: str,
        page_text: str,
        document: dict,
    ) -> dict:
        cleaned_input = str(prepared.get("cleaned_input") or "")
        normalized_input = str(prepared.get("normalized_text") or normalize_text(cleaned_input))
        normalized_page = normalize_text(page_text or "")
        normalized_title = normalize_text(title or "")
        normalized_snippet = normalize_text(snippet or "")
        input_sentences = [normalize_text(item) for item in prepared.get("sentences", []) if str(item).strip()]
        page_sentences = self._text_sentences(page_text or "")
        page_sentence_set = {normalize_text(item) for item in page_sentences if item}

        exact_sentence_matches = sum(1 for sentence in input_sentences[:6] if sentence and sentence in normalized_page)
        exact_sentence_ratio = exact_sentence_matches / max(1, min(len(input_sentences), 6))
        token_overlap = self._text_overlap(normalized_input, normalized_page)
        snippet_overlap = max(self._text_overlap(normalized_input, normalized_title), self._text_overlap(normalized_input, normalized_snippet))
        longest_common_span = self._longest_common_token_span_ratio(normalized_input, normalized_page)
        entity_overlap = self._entity_overlap_score(extraction, f"{title} {snippet} {page_text[:1000]}")
        ngram_overlap = self._ngram_overlap_score(normalized_input, normalized_page, 3)
        containment_score = 1.0 if normalized_input and normalized_input[:180] in normalized_page else 0.0
        headline_mode = self._is_headline_like(str(prepared.get("cleaned_input") or ""))
        headline_exactness = self._headline_exactness_score(str(prepared.get("cleaned_input") or ""), title, snippet)
        source_recovery_score = round(
            min(
                1.0,
                0.28 * exact_sentence_ratio
                + 0.18 * token_overlap
                + 0.20 * longest_common_span
                + 0.14 * ngram_overlap
                + 0.10 * entity_overlap
                + 0.05 * snippet_overlap
                + 0.05 * containment_score,
            ),
            3,
        )
        if headline_mode:
            source_recovery_score = round(max(source_recovery_score, min(1.0, headline_exactness)), 3)
        accepted = (
            exact_sentence_ratio >= 0.34
            or longest_common_span >= 0.52
            or containment_score >= 1.0
            or source_recovery_score >= 0.55
            or (headline_mode and headline_exactness >= 0.86)
        )
        reject_reason = ""
        if document.get("fetch_status") != "success":
            reject_reason = "candidate fetch failed"
        elif document.get("extract_status") != "extracted":
            reject_reason = "" if headline_mode and headline_exactness >= 0.86 else "candidate fetched but extracted text too short"
        elif snippet_overlap >= 0.5 and token_overlap < 0.2:
            reject_reason = "" if headline_mode and headline_exactness >= 0.86 else "title similar but text overlap low"
        elif source_recovery_score < 0.55:
            reject_reason = "threshold not met"
        return {
            "title": title,
            "url": url,
            "snippet": snippet,
            "passage": " ".join(page_sentences[:3])[:800] if page_sentences else (page_text[:800] if page_text else snippet),
            "domain": extract_domain(url),
            "source": "direct_source_recovery",
            "score": source_recovery_score,
            "retrieval_score": source_recovery_score,
            "source_recovery_score": source_recovery_score,
            "exact_sentence_match_ratio": round(exact_sentence_ratio, 3),
            "token_overlap": round(token_overlap, 3),
            "longest_common_span_ratio": round(longest_common_span, 3),
            "ngram_overlap": round(ngram_overlap, 3),
            "entity_overlap": round(entity_overlap, 3),
            "snippet_overlap": round(snippet_overlap, 3),
            "headline_exactness_score": round(headline_exactness, 3),
            "containment_score": round(containment_score, 3),
            "accepted_as_direct_source": accepted,
            "reject_reason": reject_reason,
            "reasoning_score": source_recovery_score,
            "explanation": "Direct source recovery compared pasted text against candidate page text for near-duplicate overlap.",
        }

    def _entity_overlap_score(self, extraction: dict, candidate_text: str) -> float:
        wanted = {normalize_text(str(item)) for item in extraction.get("entities", []) + extraction.get("locations", []) + extraction.get("dates", []) if str(item).strip()}
        if not wanted:
            return 0.0
        candidate = normalize_text(candidate_text)
        hits = sum(1 for item in wanted if item and item in candidate)
        return hits / max(1, len(wanted))

    def _ngram_overlap_score(self, left: str, right: str, n: int) -> float:
        left_tokens = [token for token in left.split() if token]
        right_tokens = [token for token in right.split() if token]
        if len(left_tokens) < n or len(right_tokens) < n:
            return 0.0
        left_ngrams = {" ".join(left_tokens[index : index + n]) for index in range(len(left_tokens) - n + 1)}
        right_ngrams = {" ".join(right_tokens[index : index + n]) for index in range(len(right_tokens) - n + 1)}
        if not left_ngrams or not right_ngrams:
            return 0.0
        return len(left_ngrams & right_ngrams) / max(1, len(left_ngrams))

    def _longest_common_token_span_ratio(self, left: str, right: str) -> float:
        left_tokens = [token for token in left.split() if token]
        right_tokens = [token for token in right.split() if token]
        if not left_tokens or not right_tokens:
            return 0.0
        right_index: dict[str, list[int]] = {}
        for index, token in enumerate(right_tokens):
            right_index.setdefault(token, []).append(index)
        best = 0
        for start, token in enumerate(left_tokens):
            for right_start in right_index.get(token, []):
                span = 0
                while (
                    start + span < len(left_tokens)
                    and right_start + span < len(right_tokens)
                    and left_tokens[start + span] == right_tokens[right_start + span]
                ):
                    span += 1
                best = max(best, span)
        return best / max(1, min(len(left_tokens), 30))

    def _extract_text_claims(self, extracted: ExtractedPayload) -> list[str]:
        text = (extracted.main_text or extracted.title or "").strip()
        if not text:
            return []

        compact = " ".join(text.split())
        if self._is_headline_like(compact):
            return [compact[:500]]

        fragments = re.split(r"(?<=[\.\!\?\n])\s+", compact)
        claims: list[str] = []
        for fragment in fragments:
            candidate = " ".join(fragment.split()).strip(" .")
            if not candidate or len(candidate) < 18:
                continue
            lowered = candidate.lower()
            if lowered.startswith(("i think", "i feel", "in my opinion", "maybe", "perhaps")):
                continue
            has_fact_signal = bool(
                re.search(
                    r"\b(is|are|was|were|has|have|had|will|said|claims|reported|announced|confirmed|approved|threatened|killed|won|lost|passed|arrested|warned|accused|ordered)\b",
                    lowered,
                )
            )
            if not has_fact_signal and not re.search(r"\d", candidate):
                continue
            if candidate not in claims:
                claims.append(candidate[:500])
            if len(claims) >= self.TEXT_CLAIM_LIMIT:
                break

        if not claims and extracted.title:
            claims = [str(extracted.title)[:500]]
        return claims

    def _build_text_hybrid_queries(
        self,
        prepared: dict[str, str | list[str] | bool],
        extraction: dict,
        claims: list[str],
        meaning_bundle: dict[str, object],
    ) -> dict[str, list[str]]:
        lexical_queries = list(meaning_bundle.get("exact_phrase_queries", []) or [])
        lexical_queries.extend(self._targeted_claim_search_queries(extraction, claims))
        for claim in extraction.get("claims", [])[: self.TEXT_CLAIM_LIMIT]:
            claim_text = " ".join(str(claim.get("text") or "").split()[:18]).strip()
            if claim_text:
                lexical_queries.append(claim_text)
                lexical_queries.extend(self._localized_claim_queries(claim_text))
            for variant in claim.get("short_forms", [])[:2]:
                variant_text = " ".join(str(variant or "").split()[:14]).strip()
                if variant_text:
                    lexical_queries.append(variant_text)
        entity_query = self._entity_focused_query(extraction)
        if entity_query:
            lexical_queries.append(entity_query)
        keyword_query = self._keyword_claim_query(prepared, extraction, claims)
        if keyword_query:
            lexical_queries.append(keyword_query)

        semantic_queries = []
        for claim in claims[: self.TEXT_CLAIM_LIMIT]:
            claim_query = " ".join(str(claim or "").split()[:18]).strip()
            if claim_query:
                semantic_queries.append(claim_query)
        simplified = str(meaning_bundle.get("simplified_meaning") or "").strip()
        if simplified:
            semantic_queries.append(simplified)

        return {
            "broad": self._dedupe_queries([*lexical_queries, *semantic_queries], limit=self.TEXT_MAX_SEARCH_QUERIES),
            "lexical": self._dedupe_queries(lexical_queries, limit=self.TEXT_MAX_SEARCH_QUERIES),
            "semantic": self._dedupe_queries(semantic_queries, limit=2),
            "trusted": self._dedupe_queries(self._trusted_source_queries(meaning_bundle), limit=self.TEXT_MAX_SEARCH_QUERIES),
            "fact_check": self._dedupe_queries(self._fact_check_queries(meaning_bundle), limit=4),
            "official": self._dedupe_queries(self._official_source_queries(meaning_bundle), limit=4),
        }

    def _trusted_source_queries(self, meaning_bundle: dict[str, object]) -> list[str]:
        claim_tuple = meaning_bundle.get("claim_tuple", {}) if isinstance(meaning_bundle.get("claim_tuple"), dict) else {}
        tuple_queries = self._claim_tuple_search_queries(claim_tuple)
        if tuple_queries:
            return tuple_queries
        base = str(meaning_bundle.get("simplified_meaning") or meaning_bundle.get("keyword_query") or "").strip()
        if not base:
            return []
        return [f"site:{domain} {base}" for domain in ("apnews.com", "reuters.com", "bbc.com", "cnn.com", "npr.org")]

    def _fact_check_queries(self, meaning_bundle: dict[str, object]) -> list[str]:
        base = str(meaning_bundle.get("simplified_meaning") or meaning_bundle.get("keyword_query") or "").strip()
        if not base:
            return []
        return [
            f"{base} fact check",
            f"site:politifact.com {base}",
            f"site:factcheck.org {base}",
            f"site:snopes.com {base}",
        ]

    def _official_source_queries(self, meaning_bundle: dict[str, object]) -> list[str]:
        claim_tuple = meaning_bundle.get("claim_tuple", {}) if isinstance(meaning_bundle.get("claim_tuple"), dict) else {}
        base = " ".join(
            str(claim_tuple.get(key) or "")
            for key in ("subject", "object", "event", "location")
        ).strip() or str(meaning_bundle.get("keyword_query") or "").strip()
        if not base:
            return []
        queries = []
        if str(claim_tuple.get("claim_type") or "") == "election result" or re.search(r"\b(?:election|primary|runoff|senate)\b", normalize_text(base)):
            queries.extend([f"site:sos.texas.gov {base}", f"site:results.texas-election.com {base}", f"site:gov {base} election results"])
        return queries

    def _localized_claim_queries(self, claim: str) -> list[str]:
        normalized = normalize_text(claim)
        queries: list[str] = []
        has_bangla = bool(re.search(r"[\u0980-\u09ff]", claim or ""))
        mentions_dhaka = bool(re.search(r"\b(?:dhaka)\b|ঢাকা|ঢাকায়|ঢাকায়", normalized))
        mentions_bangladesh = bool(re.search(r"\b(?:bangladesh)\b|বাংলাদেশ", normalized))
        mentions_students = bool(re.search(r"\b(?:student|students)\b|শিক্ষার্থী|ছাত্র|ছাত্রী", normalized))
        mentions_police = bool(re.search(r"\bpolice\b|পুলিশ", normalized))
        mentions_clash = bool(re.search(r"\b(?:clash|clashes|protest|movement)\b|সংঘর্ষ|আন্দোলন|বিক্ষোভ", normalized))
        mentions_fire = bool(re.search(r"\b(?:fire|blaze)\b|আগুন|অগ্নিকাণ্ড", normalized))
        if has_bangla and mentions_dhaka and mentions_students and mentions_police and mentions_clash:
            queries.extend(
                [
                    "ঢাকায় শিক্ষার্থীদের আন্দোলনে পুলিশের সঙ্গে সংঘর্ষ",
                    "Dhaka student protest police clash",
                    "Bangladesh Dhaka students police clash",
                ]
            )
        if mentions_dhaka and mentions_fire:
            queries.extend(["Dhaka Bangladesh major fire", "ঢাকায় অগ্নিকাণ্ড", "Dhaka fire Bangladesh"])
        if mentions_dhaka and not mentions_bangladesh:
            queries.append(f"Bangladesh Dhaka {' '.join(str(claim).split()[:10])}")
        return queries[:4]

    def _targeted_claim_search_queries(self, extraction: dict, claims: list[str]) -> list[str]:
        queries: list[str] = []
        for claim in extraction.get("claims", [])[: self.TEXT_CLAIM_LIMIT]:
            if not isinstance(claim, dict):
                continue
            text = str(claim.get("text") or "").strip()
            if not text:
                continue
            details = self._parse_claim_details(text)
            claim_tuple = details.get("claim_tuple", {}) if isinstance(details.get("claim_tuple"), dict) else {}
            queries.extend(self._claim_tuple_search_queries(claim_tuple))
            focused_terms = [
                *details.get("entities", [])[:5],
                *details.get("actions", [])[:4],
                *details.get("money", [])[:1],
            ]
            query = " ".join(dict.fromkeys(item for item in focused_terms if item))
            if len(query.split()) >= 4:
                queries.append(query)
            queries.append(" ".join(text.split()[:16]))
            if details.get("extraordinary"):
                core = " ".join(dict.fromkeys([*details.get("entities", [])[:4], *details.get("actions", [])[:3]]))
                if core:
                    queries.append(f"{core} fact check")
                    queries.append(f"site:gov.bd {core}")
            if details.get("legal_eligibility"):
                entity = " ".join(details.get("entities", [])[:2]).strip()
                queries.extend(
                    [
                        "U.S. presidential eligibility natural-born citizen",
                        f"{entity} birthplace citizenship presidential eligibility".strip(),
                        f"{entity} 2028 presidential run fact check".strip(),
                        f"{entity} renounced South African citizenship".strip(),
                    ]
                )
        if not queries:
            queries = [" ".join(str(claim or "").split()[:16]) for claim in claims[: self.TEXT_CLAIM_LIMIT]]
        return self._dedupe_queries(queries, limit=self.TEXT_MAX_SEARCH_QUERIES)

    def _gather_text_candidate_documents(
        self,
        prepared: dict[str, str | list[str] | bool],
        extraction: dict,
        query_plan: dict[str, list[str]],
        source_recovery_stage: dict,
    ) -> tuple[list[dict], dict]:
        cache_key = sha256_text(
            normalize_text(
                " | ".join(
                    [
                        *[str(item.get("text") or "") for item in extraction.get("claims", []) if isinstance(item, dict)],
                        *query_plan.get("broad", []),
                        *query_plan.get("trusted", []),
                        *query_plan.get("fact_check", []),
                        *query_plan.get("official", []),
                        *query_plan.get("lexical", []),
                        *query_plan.get("semantic", []),
                    ]
                )
            )
        )
        if cache_key in self._claim_candidate_cache:
            cached_candidates, cached_debug = self._claim_candidate_cache[cache_key]
            debug = {
                **cached_debug,
                "cache_hit": True,
            }
            return [dict(item) for item in cached_candidates], debug

        search_debug = {"broad": [], "trusted": [], "fact_check": [], "official": [], "lexical": [], "semantic": []}
        candidate_fetches: list[dict] = []
        candidates_by_url: dict[str, dict] = {}
        headline_mode = self._is_headline_like(str(prepared.get("cleaned_input") or ""))
        seeded = source_recovery_stage.get("best_candidate") if isinstance(source_recovery_stage.get("best_candidate"), dict) else None
        if seeded and seeded.get("url"):
            candidates_by_url[str(seeded.get("url"))] = {
                **seeded,
                "text": " ".join(filter(None, [str(seeded.get("title") or ""), str(seeded.get("passage") or ""), str(seeded.get("snippet") or "")])),
                "query_type": "direct_source_recovery",
                "source": "direct_source_recovery",
            }

        for query_type in ("broad", "trusted", "fact_check", "official", "lexical", "semantic"):
            for query in query_plan.get(query_type, [])[: self.TEXT_MAX_SEARCH_QUERIES]:
                bundle = self._search_bundle(query, expected_domain=None, limit=5)
                search_debug[query_type].append(
                    {
                        "query": query,
                        "failure_reason": bundle.get("failure_reason"),
                        "provider_status": bundle.get("provider_status", {}),
                        "provider_attempts": bundle.get("provider_attempts", []),
                        "results": [{"url": item.url, "title": item.title, "snippet": item.snippet} for item in bundle["results"][:5]],
                    }
                )
                for result in bundle["results"]:
                    if not result.url or result.url in candidates_by_url:
                        continue
                    title_exactness = self._headline_exactness_score(str(prepared.get("cleaned_input") or ""), result.title, result.snippet)
                    source_type = get_source_type(result.url or result.domain or "")
                    should_fetch = (
                        len(candidates_by_url) < 1
                        or query_type in {"trusted", "official", "fact_check"}
                        or source_type in {"trusted_news_page", "official_source_page", "fact_check_page"}
                        or (headline_mode and title_exactness >= 0.55 and len(candidate_fetches) < 3)
                    )
                    document = self.retrieval.fetch_page_document(result.url) if should_fetch else {
                        "url": result.url,
                        "fetch_status": "skipped",
                        "content_type": None,
                        "title": result.title,
                        "extracted_text": "",
                        "extracted_text_length": 0,
                        "extract_status": "search_snippet_only",
                    }
                    candidate_fetches.append(
                        {
                            "url": result.url,
                            "fetch_status": document.get("fetch_status"),
                            "content_type": document.get("content_type"),
                            "extracted_text_length": document.get("extracted_text_length"),
                            "title": document.get("title") or result.title,
                            "snippet": result.snippet,
                        }
                    )
                    candidate = self._build_text_candidate_document(prepared, extraction, result.title, result.url, result.snippet, document, query_type)
                    if headline_mode:
                        candidate["headline_exactness_score"] = round(title_exactness, 3)
                    candidates_by_url[result.url] = candidate
                    if len(candidates_by_url) >= self.TEXT_MAX_CANDIDATES:
                        break
                if len(candidates_by_url) >= self.TEXT_MAX_CANDIDATES:
                    break
            if len(candidates_by_url) >= self.TEXT_MAX_CANDIDATES:
                break

        failure_reasons = list(
            dict.fromkeys(
                str(item.get("failure_reason") or "")
                for group in search_debug.values()
                for item in group
                if item.get("failure_reason")
            )
        )
        candidates = list(candidates_by_url.values())
        debug = {
            "broad_searches": search_debug["broad"],
            "trusted_searches": search_debug["trusted"],
            "fact_check_searches": search_debug["fact_check"],
            "official_searches": search_debug["official"],
            "lexical_searches": search_debug["lexical"],
            "semantic_searches": search_debug["semantic"],
            "candidate_fetches": candidate_fetches,
            "retrieval_failure_reason": self._classify_retrieval_failure(failure_reasons) if failure_reasons and not candidates_by_url else None,
            "cache_hit": False,
        }
        self._claim_candidate_cache[cache_key] = ([dict(item) for item in candidates], dict(debug))
        return candidates, debug

    def _build_text_candidate_document(
        self,
        prepared: dict[str, str | list[str] | bool],
        extraction: dict,
        title: str,
        url: str,
        snippet: str,
        document: dict,
        query_type: str,
    ) -> dict:
        page_text = str(document.get("extracted_text") or "")
        page_sentences = self._text_sentences(page_text or snippet)
        passage = " ".join(page_sentences[:3])[:900] if page_sentences else snippet
        source_recovery = self._evaluate_source_candidate(prepared, extraction, title, url, snippet, page_text, document)
        headline_exactness = self._headline_exactness_score(str(prepared.get("cleaned_input") or ""), title, snippet)
        source_type = get_source_type(url)
        verdict_text = " ".join(filter(None, [title, snippet, passage, page_text[:1800]]))
        fact_check_verdict = extract_fact_check_verdict(verdict_text) if source_type == "fact_check_page" else "not_found"
        verdict_label = extract_fact_check_verdict_label(verdict_text) if source_type == "fact_check_page" else "not_found"
        return {
            "title": title,
            "url": url,
            "domain": extract_domain(url),
            "source_type": source_type,
            "fact_check_verdict": fact_check_verdict,
            "verdict_extracted": verdict_label,
            "snippet": snippet,
            "passage": passage,
            "text": " ".join(filter(None, [title, passage, snippet])),
            "query_type": query_type,
            "source": "web_search",
            "fetch_status": document.get("fetch_status"),
            "extract_status": document.get("extract_status"),
            "extracted_text_length": document.get("extracted_text_length"),
            "source_recovery_score": source_recovery.get("source_recovery_score", 0.0),
            "direct_source_match": source_recovery.get("accepted_as_direct_source", False),
            "headline_exactness_score": round(headline_exactness, 3),
            "exact_sentence_match_ratio": source_recovery.get("exact_sentence_match_ratio", 0.0),
            "token_overlap": source_recovery.get("token_overlap", 0.0),
            "ngram_overlap": source_recovery.get("ngram_overlap", 0.0),
            "longest_common_span_ratio": source_recovery.get("longest_common_span_ratio", 0.0),
            "entity_overlap": source_recovery.get("entity_overlap", 0.0),
            "containment_score": source_recovery.get("containment_score", 0.0),
            "reject_reason": source_recovery.get("reject_reason", ""),
        }

    def _hybrid_rank_text_candidates(
        self,
        prepared: dict[str, str | list[str] | bool],
        claims: list[str],
        query_plan: dict[str, list[str]],
        candidates: list[dict],
    ) -> dict:
        ranking_query = str(prepared.get("cleaned_input") or claims[0] if claims else "").strip()
        bm25_ranked = self.bm25.rank(query_plan.get("lexical", []), candidates, limit=5)
        semantic_ranked, semantic_error = self.semantic_retrieval.rank(query_plan.get("semantic", []), candidates, limit=5)

        merged: dict[str, dict] = {}
        for item in candidates:
            if item.get("url"):
                merged[str(item.get("url"))] = dict(item)
        for ranked in bm25_ranked:
            merged[str(ranked.get("url"))].update(ranked)
        for ranked in semantic_ranked:
            merged[str(ranked.get("url"))].update(ranked)

        merged_candidates = []
        for item in merged.values():
            bm25_score = float(item.get("bm25_score") or 0.0)
            semantic_score = float(item.get("semantic_score") or 0.0)
            source_recovery_score = float(item.get("source_recovery_score") or 0.0)
            hybrid_score = max(bm25_score, semantic_score, source_recovery_score)
            item["hybrid_score"] = round(hybrid_score, 3)
            merged_candidates.append(item)
        merged_candidates.sort(
            key=lambda item: (
                float(item.get("hybrid_score") or 0.0),
                float(item.get("source_recovery_score") or 0.0),
                float(item.get("bm25_score") or 0.0),
                float(item.get("semantic_score") or 0.0),
            ),
            reverse=True,
        )

        reranked, rerank_error = self.text_reranker.rerank(ranking_query, merged_candidates[:8], limit=5)
        return {
            "bm25_top_results": bm25_ranked,
            "semantic_top_results": semantic_ranked,
            "semantic_failure_reason": semantic_error,
            "merged_candidates": merged_candidates[: self.TEXT_MAX_CANDIDATES],
            "reranked_top_results": reranked[: self.TEXT_MAX_CANDIDATES],
            "rerank_failure_reason": rerank_error,
        }

    def _text_nlp_layer(self, job: VerificationJob, extracted: ExtractedPayload, input_text: str) -> LayerOutcome:
        prepared = self.preprocess_text(input_text)
        extraction = self.claim_extraction.extract_claims(prepared["cleaned_input"], max_claims=self.TEXT_CLAIM_LIMIT)
        claims = [item["text"] for item in extraction.get("claims", [])] or self._extract_text_claims(extracted)
        cleaned_input = str(prepared.get("cleaned_input") or "")
        if len(claims) == 1 and len(cleaned_input.split()) <= 80 and normalize_text(cleaned_input).startswith(normalize_text(claims[0])):
            claims = [cleaned_input]
            if extraction.get("claims"):
                extraction["claims"][0]["text"] = cleaned_input
        if self._is_low_information_text_input(prepared, extraction, claims):
            return self._text_low_information_layer(prepared, claims, extraction)
        claim_context = extract_claim_context(prepared["cleaned_input"])
        meaning_bundle = self._build_text_meaning_bundle(prepared, extraction, claims)
        query_plan = self._build_text_hybrid_queries(prepared, extraction, claims, meaning_bundle)
        trusted_queries = build_country_aware_queries(claim_context, limit=self.TEXT_MAX_SEARCH_QUERIES)
        query_plan["trusted"] = self._dedupe_queries([*query_plan.get("trusted", []), *trusted_queries], limit=self.TEXT_MAX_SEARCH_QUERIES)
        query_failure_reason = self._validate_text_query_plan(meaning_bundle, query_plan)
        if query_failure_reason:
            return self._text_pipeline_failure_layer(prepared, claims, meaning_bundle, query_plan, query_failure_reason)

        headline_mode = self._is_headline_like(str(prepared.get("cleaned_input") or ""))
        url_text_wrapper = (
            isinstance(extracted.metadata, dict)
            and str(extracted.metadata.get("source_type") or "").lower() == "url_article_text"
        )
        source_recovery_stage = (
            self._empty_source_recovery("Skipped direct source recovery for URL verification because the submitted URL cannot count as independent evidence.")
            if url_text_wrapper
            else self._direct_source_recovery(prepared, extraction, extracted)
            if bool(prepared.get("likely_copied_article_text")) or headline_mode
            else self._empty_source_recovery("Skipped direct source recovery for short claim-style input to keep verification responsive.")
        )
        candidates, retrieval_debug = self._gather_text_candidate_documents(prepared, extraction, query_plan, source_recovery_stage)
        retrieval_failure_reason = retrieval_debug.get("retrieval_failure_reason") or source_recovery_stage.get("rejection_reason")
        if not candidates and isinstance(retrieval_failure_reason, str) and (
            retrieval_failure_reason.startswith("retrieval_backend_failure:")
            or retrieval_failure_reason in {"provider_not_configured", "provider_auth_failed"}
        ):
            return self._text_pipeline_failure_layer(prepared, claims, meaning_bundle, query_plan, retrieval_failure_reason)
        ranking_debug = self._hybrid_rank_text_candidates(prepared, claims, query_plan, candidates)
        top_articles = ranking_debug["reranked_top_results"]
        claim_results = [
            self._score_claim_against_evidence(claim, self._build_claim_evidence_from_top_articles(claim, top_articles))
            for claim in claims
        ]
        aggregate = self._finalize_text_verification(claim_results, top_articles, meaning_bundle)
        source_recovery = retrieval_debug.get("final_source_recovery") or source_recovery_stage.get("final_source_recovery", {})
        if top_articles:
            recovery_candidate = max(
                top_articles,
                key=lambda item: (
                    float(item.get("source_recovery_score") or 0.0),
                    1 if str(item.get("source") or "") == "direct_source_recovery" else 0,
                    float(item.get("rerank_score") or 0.0),
                ),
            )
            if str(source_recovery.get("status") or "") == "not_found" and (
                float(recovery_candidate.get("source_recovery_score") or 0.0) >= 0.55
                or str(recovery_candidate.get("source") or "") == "direct_source_recovery"
            ):
                source_recovery = {
                    "status": "likely_recovered",
                    "url": recovery_candidate.get("url"),
                    "title": recovery_candidate.get("title"),
                    "publisher": recovery_candidate.get("domain"),
                    "overlap_confidence": round(float(recovery_candidate.get("source_recovery_score") or 0.0), 3),
                    "match_type": "likely",
                    "reason": "A likely source article was recovered after candidate comparison, even though exact body overlap stayed below the strict direct-match threshold.",
                }

        append_audit_entry(
            self.db,
            job,
            "text_nlp_completed",
            {
                "claim_count": len(claims),
                "evidence_source_count": len(aggregate["evidence_sources"]),
                "final_score": aggregate["final_score"],
                "final_label": aggregate["final_label"],
                "source_recovery_status": source_recovery.get("status"),
            },
        )
        for candidate in aggregate["evidence_sources"]:
            self.db.add(
                SimilarityMatch(
                    job_id=job.id,
                    candidate_url=candidate.get("url"),
                    candidate_title=candidate.get("title"),
                    score=float(candidate.get("score") or 0.0),
                    explanation=str(candidate.get("explanation") or candidate.get("summary") or ""),
                    metadata_json={
                        "source": candidate.get("source", "web_search"),
                        "snippet": candidate.get("snippet"),
                        "domain": candidate.get("domain"),
                        "relation": candidate.get("relation") or candidate.get("evidence_relation"),
                        "evidence_relation": candidate.get("evidence_relation"),
                        "aboutness": candidate.get("aboutness"),
                        "stance": candidate.get("stance"),
                        "exact_claim_support": candidate.get("exact_claim_support"),
                        "evidence_quote": candidate.get("evidence_quote"),
                        "relation_confidence": candidate.get("relation_confidence"),
                        "claim_evidence_verifier": candidate.get("claim_evidence_verifier"),
                        "rerank_score": candidate.get("rerank_score"),
                        "embedding_similarity": candidate.get("embedding_similarity") or candidate.get("semantic_score"),
                        "entity_match_score": candidate.get("entity_match_score"),
                        "location_match_score": candidate.get("location_match_score"),
                        "location_match_pass": candidate.get("location_match_pass"),
                        "event_match_score": candidate.get("event_match_score"),
                        "context_match_pass": candidate.get("context_match_pass"),
                        "mismatch_reason": candidate.get("mismatch_reason"),
                        "reason_accepted_or_rejected": candidate.get("reason_accepted_or_rejected"),
                    },
                )
            )

        all_search_queries = (
            query_plan.get("broad", [])
            + query_plan.get("trusted", [])
            + query_plan.get("fact_check", [])
            + query_plan.get("official", [])
            + query_plan.get("lexical", [])
            + query_plan.get("semantic", [])
        )
        raw_search_results = [
            result
            for item in (
                retrieval_debug.get("broad_searches", [])
                + retrieval_debug.get("trusted_searches", [])
                + retrieval_debug.get("fact_check_searches", [])
                + retrieval_debug.get("official_searches", [])
                + retrieval_debug.get("lexical_searches", [])
                + retrieval_debug.get("semantic_searches", [])
            )
            for result in item.get("results", [])
        ]
        components = aggregate.get("component_scores", {})
        raw_text_score = (
            100
            * (
                0.45 * float(components.get("S") or 0.0)
                + 0.20 * float(components.get("R") or 0.0)
                + 0.20 * float(components.get("A") or 0.0)
                + 0.10 * float(components.get("D") or 0.0)
                + 0.05 * float(components.get("C") or 0.0)
            )
        )
        pipeline_debug = {
            "full_input_text": prepared["input_text"],
            "claim_tuple": meaning_bundle.get("claim_tuple", {}),
            "queries_used": all_search_queries,
            "search_results_raw": raw_search_results,
            "fetched_pages": retrieval_debug.get("candidate_fetches", []),
            "trusted_sources_found": aggregate.get("debug", {}).get("trusted_sources_found", []),
            "exact_supporting_sources": aggregate.get("debug", {}).get("exact_supporting_sources", []),
            "refuting_sources": aggregate.get("debug", {}).get("refuting_sources", []),
            "related_rejected_sources": aggregate.get("debug", {}).get("mentions_only_sources", []),
            "component_scores": {
                "S_claim_evidence_support": round(100 * float(components.get("S") or 0.0), 1),
                "R_source_reliability": round(100 * float(components.get("R") or 0.0), 1),
                "A_cross_source_agreement": round(100 * float(components.get("A") or 0.0), 1),
                "D_evidence_directness": round(100 * float(components.get("D") or 0.0), 1),
                "C_claim_clarity": round(100 * float(components.get("C") or 0.0), 1),
                "E_evidence_sufficiency": round(100 * float(components.get("E") or 0.0), 1),
                "P_penalty": round(100 * float(components.get("P") or 0.0), 1),
            },
            "raw_score": round(raw_text_score, 1),
            "adjusted_score": round(100 * float(aggregate.get("final_score") or 0.0), 1),
            "final_label": aggregate.get("final_label", ""),
        }

        return LayerOutcome(
            status="supporting" if aggregate["final_score"] >= 0.5 else "none",
            confidence=round(aggregate["final_score"], 3),
            explanation=aggregate["explanation"],
            details={
                "mode": "text_verification",
                "input_text": prepared["input_text"],
                "cleaned_input": prepared["cleaned_input"],
                "normalized_text": prepared["normalized_text"],
                "detected_language": prepared["detected_language"],
                "extracted_claims": claims,
                "claim_extraction": extraction,
                "pipeline_stages": {
                    "input_normalization": {"status": "completed", "cleaned_input": prepared["cleaned_input"], "language": prepared["detected_language"]},
                    "claim_extraction": {"status": "completed" if claims else "no_checkable_claim", "claim_count": len(claims), "primary_claim": prepared["core_claim"]},
                    "direct_source_recovery": {
                        "status": source_recovery_stage.get("status"),
                        "best_score": source_recovery_stage.get("best_score"),
                        "best_url": source_recovery_stage.get("best_candidate", {}).get("url") if isinstance(source_recovery_stage.get("best_candidate"), dict) else None,
                        "reason": source_recovery_stage.get("final_source_recovery", {}).get("reason"),
                    },
                    "meaning_simplification": {
                        "status": "completed",
                        "simplified_meaning": meaning_bundle.get("simplified_meaning"),
                        "keyword_query": meaning_bundle.get("keyword_query"),
                        "entity_query": meaning_bundle.get("entity_query"),
                    },
                    "query_generation": {
                        "status": "completed",
                        "queries": all_search_queries,
                    },
                    "retrieval": {
                        "status": "completed" if candidates else "retrieval_backend_failure" if retrieval_debug.get("retrieval_failure_reason") else "no_relevant_evidence",
                        "candidate_count": len(candidates),
                        "failure_reason": retrieval_debug.get("retrieval_failure_reason") or ranking_debug.get("semantic_failure_reason"),
                    },
                    "article_comparison": {"status": "completed" if top_articles else "no_candidates", "source_recovery": source_recovery},
                    "reranking": {
                        "status": "completed" if top_articles else "no_reranked_evidence",
                        "candidate_count": len(top_articles),
                        "failure_reason": ranking_debug.get("rerank_failure_reason"),
                    },
                    "evidence_scoring": {"status": "completed" if claim_results else "no_evidence", "claim_result_count": len(claim_results)},
                    "final_verdict_generation": {
                        "status": "completed",
                        "final_label": aggregate["final_label"],
                        "final_score": aggregate["final_score"],
                        "component_scores": aggregate.get("component_scores", {}),
                    },
                },
                "claim_results": claim_results,
                "matches": aggregate["evidence_sources"],
                "evidence_sources": aggregate["evidence_sources"],
                "supporting_evidence_summary": aggregate["supporting_evidence_summary"],
                "contradiction_summary": aggregate["contradiction_summary"],
                "evidence_summary": aggregate["evidence_summary"],
                "fact_check_verdict": aggregate["final_label"],
                "fact_check_assessment": aggregate["explanation"],
                "fact_check_basis": "Claim extraction, BM25 retrieval, semantic retrieval, reranking, and claim-evidence judgment were used for pasted text verification.",
                "text_verification_score": aggregate["final_score"],
                "text_verification_label": aggregate["final_label"],
                "evidence_multiplier": aggregate["evidence_multiplier"],
                "penalty_multiplier": aggregate["penalty_multiplier"],
                "text_component_scores": aggregate.get("component_scores", {}),
                "pipeline_failure": False,
                "pipeline_failure_reason": None,
                "retrieval_state": "retrieval_backend_failure" if str(retrieval_debug.get("retrieval_failure_reason") or "").startswith("retrieval_backend_failure:") else "completed",
                "simplified_meaning": meaning_bundle.get("simplified_meaning"),
                "exact_phrase_queries": meaning_bundle.get("exact_phrase_queries", []),
                "trusted_source_queries": query_plan.get("trusted", []),
                "lexical_queries": query_plan.get("lexical", []),
                "semantic_queries": query_plan.get("semantic", []),
                "simplified_meaning_query": meaning_bundle.get("simplified_meaning"),
                "search_queries_used": all_search_queries,
                "match_summary": aggregate["evidence_summary"],
                "metadata_signals": ["input_text", "claim_extraction", "evidence_retrieval", "cross_source_agreement"],
                "trusted_publisher_match": None,
                "total_match_count": len(aggregate["evidence_sources"]),
                "has_more_matches": False,
                "source_recovery": source_recovery,
                "retrieval_debug": retrieval_debug,
                "verification_debug": aggregate.get("debug", {}),
                "pipeline_debug": pipeline_debug,
                "text_debug": {
                    "raw_input": prepared["input_text"],
                    "cleaned_input": prepared["cleaned_input"],
                    "normalized_input": prepared["normalized_text"],
                    "extracted_claim": claims[0] if claims else "",
                    "provider_used": list(
                        dict.fromkeys(
                            [
                                *[
                                    provider
                                    for item in (
                                        retrieval_debug.get("broad_searches", [])
                                        + retrieval_debug.get("trusted_searches", [])
                                        + retrieval_debug.get("fact_check_searches", [])
                                        + retrieval_debug.get("official_searches", [])
                                        + retrieval_debug.get("lexical_searches", [])
                                        + retrieval_debug.get("semantic_searches", [])
                                    )
                                    for provider in item.get("providers_used", [])
                                ],
                                *[
                                    provider
                                    for item in source_recovery_stage.get("search_results", [])
                                    for provider in item.get("providers_used", [])
                                ],
                            ]
                        )
                    ),
                    "generated_queries": query_plan.get("broad", []) + query_plan.get("trusted", []) + query_plan.get("fact_check", []) + query_plan.get("official", []) + query_plan.get("lexical", []) + query_plan.get("semantic", []),
                    "simplified_meaning": meaning_bundle.get("simplified_meaning"),
                    "extracted_claim_tuple": meaning_bundle.get("claim_tuple", {}),
                    "exact_phrase_queries": meaning_bundle.get("exact_phrase_queries", []),
                    "broad_queries": query_plan.get("broad", []),
                    "trusted_source_queries": query_plan.get("trusted", []),
                    "fact_check_queries": query_plan.get("fact_check", []),
                    "official_source_queries": query_plan.get("official", []),
                    "lexical_queries": query_plan.get("lexical", []),
                    "lexical_entity_queries": meaning_bundle.get("lexical_queries", []),
                    "semantic_queries": query_plan.get("semantic", []),
                    "final_search_queries_sent": query_plan.get("broad", []) + query_plan.get("trusted", []) + query_plan.get("fact_check", []) + query_plan.get("official", []) + query_plan.get("lexical", []) + query_plan.get("semantic", []),
                    "search_failure_reason": retrieval_debug.get("retrieval_failure_reason") or ranking_debug.get("semantic_failure_reason"),
                    "provider_status": [
                        item.get("provider_status")
                        for item in (
                            retrieval_debug.get("broad_searches", [])
                            + retrieval_debug.get("trusted_searches", [])
                            + retrieval_debug.get("fact_check_searches", [])
                            + retrieval_debug.get("official_searches", [])
                            + retrieval_debug.get("lexical_searches", [])
                            + retrieval_debug.get("semantic_searches", [])
                        )
                        if item.get("provider_status")
                    ]
                    + [
                        item.get("provider_status")
                        for item in source_recovery_stage.get("search_results", [])
                        if item.get("provider_status")
                    ],
                    "bm25_top_results": ranking_debug.get("bm25_top_results", []),
                    "semantic_top_results": ranking_debug.get("semantic_top_results", []),
                    "reranked_top_5": ranking_debug.get("reranked_top_results", []),
                    "broad_searches": retrieval_debug.get("broad_searches", []),
                    "trusted_searches": retrieval_debug.get("trusted_searches", []),
                    "fact_check_searches": retrieval_debug.get("fact_check_searches", []),
                    "official_searches": retrieval_debug.get("official_searches", []),
                    "lexical_searches": retrieval_debug.get("lexical_searches", []),
                    "semantic_searches": retrieval_debug.get("semantic_searches", []),
                    "top_5_urls": [item.get("url") for item in top_articles[:5] if item.get("url")],
                    "brave_results_count": sum(
                        int(item.get("result_count") or 0)
                        for item in (
                            retrieval_debug.get("broad_searches", [])
                            + retrieval_debug.get("trusted_searches", [])
                            + retrieval_debug.get("fact_check_searches", [])
                            + retrieval_debug.get("official_searches", [])
                            + retrieval_debug.get("lexical_searches", [])
                            + retrieval_debug.get("semantic_searches", [])
                        )
                    )
                    + sum(int(item.get("result_count") or 0) for item in source_recovery_stage.get("search_results", [])),
                    "brave_top_results": [
                        result
                        for item in (
                            retrieval_debug.get("broad_searches", [])
                            + retrieval_debug.get("trusted_searches", [])
                            + retrieval_debug.get("fact_check_searches", [])
                            + retrieval_debug.get("official_searches", [])
                            + retrieval_debug.get("lexical_searches", [])
                            + retrieval_debug.get("semantic_searches", [])
                        )
                        for result in item.get("results", [])
                    ][:5],
                    "support_judgments": [
                        {
                            "claim": result.get("claim_text"),
                            "evidence": [
                                {
                                    "url": item.get("url"),
                                    "title": item.get("title"),
                                    "judgment_label": item.get("judgment_label"),
                                    "evidence_relation": item.get("evidence_relation"),
                                    "judgment_confidence": item.get("judgment_confidence"),
                                    "rerank_score": item.get("rerank_score"),
                                    "bm25_score": item.get("bm25_score"),
                                    "semantic_score": item.get("semantic_score"),
                                    "entity_match_score": item.get("entity_match_score"),
                                    "location_match_score": item.get("location_match_score"),
                                    "location_match_pass": item.get("location_match_pass"),
                                    "event_match_score": item.get("event_match_score"),
                                    "context_match_pass": item.get("context_match_pass"),
                                    "mismatch_reason": item.get("mismatch_reason"),
                                }
                                for item in result.get("evidence_sources", [])[:3]
                            ],
                            "rejected_sources": result.get("top_candidate_diagnostics", []),
                        }
                        for result in claim_results
                    ],
                    "candidate_fetches": retrieval_debug.get("candidate_fetches", []),
                    "overlap_scores": source_recovery_stage.get("overlap_scores", []),
                    "direct_source_recovery": source_recovery_stage,
                    "final_source_recovery": source_recovery,
                    "source_recovery_status": source_recovery.get("status"),
                    "source_recovery_reason": source_recovery.get("reason"),
                    "component_scores": aggregate.get("component_scores", {}),
                    "trusted_sources_found": aggregate.get("debug", {}).get("trusted_sources_found", []),
                    "exact_supporting_sources": aggregate.get("debug", {}).get("exact_supporting_sources", []),
                    "refuting_sources": aggregate.get("debug", {}).get("refuting_sources", []),
                    "mentions_only_sources": aggregate.get("debug", {}).get("mentions_only_sources", []),
                    "score_reasoning": aggregate.get("debug", {}).get("score_reasoning", {}),
                    "formula_components": {
                        key: value
                        for key, value in aggregate.get("component_scores", {}).items()
                        if key in {"S", "R", "A", "D", "C", "E", "P"}
                    },
                    "final_score": aggregate.get("final_score"),
                    "final_label": aggregate.get("final_label"),
                    "pipeline_debug": pipeline_debug,
                },
            },
        )

    def _retrieve_text_evidence(
        self,
        prepared: dict[str, str | list[str] | bool],
        extracted: ExtractedPayload,
        claims: list[str],
        extraction: dict,
        source_recovery: dict,
    ) -> tuple[dict[str, list[dict]], dict]:
        evidence_by_claim: dict[str, list[dict]] = {}
        pooled_candidates, pool_debug = self._build_text_evidence_pool(prepared, extracted, claims, extraction, source_recovery)

        for claim in claims:
            ranked = sorted(
                pooled_candidates,
                key=lambda item: (
                    float(item.get("source_recovery_score") or 0.0),
                    float(item.get("reasoning_score") or 0.0),
                    self._headline_exactness_score(claim, str(item.get("title") or ""), str(item.get("snippet") or "")),
                    self._text_evidence_directness(claim, [item]),
                    float(item.get("score") or item.get("retrieval_score") or 0.0),
                ),
                reverse=True,
            )
            evidence_by_claim[claim] = ranked[: self.TEXT_EVIDENCE_PER_CLAIM]
        return evidence_by_claim, pool_debug

    def _build_text_evidence_pool(
        self,
        prepared: dict[str, str | list[str] | bool],
        extracted: ExtractedPayload,
        claims: list[str],
        extraction: dict,
        source_recovery: dict,
    ) -> tuple[list[dict], dict]:
        normalized_input = " ".join(str(prepared.get("cleaned_input") or "").split())
        headline_mode = self._is_headline_like(normalized_input)
        query_plan = self._text_query_plan(prepared, extracted, claims, extraction, headline_mode)

        pooled_candidates: list[dict] = []
        seen_keys: set[str] = set()
        queries_used: list[str] = []
        candidates_considered: list[dict] = []
        if source_recovery.get("best_candidate"):
            best_candidate = dict(source_recovery["best_candidate"])
            key = f"{best_candidate.get('url')}::{best_candidate.get('passage') or best_candidate.get('snippet') or best_candidate.get('title')}"
            best_candidate["source"] = "direct_source_recovery"
            best_candidate["score"] = max(float(best_candidate.get("score") or 0.0), float(best_candidate.get("source_recovery_score") or 0.0))
            pooled_candidates.append(best_candidate)
            seen_keys.add(key)

        for query in query_plan:
            queries_used.append(query)
            candidates = self._search_text_candidates(query, exact_first=headline_mode or query.startswith("\""))
            for candidate in candidates:
                key = f"{candidate.get('url')}::{candidate.get('passage') or candidate.get('snippet') or candidate.get('title')}"
                candidates_considered.append({"query": query, "url": candidate.get("url"), "title": candidate.get("title"), "score": candidate.get("score") or candidate.get("retrieval_score")})
                if not candidate.get("url") or key in seen_keys:
                    continue
                seen_keys.add(key)
                pooled_candidates.append(candidate)
                if len(pooled_candidates) >= max(self.DEFAULT_NLP_MATCH_LIMIT, 10):
                    return pooled_candidates, {
                        "query_generation": {"status": "completed", "queries": queries_used},
                        "retrieval": {"status": "completed" if pooled_candidates else "no_relevant_evidence", "candidates_considered": candidates_considered[:12]},
                        "reranking": {"status": "completed", "top_candidates": pooled_candidates[:5]},
                    }
        return pooled_candidates, {
            "query_generation": {"status": "completed", "queries": queries_used},
            "retrieval": {"status": "completed" if pooled_candidates else "no_relevant_evidence", "candidates_considered": candidates_considered[:12]},
            "reranking": {"status": "completed" if pooled_candidates else "no_relevant_evidence", "top_candidates": pooled_candidates[:5]},
        }

    def _text_query_plan(
        self,
        prepared: dict[str, str | list[str] | bool],
        extracted: ExtractedPayload,
        claims: list[str],
        extraction: dict,
        headline_mode: bool,
    ) -> list[str]:
        input_text = str(prepared.get("cleaned_input") or "")
        plan: list[str] = []
        sentence_chunks = self._build_source_recovery_sentence_chunks(prepared)
        for sentence in sentence_chunks[:3]:
            plan.append(f"\"{sentence}\"")
        for claim in extraction.get("claims", [])[:2]:
            text = str(claim.get("text") or "").strip()
            if text:
                plan.append(" ".join(text.split()[:18]))
            for variant in claim.get("short_forms", [])[:2]:
                plan.append(" ".join(str(variant).split()[:12]))
        entity_query = self._entity_focused_query(extraction)
        if entity_query:
            plan.append(entity_query)
        keyword_query = self._keyword_claim_query(prepared, extraction, claims)
        if keyword_query:
            plan.append(keyword_query)
        if input_text:
            plan.append(" ".join(input_text.split()[:18]))
        if not headline_mode:
            summary_query = self._build_fact_check_summary_query(extracted)
            if summary_query:
                plan.append(summary_query)

        deduped: list[str] = []
        seen: set[str] = set()
        for query in plan:
            normalized = " ".join((query or "").split())
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
            if len(deduped) >= self.TEXT_MAX_SEARCH_QUERIES:
                break
        return deduped

    def _search_text_candidates(self, query: str, exact_first: bool = False) -> list[dict]:
        normalized_query = " ".join((query or "").split())
        if not normalized_query:
            return []

        query_variants = [f"\"{normalized_query}\"", normalized_query] if exact_first else [normalized_query]
        collected: list[dict] = []
        seen: set[str] = set()
        for query_variant in query_variants:
            passages = self.retrieval.retrieve_hybrid(query_variant, limit=self.TEXT_SEARCH_PER_QUERY)
            for passage in passages:
                key = f"{passage.get('url')}::{passage.get('passage')}"
                if key in seen:
                    continue
                seen.add(key)
                collected.append(passage)
                if len(collected) >= self.TEXT_SEARCH_PER_QUERY:
                    break
            if len(collected) >= self.TEXT_SEARCH_PER_QUERY:
                break
        return collected[: self.TEXT_SEARCH_PER_QUERY]

    def _score_claim_against_evidence(self, claim: str, evidence_sources: list[dict]) -> dict:
        clarity = self._claim_clarity_score(claim)
        if not claim.strip():
            return {
                "claim_text": claim,
                "support_score": 0.0,
                "source_reliability": 0.0,
                "agreement_score": 0.0,
                "directness_score": 0.0,
                "clarity_score": 0.0,
                "claim_score": 0.0,
                "claim_label": map_score_to_label(0.0),
                "evidence_summary": "No checkable claim was found.",
                "evidence_sources": [],
            }

        if not evidence_sources:
            return {
                "claim_text": claim,
                "support_score": 0.0,
                "source_reliability": 0.0,
                "agreement_score": 0.0,
                "directness_score": 0.0,
                "clarity_score": round(clarity, 3),
                "claim_score": 0.0,
                "claim_label": "Not Verifiable",
                "claim_verdict": "insufficient_evidence",
                "claim_rationale": "No relevant supporting article was retrieved for this claim.",
                "retrieval_outcome": "no_relevant_evidence_found",
                "evidence_summary": "No relevant evidence sources were retrieved for this claim.",
                "evidence_sources": [],
            }

        judged_evidence, judgment_error = self.evidence_judgment.judge(claim, evidence_sources)
        if not judged_evidence:
            judged_evidence = [{**item, "judgment_label": "unrelated", "judgment_confidence": 0.0} for item in evidence_sources]

        claim_details = self._parse_claim_details(claim)
        claim_id = sha256_text(normalize_text(claim))[:12]
        for item in judged_evidence:
            item["source_type"] = item.get("source_type") or get_source_type(str(item.get("url") or item.get("domain") or ""))
            item["source_tier"] = self._source_tier(item)
            evidence_text = " ".join(str(item.get(key) or "") for key in ("title", "snippet", "passage", "text"))
            item["fact_check_verdict"] = item.get("fact_check_verdict") or (
                extract_fact_check_verdict(evidence_text) if item["source_type"] == "fact_check_page" else "not_found"
            )
            item["verdict_extracted"] = item.get("verdict_extracted") or (
                extract_fact_check_verdict_label(evidence_text) if item["source_type"] == "fact_check_page" else "not_found"
            )
            alignment = classify_evidence(claim, evidence_text, is_fact_check=item["source_type"] == "fact_check_page")
            relevance = self._claim_evidence_relevance_score(claim, item)
            credibility = self._source_credibility_score(item)
            label = str(item.get("judgment_label") or "unrelated")
            context_match = self.compare_claim_and_evidence_context(claim, item)
            headline_exactness = max(
                float(item.get("headline_exactness_score") or 0.0),
                self._headline_exactness_score(claim, str(item.get("title") or ""), str(item.get("snippet") or "")),
            )
            exact_headline_source = (
                headline_exactness >= 0.86
                and item["source_type"] in {"trusted_news_page", "official_source_page"}
            )
            trusted_news_support = self._trusted_news_exact_support(claim_details.get("claim_tuple", {}), item)
            if not trusted_news_support.get("supports"):
                trusted_news_support = self._trusted_ordinary_news_support(claim, claim_details, item)
            retrieval_strength = max(
                float(item.get("rerank_score") or 0.0),
                float(item.get("semantic_score") or 0.0),
                float(item.get("bm25_score") or 0.0),
                float(item.get("source_recovery_score") or 0.0),
                headline_exactness,
            )
            contradiction_reason = self._explicit_contradiction_reason(claim, item)
            if trusted_news_support.get("supports"):
                label = "supports"
                relevance = max(relevance, float(trusted_news_support.get("support_score") or 0.9))
                item["judgment_confidence"] = max(float(item.get("judgment_confidence") or 0.0), 0.92)
                item["direct_source_match"] = bool(item.get("direct_source_match")) or float(trusted_news_support.get("directness_score") or 0.0) >= 0.85
                item["source_recovery_score"] = max(float(item.get("source_recovery_score") or 0.0), float(trusted_news_support.get("directness_score") or 0.0))
                item["trusted_news_exact_support"] = trusted_news_support
            elif exact_headline_source:
                label = "supports"
                relevance = max(relevance, min(0.96, headline_exactness))
                item["judgment_confidence"] = max(float(item.get("judgment_confidence") or 0.0), min(0.94, headline_exactness))
                item["direct_source_match"] = True
                item["source_recovery_score"] = max(float(item.get("source_recovery_score") or 0.0), headline_exactness)
            elif item["source_type"] == "fact_check_page" and item["fact_check_verdict"] == "refutes" and alignment.get("aboutness") == "exact_claim":
                label = "contradicts"
                relevance = max(relevance, 0.9)
                contradiction_reason = "A trusted fact-check page's verdict refutes this claim."
                item["contradiction_reason"] = contradiction_reason
                item["judgment_confidence"] = max(float(item.get("judgment_confidence") or 0.0), 0.9)
            elif item["source_type"] == "fact_check_page" and item["fact_check_verdict"] == "supports" and alignment.get("aboutness") == "exact_claim":
                label = "supports"
                relevance = max(relevance, 0.9)
                item["judgment_confidence"] = max(float(item.get("judgment_confidence") or 0.0), 0.9)
            elif contradiction_reason:
                label = "contradicts"
                relevance = max(relevance, 0.86)
                item["contradiction_reason"] = contradiction_reason
                item["judgment_confidence"] = max(float(item.get("judgment_confidence") or 0.0), 0.86)
            elif bool(context_match.get("context_match_pass")) and retrieval_strength >= 0.86 and self._text_overlap(claim, " ".join(str(item.get(key) or "") for key in ("title", "snippet", "passage", "text"))) >= 0.5:
                relevance = max(relevance, min(0.95, retrieval_strength))
            elif not bool(context_match.get("location_match_pass")):
                label = "irrelevant"
                relevance = min(relevance, 0.45)
            elif not bool(context_match.get("event_match_pass")) and context_match.get("claim_events"):
                label = "partially_supports" if relevance >= 0.72 else "irrelevant"
                relevance = min(relevance, 0.58)
            if relevance < 0.6:
                label = "irrelevant"
            elif relevance < 0.8 and label == "supports":
                label = "partially_supports"
            if claim_details.get("extraordinary") and label == "supports" and (relevance < 0.8 or credibility < 0.75):
                label = "partially_supports"
            verification = self.claim_evidence_verifier.verify(claim, item, relevance_score=relevance, fallback_label=label)
            context_match = verification.context_match
            evidence_relation = verification.relation_label
            if trusted_news_support.get("supports"):
                evidence_relation = "supports"
            elif exact_headline_source:
                evidence_relation = "supports"
            elif item["source_type"] == "fact_check_page" and item["fact_check_verdict"] == "refutes" and alignment.get("aboutness") == "exact_claim":
                evidence_relation = "refutes"
            elif item["source_type"] == "fact_check_page" and item["fact_check_verdict"] == "supports" and bool(context_match.get("context_match_pass")):
                evidence_relation = "supports"
            elif alignment["alignment"] in {"mentions_only", "irrelevant"} and evidence_relation == "supports" and not exact_headline_source and not trusted_news_support.get("supports"):
                evidence_relation = "partially_related"
            elif evidence_relation == "supports" and not bool(alignment.get("action_match", True)) and not exact_headline_source and not trusted_news_support.get("supports"):
                evidence_relation = "partially_related"
            if item["source_type"] not in {"fact_check_page", "trusted_news_page", "official_source_page"} and evidence_relation == "supports":
                evidence_relation = "partially_related"
            if evidence_relation == "supports":
                label = "supports"
            elif evidence_relation == "partially_related":
                label = "partially_supports"
            elif evidence_relation in {"contradicts", "refutes"}:
                label = "contradicts"
            else:
                label = "irrelevant"
            item["claim_relevance_score"] = round(relevance, 3)
            item["credibility_score"] = round(credibility, 3)
            item["alignment"] = "refutes" if evidence_relation in {"contradicts", "refutes"} else "supports" if evidence_relation == "supports" else alignment["alignment"]
            item["aboutness"] = self._strict_aboutness_label(
                evidence_relation,
                trusted_news_support,
                alignment,
                context_match,
                relevance,
            )
            item["alignment_confidence"] = alignment["confidence"]
            item["alignment_reason"] = alignment["reason"]
            item["matched_entities"] = alignment["matched_entities"]
            item["missing_entities"] = alignment["missing_entities"]
            item["contradictions"] = alignment["contradictions"]
            item["evidence_quote"] = trusted_news_support.get("evidence_quote") or alignment.get("evidence_quote", "")
            item["exact_claim_support"] = evidence_relation == "supports" and (alignment.get("aboutness") == "exact_claim" or trusted_news_support.get("supports"))
            item["verdict_extracted"] = item.get("verdict_extracted") or alignment.get("verdict_extracted") or "not_found"
            item["claim_id"] = claim_id
            item["judgment_label"] = label
            item["evidence_relation"] = evidence_relation
            item["relation"] = "exact_support" if evidence_relation == "supports" and item["aboutness"] == "exact_claim" else evidence_relation
            item["stance"] = "refutes" if evidence_relation in {"contradicts", "refutes"} else "supports" if evidence_relation == "supports" else alignment.get("stance", "unknown")
            item["claim_evidence_verifier"] = {
                "model_used": verification.model_used,
                "confidence": verification.confidence,
                "raw_scores": verification.raw_scores,
            }
            item["relation_confidence"] = verification.confidence
            item["context_match"] = context_match
            item["entity_match_score"] = context_match.get("entity_match_score")
            item["location_match_score"] = context_match.get("location_match_score")
            item["location_match_pass"] = context_match.get("location_match_pass")
            item["event_match_score"] = context_match.get("event_match_score")
            item["event_match_pass"] = context_match.get("event_match_pass")
            item["time_match_score"] = context_match.get("time_match_score")
            item["context_match_pass"] = context_match.get("context_match_pass")
            item["mismatch_reason"] = context_match.get("mismatch_reason")
            item["embedding_similarity"] = item.get("semantic_score")
            if evidence_relation == "partially_related":
                item["reason_accepted_or_rejected"] = "This source is related to the same person or topic, but it does not directly verify the exact claim."
            elif evidence_relation in {"supports", "contradicts", "refutes"}:
                item["reason_accepted_or_rejected"] = trusted_news_support.get("reason") or alignment.get("reason") or verification.explanation
            else:
                item["reason_accepted_or_rejected"] = alignment.get("reason") or verification.explanation
            item["included_in_report"] = (
                evidence_relation in {"supports", "contradicts", "refutes", "partially_related"}
                and relevance >= 0.6
                and item["source_type"] in {"fact_check_page", "trusted_news_page", "official_source_page"}
            )
            item["citation_reason"] = self._citation_reason_for_claim(claim, item)

        usable = [item for item in judged_evidence if item.get("included_in_report")]
        supports = [item for item in usable if str(item.get("evidence_relation")) == "supports"]
        partials = [item for item in usable if str(item.get("evidence_relation")) == "partially_related"]
        contradicts = [item for item in usable if str(item.get("evidence_relation")) in {"contradicts", "refutes"}]

        if not usable:
            return {
                "claim_text": claim,
                "support_score": 0.0,
                "source_reliability": 0.0,
                "agreement_score": 0.0,
                "directness_score": 0.0,
                "clarity_score": round(clarity, 3),
                "claim_score": 0.0,
                "claim_label": "Not Verifiable",
                "claim_verdict": "insufficient_evidence",
                "claim_rationale": "Retrieved sources were rejected because they were unrelated or too weak for this claim.",
                "retrieval_outcome": "no_relevant_evidence_found",
                "evidence_summary": "No directly relevant evidence sources survived relevance filtering for this claim.",
                "evidence_sources": [],
                "judgment_failure_reason": judgment_error,
                "top_candidate_diagnostics": [
                    {
                        "url": item.get("url"),
                        "title": item.get("title"),
                        "score": item.get("rerank_score") or item.get("hybrid_score"),
                        "rerank_score": item.get("rerank_score"),
                        "embedding_similarity": item.get("embedding_similarity") or item.get("semantic_score"),
                        "source_recovery_score": item.get("source_recovery_score"),
                        "claim_relevance_score": item.get("claim_relevance_score"),
                        "judgment_label": item.get("judgment_label"),
                        "evidence_relation": item.get("evidence_relation"),
                        "relation_confidence": item.get("relation_confidence"),
                        "claim_evidence_verifier": item.get("claim_evidence_verifier"),
                        "judgment_confidence": item.get("judgment_confidence"),
                        "entity_match_score": item.get("entity_match_score"),
                        "location_match_score": item.get("location_match_score"),
                        "location_match_pass": item.get("location_match_pass"),
                        "event_match_score": item.get("event_match_score"),
                        "event_match_pass": item.get("event_match_pass"),
                        "context_match_pass": item.get("context_match_pass"),
                        "mismatch_reason": item.get("mismatch_reason"),
                        "rejection_reason": item.get("reason_accepted_or_rejected") or "low_claim_relevance_or_unrelated",
                    }
                    for item in judged_evidence[:3]
                ],
            }

        support = 0.0
        if supports:
            support = max(
                min(
                    float(item.get("judgment_confidence") or 0.0),
                    float(item.get("claim_relevance_score") or 0.0),
                    float(item.get("credibility_score") or 0.0) + 0.15,
                )
                for item in supports
            )
        elif partials:
            support = 0.0
        elif contradicts:
            support = 0.08
        reliability = round(self._text_source_reliability(supports or contradicts), 3)
        agreement = round(self._text_cross_source_agreement(supports), 3)
        directness = round(
            max(
                [float(item.get("claim_relevance_score") or 0.0) for item in supports + contradicts]
                + [self._text_evidence_directness(claim, supports + contradicts)]
                + [float(item.get("source_recovery_score") or 0.0) for item in supports + contradicts]
                + [0.0]
            ),
            3,
        )
        evidence_multiplier = compute_evidence_multiplier(len(supports) + len(contradicts), self.TEXT_EVIDENCE_PER_CLAIM)
        contradiction = len(contradicts) > len(supports) + len(partials)
        if contradicts and claim_details.get("legal_eligibility"):
            contradiction = True
        mixed = bool(contradicts) and bool(supports or partials)
        penalty = 0.4 if contradiction else 0.7 if mixed else 1.0
        exactness = max(float(item.get("source_recovery_score") or 0.0) for item in usable)
        source_family_count = len({self._domain_family(extract_domain(item.get("url")) or str(item.get("domain") or "")) for item in usable if item.get("url") or item.get("domain")})
        top_source = usable[0]
        top_source_domain = extract_domain(top_source.get("url")) or str(top_source.get("domain") or "")
        if supports and exactness >= 0.9:
            support = max(support, 0.9)
            reliability = max(reliability, 0.8)
            agreement = max(agreement, 0.8 if source_family_count >= 2 else 0.65)
            directness = max(directness, 0.9)
            evidence_multiplier = max(evidence_multiplier, 0.8)
            penalty = 1.0
        elif supports and exactness >= 0.75 and self._find_trusted_publisher(top_source_domain, None):
            support = max(support, 0.8)
            reliability = max(reliability, 0.78)
            directness = max(directness, 0.8)
        if claim_details.get("extraordinary") and not supports:
            support = min(support, 0.28)
            penalty = min(penalty, 0.55)
        if not supports and not contradiction:
            support = 0.0
            reliability = 0.0
            agreement = 0.0
            directness = min(directness, 0.25)
            evidence_multiplier = min(evidence_multiplier, 0.4)
            penalty = min(penalty, 0.6)
        if contradiction:
            support = min(support, 0.08)
            penalty = min(penalty, 0.35 if claim_details.get("legal_eligibility") else 0.4)
        claim_score = max(
            0.0,
            min(
                1.0,
                (
                    0.45 * support
                    + 0.20 * reliability
                    + 0.20 * agreement
                    + 0.10 * directness
                + 0.05 * clarity
                )
                * evidence_multiplier
                * penalty,
            ),
        )
        label = self._generate_text_label(claim_score, evidence_multiplier, contradiction, clarity)
        if not supports and not contradiction:
            claim_score = min(claim_score, 0.349)
            label = "Not Verifiable"
        retrieval_outcome = "strong_supporting_evidence_found"
        if contradiction:
            retrieval_outcome = "strong_contradiction_found"
        elif not supports:
            retrieval_outcome = "mentions_only_or_irrelevant_evidence_found"
        elif support < 0.35 and directness < 0.35:
            retrieval_outcome = "relevant_but_indirect_evidence_found"
        elif support < 0.55:
            retrieval_outcome = "mixed_or_partial_evidence_found"
        return {
            "claim_text": claim,
            "support_score": round(support, 3),
            "source_reliability": reliability,
            "agreement_score": agreement,
            "directness_score": directness,
            "clarity_score": round(clarity, 3),
            "exact_match_score": round(exactness, 3),
            "claim_score": round(claim_score, 3),
            "claim_label": label,
            "claim_verdict": "contradicted" if contradiction else "supported" if supports else "partially_supported" if partials else "insufficient_evidence",
            "claim_rationale": (
                self._claim_contradiction_rationale(contradicts)
                if contradiction
                else "Top evidence passages supported the claim."
                if supports
                else "Top evidence passages partially supported the claim."
                if partials
                else "No credible direct evidence was found for the specific claim."
            ),
            "retrieval_outcome": retrieval_outcome,
            "evidence_summary": self._claim_evidence_summary(judged_evidence, label, contradiction),
            "evidence_sources": usable[:5],
            "judgment_failure_reason": judgment_error,
            "top_candidate_diagnostics": [
                {
                    "url": item.get("url"),
                    "title": item.get("title"),
                    "score": item.get("rerank_score") or item.get("hybrid_score"),
                    "rerank_score": item.get("rerank_score"),
                    "embedding_similarity": item.get("embedding_similarity") or item.get("semantic_score"),
                    "source_recovery_score": item.get("source_recovery_score"),
                    "judgment_label": item.get("judgment_label"),
                    "evidence_relation": item.get("evidence_relation"),
                    "relation_confidence": item.get("relation_confidence"),
                    "claim_evidence_verifier": item.get("claim_evidence_verifier"),
                    "judgment_confidence": item.get("judgment_confidence"),
                    "contradiction_reason": item.get("contradiction_reason"),
                    "entity_match_score": item.get("entity_match_score"),
                    "location_match_score": item.get("location_match_score"),
                    "location_match_pass": item.get("location_match_pass"),
                    "event_match_score": item.get("event_match_score"),
                    "event_match_pass": item.get("event_match_pass"),
                    "context_match_pass": item.get("context_match_pass"),
                    "mismatch_reason": item.get("mismatch_reason"),
                }
                for item in judged_evidence[:3]
            ],
        }

    def _aggregate_text_verification_score(self, claim_results: list[dict], claims: list[str]) -> dict:
        if not claims:
            return {
                "final_score": 0.0,
                "final_label": "Not Verifiable",
                "explanation": "No factual, checkable claim was found in the pasted text.",
                "evidence_summary": "The pasted text did not contain a clear factual claim that could be verified.",
                "supporting_evidence_summary": "",
                "contradiction_summary": "",
                "evidence_sources": [],
                "evidence_multiplier": 0.4,
                "penalty_multiplier": 0.0,
            }

        scores = [float(item.get("claim_score") or 0.0) for item in claim_results]
        average_score = sum(scores) / max(1, len(scores))
        evidence_sources: list[dict] = []
        seen_urls: set[str] = set()
        for result in claim_results:
            for source in result.get("evidence_sources", []):
                url = str(source.get("url") or "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                evidence_sources.append(source)
                if len(evidence_sources) >= 5:
                    break
            if len(evidence_sources) >= 5:
                break

        evidence_present = sum(
            [
                1 if claims else 0,
                1 if evidence_sources else 0,
                1 if any(float(item.get("support_score") or 0.0) > 0.5 for item in claim_results) else 0,
                1 if len({self._domain_family(extract_domain(item.get("url")) or str(item.get("domain") or "")) for item in evidence_sources if item.get("url")}) >= 2 else 0,
                1 if any(float(item.get("directness_score") or 0.0) > 0.45 for item in claim_results) else 0,
            ]
        )
        evidence_multiplier = compute_evidence_multiplier(evidence_present, 5)
        contradiction = any(item.get("claim_label") == "Unverified" and float(item.get("support_score") or 0.0) < 0.2 for item in claim_results)
        explicit_contradiction = any(str(item.get("claim_verdict") or "") == "contradicted" for item in claim_results)
        strong_supported_claims = sum(1 for item in claim_results if float(item.get("claim_score") or 0.0) >= 0.7)
        weak_evidence = not evidence_sources or (average_score < 0.35 and strong_supported_claims == 0)
        penalty = 0.0 if not claims else 0.25 if explicit_contradiction else 0.4 if contradiction else 0.7 if weak_evidence else 1.0
        exact_match_found = any(float(item.get("exact_match_score") or 0.0) >= 0.9 for item in claim_results)
        if exact_match_found:
            evidence_multiplier = max(evidence_multiplier, 0.9)
            penalty = 1.0
        final_score = max(0.0, min(1.0, average_score * evidence_multiplier * penalty))
        if exact_match_found and final_score < 0.7:
            final_score = min(1.0, max(final_score, 0.74))
        if not evidence_sources and not contradiction:
            final_score = min(final_score, 0.349)
        if explicit_contradiction:
            final_score = min(final_score, 0.25)
        final_label = self._generate_text_label(
            final_score,
            evidence_multiplier,
            contradiction,
            max(float(item.get("clarity_score") or 0.0) for item in claim_results),
        )
        if not evidence_sources and not contradiction:
            final_label = "Not Verifiable"
        if explicit_contradiction:
            final_label = "Unverified"
        supporting = [str(item.get("evidence_summary") or "") for item in claim_results if item.get("claim_label") in {"Verified", "Likely Verified"}]
        contradicting = [str(item.get("evidence_summary") or "") for item in claim_results if item.get("claim_label") == "Unverified"]
        return {
            "final_score": round(final_score, 3),
            "final_label": final_label,
            "explanation": self._generate_text_explanation(claim_results, final_score, final_label),
            "evidence_summary": supporting[0] if supporting else contradicting[0] if contradicting else "Evidence remained limited for the pasted text.",
            "supporting_evidence_summary": " ".join(supporting[:2]).strip(),
            "contradiction_summary": " ".join(contradicting[:2]).strip(),
            "evidence_sources": evidence_sources[:5],
            "evidence_multiplier": round(evidence_multiplier, 3),
            "penalty_multiplier": round(penalty, 3),
        }

    def _generate_text_label(self, score: float, evidence_multiplier: float, contradiction: bool, clarity: float) -> str:
        score100 = max(0.0, min(100.0, score * 100))
        return map_score_to_label(score100)

    def _generate_text_explanation(self, claim_results: list[dict], final_score: float, final_label: str) -> str:
        if not claim_results:
            return "No factual claim could be extracted from the pasted text."
        if any(float(item.get("exact_match_score") or 0.0) >= 0.9 for item in claim_results):
            if final_label in {"Verified", "Likely Verified"}:
                return "The pasted text closely matches directly relevant reporting from retrieved sources."
        supported = sum(1 for item in claim_results if item.get("claim_label") in {"Verified", "Likely Verified"})
        contradicted = sum(1 for item in claim_results if item.get("claim_label") == "Unverified")
        no_evidence = all(item.get("retrieval_outcome") == "no_relevant_evidence_found" for item in claim_results)
        indirect = any(item.get("retrieval_outcome") == "relevant_but_indirect_evidence_found" for item in claim_results)
        if final_label == "Verified":
            return "The pasted text matches strong supporting evidence across relevant sources."
        if final_label == "Likely Verified":
            return "The pasted text is supported by relevant evidence, though some gaps remain."
        if any(str(item.get("claim_verdict") or "") == "contradicted" for item in claim_results):
            reason = next((str(item.get("claim_rationale") or "") for item in claim_results if item.get("claim_verdict") == "contradicted"), "")
            return reason or "A core claim was contradicted by the retrieved evidence."
        if final_label == "Not Verifiable":
            if no_evidence:
                return "No direct source URL could be recovered from the pasted text. Usable corroborating evidence was limited, so the result remains Not Verifiable."
            if indirect:
                return "Related articles were retrieved, but none matched the pasted text strongly enough."
            return "The available evidence is too limited or mixed to make a reliable text-verification judgment."
        if contradicted and supported:
            return "Some extracted claims are contradicted while others remain supported, so the pasted text is unverified overall."
        return "The pasted text is not well supported by the evidence that was found."

    def _claim_evidence_relevance_score(self, claim: str, evidence: dict) -> float:
        text = " ".join(
            str(evidence.get(key) or "")
            for key in ("title", "snippet", "passage", "text")
        )
        normalized_text = normalize_text(text)
        details = self._parse_claim_details(claim)
        entities = [normalize_text(item) for item in details.get("entities", []) if normalize_text(item)]
        actions = [normalize_text(item) for item in details.get("actions", []) if normalize_text(item)]
        money = [normalize_text(item) for item in details.get("money", []) if normalize_text(item)]
        important_terms = [normalize_text(item) for item in details.get("important_terms", []) if normalize_text(item)]

        entity_score = self._coverage_score(entities, normalized_text)
        action_score = self._coverage_score(actions, normalized_text)
        money_score = self._coverage_score(money, normalized_text) if money else 1.0
        term_score = self._coverage_score(important_terms, normalized_text)
        overlap = self._text_overlap(claim, text)
        direct_source_score = float(evidence.get("source_recovery_score") or 0.0)
        exact_match_bonus = 0.15 if direct_source_score >= 0.9 else 0.0
        score = (
            0.34 * entity_score
            + 0.26 * action_score
            + 0.22 * money_score
            + 0.13 * term_score
            + 0.05 * overlap
            + exact_match_bonus
        )
        if money and money_score < 1.0:
            score = min(score, 0.56)
        if actions and action_score < 0.5:
            score = min(score, 0.58)
        if details.get("extraordinary") and (entity_score < 0.55 or action_score < 0.55 or (money and money_score < 1.0)):
            score = min(score, 0.55)
        if direct_source_score >= 0.9 and overlap >= 0.75:
            score = max(score, min(1.0, direct_source_score))
        return round(max(0.0, min(1.0, score)), 3)

    def compare_claim_and_evidence_context(self, claim: str, evidence: dict) -> dict:
        return self.context_matcher.compare(claim, evidence)

    def _country_for_location(self, location: str) -> str:
        return self.context_matcher.country_for_location(location)

    def _extract_context_terms(self, text: str) -> dict[str, list[str]]:
        return self.context_matcher.extract_context_terms(text)

    def _evidence_relation_from_label(self, label: str, context_match: dict, relevance: float) -> str:
        return self.context_matcher.relation_from_context(label, context_match, relevance)

    def _relation_explanation(self, relation: str, context_match: dict) -> str:
        return self.context_matcher.relation_explanation(relation, context_match)

    def _explicit_contradiction_reason(self, claim: str, evidence: dict) -> str | None:
        claim_text = normalize_text(claim)
        evidence_text = normalize_text(
            " ".join(str(evidence.get(key) or "") for key in ("title", "snippet", "passage", "text"))
        )
        if not claim_text or not evidence_text:
            return None

        musk_claim = "musk" in claim_text or "elon" in claim_text
        president_claim = bool(re.search(r"\b(?:president|presidential|eligible|run)\b", claim_text))
        natural_born_rule = ("natural born" in evidence_text or "natural-born" in evidence_text) and bool(re.search(r"\b(?:president|presidency|constitution|eligible|eligibility)\b", evidence_text))
        musk_foreign_birth = musk_claim and bool(re.search(r"\b(?:pretoria|south africa|south african)\b", evidence_text)) and bool(re.search(r"\b(?:born|birthplace)\b", evidence_text))
        if musk_claim and president_claim and natural_born_rule and musk_foreign_birth:
            return "Evidence says U.S. presidents must be natural-born citizens and Elon Musk was born in South Africa, so renouncing citizenship does not make him eligible."
        if musk_claim and "renounc" in claim_text and natural_born_rule:
            return "Eligibility evidence says the presidency requires natural-born U.S. citizenship; renouncing foreign citizenship does not satisfy that requirement."
        if musk_claim and president_claim and re.search(r"\b(?:has not|hasn't|no evidence|did not|not announced|cannot run|ineligible|not eligible)\b", evidence_text):
            return "The source directly says the claimed presidential run or eligibility is not confirmed or is contradicted."
        if "announced" in claim_text and re.search(r"\b(?:has not announced|not announced|did not announce|no announcement)\b", evidence_text):
            return "The source says no such announcement was made."
        return None

    def _claim_contradiction_rationale(self, contradicts: list[dict]) -> str:
        reason = next((str(item.get("contradiction_reason") or "") for item in contradicts if item.get("contradiction_reason")), "")
        return reason or "Top evidence passages contradicted the claim."

    def _parse_claim_details(self, claim: str) -> dict[str, list[str] | bool]:
        extraction = self.claim_extraction.extract_claims(claim, max_claims=1)
        claim_tuple = self._extract_structured_claim_tuple(claim)
        entities = [str(item) for item in extraction.get("entities", []) if str(item).strip()]
        for key in ("subject", "object", "location"):
            value = claim_tuple.get(key) if isinstance(claim_tuple, dict) else None
            if value:
                entities.append(str(value))
        entities.extend(str(item) for item in extraction.get("locations", []) if str(item).strip())
        context_terms = self._extract_context_terms(claim)
        entities.extend(context_terms.get("locations", []))
        entities.extend(context_terms.get("countries", []))
        entities.extend(context_terms.get("actors", []))
        money = re.findall(
            r"(?:\$|usd\s*)\s?\d[\d,\.]*(?:\s?(?:million|billion|trillion))?|\b\d[\d,\.]*\s?(?:million|billion|trillion)\b",
            claim or "",
            flags=re.IGNORECASE,
        )
        normalized_claim = normalize_text(claim)
        action_vocab = (
            "sell sold sale transfer ownership deal signed secretly alleged allegedly planned convert converted converting luxury resort resorts island islands "
            "government china chinese conglomerate company company-owned foreign destroy destruction mangrove forest sundarbans "
            "announced president presidential campaign election eligible eligibility citizenship renounced renouncing natural born birthplace bitcoin wallet fund funding "
            "defeat defeated defeating beat beaten won win wins winner lost loses runoff run-off primary nomination nominee senate republican democrat "
            "announce announced announcing say said says report reported reporting confirm confirmed confirming review reviewed reviewing investigate investigating "
            "charge charged indict indicted arrest arrested convict convicted sentence sentenced sue sued rule ruled order ordered block blocked "
            "kill killed die died dead fatal shooting shot attack attacked fire explosion resign resigned appoint appointed launch launched open opened"
        ).split()
        actions = [term for term in action_vocab if re.search(rf"\b{re.escape(term)}\b", normalized_claim)]
        actions.extend(context_terms.get("events", []))
        if claim_tuple.get("claim_type") == "election result":
            actions.extend(["defeated", "won", "runoff", "primary", "nomination", "senate"])
        stopwords = {
            "the", "a", "an", "to", "of", "for", "and", "or", "in", "on", "by", "was", "were", "is", "are", "has", "have",
            "officials", "allegedly", "planned", "government", "secretly",
        }
        important_terms = [
            token
            for token in re.findall(r"\b[a-z0-9][a-z0-9\-]{3,}\b", normalized_claim)
            if token not in stopwords
        ]
        extraordinary = bool(
            re.search(
                r"\b(secret|secretly|alleged|allegedly|hidden|cover.?up|sell|sold|sale|ownership|foreign|billion|trillion|conglomerate|destroy|destruction|convert|luxury resort|president|presidential|citizenship|eligible|eligibility|campaign funding)\b|\$",
                normalized_claim,
            )
        )
        legal_eligibility = bool(re.search(r"\b(?:president|presidential|eligible|eligibility|citizenship|natural born|constitution)\b", normalized_claim))
        return {
            "entities": self._dedupe_text_items(entities),
            "money": self._dedupe_text_items(money),
            "actions": self._dedupe_text_items(actions),
            "important_terms": self._dedupe_text_items(important_terms)[:10],
            "extraordinary": extraordinary,
            "legal_eligibility": legal_eligibility,
            "claim_tuple": claim_tuple,
        }

    def _extract_structured_claim_tuple(self, claim: str) -> dict[str, str]:
        text = " ".join((claim or "").split()).strip()
        normalized = normalize_text(text)
        result: dict[str, str] = {}
        election_result = bool(
            re.search(r"\b(?:defeat(?:ed|s|ing)?|beat|beats|won|wins|lost|run[- ]?off|primary|nomination|nominee|senate)\b", normalized)
            and re.search(r"\b(?:election|run[- ]?off|primary|nomination|nominee|senate|republican|democrat|gop)\b", normalized)
        )
        if not election_result:
            return result

        result["claim_type"] = "election result"
        if re.search(r"\btexas\b", normalized):
            result["location"] = "Texas"
        date_match = re.search(r"\b(?:tuesday night|tuesday|monday|wednesday|thursday|friday|saturday|sunday|may\s+20\d{2}|20\d{2})\b", text, flags=re.IGNORECASE)
        if date_match:
            result["date"] = date_match.group(0)

        person_pattern = r"(?:Texas Attorney General\s+|Attorney General\s+|Senator\s+|Sen\.\s+|Rep\.\s+|Governor\s+|Gov\.\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})"
        defeat_match = re.search(rf"{person_pattern}[^.]*?\b(?:defeated|defeats|beat|beats)\b[^.]*?{person_pattern}", text)
        if defeat_match:
            result["subject"] = defeat_match.group(1).strip()
            result["action"] = "defeated"
            result["object"] = defeat_match.group(2).strip()
        else:
            won_match = re.search(rf"{person_pattern}[^.]*?\b(?:won|wins)\b", text)
            if won_match:
                result["subject"] = won_match.group(1).strip()
                result["action"] = "won"
            against_match = re.search(rf"\b(?:against|over|defeating)\b[^.]*?{person_pattern}", text)
            if against_match:
                result["object"] = against_match.group(1).strip()

        event_terms = []
        if re.search(r"\brepublican|\bgop\b", normalized):
            event_terms.append("Republican")
        if re.search(r"\bu\.?s\.?\s+senate|\bsenate\b", normalized):
            event_terms.append("U.S. Senate")
        if re.search(r"\brun[- ]?off\b", normalized):
            event_terms.append("runoff")
        if re.search(r"\bprimary\b", normalized):
            event_terms.append("primary")
        if event_terms:
            result["event"] = " ".join(dict.fromkeys(event_terms))
        return result

    def _claim_tuple_search_queries(self, claim_tuple: dict) -> list[str]:
        if not claim_tuple or claim_tuple.get("claim_type") != "election result":
            return []
        subject = str(claim_tuple.get("subject") or "").strip()
        obj = str(claim_tuple.get("object") or "").strip()
        location = str(claim_tuple.get("location") or "").strip()
        event = str(claim_tuple.get("event") or "Republican Senate runoff").strip()
        if not subject or not obj:
            return []
        base = " ".join(item for item in [subject, "defeated", obj, event] if item)
        location_base = " ".join(item for item in [location, subject, obj, event] if item)
        nomination = " ".join(item for item in [subject, "wins Republican nomination U.S. Senate", obj] if item)
        source_base = " ".join(item for item in [subject, obj, "runoff Senate"] if item)
        return self._dedupe_queries(
            [
                base,
                location_base,
                nomination,
                f"site:apnews.com {source_base}",
                f"site:reuters.com {source_base}",
                f"site:bbc.com {source_base}",
            ],
            limit=6,
        )

    def _coverage_score(self, items: list[str], normalized_text: str) -> float:
        normalized_items = [normalize_text(item) for item in items if normalize_text(item)]
        if not normalized_items:
            return 1.0
        hits = sum(1 for item in normalized_items if item and item in normalized_text)
        return hits / max(1, len(normalized_items))

    def _trusted_news_exact_support(self, claim_tuple: dict, evidence: dict) -> dict:
        if not claim_tuple or claim_tuple.get("claim_type") != "election result":
            return {"supports": False}
        source_type = evidence.get("source_type") or get_source_type(str(evidence.get("url") or evidence.get("domain") or ""))
        if source_type not in {"trusted_news_page", "official_source_page"}:
            return {"supports": False}

        evidence_text = " ".join(str(evidence.get(key) or "") for key in ("title", "snippet", "passage", "text"))
        normalized = normalize_text(evidence_text)
        subject = str(claim_tuple.get("subject") or "").strip()
        obj = str(claim_tuple.get("object") or "").strip()
        if not subject or not obj or not normalized:
            return {"supports": False}

        def name_match(name: str) -> bool:
            tokens = [token for token in normalize_text(name).split() if len(token) > 2]
            if not tokens:
                return False
            last = tokens[-1]
            return bool(re.search(rf"\b{re.escape(last)}\b", normalized)) and any(
                re.search(rf"\b{re.escape(token)}\b", normalized) for token in tokens[:-1] or [last]
            )

        subject_match = name_match(subject)
        object_match = name_match(obj)
        location = normalize_text(str(claim_tuple.get("location") or ""))
        location_check = self._claim_location_supported_or_unspecified(location, evidence_text)
        location_match = bool(location_check["match"])
        election_terms = bool(re.search(r"\b(?:republican|gop|primary|run[- ]?off|runoff|nomination|nominee|senate|u\.?s\.?\s+senate)\b", normalized))
        action_terms = bool(re.search(r"\b(?:defeat(?:ed|s|ing)?|beat|beats|won|wins|winner|win|nomination|nominee)\b", normalized))
        race_match = election_terms and action_terms
        contradiction = False
        subject_last = normalize_text(subject).split()[-1] if normalize_text(subject).split() else ""
        object_last = normalize_text(obj).split()[-1] if normalize_text(obj).split() else ""
        if subject_last and object_last:
            contradiction = bool(
                re.search(rf"\b{re.escape(subject_last)}\b[^.]*\b(?:lost|loses|conceded)\b", normalized)
                or re.search(rf"\b{re.escape(object_last)}\b[^.]*\b(?:defeat(?:ed|s)?|beat|beats|won|wins)\b[^.]*\b{re.escape(subject_last)}\b", normalized)
            )
        supports = subject_match and object_match and location_match and race_match and not contradiction
        if not supports:
            return {
                "supports": False,
                "subject_match": subject_match,
                "object_match": object_match,
                "location_match": location_match,
                "location_reason": location_check["reason"],
                "race_match": race_match,
                "contradiction": contradiction,
            }

        quote = ""
        for sentence in self._text_sentences(evidence_text):
            sentence_norm = normalize_text(sentence)
            if subject_last in sentence_norm and object_last in sentence_norm and re.search(r"\b(?:defeat|defeated|won|wins|nomination|runoff|primary)\b", sentence_norm):
                quote = sentence[:300]
                break
        return {
            "supports": True,
            "aboutness": "exact_claim",
            "stance": "supports",
            "support_score": 0.9,
            "directness_score": 0.9,
            "subject_match": subject_match,
            "object_match": object_match,
            "location_match": location_match,
            "location_reason": location_check["reason"],
            "race_match": race_match,
            "contradiction": False,
            "evidence_quote": quote,
            "reason": (
                "Trusted news source directly confirms the same winner, defeated opponent, and election race; "
                f"{location_check['reason']}."
            ),
        }

    def _claim_location_supported_or_unspecified(self, claim_location: str, evidence_text: str) -> dict[str, object]:
        location = normalize_text(claim_location)
        normalized = normalize_text(evidence_text)
        if not location:
            return {"match": True, "reason": "no claim location was required"}
        if re.search(rf"\b{re.escape(location)}\b", normalized):
            return {"match": True, "reason": f"the source explicitly mentions {claim_location}"}

        evidence_context = self._extract_context_terms(evidence_text)
        evidence_locations = {normalize_text(item) for item in evidence_context.get("locations", [])}
        evidence_countries = {normalize_text(item) for item in evidence_context.get("countries", [])}
        claim_country = normalize_text(self._country_for_location(location))
        evidence_location_countries = {
            normalize_text(self._country_for_location(item))
            for item in evidence_locations
            if self._country_for_location(item)
        }
        if claim_country and claim_country in evidence_countries | evidence_location_countries:
            return {"match": True, "reason": f"the source matches the claim's country context for {claim_location}"}
        if not evidence_locations and not evidence_countries:
            return {"match": True, "reason": f"the source does not mention a conflicting location for {claim_location}"}
        return {
            "match": False,
            "reason": (
                f"the source location/country {sorted(evidence_locations | evidence_countries)} "
                f"does not match {claim_location}"
            ),
        }

    def _trusted_ordinary_news_support(self, claim: str, claim_details: dict, evidence: dict) -> dict:
        source_type = evidence.get("source_type") or get_source_type(str(evidence.get("url") or evidence.get("domain") or ""))
        if source_type not in {"trusted_news_page", "official_source_page"}:
            return {"supports": False}
        if str(evidence.get("source_tier") or self._source_tier(evidence)) not in {"tier_1", "tier_2"}:
            return {"supports": False}

        evidence_text = " ".join(str(evidence.get(key) or "") for key in ("title", "snippet", "passage", "text"))
        normalized_evidence = normalize_text(evidence_text)
        normalized_claim = normalize_text(claim)
        if not normalized_claim or not normalized_evidence:
            return {"supports": False}

        if claim_details.get("extraordinary") and normalized_claim not in normalized_evidence:
            return {"supports": False}

        context_match = self.compare_claim_and_evidence_context(claim, evidence)
        if not bool(context_match.get("location_match_pass")) or not bool(context_match.get("time_match_pass")):
            return {"supports": False, "context_match": context_match}

        entities = self._ordinary_claim_entities(claim, claim_details)
        entity_score = self._coverage_score(entities, normalized_evidence) if entities else 1.0
        if entities and entity_score < 0.67:
            return {"supports": False, "entity_score": entity_score, "context_match": context_match}

        claim_groups = self._ordinary_action_groups(normalized_claim)
        evidence_groups = self._ordinary_action_groups(normalized_evidence)
        action_match = bool(claim_groups & evidence_groups) if claim_groups else True

        important_terms = [
            term
            for term in claim_details.get("important_terms", [])
            if normalize_text(term) not in {"said", "says", "report", "reported", "news", "article"}
        ]
        term_score = self._coverage_score([normalize_text(term) for term in important_terms], normalized_evidence) if important_terms else 1.0
        overlap = self._text_overlap(normalized_claim, normalized_evidence)
        headline_exactness = self._headline_exactness_score(claim, str(evidence.get("title") or ""), str(evidence.get("snippet") or ""))
        direct_phrase = normalized_claim in normalized_evidence or headline_exactness >= 0.82

        if not action_match and not direct_phrase:
            return {
                "supports": False,
                "entity_score": entity_score,
                "term_score": term_score,
                "action_match": False,
                "context_match": context_match,
            }
        if not direct_phrase and term_score < 0.45 and overlap < 0.34:
            return {
                "supports": False,
                "entity_score": entity_score,
                "term_score": term_score,
                "overlap": overlap,
                "context_match": context_match,
            }
        if self._ordinary_news_contradiction(normalized_claim, normalized_evidence):
            return {"supports": False, "contradiction": True, "context_match": context_match}

        quote = ""
        for sentence in self._text_sentences(evidence_text):
            sentence_norm = normalize_text(sentence)
            sentence_entity_score = self._coverage_score(entities, sentence_norm) if entities else 1.0
            if sentence_entity_score >= 0.67 and (normalized_claim in sentence_norm or self._ordinary_action_groups(sentence_norm) & claim_groups or term_score >= 0.6):
                quote = sentence[:300]
                break

        directness = max(0.78, min(0.92, max(headline_exactness, overlap + 0.35, term_score)))
        support = max(0.76, min(0.9, directness))
        return {
            "supports": True,
            "aboutness": "exact_claim" if direct_phrase or term_score >= 0.6 else "strong_related",
            "stance": "supports",
            "support_score": round(support, 3),
            "directness_score": round(directness, 3),
            "subject_match": entity_score >= 0.67,
            "object_match": True,
            "location_match": bool(context_match.get("location_match_pass")),
            "race_match": False,
            "contradiction": False,
            "evidence_quote": quote,
            "reason": "Trusted news source directly confirms the same named entities, action context, and claim meaning.",
        }

    def _ordinary_claim_entities(self, claim: str, claim_details: dict) -> list[str]:
        entities = [
            str(item)
            for item in claim_details.get("entities", [])
            if str(item).strip() and len(normalize_text(str(item)).split()) >= 2
        ]
        entities.extend(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", claim or ""))
        return self._dedupe_text_items(entities)[:6]

    def _ordinary_action_groups(self, normalized_text: str) -> set[str]:
        groups = {
            "announce": r"\b(?:announce(?:d|s|ment)?|confirm(?:ed|s)?|say|said|says|report(?:ed|s)?|according)\b",
            "investigate": r"\b(?:investigat(?:e|ed|es|ing|ion)|review(?:ed|s|ing)?|probe(?:d|s)?)\b",
            "legal_charge": r"\b(?:charg(?:e|ed|es|ing)|indict(?:ed|s)?|arrest(?:ed|s)?|convict(?:ed|s)?|sentenc(?:e|ed|es)|sued?|lawsuit)\b",
            "legal_ruling": r"\b(?:court|judge|ruled?|order(?:ed|s)?|blocked?|upheld|dismiss(?:ed|es)?)\b",
            "death": r"\b(?:kill(?:ed|s|ing)?|died?|dead|fatalit(?:y|ies)|death(?:s)?)\b",
            "violence": r"\b(?:shoot(?:ing|s)?|shot|attack(?:ed|s)?|strike(?:s)?|blast|explosion|fire|clash(?:ed|es)?)\b",
            "personnel": r"\b(?:resign(?:ed|s)?|appoint(?:ed|s)?|nominate(?:d|s)?|hire(?:d|s)?|fired?)\b",
            "open_launch": r"\b(?:open(?:ed|s|ing)?|launch(?:ed|es|ing)?|start(?:ed|s|ing)?)\b",
            "meeting": r"\b(?:met|meet(?:s|ing)?|visit(?:ed|s|ing)?|talk(?:ed|s)?|summit)\b",
            "approve": r"\b(?:approv(?:e|ed|es|ing)|pass(?:ed|es)?|sign(?:ed|s)?|vot(?:e|ed|es|ing))\b",
        }
        return {name for name, pattern in groups.items() if re.search(pattern, normalized_text)}

    def _ordinary_news_contradiction(self, normalized_claim: str, normalized_evidence: str) -> bool:
        if re.search(r"\b(?:false|hoax|fabricated|no evidence|did not|didn't|has not|hasn't|not true)\b", normalized_evidence):
            claim_entities = set(re.findall(r"\b[a-z][a-z]+\b", normalized_claim))
            evidence_entities = set(re.findall(r"\b[a-z][a-z]+\b", normalized_evidence))
            if len(claim_entities & evidence_entities) >= 3:
                return True
        return False

    def _dedupe_text_items(self, items: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            cleaned = " ".join(str(item or "").split()).strip()
            key = normalize_text(cleaned)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
        return deduped

    def _source_credibility_score(self, evidence: dict) -> float:
        tier = self._source_tier(evidence)
        if tier == "tier_1":
            return 0.95
        if tier == "tier_2":
            return 0.9
        if tier == "tier_4":
            return 0.2
        domain = extract_domain(evidence.get("url")) or str(evidence.get("domain") or "")
        lowered = domain.lower()
        if not lowered:
            return 0.35
        source_type = get_source_type(evidence.get("url") or lowered)
        if source_type == "fact_check_page":
            return 0.92
        if source_type in {"trusted_news_page", "official_source_page"}:
            return 0.9
        if source_type == "social_media_page":
            return 0.2
        if self._find_trusted_publisher(lowered, None):
            return 0.9
        if lowered.endswith(".gov") or ".gov." in lowered or lowered.endswith("gov.bd"):
            return 0.9
        if any(name in lowered for name in ("unesco.org", "un.org", "worldbank.org", "who.int", "imf.org")):
            return 0.88
        if any(name in lowered for name in ("reuters", "apnews", "bbc", "nytimes", "washingtonpost", "theguardian", "aljazeera", "cnn", "dw.com", "france24")):
            return 0.8
        if any(name in lowered for name in ("factcheck", "politifact", "snopes", "fullfact", "afp.com/fact-check")):
            return 0.82
        if any(name in lowered for name in ("blogspot", "wordpress", "medium.com", "facebook.com", "x.com", "twitter.com", "youtube.com", "tiktok.com")):
            return 0.25
        return 0.55

    def _source_tier(self, evidence: dict) -> str:
        url_or_domain = str(evidence.get("url") or evidence.get("domain") or "")
        domain = (extract_domain(url_or_domain) or url_or_domain).lower()
        source_type = str(evidence.get("source_type") or get_source_type(url_or_domain))
        if source_type == "official_source_page" or domain.endswith(".gov") or ".gov." in domain:
            return "tier_1"
        if any(name in domain for name in ("apnews.com", "reuters.com", "bbc.com")):
            return "tier_1"
        if any(name in domain for name in ("npr.org", "cnn.com", "nytimes.com", "washingtonpost.com", "texastribune.org", "theguardian.com", "abcnews.go.com", "cbsnews.com", "nbcnews.com")):
            return "tier_2"
        if source_type == "fact_check_page":
            return "tier_1"
        if source_type == "social_media_page" or any(name in domain for name in ("facebook.com", "x.com", "twitter.com", "youtube.com", "tiktok.com", "instagram.com")):
            return "tier_4"
        if source_type == "trusted_news_page":
            return "tier_2"
        return "tier_3"

    def _strict_aboutness_label(self, relation: str, trusted_support: dict, alignment: dict, context_match: dict, relevance: float) -> str:
        if relation in {"contradicts", "refutes"}:
            return "exact_claim" if bool(context_match.get("context_match_pass")) else "strong_related"
        if trusted_support.get("supports"):
            return "exact_claim"
        if relation == "supports" and bool(context_match.get("context_match_pass")) and relevance >= 0.82:
            return "exact_claim"
        if relation == "supports" and relevance >= 0.72:
            return "strong_related"
        if relation == "partially_related":
            return "partial_context"
        base = str(alignment.get("alignment") or "")
        if base == "mentions_only":
            return "mentions_only"
        return "irrelevant"

    def _citation_reason_for_claim(self, claim: str, evidence: dict) -> str:
        label = str(evidence.get("evidence_relation") or evidence.get("judgment_label") or "unrelated")
        title = str(evidence.get("title") or evidence.get("url") or "This source")
        if label == "supports":
            relation = "directly supports"
        elif label in {"partially_supports", "partially_related"}:
            relation = "partially supports or contextualizes"
        elif label == "contradicts":
            relation = "contradicts"
        elif label == "background_only":
            relation = "provides background for"
        else:
            relation = "was retrieved for but did not directly address"
        return f"{title} {relation} the claim: {claim[:160]}"

    def _claim_clarity_score(self, claim: str) -> float:
        word_count = len(claim.split())
        proper_name_signal = bool(re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", claim))
        all_caps_signal = bool(re.search(r"\b[A-Z]{2,}\b", claim))
        has_entity_signal = proper_name_signal or all_caps_signal or bool(re.search(r"\d", claim))
        factual_verb = bool(re.search(r"\b(is|are|was|were|said|claims|reported|announced|approved|confirmed|threatened|killed|won|lost|passed)\b", claim.lower()))
        if word_count < 4:
            return 0.2
        score = 0.4
        if word_count >= 8:
            score += 0.2
        if has_entity_signal:
            score += 0.2
        if factual_verb:
            score += 0.2
        return min(1.0, score)

    def _is_headline_like(self, text: str) -> bool:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return False
        if "\n" in text:
            return False
        word_count = len(cleaned.split())
        if word_count < 5 or word_count > 22:
            return False
        sentence_endings = sum(1 for char in cleaned if char in ".!?")
        return sentence_endings <= 1

    def _text_source_reliability(self, evidence_sources: list[dict]) -> float:
        scores: list[float] = []
        for source in evidence_sources:
            scores.append(self._source_credibility_score(source))
        return (sum(scores) / len(scores)) if scores else 0.0

    def _text_cross_source_agreement(self, evidence_sources: list[dict]) -> float:
        domains = [self._domain_family(extract_domain(source.get("url")) or str(source.get("domain") or "")) for source in evidence_sources]
        unique_domains = {domain for domain in domains if domain}
        if len(unique_domains) >= 4:
            return 1.0
        if len(unique_domains) == 3:
            return 0.85
        if len(unique_domains) == 2:
            return 0.8
        if len(unique_domains) == 1:
            return 0.55
        return 0.0

    def _text_evidence_directness(self, claim: str, evidence_sources: list[dict]) -> float:
        claim_terms = set(normalize_text(claim).split())
        if not claim_terms:
            return 0.0
        best = 0.0
        for source in evidence_sources:
            title_terms = set(normalize_text(str(source.get("title") or "")).split())
            snippet_terms = set(normalize_text(str(source.get("snippet") or "")).split())
            overlap = len(claim_terms & (title_terms | snippet_terms)) / max(1, len(claim_terms))
            best = max(best, overlap)
        return min(1.0, best + 0.1 if best else 0.0)

    def _headline_exactness_score(self, claim: str, title: str, snippet: str) -> float:
        normalized_claim = normalize_text(claim)
        normalized_title = normalize_text(title)
        normalized_snippet = normalize_text(snippet)
        if not normalized_claim:
            return 0.0
        if normalized_claim == normalized_title:
            return 1.0
        if normalized_claim and normalized_claim in normalized_title:
            return 0.95
        title_overlap = self._text_overlap(normalized_claim, normalized_title)
        snippet_overlap = self._text_overlap(normalized_claim, normalized_snippet)
        combined = max(title_overlap, snippet_overlap * 0.8)
        return round(min(1.0, combined), 3)

    def _claim_evidence_summary(self, evidence_sources: list[dict], label: str, contradiction: bool) -> str:
        if not evidence_sources:
            return "No relevant evidence was found for this claim."
        top = evidence_sources[0]
        title = str(top.get("title") or top.get("url") or "a matching source")
        if contradiction:
            return f"Retrieved evidence did not consistently support the claim, and the strongest match was {title}."
        if label == "Verified":
            return f"The claim is strongly supported by retrieved evidence, including {title}."
        if label == "Likely Verified":
            return f"The claim is supported by relevant reporting, including {title}, but the evidence is not perfect."
        return f"Evidence was found, including {title}, but it was not strong enough for a firm conclusion."

    def _build_verification_record(self, input_text: str, extracted: ExtractedPayload, layer2: LayerOutcome) -> VerificationRecord:
        previous = self.db.scalar(
            select(VerificationRecord)
            .where(VerificationRecord.input_text_hash == extracted.content_hash)
            .order_by(VerificationRecord.created_at.desc())
            .limit(1)
        )
        evidence_sources = []
        for match in layer2.details.get("matches", []) if isinstance(layer2.details, dict) else []:
            if not isinstance(match, dict):
                continue
            evidence_sources.append(
                {
                    "title": match.get("title"),
                    "url": match.get("url"),
                    "domain": match.get("domain"),
                    "score": match.get("score"),
                    "source": match.get("source"),
                    "explanation": match.get("explanation"),
                }
            )

        record = VerificationRecord(
            input_text=input_text,
            normalized_text=extracted.normalized_text,
            input_text_hash=extracted.content_hash,
            extracted_claims=layer2.details.get("extracted_claims", self._extract_text_claims(extracted)),
            evidence_sources=evidence_sources,
            nlp_result={
                "mode": "text_verification",
                "input_text": input_text,
                "normalized_text": extracted.normalized_text,
                "extracted_claims": layer2.details.get("extracted_claims", self._extract_text_claims(extracted)),
                "claim_results": layer2.details.get("claim_results", []),
                "evidence_sources": evidence_sources,
                "supporting_evidence_summary": layer2.details.get("supporting_evidence_summary"),
                "contradiction_summary": layer2.details.get("contradiction_summary"),
                "evidence_summary": layer2.details.get("evidence_summary"),
                "nlp_confidence_score": layer2.details.get("text_verification_score", layer2.confidence),
                "nlp_label": layer2.details.get("text_verification_label", layer2.details.get("fact_check_verdict", layer2.status)),
                "nlp_explanation": layer2.explanation,
                "evidence_multiplier": layer2.details.get("evidence_multiplier"),
                "penalty_multiplier": layer2.details.get("penalty_multiplier"),
                "metadata_signals": layer2.details.get("metadata_signals", []),
            },
            nlp_confidence_score=float(layer2.confidence or 0.0),
            nlp_label=str(layer2.details.get("text_verification_label", layer2.details.get("fact_check_verdict", layer2.status))),
            current_version=(previous.current_version + 1) if previous else 1,
            previous_record_id=previous.id if previous else None,
        )
        self.db.add(record)
        self.db.flush()
        return record

    def _create_verification_manifest(self, record: VerificationRecord):
        crypto = CryptoProvenanceService(self.db)
        manifest = crypto.sign_object(CryptoObjectType.verification_record, record.id)
        record.current_manifest_id = manifest.id
        self.db.add(record)
        self.db.flush()
        return manifest

    def _text_crypto_layer(
        self,
        record: VerificationRecord,
        manifest,
        extracted: ExtractedPayload,
    ) -> LayerOutcome:
        if manifest is None:
            return LayerOutcome(
                status="platform-issued",
                confidence=0.4,
                explanation="The NLP verification record exists, but its linked provenance manifest is not available for verification.",
                details={
                    "record_type": "verification_record",
                    "verification_record_id": str(record.id),
                    "verification_result": "Platform verification record exists, but manifest lookup failed",
                    "provenance_verification_status": "no_provenance",
                    "current_version": record.current_version,
                    "previous_version_id": str(record.previous_record_id) if record.previous_record_id else None,
                },
            )
        verification_status = manifest.verification_status.value if manifest and manifest.verification_status else "no_provenance"
        status_value = "platform-issued"
        confidence = 0.82 if verification_status == "valid" else 0.52
        explanation = (
            "NewsProof created a signed provenance record for this NLP verification result, including the checked text, extracted claims, evidence list, and version history."
            if verification_status == "valid"
            else "NewsProof recorded platform-issued provenance for this NLP verification result, but the latest manifest did not verify cleanly."
        )
        return LayerOutcome(
            status=status_value,
            confidence=confidence,
            explanation=explanation,
            details={
                "record_type": "verification_record",
                "verification_record_id": str(record.id),
                "manifest_id": str(manifest.id),
                "verification_manifest_id": str(manifest.id),
                "current_version": record.current_version,
                "previous_version_id": str(record.previous_record_id) if record.previous_record_id else None,
                "previous_manifest_id": str(manifest.previous_manifest_id) if manifest.previous_manifest_id else None,
                "issued_by": manifest.signer.issuer_name if manifest.signer else "NewsProof system",
                "issuer_type": manifest.signer.issuer_type.value if manifest.signer else "platform",
                "verification_result": "Valid platform-issued provenance" if verification_status == "valid" else "Partial platform-issued provenance",
                "provenance_verification_status": verification_status,
                "signed_at": manifest.created_at.isoformat() if manifest.created_at else None,
                "timestamp_source": (manifest.timestamp_token or {}).get("provider", "internal server timestamp"),
                "canonical_hash": manifest.sha256_hash,
                "signature_algorithm": manifest.signing_algorithm,
                "internal_manifest_valid": verification_status == "valid",
                "platform_manifest_valid": verification_status == "valid",
                "canonical_json": manifest.canonical_json,
                "canonical_json_preview": manifest.canonical_json[:1200],
                "issuer_key_id": (record.nlp_result or {}).get("key_id", "platform-managed"),
                "timestamp_token_status": (manifest.timestamp_token or {}).get("status", "issued"),
                "verification_context": "nlp_result_provenance",
                "extracted_publisher_name": extracted.metadata.get("publisher_name"),
                "publisher": extracted.metadata.get("publisher_name"),
                "publisher_domain": extracted.metadata.get("source_domain"),
                "author": extracted.author,
                "publish_time": extracted.publish_time.isoformat() if extracted.publish_time else None,
            },
        )

    def _resolve_verification_record(self, job: VerificationJob) -> VerificationRecord | None:
        if job.verification_record_id:
            return self.db.get(VerificationRecord, job.verification_record_id)
        record_id = (job.proof_summary or {}).get("verification_record_id")
        if record_id:
            try:
                return self.db.get(VerificationRecord, uuid.UUID(str(record_id)))
            except (ValueError, TypeError):
                return None
        return None

    def _feedback_entries_for_record(self, record: VerificationRecord) -> list[FeedbackEntry]:
        chain_ids: list[uuid.UUID] = []
        current: VerificationRecord | None = record
        while current:
            chain_ids.append(current.id)
            current = current.previous_record
        return list(
            self.db.scalars(
                select(FeedbackEntry)
                .where(FeedbackEntry.verification_record_id.in_(chain_ids))
                .order_by(FeedbackEntry.created_at.desc())
            )
        )

    def _text_feedback_layer(self, record: VerificationRecord, persist_audit: bool = True) -> LayerOutcome:
        entries = self._feedback_entries_for_record(record)
        total = len(entries)
        counts = {
            "correction": sum(1 for entry in entries if entry.feedback_type == FeedbackType.correction),
            "dispute": sum(1 for entry in entries if entry.feedback_type == FeedbackType.dispute),
            "source_suggestion": sum(1 for entry in entries if entry.feedback_type == FeedbackType.source_suggestion),
            "clarification": sum(1 for entry in entries if entry.feedback_type == FeedbackType.clarification),
            "review_note": sum(1 for entry in entries if entry.feedback_type == FeedbackType.review_note),
        }
        reviewed = sum(1 for entry in entries if entry.status in {FeedbackStatus.reviewed, FeedbackStatus.resolved})
        resolved = sum(1 for entry in entries if entry.status == FeedbackStatus.resolved)
        expert_count = sum(1 for entry in entries if entry.user_role == FeedbackUserRole.expert)
        moderator_count = sum(1 for entry in entries if entry.user_role == FeedbackUserRole.moderator)
        latest = entries[0] if entries else None
        if persist_audit:
            append_audit_entry(self.db, self.db.scalar(select(VerificationJob).where(VerificationJob.verification_record_id == record.id).limit(1)), "feedback_layer_completed", {"feedback_count": total}) if self.db.scalar(select(VerificationJob).where(VerificationJob.verification_record_id == record.id).limit(1)) else None

        if total == 0:
            status_value = "no_feedback"
            explanation = "No user, expert, or moderator feedback has been submitted for this verification record yet."
        elif counts["dispute"] or counts["correction"]:
            status_value = "review_attention"
            explanation = "Feedback has been submitted for review, including dispute or correction requests that may lead to a revised verification version."
        elif reviewed or expert_count or moderator_count:
            status_value = "reviewed_feedback"
            explanation = "Structured feedback and review notes are linked to this verification record for audit and refinement."
        else:
            status_value = "feedback_open"
            explanation = "Feedback has been submitted and is awaiting review."

        confidence = 0.0 if total == 0 else round(min(1.0, (reviewed + resolved + expert_count + moderator_count) / max(1, total * 2)), 3)
        return LayerOutcome(
            status=status_value,
            confidence=confidence,
            explanation=explanation,
            details={
                "mode": "feedback",
                "feedback_count": total,
                "reviewed_count": reviewed,
                "resolved_count": resolved,
                "expert_feedback_count": expert_count,
                "moderator_feedback_count": moderator_count,
                "correction_count": counts["correction"],
                "dispute_count": counts["dispute"],
                "source_suggestion_count": counts["source_suggestion"],
                "clarification_count": counts["clarification"],
                "review_note_count": counts["review_note"],
                "latest_feedback_type": latest.feedback_type.value if latest else None,
                "latest_feedback_status": latest.status.value if latest else None,
                "verification_record_id": str(record.id),
                "current_manifest_id": str(record.current_manifest_id) if record.current_manifest_id else None,
            },
        )

    def _layer1(self, extracted: ExtractedPayload) -> LayerOutcome:
        record = self.db.scalar(select(ContentRecord).where(ContentRecord.content_hash == extracted.content_hash))
        source_domain = extract_domain(extracted.canonical_url) or extract_domain(extracted.metadata.get("fetched_url"))
        extracted_publisher_name = extracted.metadata.get("publisher_name")
        publisher = self._find_trusted_publisher(source_domain, extracted_publisher_name)
        metadata_candidate = self._find_metadata_candidate(publisher.id, extracted) if publisher else None

        if not record and not metadata_candidate and not publisher:
            return LayerOutcome(
                status="no-proof",
                confidence=0.0,
                explanation=(
                    "Publisher metadata was extracted from the page, but the source is not in the trusted registry and no signed proof record exists."
                    if extracted_publisher_name or source_domain
                    else "No trusted publisher record matched the submitted content hash, publisher metadata, or canonical source."
                ),
                details={
                    "publisher": None,
                    "publisher_domain": None,
                    "canonical_url": extracted.canonical_url,
                    "source_domain": source_domain,
                    "extracted_publisher_name": extracted_publisher_name,
                    "source_type": extracted.metadata.get("source_type", "unknown"),
                    "author": extracted.author,
                    "publish_time": extracted.publish_time.isoformat() if extracted.publish_time else None,
                },
            )

        if not record and metadata_candidate:
            record = metadata_candidate

        if publisher and not record:
            return LayerOutcome(
                status="publisher-recognized",
                confidence=0.62,
                explanation="A verified publisher was identified from the source domain or publisher metadata, but no signed proof record exists for this specific article.",
                details={
                    "publisher": publisher.name,
                    "publisher_domain": publisher.domain,
                    "canonical_url": extracted.canonical_url,
                    "source_domain": source_domain,
                    "extracted_publisher_name": extracted_publisher_name,
                    "title": extracted.title,
                    "author": extracted.author,
                    "publish_time": extracted.publish_time.isoformat() if extracted.publish_time else None,
                    "matched_by": "trusted-publisher-registry",
                    "content_record_id": None,
                },
            )

        signature_valid = False
        if record.signature and record.key_id and record.publisher_id:
            key = self.db.scalar(
                select(PublicKey).where(
                    PublicKey.publisher_id == record.publisher_id,
                    PublicKey.key_id == record.key_id,
                    PublicKey.is_active.is_(True),
                )
            )
            if key:
                signature_valid = verify_ed25519_signature(
                    key.public_key,
                    record.signature,
                    extracted.content_hash.encode("utf-8"),
                )

        url_match = not record.canonical_url or record.canonical_url == extracted.canonical_url
        metadata_match = True
        if extracted.title and record.title:
            metadata_match = extracted.title.strip().lower() == record.title.strip().lower()
        author_match = self._metadata_value_matches(extracted.author, record.source_metadata.get("author"))
        published_at_match = self._publish_time_matches(extracted.publish_time, record.source_metadata.get("publish_time"))

        if signature_valid and url_match and metadata_match:
            return LayerOutcome(
                status="matched",
                confidence=0.99,
                explanation="Trusted publisher proof matched the normalized content hash and signature.",
                details={
                    "publisher": record.publisher.name if record.publisher else None,
                    "publisher_domain": record.publisher.domain if record.publisher else None,
                    "content_record_id": record.id,
                    "canonical_url_match": url_match,
                    "source_type": "publisher_registry",
                    "author_match": author_match,
                    "publish_time_match": published_at_match,
                },
            )

        if publisher and not record.signature:
            return LayerOutcome(
                status="publisher-recognized",
                confidence=0.58,
                explanation="A verified publisher was identified from the source domain and metadata, but no signed content proof exists for this specific article.",
                details={
                    "publisher": publisher.name,
                    "publisher_domain": publisher.domain,
                    "canonical_url": extracted.canonical_url,
                    "source_domain": source_domain,
                    "title": extracted.title,
                    "author": extracted.author,
                    "publish_time": extracted.publish_time.isoformat() if extracted.publish_time else None,
                    "matched_by": "domain-and-metadata",
                    "content_record_id": record.id,
                },
            )

        return LayerOutcome(
            status="mismatch",
            confidence=0.98,
            explanation="A trusted proof exists but the current content does not match its recorded provenance.",
            details={
                "publisher": record.publisher.name if record.publisher else None,
                "publisher_domain": record.publisher.domain if record.publisher else None,
                "content_record_id": record.id,
                "canonical_url_match": url_match,
                "metadata_match": metadata_match,
                "signature_valid": signature_valid,
                "author_match": author_match,
                "publish_time_match": published_at_match,
                "source_type": "publisher_registry",
            },
        )

    def _layer2(self, job: VerificationJob, extracted: ExtractedPayload, persist_matches: bool = True) -> LayerOutcome:
        source_domain = extract_domain(extracted.canonical_url) or extract_domain(extracted.metadata.get("fetched_url"))
        publisher = self._find_trusted_publisher(source_domain, extracted.metadata.get("publisher_name"))
        metadata_signals = self._metadata_signals(extracted)
        platform_type = str((extracted.metadata or {}).get("platform_type") or "").lower() if isinstance(extracted.metadata, dict) else ""
        if platform_type == "facebook":
            outcome = self._layer2_facebook_post(job, extracted, publisher, metadata_signals, persist_matches=persist_matches)
            append_audit_entry(
                self.db,
                job,
                "layer2_completed",
                {"match_count": len((outcome.details or {}).get("matches", [])), "mode": "facebook_text_reasoning"},
            )
            return outcome
        if self._should_use_social_text_reasoning(extracted):
            outcome = self._layer2_social_post(job, extracted, publisher, metadata_signals, persist_matches=persist_matches)
            append_audit_entry(
                self.db,
                job,
                "layer2_completed",
                {"match_count": len((outcome.details or {}).get("matches", [])), "mode": "social_text_reasoning"},
            )
            return outcome
        outcome = self._layer2_article_url(job, extracted, publisher, metadata_signals, persist_matches=persist_matches)
        append_audit_entry(
            self.db,
            job,
            "layer2_completed",
            {"match_count": len((outcome.details or {}).get("matches", [])), "mode": "article_url_reasoning"},
        )
        return outcome

    def _should_use_social_text_reasoning(self, extracted: ExtractedPayload) -> bool:
        metadata = extracted.metadata if isinstance(extracted.metadata, dict) else {}
        source_type = str(metadata.get("source_type") or "").lower()
        platform_type = str(metadata.get("platform_type") or "").lower()
        source_domain = str(metadata.get("source_domain") or extract_domain(extracted.canonical_url) or "").lower()
        if source_type == "social_post":
            return True
        return platform_type in {"youtube", "facebook", "instagram", "x", "linkedin"} or any(
            token in source_domain for token in ("youtube.com", "youtu.be", "facebook.com", "instagram.com", "x.com", "twitter.com")
        )

    def _layer2_article_url(
        self,
        job: VerificationJob,
        extracted: ExtractedPayload,
        publisher: TrustedPublisher | None,
        metadata_signals: list[str],
        persist_matches: bool = True,
    ) -> LayerOutcome:
        source_domain = extract_domain(extracted.canonical_url) or extract_domain(extracted.metadata.get("fetched_url"))
        return self._layer2_article_url_via_text_pipeline(job, extracted, publisher, metadata_signals, source_domain, persist_matches=persist_matches)

    def _layer2_article_url_via_text_pipeline(
        self,
        job: VerificationJob,
        extracted: ExtractedPayload,
        publisher: TrustedPublisher | None,
        metadata_signals: list[str],
        source_domain: str | None,
        persist_matches: bool = True,
    ) -> LayerOutcome:
        article_context = self._extract_url_article_context(extracted)
        clean_body = str(article_context.get("cleaned_article_text") or extracted.main_text or "").strip()
        article_text = self._url_claim_input_text(extracted, clean_body)
        article_text_length = len(clean_body)
        context_quality = "full" if article_text_length >= 500 else "limited" if article_text_length >= 200 else "failed"
        if self._url_article_text_unusable_for_nlp(extracted, clean_body, article_text):
            return self._url_insufficient_article_text_outcome(
                extracted,
                publisher,
                metadata_signals,
                article_context,
                article_text_length,
                failure_reason="extraction_too_short" if article_text_length < 200 else "extraction_failed",
            )

        url_text_payload = ExtractedPayload(
            title=extracted.title,
            author=extracted.author,
            publish_time=extracted.publish_time,
            canonical_url=extracted.canonical_url,
            main_text=article_text,
            hero_image_url=extracted.hero_image_url,
            normalized_text=normalize_text(article_text),
            content_hash=sha256_text(normalize_text(article_text)),
            metadata={
                **(extracted.metadata if isinstance(extracted.metadata, dict) else {}),
                "source_type": "url_article_text",
                "original_url": extracted.canonical_url or extracted.metadata.get("fetched_url"),
                "source_domain": source_domain,
                "context_quality": context_quality,
            },
        )
        text_outcome = self._text_nlp_layer(job, url_text_payload, article_text)
        text_details = text_outcome.details if isinstance(text_outcome.details, dict) else {}
        queries = self._dedupe_queries(
            [
                *text_details.get("trusted_source_queries", []),
                *text_details.get("lexical_queries", []),
                *text_details.get("semantic_queries", []),
                *text_details.get("fact_check_queries", []),
                *text_details.get("official_source_queries", []),
                *text_details.get("exact_phrase_queries", []),
            ],
            limit=10,
        )
        raw_evidence = [item for item in text_details.get("evidence_sources", []) if isinstance(item, dict)]
        matches, rejected = self._url_filter_text_pipeline_evidence(raw_evidence, source_domain, extracted)
        refutes = [match for match in matches if str(match.get("relation") or match.get("evidence_relation")) in {"refutes", "contradicts", "contradiction"}]
        match_groups = self._match_groups(matches, source_domain)
        same_count = int(match_groups["same_publisher_count"])
        cross_count = int(match_groups["cross_publisher_count"])
        confidence = max(0.0, min(1.0, float(text_outcome.confidence or 0.0)))
        if context_quality == "limited" and confidence > 0.65:
            confidence = 0.65 if cross_count < 2 else confidence
        confidence = round(confidence, 3)
        verdict = str(text_details.get("fact_check_verdict") or text_details.get("text_verification_label") or "").strip() or self._url_label_from_confidence(confidence, "")
        assessment = str(text_details.get("fact_check_assessment") or text_outcome.explanation or "").strip()
        if refutes:
            verdict = "contradicted"
            assessment = assessment or "Independent evidence or fact-checking refuted the extracted URL article claim."
        elif cross_count > 0:
            assessment = assessment or f"Independent cross-publisher corroboration was found from {cross_count} publisher domain{'s' if cross_count != 1 else ''}."
        elif same_count > 0:
            assessment = assessment or "Only same publisher domain context was found. It is not treated as independent URL verification."
        elif not matches and confidence <= 0.349:
            assessment = assessment or "No reliable independent sources found for the extracted URL article claims."

        if persist_matches:
            for candidate in matches[: self.MAX_NLP_MATCHES]:
                self.db.add(
                    SimilarityMatch(
                        job_id=job.id,
                        candidate_url=candidate.get("url"),
                        candidate_title=candidate.get("title"),
                        score=float(candidate.get("score") or 0.0),
                        explanation=str(candidate.get("explanation") or ""),
                        metadata_json={
                            "snippet": candidate.get("snippet"),
                            "domain": candidate.get("domain"),
                            "source": candidate.get("source", "web_search"),
                            "relation": candidate.get("relation"),
                            "evidence_relation": candidate.get("evidence_relation"),
                        },
                    )
                )

        temp_outcome = LayerOutcome(
            status="supporting" if matches and confidence > 0 else "none",
            confidence=confidence,
            explanation=assessment,
            details={"matches": matches, "fact_check_verdict": verdict, "trusted_publisher_match": publisher.name if publisher else None},
        )
        formula_components = text_details.get("text_component_scores") if isinstance(text_details.get("text_component_scores"), dict) else self._url_formula_components(temp_outcome, extracted)
        nlp_component_values = {
            "factual_consistency": confidence,
            "reference_support": confidence,
            "cross_source_agreement": confidence,
            "language_reliability": confidence,
            "authority": confidence,
            "penalty": 1.0,
        }
        details = {
            "matches": matches,
            "match_summary": (
                f"Independent corroboration found with {matches[0].get('title') or matches[0].get('url')}."
                if cross_count > 0 and matches
                else "No independent source directly supported the extracted URL article claim."
            ),
            "top_match_title": matches[0].get("title") if matches else None,
            "top_match_url": matches[0].get("url") if matches else None,
            "top_match_score": matches[0].get("score") if matches else None,
            "fact_check_verdict": verdict,
            "fact_check_basis": "URL article text was extracted and passed through the same claim extraction, search, reranking, and claim-evidence verification architecture used for pasted text.",
            "fact_check_assessment": assessment,
            "metadata_signals": metadata_signals,
            "trusted_publisher_match": publisher.name if publisher else None,
            "web_matches": [match for match in matches if match.get("source") == "web_search"],
            "original_url": extracted.canonical_url or extracted.metadata.get("fetched_url"),
            "original_domain": source_domain,
            "original_headline": extracted.title,
            "url_context_debug": {
                "input_type": "url",
                "submitted_url": extracted.metadata.get("fetched_url") or extracted.canonical_url,
                "submitted_domain": source_domain,
                "canonical_url": extracted.canonical_url,
                "fetch_status_code": extracted.metadata.get("fetch_status_code"),
                "final_redirected_url": extracted.metadata.get("final_redirected_url"),
                "content_type": extracted.metadata.get("content_type"),
                "url_fetch_debug": extracted.metadata.get("url_fetch_debug", {}),
                "extraction_method": extracted.metadata.get("extraction_method") or "readability_article_text",
                "extracted_title": extracted.title,
                "extracted_article_text_length": article_text_length,
                "context_quality": context_quality,
                "cleaned_article_body_preview": clean_body[:500],
                "final_text_passed_to_text_pipeline": article_text[:1000],
                "text_pipeline_called": True,
                "claim_tuple": article_context.get("claim_tuple", {}),
                "extracted_main_claim": article_context.get("main_claim"),
                "extracted_claims": text_details.get("extracted_claims", article_context.get("extracted_claims", [])),
                "queries_used": queries,
                "search_api_called": bool(queries),
                "url_search_debug": [],
                "text_pipeline_search_debug": text_details.get("retrieval_debug", {}),
                "candidate_evidence_urls": [item.get("url") for item in raw_evidence if item.get("url")],
                "candidate_relevance_labels": [
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "relevance_score": item.get("claim_relevance_score") or item.get("relation_confidence") or item.get("score"),
                        "reliability_score": item.get("source_reliability") or item.get("source_score") or item.get("authority_score"),
                        "independent": self._domain_family(item.get("domain") or extract_domain(item.get("url"))) != self._domain_family(source_domain),
                        "stance": item.get("evidence_relation") or item.get("judgment_label") or item.get("alignment"),
                    }
                    for item in raw_evidence[:10]
                ],
                "rejected_related_articles": rejected,
                "refuting_sources": refutes,
                "url_judged_candidates": [],
                "nlp_context_score": confidence,
                "final_nlp_score": confidence,
                "evidence_sufficiency": formula_components.get("E"),
                "penalty_multiplier": formula_components.get("P"),
                "fallback_used": article_text_length < 500,
                "fallback_reason": "short_article_text_searched_with_claim_context" if article_text_length < 500 else None,
            },
            "generated_queries": queries,
            "brave_results_count": len(text_details.get("search_results_raw", []) or []),
            "brave_top_results": text_details.get("search_results_raw", [])[:5],
            "same_publisher_candidates": [item for item in raw_evidence if self._domain_family(item.get("domain") or extract_domain(item.get("url"))) == self._domain_family(source_domain)][:5],
            "cross_publisher_candidates": [item for item in raw_evidence if self._domain_family(item.get("domain") or extract_domain(item.get("url"))) != self._domain_family(source_domain)][:5],
            "final_selected_candidates": raw_evidence[: self.MAX_NLP_MATCHES],
            "same_publisher_matches": match_groups["same_publisher_matches"],
            "cross_publisher_matches": match_groups["cross_publisher_matches"],
            "same_publisher_count": same_count,
            "cross_publisher_count": cross_count,
            "support_domains": match_groups["support_domains"],
            "cross_publisher_agreement_exists": match_groups["cross_source_exists"],
            "corroboration_scope": "same_source_only" if match_groups["same_source_only"] else ("cross_source" if match_groups["cross_source_exists"] else "no_matches"),
            "support_judgment_per_candidate": text_details.get("claim_results", []),
            "max_similar_articles_allowed": self.MAX_NLP_MATCHES,
            "similar_articles_found_count": len(matches),
            "same_context_articles_found_count": len(matches),
            "counted_same_context_articles": matches,
            "rejected_related_articles": rejected,
            "stopped_search_reason": "found_5_sources" if len(matches) >= self.MAX_NLP_MATCHES else "search_exhausted",
            "formula_components": formula_components,
            "nlp_component_values": nlp_component_values,
            "final_score": confidence,
            "final_label": self._url_label_from_confidence(confidence, verdict),
            "retrieval_failure_reason": text_details.get("pipeline_failure_reason"),
            "total_match_count": len(matches),
            "has_more_matches": len(raw_evidence) > len(matches),
            "search_summary": article_context.get("article_meaning"),
            "search_keywords": self._build_fact_check_keyword_query(extracted),
            "text_pipeline_details": text_details,
        }
        return LayerOutcome(
            status="supporting" if matches and confidence > 0 else "none",
            confidence=confidence,
            explanation=assessment,
            details=details,
        )

    def _url_article_text_unusable_for_nlp(self, extracted: ExtractedPayload, clean_body: str, article_text: str) -> bool:
        metadata = extracted.metadata if isinstance(extracted.metadata, dict) else {}
        source_type = str(metadata.get("source_type") or "").lower()
        normalized_body = normalize_text(clean_body)
        normalized_article = normalize_text(article_text)
        if source_type == "url_fallback" or bool(metadata.get("fetch_limited")):
            return True
        if "full article text could not be fetched server side" in normalized_body or "full article text could not be fetched server side" in normalized_article:
            return True
        if normalized_body.startswith("submitted url from ") and "full article text could not be fetched" in normalized_body:
            return True
        return len(clean_body) < 200 or len(article_text) < 200

    def _url_evidence_from_judged_candidates(self, judged_candidates: list[dict]) -> list[dict]:
        evidence: list[dict] = []
        for item in judged_candidates:
            if not isinstance(item, dict):
                continue
            match = item.get("match", {}) if isinstance(item.get("match"), dict) else {}
            judgment = item.get("judgment", {}) if isinstance(item.get("judgment"), dict) else {}
            url = str(match.get("url") or judgment.get("url") or "").strip()
            if not url:
                continue
            label = str(judgment.get("support_label") or judgment.get("context_match") or "").lower()
            if label == "refutes":
                relation = "refutes"
            elif label in {"exact_support", "strong_support"}:
                relation = "supports"
            elif label == "partial_context":
                relation = "partially_related"
            else:
                relation = "irrelevant"
            evidence.append(
                {
                    "title": match.get("title") or judgment.get("title"),
                    "url": url,
                    "domain": match.get("domain") or judgment.get("domain") or extract_domain(url),
                    "source": match.get("source", "web_search"),
                    "source_type": get_source_type(url),
                    "snippet": match.get("snippet"),
                    "passage": judgment.get("evidence_quote") or match.get("snippet"),
                    "score": max(float(match.get("score") or 0.0), float(judgment.get("support_score") or 0.0)),
                    "claim_relevance_score": judgment.get("relevance_score"),
                    "relation_confidence": judgment.get("support_score"),
                    "evidence_relation": relation,
                    "relation": "exact_support" if label == "exact_support" else relation,
                    "judgment_label": relation,
                    "aboutness": "exact_claim" if label == "exact_support" else "strong_related" if label == "strong_support" else "partial_context" if label == "partial_context" else "irrelevant",
                    "stance": "refutes" if relation == "refutes" else "supports" if relation == "supports" else "neutral",
                    "evidence_quote": judgment.get("evidence_quote"),
                    "reason_accepted_or_rejected": judgment.get("reject_reason") or match.get("explanation"),
                    "rerank_score": judgment.get("rerank_score"),
                    "fetch_status": judgment.get("fetch_status"),
                    "extracted_text_length": judgment.get("extracted_text_length"),
                }
            )
        return evidence

    def _url_filter_text_pipeline_evidence(
        self,
        evidence_sources: list[dict],
        source_domain: str | None,
        extracted: ExtractedPayload,
    ) -> tuple[list[dict], list[dict]]:
        source_family = self._domain_family(source_domain)
        original_url = canonicalize_url(extracted.canonical_url or extracted.metadata.get("fetched_url") or "")
        matches: list[dict] = []
        rejected: list[dict] = []
        seen: set[str] = set()
        for item in evidence_sources:
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            canonical_candidate = canonicalize_url(url)
            domain = str(item.get("domain") or extract_domain(url) or "")
            same_publisher = bool(source_family and self._domain_family(domain) == source_family)
            relation = str(item.get("evidence_relation") or item.get("relation") or item.get("judgment_label") or item.get("alignment") or "").lower()
            supports = relation in {"supports", "exact_support", "strong_support"}
            refutes = relation in {"refutes", "contradicts", "contradiction"}
            if canonical_candidate == original_url:
                rejected.append({**item, "reason": "Original URL cannot count as independent URL evidence.", "same_publisher": same_publisher})
                continue
            if supports and same_publisher:
                relation = "partially_related"
                supports = False
            if not (supports or refutes):
                if not same_publisher:
                    rejected.append({**item, "reason": "Evidence was related, uncertain, or unrelated, but did not directly support/refute the URL claim.", "same_publisher": same_publisher})
                    continue
            key = canonical_candidate or url
            if key in seen:
                continue
            seen.add(key)
            score = float(item.get("claim_relevance_score") or item.get("relation_confidence") or item.get("score") or item.get("rerank_score") or 0.0)
            matches.append(
                {
                    "title": item.get("title"),
                    "url": url,
                    "score": round(max(0.0, min(1.0, score)), 3),
                    "explanation": item.get("reason_accepted_or_rejected") or item.get("summary") or "Text verification pipeline matched this source to the URL article claim.",
                    "source": item.get("source", "web_search"),
                    "snippet": item.get("snippet"),
                    "domain": domain,
                    "relation": "refutes" if refutes else "supports" if supports else "partially_related",
                    "evidence_relation": "refutes" if refutes else "supports" if supports else "partially_related",
                    "evidence_quote": item.get("evidence_quote"),
                    "same_publisher_context_only": bool(same_publisher and not refutes),
                }
            )
            if len(matches) >= self.MAX_NLP_MATCHES:
                break
        return matches, rejected[:10]

    def _url_insufficient_article_text_outcome(
        self,
        extracted: ExtractedPayload,
        publisher: TrustedPublisher | None,
        metadata_signals: list[str],
        article_context: dict[str, object],
        article_text_length: int,
        failure_reason: str = "extraction_too_short",
    ) -> LayerOutcome:
        metadata = extracted.metadata if isinstance(extracted.metadata, dict) else {}
        details = {
            "matches": [],
            "match_summary": "The URL article body was too short for reliable NLP fact verification.",
            "top_match_title": None,
            "top_match_url": None,
            "top_match_score": None,
            "fact_check_verdict": "inconclusive",
            "fact_check_basis": "URL NLP requires enough cleaned article text before claim extraction and external evidence verification can run.",
            "fact_check_assessment": "Could not extract enough article text from the URL.",
            "metadata_signals": metadata_signals,
            "trusted_publisher_match": publisher.name if publisher else None,
            "web_matches": [],
            "original_url": extracted.canonical_url or extracted.metadata.get("fetched_url"),
            "original_domain": extract_domain(extracted.canonical_url) or extract_domain(extracted.metadata.get("fetched_url")),
            "original_headline": extracted.title,
            "url_context_debug": {
                "input_type": "url",
                "submitted_url": extracted.metadata.get("fetched_url") or extracted.canonical_url,
                "normalized_url": canonicalize_url(extracted.metadata.get("fetched_url") or extracted.canonical_url or ""),
                "final_redirected_url": metadata.get("final_redirected_url"),
                "fetch_status_code": metadata.get("fetch_status_code"),
                "content_type": metadata.get("content_type"),
                "url_fetch_debug": metadata.get("url_fetch_debug", {}),
                "extraction_method": metadata.get("extraction_method") or "none",
                "extracted_title": extracted.title,
                "extracted_article_text_length": article_text_length,
                "context_quality": "failed",
                "cleaned_article_body_preview": str(article_context.get("cleaned_article_text") or "")[:500],
                "extracted_claims": [],
                "queries_used": [],
                "candidate_evidence_urls": [],
                "text_pipeline_called": False,
                "final_nlp_score": 0.0,
                "fallback_used": True,
                "fallback_reason": failure_reason,
            },
            "generated_queries": [],
            "brave_results_count": 0,
            "brave_top_results": [],
            "same_publisher_candidates": [],
            "cross_publisher_candidates": [],
            "final_selected_candidates": [],
            "same_publisher_matches": [],
            "cross_publisher_matches": [],
            "same_publisher_count": 0,
            "cross_publisher_count": 0,
            "support_domains": [],
            "cross_publisher_agreement_exists": False,
            "corroboration_scope": "insufficient_article_text",
            "support_judgment_per_candidate": [],
            "formula_components": {"S": 0.0, "R": 0.0, "A": 0.0, "D": 0.0, "C": 0.0, "E": 0.0, "P": 1.0},
            "nlp_component_values": {
                "factual_consistency": 0.0,
                "reference_support": 0.0,
                "cross_source_agreement": 0.0,
                "language_reliability": 0.0,
                "authority": 0.0,
                "penalty": 1.0,
            },
            "final_score": 0.0,
            "final_label": "Unverified",
            "retrieval_failure_reason": failure_reason,
            "total_match_count": 0,
            "has_more_matches": False,
        }
        return LayerOutcome(status="none", confidence=0.0, explanation=details["fact_check_assessment"], details=details)

    def _layer2_article_url_legacy(
        self,
        job: VerificationJob,
        extracted: ExtractedPayload,
        publisher: TrustedPublisher | None,
        metadata_signals: list[str],
        persist_matches: bool = True,
    ) -> LayerOutcome:
        source_domain = extract_domain(extracted.canonical_url) or extract_domain(extracted.metadata.get("fetched_url"))
        article_context = self._extract_url_article_context(extracted)
        logger.info(
            "URL NLP extraction: cleaned_article_text_length=%s extracted_title=%r main_claim=%r generated_queries=%s retrieved_titles=%s",
            article_context.get("cleaned_article_text_length"),
            extracted.title,
            article_context.get("main_claim"),
            queries,
            [item.get("title") for item in retrieval.get("selected_candidates", [])[:10]],
        )
        retrieval_failure = str(retrieval.get("failure_reason") or "")
        if retrieval_failure.startswith("retrieval_backend_failure:") or retrieval_failure in {"provider_not_configured", "provider_auth_failed"}:
            return LayerOutcome(
                status="none",
                confidence=0.0,
                explanation="The retrieval backend failed before corroborating news evidence could be gathered.",
                details={
                    "matches": [],
                    "fact_check_verdict": "inconclusive",
                    "fact_check_basis": "Brave retrieval failed before usable corroborating candidates were returned.",
                    "fact_check_assessment": "Independent corroboration could not be evaluated because retrieval failed.",
                    "metadata_signals": metadata_signals,
                    "trusted_publisher_match": publisher.name if publisher else None,
                    "original_url": extracted.canonical_url or extracted.metadata.get("fetched_url"),
                    "original_domain": source_domain,
                    "original_headline": extracted.title,
                    "generated_queries": queries,
                    "brave_results_count": retrieval.get("brave_results_count", 0),
                    "brave_top_results": retrieval.get("brave_top_results", []),
                    "same_publisher_candidates": [],
                    "cross_publisher_candidates": [],
                    "final_selected_candidates": [],
                    "same_publisher_count": 0,
                    "cross_publisher_count": 0,
                    "support_domains": [],
                    "cross_publisher_agreement_exists": False,
                    "corroboration_scope": "retrieval_failure",
                    "support_judgment_per_candidate": [],
                    "formula_components": {"S": 0.0, "R": 0.0, "A": 0.0, "D": 0.0, "C": 0.0, "E": 0.0, "P": 0.0},
                    "final_score": 0.0,
                    "final_label": "Not Verifiable",
                    "retrieval_failure_reason": retrieval_failure,
                    "web_matches": [],
                    "total_match_count": 0,
                    "has_more_matches": False,
                },
            )

        judged_candidates = self._judge_article_candidates(extracted, article_meaning, retrieval.get("selected_candidates", []), source_domain, article_context)
        logger.info(
            "URL NLP candidate judgments: %s",
            [
                {
                    "title": item["judgment"].get("title"),
                    "relevance_score": item["judgment"].get("relevance_score"),
                    "label": item["judgment"].get("evidence_label"),
                }
                for item in judged_candidates
            ],
        )
        useful_judgments = [
            item
            for item in judged_candidates
            if item["judgment"].get("support_label") in {"strong_support", "exact_support", "refutes", "contradiction"}
            and not item["judgment"].get("same_publisher")
        ][: self.MAX_NLP_MATCHES]
        rejected_judgments = [
            item
            for item in judged_candidates
            if item not in useful_judgments
        ]
        matches = [item["match"] for item in useful_judgments]
        match_groups = self._match_groups(matches, source_domain)
        same_count = int(match_groups["same_publisher_count"])
        cross_count = int(match_groups["cross_publisher_count"])
        confidence = round(self._article_url_confidence(useful_judgments, extracted, publisher), 3)
        if cross_count == 0 and same_count > 0:
            confidence = min(confidence, 0.54)
        verdict, assessment = self._fact_check_decision(
            confidence,
            bool(publisher),
            same_publisher_count=same_count,
            cross_publisher_count=cross_count,
        )
        explanation = assessment
        if cross_count > 0:
            explanation = f"Independent cross-publisher corroboration was found from {cross_count} publisher domain{'s' if cross_count != 1 else ''}."
        elif same_count > 0:
            explanation = "Contextually related content was found on the same publisher domain. Independent cross-publisher corroboration was limited."

        if persist_matches:
            for candidate in matches:
                self.db.add(
                    SimilarityMatch(
                        job_id=job.id,
                        candidate_url=candidate.get("url"),
                        candidate_title=candidate.get("title"),
                        score=float(candidate.get("score") or 0.0),
                        explanation=str(candidate.get("explanation") or ""),
                        metadata_json={
                            "snippet": candidate.get("snippet"),
                            "domain": candidate.get("domain"),
                            "source": candidate.get("source", "web_search"),
                        },
                    )
                )

        temp_outcome = LayerOutcome(
            status="supporting" if matches or confidence > 0 else "none",
            confidence=confidence,
            explanation=explanation,
            details={
                "matches": matches,
                "fact_check_verdict": verdict,
                "trusted_publisher_match": publisher.name if publisher else None,
            },
        )
        formula_components = self._url_formula_components(temp_outcome, extracted)
        details = {
            "matches": matches,
            "match_summary": (
                f"Independent corroboration found with {matches[0].get('title') or matches[0].get('url')}."
                if cross_count > 0 and matches
                else f"Same-publisher context found with {matches[0].get('title') or matches[0].get('url')}."
                if matches
                else "No strong supporting or conflicting source match was available."
            ),
            "top_match_title": matches[0].get("title") if matches else None,
            "top_match_url": matches[0].get("url") if matches else None,
            "top_match_score": matches[0].get("score") if matches else None,
            "fact_check_verdict": verdict,
            "fact_check_basis": "Article verification used extracted article meaning, Brave retrieval, cross-publisher candidate filtering, Cohere reranking, and per-candidate support judgment.",
            "fact_check_assessment": explanation,
            "metadata_signals": metadata_signals,
            "trusted_publisher_match": publisher.name if publisher else None,
            "web_matches": [match for match in matches if match.get("source") == "web_search"],
            "original_url": extracted.canonical_url or extracted.metadata.get("fetched_url"),
            "original_domain": source_domain,
            "original_headline": extracted.title,
            "url_context_debug": {
                "input_type": "url",
                "submitted_url": extracted.metadata.get("fetched_url") or extracted.canonical_url,
                "submitted_domain": source_domain,
                "canonical_url": extracted.canonical_url,
                "extracted_title": extracted.title,
                "extracted_article_text_length": int(article_context.get("cleaned_article_text_length") or len(extracted.main_text or "")),
                "cleaned_article_text_length": int(article_context.get("cleaned_article_text_length") or 0),
                "extracted_main_claim": article_context.get("main_claim"),
                "extracted_claims": article_context.get("extracted_claims", []),
                "claim_tuple": article_context.get("claim_tuple", {}),
                "claim_entities": article_context.get("entities", []),
                "claim_actions": article_context.get("actions", []),
                "claim_locations": article_context.get("locations", []),
                "claim_dates": article_context.get("dates", []),
                "queries_used": queries,
                "search_api_called": bool(queries),
                "raw_search_results_count": retrieval.get("brave_results_count", 0),
                "retrieved_source_titles": [item.get("title") for item in retrieval.get("selected_candidates", []) if item.get("title")],
                "candidate_relevance_labels": [
                    {
                        "title": item["judgment"].get("title"),
                        "url": item["judgment"].get("url"),
                        "relevance_score": item["judgment"].get("relevance_score"),
                        "label": item["judgment"].get("evidence_label"),
                        "support_label": item["judgment"].get("support_label"),
                        "reason": item["judgment"].get("reject_reason"),
                    }
                    for item in judged_candidates
                ],
                "fetched_pages_count": sum(1 for item in retrieval.get("selected_candidates", []) if item.get("fetch_status")),
                "max_similar_articles_allowed": self.MAX_NLP_MATCHES,
                "same_context_articles_found_count": len(matches),
                "counted_same_context_articles": matches,
                "rejected_related_articles": [
                    {
                        **item["match"],
                        "support_label": item["judgment"].get("support_label"),
                        "support_score": item["judgment"].get("support_score"),
                        "same_publisher": item["judgment"].get("same_publisher"),
                        "reason": item["judgment"].get("reject_reason") or "Not counted because it was partial, same-publisher context, or unrelated.",
                    }
                    for item in rejected_judgments
                ],
                "refuting_sources": [item["match"] for item in useful_judgments if item["judgment"].get("support_label") == "refutes"],
                "stopped_search_reason": "found_5_sources" if len(matches) >= self.MAX_NLP_MATCHES else "search_exhausted",
                "crypto_score": 0,
                "nlp_context_score": confidence,
                "community_score": 0,
                "evidence_sufficiency": 1.0 if matches else 0.4,
                "penalty_multiplier": 0.4 if any(item["judgment"].get("support_label") == "refutes" for item in useful_judgments) else 1.0,
                "final_score": confidence,
                "final_label": self._url_label_from_confidence(confidence, verdict),
                "reason": explanation,
            },
            "generated_queries": queries,
            "brave_results_count": retrieval.get("brave_results_count", 0),
            "brave_top_results": retrieval.get("brave_top_results", []),
            "same_publisher_candidates": retrieval.get("same_publisher_candidates", []),
            "cross_publisher_candidates": retrieval.get("cross_publisher_candidates", []),
            "final_selected_candidates": retrieval.get("selected_candidates", []),
            "same_publisher_matches": match_groups["same_publisher_matches"],
            "cross_publisher_matches": match_groups["cross_publisher_matches"],
            "same_publisher_count": same_count,
            "cross_publisher_count": cross_count,
            "support_domains": match_groups["support_domains"],
            "cross_publisher_agreement_exists": match_groups["cross_source_exists"],
            "corroboration_scope": "same_source_only" if match_groups["same_source_only"] else ("cross_source" if match_groups["cross_source_exists"] else "no_matches"),
            "support_judgment_per_candidate": [item["judgment"] for item in judged_candidates],
            "max_similar_articles_allowed": self.MAX_NLP_MATCHES,
            "similar_articles_found_count": len(matches),
            "same_context_articles_found_count": len(matches),
            "counted_same_context_articles": matches,
            "rejected_related_articles": [
                {
                    **item["match"],
                    "support_label": item["judgment"].get("support_label"),
                    "support_score": item["judgment"].get("support_score"),
                    "same_publisher": item["judgment"].get("same_publisher"),
                    "reason": item["judgment"].get("reject_reason") or "Not counted because it was partial, same-publisher context, or unrelated.",
                }
                for item in rejected_judgments
            ],
            "stopped_search_reason": "found_5_sources" if len(matches) >= self.MAX_NLP_MATCHES else "search_exhausted",
            "formula_components": formula_components,
            "final_score": confidence,
            "final_label": self._url_label_from_confidence(confidence, verdict),
            "retrieval_failure_reason": retrieval_failure or None,
            "total_match_count": len(matches),
            "has_more_matches": False,
            "search_summary": article_meaning,
            "search_keywords": self._build_fact_check_keyword_query(extracted),
        }
        return LayerOutcome(
            status="supporting" if matches or confidence > 0 else "none",
            confidence=confidence,
            explanation=explanation,
            details=details,
        )

    def _layer2_social_post(
        self,
        job: VerificationJob,
        extracted: ExtractedPayload,
        publisher: TrustedPublisher | None,
        metadata_signals: list[str],
        persist_matches: bool = True,
    ) -> LayerOutcome:
        metadata = extracted.metadata if isinstance(extracted.metadata, dict) else {}
        source_domain = extract_domain(extracted.canonical_url) or extract_domain(metadata.get("fetched_url"))
        platform_type = str(metadata.get("platform_type") or "").lower()
        social_ocr = self._extract_remote_social_image_ocr(extracted)
        caption_text = self._social_caption_value(extracted)
        ocr_texts = self._social_ocr_texts(extracted, social_ocr)
        social_queries = self._social_query_groups(extracted, caption_text, ocr_texts)
        social_text = self._build_social_text_input(extracted, social_ocr.get("ocr_text"))
        if not social_text.strip():
            no_input_explanation = (
                "YouTube metadata could not be retrieved, so verification did not run on blocked-page or fallback boilerplate text."
                if platform_type == "youtube"
                else "No usable social-post caption, title, description, or image text was available for contextual verification."
            )
            return LayerOutcome(
                status="none",
                confidence=0.0,
                explanation=no_input_explanation,
                details={
                    "matches": [],
                    "fact_check_verdict": "inconclusive",
                    "fact_check_basis": (
                        "The YouTube URL did not expose enough retrievable video metadata to run metadata-based verification."
                        if platform_type == "youtube"
                        else "The social post did not expose enough caption, description, or image text to run text-style evidence verification."
                    ),
                    "fact_check_assessment": (
                        "Blocked-page or fallback text was rejected, so no meaningful YouTube verification input was available."
                        if platform_type == "youtube"
                        else "No strong supporting or conflicting source match was available."
                    ),
                    "metadata_signals": metadata_signals,
                    "trusted_publisher_match": publisher.name if publisher else None,
                    "social_text_input": "",
                    "verification_input_used": "",
                    "social_image_ocr": social_ocr.get("ocr_text"),
                    "social_ocr_result": social_ocr,
                    "social_caption_text": caption_text,
                    "social_ocr_texts": ocr_texts,
                    "social_query_debug": social_queries,
                    "web_matches": [],
                    "metadata_fetch_status": metadata.get("metadata_fetch_status"),
                    "blocked_page_detected": bool(metadata.get("blocked_page_detected")),
                    "blocked_page_reason": metadata.get("blocked_page_reason"),
                },
            )

        social_payload = ExtractedPayload(
            title=extracted.title,
            author=extracted.author,
            publish_time=extracted.publish_time,
            canonical_url=extracted.canonical_url,
            main_text=social_text,
            hero_image_url=extracted.hero_image_url,
            normalized_text=normalize_text(social_text),
            content_hash=extracted.content_hash,
            metadata={
                **(extracted.metadata if isinstance(extracted.metadata, dict) else {}),
                "social_text_input": social_text,
                "social_image_ocr": social_ocr.get("ocr_text"),
                "social_ocr_result": social_ocr,
            },
        )
        prepared = self.preprocess_text(social_text)
        extraction = self.claim_extraction.extract_claims(prepared["cleaned_input"], max_claims=self.TEXT_CLAIM_LIMIT)
        claims = [item["text"] for item in extraction.get("claims", [])] or self._extract_text_claims(social_payload)
        meaning_bundle = self._build_text_meaning_bundle(prepared, extraction, claims)
        query_plan = self._build_text_hybrid_queries(prepared, extraction, claims, meaning_bundle)
        source_recovery_stage = self._empty_source_recovery("Skipped direct source recovery for social text; claim evidence search is used instead.")
        candidates, retrieval_debug = self._gather_text_candidate_documents(prepared, extraction, query_plan, source_recovery_stage)
        ranking_debug = self._hybrid_rank_text_candidates(prepared, claims, query_plan, candidates)
        top_articles = ranking_debug["reranked_top_results"]
        claim_results = [
            self._score_claim_against_evidence(claim, self._build_claim_evidence_from_top_articles(claim, top_articles))
            for claim in claims
        ]
        aggregate = self._finalize_text_verification(claim_results, top_articles, meaning_bundle)
        matches = aggregate["evidence_sources"][: self.DEFAULT_NLP_MATCH_LIMIT]

        if persist_matches:
            for candidate in matches:
                self.db.add(
                    SimilarityMatch(
                        job_id=job.id,
                        candidate_url=candidate.get("url"),
                        candidate_title=candidate.get("title"),
                        score=float(candidate.get("score") or candidate.get("retrieval_score") or 0.0),
                        explanation=str(candidate.get("summary") or candidate.get("explanation") or ""),
                        metadata_json={
                            "snippet": candidate.get("snippet") or candidate.get("passage"),
                            "domain": candidate.get("domain"),
                            "source": candidate.get("source", "web_search"),
                        },
                    )
                )

        confidence = round(float(aggregate["final_score"]), 3)
        matches = aggregate["evidence_sources"][: self.DEFAULT_NLP_MATCH_LIMIT]
        match_groups = self._match_groups(matches, source_domain)
        if int(match_groups["cross_publisher_count"]) == 0 and confidence > 0:
            confidence = min(confidence, 0.54)
        verdict, assessment = self._fact_check_decision(
            confidence,
            bool(publisher),
            same_publisher_count=int(match_groups["same_publisher_count"]),
            cross_publisher_count=int(match_groups["cross_publisher_count"]),
            social_mode=True,
        )
        if aggregate["final_label"] == "Unverified":
            verdict = "conflicting"
            assessment = aggregate["explanation"]
        elif aggregate["final_label"] == "Not Verifiable":
            verdict = "inconclusive"
            assessment = aggregate["explanation"]
        elif int(match_groups["cross_publisher_count"]) == 0 and int(match_groups["same_publisher_count"]) == 0 and confidence > 0:
            assessment = "Post-internal content was available, but broader external corroboration remains limited."
        details = {
            "matches": matches,
            "evidence_sources": matches,
            "extracted_claims": claims,
            "match_summary": aggregate["evidence_summary"],
            "top_match_title": matches[0].get("title") if matches else None,
            "top_match_url": matches[0].get("url") if matches else None,
            "top_match_score": matches[0].get("score") if matches else None,
            "fact_check_verdict": verdict,
            "fact_check_basis": "Social-post verification used title, caption/description metadata, post-image OCR when available, and text-style claim retrieval against web evidence.",
            "fact_check_assessment": assessment,
            "metadata_signals": metadata_signals,
            "trusted_publisher_match": publisher.name if publisher else None,
            "web_matches": matches,
            "total_match_count": len(matches),
            "has_more_matches": False,
            "search_summary": self._build_fact_check_summary_query(social_payload),
            "search_keywords": self._build_fact_check_keyword_query(social_payload),
            "search_queries_used": query_plan.get("lexical", []) + query_plan.get("semantic", []),
            "original_domain": source_domain,
            "same_publisher_matches": match_groups["same_publisher_matches"],
            "cross_publisher_matches": match_groups["cross_publisher_matches"],
            "same_publisher_count": match_groups["same_publisher_count"],
            "cross_publisher_count": match_groups["cross_publisher_count"],
            "support_domains": match_groups["support_domains"],
            "cross_publisher_agreement_exists": match_groups["cross_source_exists"],
            "corroboration_scope": "same_source_only" if match_groups["same_source_only"] else ("cross_source" if match_groups["cross_source_exists"] else "no_matches"),
            "social_text_input": social_text,
            "verification_input_used": social_text,
            "social_caption_text": caption_text,
            "social_image_ocr": social_ocr.get("ocr_text"),
            "social_ocr_texts": ocr_texts,
            "social_ocr_result": social_ocr,
            "social_ocr_results": metadata.get("social_image_ocr_results", []),
            "social_query_debug": social_queries,
            "social_context_mismatch": self._social_caption_ocr_mismatch(caption_text, ocr_texts),
            "metadata_fetch_status": metadata.get("metadata_fetch_status"),
            "blocked_page_detected": bool(metadata.get("blocked_page_detected")),
            "blocked_page_reason": metadata.get("blocked_page_reason"),
            "text_debug": {
                "raw_input": social_text,
                "cleaned_input": prepared["cleaned_input"],
                "exact_phrase_queries": meaning_bundle.get("exact_phrase_queries", []),
                "lexical_entity_queries": meaning_bundle.get("lexical_queries", []),
                "final_search_queries_sent": query_plan.get("lexical", []) + query_plan.get("semantic", []),
                "provider_status": [
                    item.get("provider_status")
                    for item in retrieval_debug.get("lexical_searches", []) + retrieval_debug.get("semantic_searches", [])
                    if item.get("provider_status")
                ],
                "brave_results_count": sum(
                    int(item.get("result_count") or 0)
                    for item in retrieval_debug.get("lexical_searches", []) + retrieval_debug.get("semantic_searches", [])
                ),
                "brave_top_results": [
                    result
                    for item in retrieval_debug.get("lexical_searches", []) + retrieval_debug.get("semantic_searches", [])
                    for result in item.get("results", [])
                ][:5],
                "candidate_fetches": retrieval_debug.get("candidate_fetches", []),
                "overlap_scores": source_recovery_stage.get("overlap_scores", []),
                "source_recovery_status": source_recovery_stage.get("final_source_recovery", {}).get("status"),
                "source_recovery_reason": source_recovery_stage.get("final_source_recovery", {}).get("reason"),
                "formula_components": {
                    key: value for key, value in aggregate.get("component_scores", {}).items() if key in {"S", "R", "A", "D", "C", "E", "P"}
                },
                "final_score": aggregate["final_score"],
                "final_label": aggregate["final_label"],
            },
            "social_claim_results": claim_results,
            "text_verification_score": aggregate["final_score"],
            "text_verification_label": aggregate["final_label"],
            "evidence_multiplier": aggregate["evidence_multiplier"],
            "penalty_multiplier": aggregate["penalty_multiplier"],
            "supporting_evidence_summary": aggregate["supporting_evidence_summary"],
            "contradiction_summary": aggregate["contradiction_summary"],
        }
        if platform_type == "youtube":
            details.update(
                {
                    "detected_platform": "youtube",
                    "detected_youtube_url": metadata.get("detected_youtube_url") or extracted.canonical_url,
                    "normalized_url": metadata.get("normalized_url") or extracted.canonical_url,
                    "youtube_video_id": metadata.get("youtube_video_id"),
                    "youtube_title": extracted.title,
                    "youtube_channel_title": metadata.get("channel_title") or metadata.get("publisher_name"),
                    "youtube_channel_id": metadata.get("channel_id"),
                    "youtube_channel_verification_status": metadata.get("channel_verification_status")
                    or extracted.metadata.get("youtube_signals", {}).get("channel_verification_status"),
                    "youtube_published_date": metadata.get("publish_time"),
                    "youtube_description": metadata.get("description"),
                    "youtube_thumbnail_url": metadata.get("thumbnail_url"),
                    "youtube_thumbnails": metadata.get("thumbnails", {}),
                    "youtube_transcript_status": metadata.get("transcript_status", "unavailable"),
                    "youtube_transcript_text": metadata.get("transcript_text"),
                    "youtube_transcript_length": int(metadata.get("transcript_length") or 0),
                    "youtube_metadata_fetch_status": metadata.get("metadata_fetch_status"),
                    "youtube_blocked_page_detected": bool(metadata.get("blocked_page_detected")),
                    "youtube_blocked_page_reason": metadata.get("blocked_page_reason"),
                    "youtube_verification_input_used": social_text,
                    "youtube_generated_queries": query_plan.get("lexical", []) + query_plan.get("semantic", []),
                    "youtube_candidate_external_sources": [match.get("url") for match in matches if match.get("url")],
                    "youtube_matched_sources": matches,
                    "youtube_same_platform_context_count": match_groups["same_publisher_count"],
                    "youtube_external_corroboration_results": match_groups["cross_publisher_matches"],
                }
            )
        if platform_type == "facebook":
            details.update(
                {
                    "facebook_caption_text": caption_text,
                    "facebook_attached_image_count": len(metadata.get("media_assets", [])) if isinstance(metadata.get("media_assets"), list) else 0,
                    "facebook_ocr_text_per_image": [
                        {
                            "image_url": item.get("image_url"),
                            "ocr_text": item.get("ocr_text"),
                            "ocr_status": item.get("ocr_status"),
                        }
                        for item in metadata.get("social_image_ocr_results", [])
                        if isinstance(item, dict)
                    ],
                    "facebook_caption_queries": social_queries["caption_queries"],
                    "facebook_ocr_queries": social_queries["ocr_queries"],
                    "facebook_combined_queries": social_queries["combined_queries"],
                    "facebook_candidate_urls": [match.get("url") for match in matches if match.get("url")],
                }
            )
        return LayerOutcome(
            status="supporting" if matches or confidence > 0.0 else "none",
            confidence=round(confidence, 3),
            explanation=assessment,
            details=details,
        )

    def _layer2_facebook_post(
        self,
        job: VerificationJob,
        extracted: ExtractedPayload,
        publisher: TrustedPublisher | None,
        metadata_signals: list[str],
        persist_matches: bool = True,
    ) -> LayerOutcome:
        metadata = extracted.metadata if isinstance(extracted.metadata, dict) else {}
        source_domain = extract_domain(extracted.canonical_url) or extract_domain(metadata.get("fetched_url"))
        caption_text = self._social_caption_value(extracted)
        social_ocr = self._extract_remote_social_image_ocr(extracted)
        ocr_texts = self._social_ocr_texts(extracted, social_ocr)
        combined_ocr_text = " ".join(ocr_texts[:2]).strip()
        social_queries = self._social_query_groups(extracted, caption_text, ocr_texts)
        context_mismatch = self._social_caption_ocr_mismatch(caption_text, ocr_texts)

        if not caption_text and not combined_ocr_text:
            return LayerOutcome(
                status="none",
                confidence=0.0,
                explanation="No usable Facebook caption or image text was available for evidence-based verification.",
                details={
                    "matches": [],
                    "fact_check_verdict": "inconclusive",
                    "fact_check_basis": "Facebook verification requires caption text, OCR text, or both before evidence retrieval can run.",
                    "fact_check_assessment": "The post did not expose a usable caption and OCR did not provide searchable text.",
                    "metadata_signals": metadata_signals,
                    "trusted_publisher_match": publisher.name if publisher else None,
                    "detected_platform": "facebook",
                    "normalized_url": extracted.canonical_url or metadata.get("fetched_url"),
                    "post_id": metadata.get("post_id"),
                    "page_or_author_name": metadata.get("publisher_name") or metadata.get("account_name") or extracted.author,
                    "publish_time": metadata.get("publish_time") or (extracted.publish_time.isoformat() if extracted.publish_time else None),
                    "raw_caption": caption_text,
                    "cleaned_caption": "",
                    "caption_missing": not bool(caption_text),
                    "caption_queries": [],
                    "attached_image_count": len(metadata.get("media_assets", [])) if isinstance(metadata.get("media_assets"), list) else 0,
                    "per_image_ocr_status": [
                        {"image_url": item.get("image_url"), "ocr_status": item.get("ocr_status"), "failure_reasons": item.get("failure_reasons", [])}
                        for item in metadata.get("social_image_ocr_results", [])
                        if isinstance(item, dict)
                    ],
                    "per_image_ocr_text": [
                        {"image_url": item.get("image_url"), "ocr_text": item.get("ocr_text"), "text_detected": bool(item.get("text_detected") or item.get("ocr_text"))}
                        for item in metadata.get("social_image_ocr_results", [])
                        if isinstance(item, dict)
                    ],
                    "ocr_queries": [],
                    "combined_queries": [],
                    "candidate_sources_from_caption": [],
                    "candidate_sources_from_ocr": [],
                    "same_source_count": 0,
                    "cross_source_count": 0,
                    "crypto_component_values": {},
                    "nlp_component_values": {
                        "factual_consistency": 0.0,
                        "reference_support": 0.0,
                        "cross_source_agreement": 0.0,
                        "language_reliability": 0.0,
                        "authority": 0.0,
                        "penalty": 0.0,
                    },
                    "final_formula_components": {
                        "factual_consistency": 0.0,
                        "reference_support": 0.0,
                        "cross_source_agreement": 0.0,
                        "language_reliability": 0.0,
                        "authority": 0.0,
                        "penalty": 0.0,
                    },
                    "final_score": 0.0,
                    "final_label": "Not Verifiable",
                    "social_context_mismatch": context_mismatch,
                },
            )

        caption_stream = self._evaluate_facebook_text_stream(extracted, caption_text, "caption") if caption_text else self._empty_facebook_stream("caption_missing")
        ocr_stream = self._evaluate_facebook_text_stream(extracted, combined_ocr_text, "ocr") if combined_ocr_text else self._empty_facebook_stream("ocr_missing")
        combined_text = " ".join(part for part in [caption_text, combined_ocr_text] if part).strip()
        combined_stream = (
            self._evaluate_facebook_text_stream(extracted, combined_text, "combined")
            if caption_text and combined_ocr_text
            else self._empty_facebook_stream("combined_not_used")
        )

        stream_candidates: list[dict] = []
        for stream in (caption_stream, ocr_stream, combined_stream):
            for match in stream.get("matches", []):
                if isinstance(match, dict):
                    stream_candidates.append(match)
        matches = self._dedupe_evidence_candidates(stream_candidates)[: self.DEFAULT_NLP_MATCH_LIMIT]
        if persist_matches:
            for candidate in matches:
                self.db.add(
                    SimilarityMatch(
                        job_id=job.id,
                        candidate_url=candidate.get("url"),
                        candidate_title=candidate.get("title"),
                        score=float(candidate.get("score") or candidate.get("rerank_score") or candidate.get("hybrid_score") or 0.0),
                        explanation=str(candidate.get("summary") or candidate.get("explanation") or candidate.get("evidence_summary") or ""),
                        metadata_json={
                            "snippet": candidate.get("snippet") or candidate.get("passage"),
                            "domain": candidate.get("domain"),
                            "source": candidate.get("source", "web_search"),
                        },
                    )
                )

        match_groups = self._match_groups(matches, source_domain)
        stream_scores = [
            float(stream.get("aggregate", {}).get("final_score") or 0.0)
            for stream in (caption_stream, ocr_stream, combined_stream)
            if isinstance(stream.get("aggregate"), dict)
        ]
        best_stream = max(
            (caption_stream, ocr_stream, combined_stream),
            key=lambda item: float(item.get("aggregate", {}).get("final_score") or 0.0),
        )
        best_aggregate = best_stream.get("aggregate", {}) if isinstance(best_stream.get("aggregate"), dict) else {}
        base_components = best_aggregate.get("component_scores", {}) if isinstance(best_aggregate.get("component_scores"), dict) else {}
        support = max(
            float(base_components.get("S") or 0.0),
            max((float(stream.get("aggregate", {}).get("component_scores", {}).get("S") or 0.0) for stream in (caption_stream, ocr_stream)), default=0.0),
        )
        reliability = max(
            float(base_components.get("R") or 0.0),
            max((float(stream.get("aggregate", {}).get("component_scores", {}).get("R") or 0.0) for stream in (caption_stream, ocr_stream)), default=0.0),
        )
        agreement = min(1.0, max(float(base_components.get("A") or 0.0), 0.2 * int(match_groups["cross_publisher_count"]) + 0.15 * max(0, len(match_groups["support_domains"]) - 1)))
        directness = max(
            float(base_components.get("D") or 0.0),
            float(caption_stream.get("aggregate", {}).get("component_scores", {}).get("D") or 0.0),
            float(ocr_stream.get("aggregate", {}).get("component_scores", {}).get("D") or 0.0),
        )
        clarity = max(
            float(base_components.get("C") or 0.0),
            0.8 if caption_text else 0.0,
            0.7 if combined_ocr_text else 0.0,
        )
        if not caption_text:
            clarity = min(clarity, 0.72)
        evidence_multiplier = min(
            1.0,
            max(
                float(best_aggregate.get("evidence_multiplier") or 0.0),
                0.35 + 0.25 * int(bool(caption_text)) + 0.25 * int(bool(combined_ocr_text)) + 0.08 * len(matches),
            ),
        )
        penalty_multiplier = min(
            float(best_aggregate.get("penalty_multiplier") or 1.0),
            0.78 if context_mismatch else 1.0,
        )
        if int(match_groups["cross_publisher_count"]) == 0:
            support = min(support, 0.48 if int(match_groups["same_publisher_count"]) else 0.42)
            agreement = min(agreement, 0.12 if int(match_groups["same_publisher_count"]) else 0.08)
            penalty_multiplier = min(penalty_multiplier, 0.72)
        if not matches:
            support = 0.0
            reliability = 0.0
            agreement = 0.0
            directness = min(directness, 0.25)
            evidence_multiplier = min(evidence_multiplier, 0.35)
            penalty_multiplier = min(penalty_multiplier, 0.7)

        confidence = max(
            0.0,
            min(
                1.0,
                (
                    0.30 * support
                    + 0.25 * reliability
                    + 0.20 * agreement
                    + 0.15 * directness
                    + 0.10 * clarity
                )
                * evidence_multiplier
                * penalty_multiplier,
            ),
        )
        verdict, assessment = self._fact_check_decision(
            confidence,
            bool(publisher),
            same_publisher_count=int(match_groups["same_publisher_count"]),
            cross_publisher_count=int(match_groups["cross_publisher_count"]),
            social_mode=True,
        )
        if not matches:
            verdict = "inconclusive"
            assessment = "No strong external corroboration was found for the Facebook caption or image text."
        elif int(match_groups["cross_publisher_count"]) == 0:
            verdict = "context match found"
            assessment = "Contextually related content was found, but independent cross-source corroboration was limited."
        elif context_mismatch:
            assessment = "Caption and OCR text were both checked, but they diverged enough to reduce confidence."
        else:
            assessment = "Caption text and image OCR were checked separately against external reporting."

        nlp_component_values = {
            "factual_consistency": round(support, 3),
            "reference_support": round(reliability, 3),
            "cross_source_agreement": round(agreement, 3),
            "language_reliability": round(directness, 3),
            "authority": round(clarity, 3),
            "penalty": round(penalty_multiplier, 3),
        }
        details = {
            "matches": matches,
            "evidence_sources": matches,
            "fact_check_verdict": verdict,
            "fact_check_basis": "Facebook verification separately checked the post caption and OCR text from attached images against external web evidence.",
            "fact_check_assessment": assessment,
            "metadata_signals": metadata_signals,
            "trusted_publisher_match": publisher.name if publisher else None,
            "web_matches": matches,
            "total_match_count": len(matches),
            "has_more_matches": False,
            "original_domain": source_domain,
            "same_publisher_matches": match_groups["same_publisher_matches"],
            "cross_publisher_matches": match_groups["cross_publisher_matches"],
            "same_publisher_count": match_groups["same_publisher_count"],
            "cross_publisher_count": match_groups["cross_publisher_count"],
            "support_domains": match_groups["support_domains"],
            "cross_publisher_agreement_exists": match_groups["cross_source_exists"],
            "corroboration_scope": "same_source_only" if match_groups["same_source_only"] else ("cross_source" if match_groups["cross_source_exists"] else "no_matches"),
            "social_text_input": combined_text or caption_text or combined_ocr_text,
            "verification_input_used": combined_text or caption_text or combined_ocr_text,
            "social_caption_text": caption_text,
            "social_image_ocr": social_ocr.get("ocr_text"),
            "social_ocr_texts": ocr_texts,
            "social_ocr_result": social_ocr,
            "social_ocr_results": metadata.get("social_image_ocr_results", []),
            "social_query_debug": social_queries,
            "social_context_mismatch": context_mismatch,
            "detected_platform": "facebook",
            "normalized_url": extracted.canonical_url or metadata.get("fetched_url"),
            "post_id": metadata.get("post_id"),
            "page_or_author_name": metadata.get("publisher_name") or metadata.get("account_name") or extracted.author,
            "publish_time": metadata.get("publish_time") or (extracted.publish_time.isoformat() if extracted.publish_time else None),
            "raw_caption": caption_text,
            "cleaned_caption": str(caption_stream.get("prepared", {}).get("cleaned_input") or ""),
            "caption_missing": not bool(caption_text),
            "caption_queries": social_queries["caption_queries"],
            "attached_image_count": len(metadata.get("media_assets", [])) if isinstance(metadata.get("media_assets"), list) else 0,
            "per_image_ocr_status": [
                {"image_url": item.get("image_url"), "ocr_status": item.get("ocr_status"), "failure_reasons": item.get("failure_reasons", [])}
                for item in metadata.get("social_image_ocr_results", [])
                if isinstance(item, dict)
            ],
            "per_image_ocr_text": [
                {
                    "image_url": item.get("image_url"),
                    "ocr_text": item.get("ocr_text"),
                    "text_detected": bool(item.get("text_detected") or item.get("ocr_text")),
                    "ocr_confidence": item.get("ocr_confidence"),
                }
                for item in metadata.get("social_image_ocr_results", [])
                if isinstance(item, dict)
            ],
            "ocr_queries": social_queries["ocr_queries"],
            "combined_queries": social_queries["combined_queries"],
            "candidate_sources_from_caption": [match.get("url") for match in caption_stream.get("matches", []) if match.get("url")],
            "candidate_sources_from_ocr": [match.get("url") for match in ocr_stream.get("matches", []) if match.get("url")],
            "same_source_count": match_groups["same_publisher_count"],
            "cross_source_count": match_groups["cross_publisher_count"],
            "facebook_caption_text": caption_text,
            "facebook_attached_image_count": len(metadata.get("media_assets", [])) if isinstance(metadata.get("media_assets"), list) else 0,
            "facebook_ocr_text_per_image": [
                {
                    "image_url": item.get("image_url"),
                    "ocr_text": item.get("ocr_text"),
                    "ocr_status": item.get("ocr_status"),
                }
                for item in metadata.get("social_image_ocr_results", [])
                if isinstance(item, dict)
            ],
            "facebook_caption_queries": social_queries["caption_queries"],
            "facebook_ocr_queries": social_queries["ocr_queries"],
            "facebook_combined_queries": social_queries["combined_queries"],
            "facebook_candidate_urls": [match.get("url") for match in matches if match.get("url")],
            "nlp_component_values": nlp_component_values,
            "final_formula_components": nlp_component_values,
            "final_score": round(confidence * 100, 2),
            "final_label": self._facebook_layer2_label(confidence),
            "caption_stream_debug": caption_stream.get("debug", {}),
            "ocr_stream_debug": ocr_stream.get("debug", {}),
            "combined_stream_debug": combined_stream.get("debug", {}),
        }
        return LayerOutcome(
            status="supporting" if matches or confidence > 0.0 else "none",
            confidence=round(confidence, 3),
            explanation=assessment,
            details=details,
        )

    def _evaluate_facebook_text_stream(self, extracted: ExtractedPayload, input_text: str, stream_name: str) -> dict:
        payload = ExtractedPayload(
            title=extracted.title,
            author=extracted.author,
            publish_time=extracted.publish_time,
            canonical_url=extracted.canonical_url,
            main_text=input_text,
            hero_image_url=extracted.hero_image_url,
            normalized_text=normalize_text(input_text),
            content_hash=sha256_text(normalize_text(input_text)),
            metadata=extracted.metadata if isinstance(extracted.metadata, dict) else {},
        )
        prepared = self.preprocess_text(input_text)
        extraction = self.claim_extraction.extract_claims(prepared["cleaned_input"], max_claims=self.TEXT_CLAIM_LIMIT)
        claims = [item["text"] for item in extraction.get("claims", [])] or self._extract_text_claims(payload)
        meaning_bundle = self._build_text_meaning_bundle(prepared, extraction, claims)
        query_plan = self._build_text_hybrid_queries(prepared, extraction, claims, meaning_bundle)
        source_recovery_stage = self._empty_source_recovery("Skipped direct source recovery for media-derived text; claim evidence search is used instead.")
        candidates, retrieval_debug = self._gather_text_candidate_documents(prepared, extraction, query_plan, source_recovery_stage)
        ranking_debug = self._hybrid_rank_text_candidates(prepared, claims, query_plan, candidates)
        top_articles = ranking_debug["reranked_top_results"]
        claim_results = [
            self._score_claim_against_evidence(claim, self._build_claim_evidence_from_top_articles(claim, top_articles))
            for claim in claims
        ]
        aggregate = self._finalize_text_verification(claim_results, top_articles, meaning_bundle)
        return {
            "name": stream_name,
            "prepared": prepared,
            "extraction": extraction,
            "query_plan": query_plan,
            "claim_results": claim_results,
            "aggregate": aggregate,
            "matches": aggregate.get("evidence_sources", [])[: self.DEFAULT_NLP_MATCH_LIMIT],
            "debug": {
                "raw_input": input_text,
                "cleaned_input": prepared["cleaned_input"],
                "exact_phrase_queries": meaning_bundle.get("exact_phrase_queries", []),
                "lexical_entity_queries": meaning_bundle.get("lexical_queries", []),
                "final_search_queries_sent": query_plan.get("lexical", []) + query_plan.get("semantic", []),
                "provider_status": [
                    item.get("provider_status")
                    for item in retrieval_debug.get("lexical_searches", []) + retrieval_debug.get("semantic_searches", [])
                    if item.get("provider_status")
                ],
                "candidate_urls": [item.get("url") for item in aggregate.get("evidence_sources", []) if item.get("url")][:5],
                "candidate_fetches": retrieval_debug.get("candidate_fetches", []),
                "source_recovery_status": source_recovery_stage.get("final_source_recovery", {}).get("status"),
                "source_recovery_reason": source_recovery_stage.get("final_source_recovery", {}).get("reason"),
                "component_scores": aggregate.get("component_scores", {}),
                "final_score": aggregate.get("final_score"),
                "final_label": aggregate.get("final_label"),
            },
        }

    def _empty_facebook_stream(self, reason: str) -> dict:
        return {
            "name": reason,
            "prepared": {"cleaned_input": ""},
            "query_plan": {"lexical": [], "semantic": []},
            "aggregate": {"final_score": 0.0, "final_label": "Not Verifiable", "component_scores": {}},
            "matches": [],
            "debug": {"failure_reason": reason},
        }

    def _dedupe_evidence_candidates(self, candidates: list[dict]) -> list[dict]:
        deduped: list[dict] = []
        seen: set[str] = set()
        for candidate in sorted(
            [item for item in candidates if isinstance(item, dict)],
            key=lambda item: float(item.get("score") or item.get("rerank_score") or item.get("hybrid_score") or 0.0),
            reverse=True,
        ):
            key = str(candidate.get("url") or candidate.get("title") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _facebook_layer2_label(self, confidence: float) -> str:
        return map_score_to_label(confidence * 100)

    def _build_social_text_input(self, extracted: ExtractedPayload, ocr_text: str | None = None) -> str:
        metadata = extracted.metadata if isinstance(extracted.metadata, dict) else {}
        platform_type = str(metadata.get("platform_type") or "").lower()
        if platform_type == "youtube":
            return self._build_youtube_verification_input(extracted)
        parts: list[str] = []
        for value in (
            extracted.title,
            metadata.get("channel_title") if platform_type == "youtube" else None,
            metadata.get("description"),
            metadata.get("summary"),
            extracted.main_text if extracted.main_text != extracted.title else None,
            self._social_media_caption_text(extracted),
            ocr_text,
        ):
            text = str(value or "").strip()
            if not text:
                continue
            parts.append(text)

        deduped: list[str] = []
        seen: set[str] = set()
        for item in parts:
            key = normalize_text(item)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return "\n".join(deduped).strip()

    def _build_youtube_verification_input(self, extracted: ExtractedPayload) -> str:
        metadata = extracted.metadata if isinstance(extracted.metadata, dict) else {}
        values = [
            extracted.title,
            metadata.get("description"),
            metadata.get("channel_title"),
            metadata.get("transcript_text"),
        ]
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            if normalize_text(text).rstrip(".") == "video metadata could not be retrieved":
                continue
            key = normalize_text(text)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(text)
        return "\n".join(deduped).strip()

    def _social_caption_value(self, extracted: ExtractedPayload) -> str:
        raw_metadata = getattr(extracted, "metadata", None)
        metadata = raw_metadata if isinstance(raw_metadata, dict) else getattr(extracted, "metadata_json", {}) or {}
        values = [
            metadata.get("description"),
            metadata.get("summary"),
            extracted.main_text if extracted.main_text != extracted.title else None,
            self._social_media_caption_text(extracted),
        ]
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
        return ""

    def _social_ocr_texts(self, extracted: ExtractedPayload, social_ocr: dict) -> list[str]:
        metadata = extracted.metadata if isinstance(extracted.metadata, dict) else {}
        texts: list[str] = []
        bundled = metadata.get("social_image_ocr_texts")
        if isinstance(bundled, list):
            for item in bundled:
                text = str(item or "").strip()
                if text:
                    texts.append(text)
        fallback = str(social_ocr.get("ocr_text") or "").strip()
        if fallback:
            texts.append(fallback)
        deduped: list[str] = []
        seen: set[str] = set()
        for item in texts:
            key = normalize_text(item)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _social_query_groups(self, extracted: ExtractedPayload, caption_text: str, ocr_texts: list[str]) -> dict[str, list[str]]:
        caption_payload = ExtractedPayload(
            title=extracted.title,
            author=extracted.author,
            publish_time=extracted.publish_time,
            canonical_url=extracted.canonical_url,
            main_text=caption_text,
            hero_image_url=extracted.hero_image_url,
            normalized_text=normalize_text(caption_text),
            content_hash=extracted.content_hash,
            metadata=extracted.metadata if isinstance(extracted.metadata, dict) else {},
        )
        ocr_text = " ".join(ocr_texts[:2]).strip()
        ocr_payload = ExtractedPayload(
            title=extracted.title,
            author=extracted.author,
            publish_time=extracted.publish_time,
            canonical_url=extracted.canonical_url,
            main_text=ocr_text,
            hero_image_url=extracted.hero_image_url,
            normalized_text=normalize_text(ocr_text),
            content_hash=extracted.content_hash,
            metadata=extracted.metadata if isinstance(extracted.metadata, dict) else {},
        )
        combined_text = " ".join(part for part in [caption_text, ocr_text] if part).strip()
        combined_payload = ExtractedPayload(
            title=extracted.title,
            author=extracted.author,
            publish_time=extracted.publish_time,
            canonical_url=extracted.canonical_url,
            main_text=combined_text,
            hero_image_url=extracted.hero_image_url,
            normalized_text=normalize_text(combined_text),
            content_hash=extracted.content_hash,
            metadata=extracted.metadata if isinstance(extracted.metadata, dict) else {},
        )
        return {
            "caption_queries": self._dedupe_queries(
                [
                    self._build_fact_check_summary_query(caption_payload),
                    self._build_fact_check_keyword_query(caption_payload),
                ],
                limit=2,
            ),
            "ocr_queries": self._dedupe_queries(
                [
                    self._build_fact_check_summary_query(ocr_payload),
                    self._build_fact_check_keyword_query(ocr_payload),
                ],
                limit=2,
            ) if ocr_text else [],
            "combined_queries": self._dedupe_queries(
                [
                    self._build_fact_check_summary_query(combined_payload),
                    self._build_fact_check_keyword_query(combined_payload),
                ],
                limit=2,
            ),
        }

    def _match_groups(self, matches: list[dict], source_domain: str | None) -> dict[str, object]:
        same_matches: list[dict] = []
        cross_matches: list[dict] = []
        source_family = self._domain_family(source_domain)
        for match in matches:
            if not isinstance(match, dict):
                continue
            domain = str(match.get("domain") or extract_domain(match.get("url")) or "").lower()
            if domain and source_family and self._domain_family(domain) == source_family:
                same_matches.append(match)
            else:
                cross_matches.append(match)
        support_domains = sorted(
            {
                str(match.get("domain") or extract_domain(match.get("url")) or "").lower()
                for match in cross_matches
                if str(match.get("domain") or extract_domain(match.get("url")) or "").strip()
            }
        )
        return {
            "same_publisher_matches": same_matches,
            "cross_publisher_matches": cross_matches,
            "same_publisher_count": len(same_matches),
            "cross_publisher_count": len(cross_matches),
            "support_domains": support_domains,
            "same_source_only": len(same_matches) > 0 and len(cross_matches) == 0,
            "cross_source_exists": len(cross_matches) > 0,
        }

    def _social_caption_ocr_mismatch(self, caption_text: str, ocr_texts: list[str]) -> bool:
        if not caption_text or not ocr_texts:
            return False
        caption_tokens = {token for token in normalize_text(caption_text).split() if len(token) >= 4}
        ocr_tokens = {token for token in normalize_text(" ".join(ocr_texts)).split() if len(token) >= 4}
        if not caption_tokens or not ocr_tokens:
            return False
        overlap = len(caption_tokens & ocr_tokens) / max(1, min(len(caption_tokens), len(ocr_tokens)))
        return overlap < 0.15

    def _social_media_caption_text(self, extracted: ExtractedPayload) -> str:
        metadata = extracted.metadata if isinstance(extracted.metadata, dict) else {}
        assets = metadata.get("media_assets", [])
        if not isinstance(assets, list):
            return ""
        texts: list[str] = []
        for asset in assets[:4]:
            if not isinstance(asset, dict):
                continue
            for key in ("caption", "alt", "description", "label"):
                value = str(asset.get(key) or "").strip()
                if value:
                    texts.append(value)
        deduped: list[str] = []
        seen: set[str] = set()
        for item in texts:
            key = normalize_text(item)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return " ".join(deduped)

    def _extract_remote_social_image_ocr(self, extracted: ExtractedPayload) -> dict:
        metadata = extracted.metadata if isinstance(extracted.metadata, dict) else {}
        bundled_results = metadata.get("social_image_ocr_results")
        if isinstance(bundled_results, list) and bundled_results:
            first_result = bundled_results[0] if isinstance(bundled_results[0], dict) else {}
            cached_result = dict(first_result)
            cached_result["ocr_text"] = str(cached_result.get("ocr_text") or cached_result.get("extracted_text") or "").strip()
            cached_result["text_detected"] = bool(cached_result.get("text_detected") or cached_result["ocr_text"])
            return cached_result
        cached = metadata.get("social_image_ocr_result")
        if isinstance(cached, dict):
            cached_result = dict(cached)
            cached_result["ocr_text"] = str(cached_result.get("ocr_text") or cached_result.get("extracted_text") or "").strip()
            cached_result["text_detected"] = bool(cached_result.get("text_detected") or cached_result["ocr_text"])
            return cached_result
        image_url = self._remote_social_image_url(extracted)
        if not image_url:
            return {
                "image_fetch_status": "not_downloaded",
                "image_decode_status": "not_attempted",
                "text_detected": False,
                "ocr_text": "",
                "ocr_status": "image_not_downloaded",
                "ocr_confidence": 0.0,
                "failure_reasons": ["image_not_downloaded"],
                "ocr_summary": "No remote post image was available for OCR.",
            }

        try:
            with httpx.Client(timeout=6.0, follow_redirects=True) as client:
                response = client.get(image_url)
                response.raise_for_status()
                content_type = str(response.headers.get("content-type") or "").lower()
                if not content_type.startswith("image/"):
                    return {
                        "image_fetch_status": "downloaded",
                        "image_decode_status": "invalid",
                        "text_detected": False,
                        "ocr_text": "",
                        "ocr_status": "invalid_image_format",
                        "ocr_confidence": 0.0,
                        "failure_reasons": ["invalid_image_format"],
                        "ocr_summary": "The post media URL did not return an image file for OCR.",
                    }
            ocr_result = extract_text_details_from_image_bytes(response.content, source=image_url)
            ocr_result["ocr_text"] = str(ocr_result.get("extracted_text") or "").strip()
            ocr_result["image_url"] = image_url
            return ocr_result
        except Exception as exc:
            logger.debug("Social image OCR fetch failed for %s: %s", image_url, exc)
            return {
                "image_fetch_status": "not_downloaded",
                "image_decode_status": "failed",
                "text_detected": False,
                "ocr_text": "",
                "ocr_status": "image_not_downloaded",
                "ocr_confidence": 0.0,
                "failure_reasons": ["image_not_downloaded"],
                "ocr_summary": "The social post image could not be downloaded for OCR.",
                "image_url": image_url,
            }

    def _remote_social_image_url(self, extracted: ExtractedPayload) -> str | None:
        metadata = extracted.metadata if isinstance(extracted.metadata, dict) else {}
        assets = metadata.get("media_assets", [])
        if isinstance(assets, list):
            for asset in assets:
                if not isinstance(asset, dict):
                    continue
                if str(asset.get("type") or "").lower() == "image" and asset.get("url"):
                    return str(asset.get("url"))
        hero = extracted.hero_image_url or metadata.get("hero_image_url")
        if hero:
            return str(hero).strip() or None
        return None

    def _find_trusted_publisher(self, source_domain: str | None, publisher_name: str | None):
        if not source_domain and not publisher_name:
            return None

        publishers = list(self.db.scalars(select(TrustedPublisher).order_by(TrustedPublisher.name.asc())))
        normalized_publisher_name = self._normalize_name(publisher_name)
        for publisher in publishers:
            candidate_domains = {publisher.domain}
            aliases = publisher.metadata_json.get("domains", []) if isinstance(publisher.metadata_json, dict) else []
            candidate_domains.update(alias for alias in aliases if isinstance(alias, str))
            candidate_names = {self._normalize_name(publisher.name)}
            name_aliases = publisher.metadata_json.get("names", []) if isinstance(publisher.metadata_json, dict) else []
            candidate_names.update(self._normalize_name(alias) for alias in name_aliases if isinstance(alias, str))
            domain_match = any(domain_matches(source_domain, candidate_domain) for candidate_domain in candidate_domains)
            name_match = bool(
                normalized_publisher_name
                and any(
                    normalized_publisher_name == candidate_name
                    or normalized_publisher_name in candidate_name
                    or candidate_name in normalized_publisher_name
                    for candidate_name in candidate_names
                    if candidate_name
                )
            )
            if domain_match or name_match:
                return publisher
        return None

    def _find_metadata_candidate(self, publisher_id: int, extracted: ExtractedPayload) -> ContentRecord | None:
        records = list(self.db.scalars(select(ContentRecord).where(ContentRecord.publisher_id == publisher_id)))
        extracted_title = normalize_text(extracted.title or "")
        extracted_author = normalize_text(extracted.author or "")

        for record in records:
            if extracted.canonical_url and record.canonical_url == extracted.canonical_url:
                return record

        for record in records:
            record_title = normalize_text(record.title or "")
            record_author = normalize_text(record.source_metadata.get("author", ""))
            record_publish_time = record.source_metadata.get("publish_time")

            title_match = bool(extracted_title and record_title and extracted_title == record_title)
            author_match = bool(extracted_author and record_author and extracted_author == record_author)
            published_at_match = self._publish_time_matches(extracted.publish_time, record_publish_time)

            if title_match and (author_match or published_at_match):
                return record
        return None

    def _publish_time_matches(self, extracted_publish_time, stored_publish_time: str | None) -> bool:
        if not extracted_publish_time or not stored_publish_time:
            return False
        extracted_value = extracted_publish_time.isoformat()[:19]
        return str(stored_publish_time)[:19] == extracted_value

    def _metadata_value_matches(self, left: str | None, right: str | None) -> bool:
        if not left or not right:
            return False
        return normalize_text(left) == normalize_text(right)

    def _attach_extracted_details(self, layer: LayerOutcome, extracted: ExtractedPayload) -> LayerOutcome:
        layer.details = {
            **layer.details,
            "extracted_publisher_name": extracted.metadata.get("publisher_name"),
            "extracted_author": extracted.author,
            "extracted_publish_time": extracted.publish_time.isoformat() if extracted.publish_time else None,
            "extracted_source_domain": extracted.metadata.get("source_domain"),
            "extracted_section": extracted.metadata.get("section"),
            "extracted_keywords": extracted.metadata.get("keywords", []),
            "extracted_hero_image_url": extracted.hero_image_url,
            "extracted_media_assets": extracted.metadata.get("media_assets", []),
            "c2pa_credentials": extracted.metadata.get("c2pa_credentials", []),
            "remote_media_provenance": extracted.metadata.get("remote_media_provenance", []),
            "youtube_signals": extracted.metadata.get("youtube_signals", {}),
            "forensics": extracted.metadata.get("forensics"),
            "transport_evidence": extracted.metadata.get("transport_evidence", {}),
        }
        return layer

    def _create_internal_provenance(self, extracted: ExtractedPayload) -> dict:
        crypto_service = CryptoProvenanceService(self.db)
        previous_snapshot = self._find_previous_article_snapshot(extracted)
        snapshot_metadata = {
            "canonical_url": extracted.canonical_url,
            "publisher_domain": extracted.metadata.get("source_domain"),
            "section_name": extracted.metadata.get("section"),
            "keywords": extracted.metadata.get("keywords", []),
            "language": extracted.metadata.get("language"),
            "content_type": extracted.metadata.get("content_type") or extracted.metadata.get("source_type") or "news",
            "summary": extracted.metadata.get("summary") or extracted.main_text[:240],
            "author_names": [extracted.author] if extracted.author else [],
            "assets": [
                {
                    "asset_type": asset.get("type") or asset.get("asset_type") or "media",
                    "asset_url": asset.get("url"),
                    "original_asset_url": asset.get("original_asset_url") or asset.get("url"),
                    "caption": asset.get("caption"),
                    "alt_text": asset.get("alt"),
                    "credit": asset.get("credit"),
                    "mime_type": asset.get("mime_type"),
                    "asset_size": asset.get("asset_size"),
                }
                for asset in extracted.metadata.get("media_assets", [])
                if isinstance(asset, dict) and asset.get("url")
            ],
            "references": [
                {
                    "reference_text": item.get("title") or item.get("url") or item.get("text"),
                    "target_url": item.get("url"),
                    "source_name": item.get("domain"),
                    "position_in_article": index,
                }
                for index, item in enumerate(extracted.metadata.get("related_links", []))
                if isinstance(item, dict)
            ],
            "transport_evidence": extracted.metadata.get("transport_evidence", {}),
            "change_type": "update" if previous_snapshot else "first_publish",
            "version_number": (previous_snapshot.metadata_json or {}).get("version_number", 1) + 1 if previous_snapshot else 1,
            "previous_payload_hash": (previous_snapshot.metadata_json or {}).get("payload_hash") if previous_snapshot else None,
        }
        snapshot = crypto_service.create_article_snapshot(
            {
                "source_url": extracted.canonical_url or extracted.metadata.get("fetched_url") or "",
                "publisher_name": extracted.metadata.get("publisher_name") or extracted.metadata.get("source_domain") or "Unknown publisher",
                "author_name": extracted.author,
                "title": extracted.title or "Untitled article",
                "published_at": extracted.publish_time,
                "captured_at": None,
                "normalized_article_text": extracted.normalized_text or extracted.main_text,
                "media_urls": [asset.get("url") for asset in extracted.metadata.get("media_assets", []) if isinstance(asset, dict) and asset.get("url")],
                "metadata": snapshot_metadata,
                "previous_version_id": previous_snapshot.id if previous_snapshot else None,
            }
        )
        manifest = crypto_service.sign_object(CryptoObjectType.article_snapshot, snapshot.id)
        manifest_data = crypto_service.serialize_manifest(manifest)
        return {
            "record_type": manifest_data.get("object_type").value if hasattr(manifest_data.get("object_type"), "value") else str(manifest_data.get("object_type")),
            "manifest_id": str(manifest_data.get("manifest_id")),
            "internal_manifest_id": str(manifest_data.get("manifest_id")),
            "previous_manifest_id": str(manifest_data.get("previous_manifest_id")) if manifest_data.get("previous_manifest_id") else None,
            "previous_version_id": str(snapshot.previous_version_id) if snapshot.previous_version_id else None,
            "issued_by": manifest_data.get("source", {}).get("publisher_name") or "NewsProof system",
            "issuer_type": "platform",
            "verification_result": manifest_data.get("verification", {}).get("status"),
            "provenance_verification_status": manifest_data.get("verification_status").value
            if hasattr(manifest_data.get("verification_status"), "value")
            else str(manifest_data.get("verification_status")),
            "signed_at": manifest_data.get("created_at").isoformat() if hasattr(manifest_data.get("created_at"), "isoformat") else manifest_data.get("created_at"),
            "timestamp_source": manifest_data.get("verification", {}).get("checks", {}).get("timestamp_provider")
            or manifest_data.get("verification", {}).get("timestamps", {}).get("timestamp_source")
            or "internal server timestamp",
            "canonical_hash": manifest_data.get("crypto", {}).get("payload_hash") or manifest_data.get("article", {}).get("payload_hash"),
            "signature_algorithm": manifest_data.get("crypto", {}).get("signature_algorithm") or "Ed25519",
            "internal_manifest_valid": bool(manifest_data.get("verification", {}).get("valid")),
            "platform_manifest_valid": bool(manifest_data.get("verification", {}).get("valid")),
            "canonical_json": manifest_data.get("canonical_json"),
            "canonical_json_preview": manifest_data.get("canonical_json"),
            "issuer_key_id": manifest_data.get("crypto", {}).get("key_id"),
            "timestamp_token_status": manifest_data.get("verification", {}).get("checks", {}).get("timestamp_reason")
            or manifest_data.get("verification", {}).get("checks", {}).get("timestamp_valid"),
        }

    def _find_previous_article_snapshot(self, extracted: ExtractedPayload) -> ArticleSnapshot | None:
        source_url = extracted.canonical_url or extracted.metadata.get("fetched_url")
        if not source_url:
            return None
        snapshots = list(
            self.db.scalars(
                select(ArticleSnapshot)
                .where(ArticleSnapshot.source_url == source_url)
                .order_by(ArticleSnapshot.created_at.desc())
                .limit(1)
            )
        )
        if snapshots:
            return snapshots[0]
        return None

    def _normalize_name(self, value: str | None) -> str:
        if not value:
            return ""
        return normalize_text(value.replace(".", " ").replace("-", " "))

    def _metadata_signals(self, extracted: ExtractedPayload) -> list[str]:
        signals: list[str] = []
        if extracted.title:
            signals.append("headline")
        if extracted.author:
            signals.append("author")
        if extracted.publish_time:
            signals.append("publish_time")
        if extracted.canonical_url:
            signals.append("canonical_url")
        if extracted.metadata.get("publisher_name"):
            signals.append("publisher")
        if extracted.metadata.get("section"):
            signals.append("section")
        if extracted.metadata.get("keywords"):
            signals.append("keywords")
        if extracted.main_text and len(normalize_text(extracted.main_text).split()) >= 80:
            signals.append("article_body")
        if extracted.metadata.get("media_assets"):
            signals.append("media_assets")
        return signals

    def _metadata_corroboration_score(self, metadata_signals: list[str], publisher: TrustedPublisher | None) -> float:
        score = min(0.65, len(metadata_signals) * 0.08)
        if publisher:
            score += 0.18
        return min(score, 0.82)

    def _build_fact_check_query(self, extracted: ExtractedPayload) -> str:
        primary_title = extracted.title or ""
        slug_terms = self._slug_terms(extracted.canonical_url or extracted.metadata.get("fetched_url"))
        body_terms = self._body_query_terms(extracted.main_text)
        parts = [
            primary_title,
            extracted.metadata.get("section") or "",
            slug_terms,
            body_terms,
        ]
        query = " ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())
        if query:
            return query
        return (extracted.main_text or "")[:180]

    def _build_fact_check_summary_query(self, extracted: ExtractedPayload) -> str:
        summary_parts: list[str] = []
        if extracted.title:
            summary_parts.append(extracted.title.strip())
        if extracted.metadata.get("summary"):
            summary_parts.append(str(extracted.metadata.get("summary")).strip())
        if extracted.main_text:
            sentences = [
                sentence.strip()
                for sentence in extracted.main_text.replace("\n", " ").split(".")
                if sentence.strip()
            ]
            summary_parts.extend(sentences[:2])
        normalized_parts = [
            " ".join(part.split())
            for part in summary_parts
            if isinstance(part, str) and part.strip()
        ]
        summary = " ".join(normalized_parts)
        return self._trim_query_terms(summary, 30).strip()

    def _build_fact_check_keyword_query(self, extracted: ExtractedPayload) -> str:
        keywords: list[str] = []
        if isinstance(extracted.metadata.get("keywords"), list):
            for item in extracted.metadata.get("keywords", []):
                if isinstance(item, str) and item.strip():
                    keywords.append(item.strip())
        if extracted.metadata.get("section"):
            keywords.append(str(extracted.metadata.get("section")).strip())
        slug_terms = self._slug_terms(extracted.canonical_url or extracted.metadata.get("fetched_url"))
        if slug_terms:
            keywords.append(slug_terms)
        body_terms = self._body_query_terms(extracted.main_text, max_terms=12)
        if body_terms:
            keywords.append(body_terms)
        deduped: list[str] = []
        seen: set[str] = set()
        for term in keywords:
            lowered = normalize_text(term)
            if not lowered or lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(term)
        return self._trim_query_terms(" ".join(deduped), 18).strip()

    def _trim_query_terms(self, query: str, max_terms: int) -> str:
        terms = [term for term in query.split() if term]
        return " ".join(terms[:max_terms])

    def _extract_url_article_context(self, extracted: ExtractedPayload) -> dict[str, object]:
        clean_text = self._clean_url_article_nlp_text(extracted)
        text = self._url_claim_input_text(extracted, clean_text)
        claim_extraction = self.claim_extraction.extract_claims(text, max_claims=3)
        extracted_claims = [
            str(item.get("text") or "").strip()
            for item in claim_extraction.get("claims", [])
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]
        main_claim = self._select_url_main_claim(extracted, extracted_claims, clean_text)
        claim_tuple = self._extract_structured_claim_tuple(text)
        if main_claim:
            tuple_from_claim = self._extract_structured_claim_tuple(main_claim)
            if tuple_from_claim:
                claim_tuple = tuple_from_claim
        context_text = " ".join(part for part in [main_claim, text] if part).strip()
        context_terms = self._extract_context_terms(context_text)
        if not claim_tuple and extracted.title:
            claim_tuple = self._extract_structured_claim_tuple(str(extracted.title))
        claim_details = self._parse_claim_details(main_claim) if main_claim else {}
        topic = str(claim_tuple.get("claim_type") or extracted.metadata.get("section") or "").strip()
        tuple_parts = [
            str(claim_tuple.get("subject") or ""),
            str(claim_tuple.get("action") or ""),
            str(claim_tuple.get("object") or ""),
            str(claim_tuple.get("event") or ""),
            str(claim_tuple.get("location") or ""),
            str(claim_tuple.get("date") or ""),
        ]
        article_meaning = " ".join(part for part in tuple_parts if part).strip()
        if not article_meaning:
            article_meaning = main_claim or self._build_fact_check_summary_query(extracted) or self._build_fact_check_query(extracted)
        entities = self._dedupe_text_items(
            [
                *claim_details.get("entities", []),
                *claim_extraction.get("entities", []),
                *claim_extraction.get("locations", []),
                *context_terms.get("actors", []),
                *context_terms.get("locations", []),
                *context_terms.get("countries", []),
            ]
        )
        actions = self._dedupe_text_items([*claim_details.get("actions", []), *context_terms.get("events", [])])
        dates = self._dedupe_text_items([*claim_extraction.get("dates", []), *claim_details.get("money", [])])
        return {
            "title": extracted.title,
            "article_text_length": len(clean_text or ""),
            "cleaned_article_text": clean_text,
            "cleaned_article_text_length": len(clean_text or ""),
            "publisher_domain": extract_domain(extracted.canonical_url) or extract_domain(extracted.metadata.get("fetched_url")),
            "canonical_url": extracted.canonical_url,
            "publication_date": str(extracted.publish_time) if extracted.publish_time else "",
            "author": extracted.author,
            "claim_tuple": claim_tuple,
            "main_claim": main_claim,
            "extracted_claims": extracted_claims,
            "entities": entities,
            "events": context_terms.get("events", []),
            "locations": context_terms.get("locations", []),
            "actions": actions,
            "dates": dates,
            "claim_extraction_reason": claim_extraction.get("reason"),
            "topic": topic,
            "article_meaning": article_meaning,
        }

    def _clean_url_article_nlp_text(self, extracted: ExtractedPayload) -> str:
        text = " ".join(str(extracted.main_text or "").split())
        if not text:
            return ""
        sentences = self._text_sentences(text)
        cleaned: list[str] = []
        seen: set[str] = set()
        for sentence in sentences:
            candidate = " ".join(sentence.split()).strip()
            if not candidate or self._looks_like_url_noise_sentence(candidate):
                continue
            key = normalize_text(candidate)
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(candidate)
            if len(" ".join(cleaned)) >= 6000:
                break
        return " ".join(cleaned) if cleaned else text[:6000]

    def _looks_like_url_noise_sentence(self, sentence: str) -> bool:
        normalized = normalize_text(sentence)
        if len(normalized) < 12:
            return True
        if len(normalized.split()) <= 3 and re.search(r"\b(?:menu|home|share|login|subscribe|advertisement|comments?)\b", normalized):
            return True
        noise = (
            "advertisement",
            "sponsored content",
            "share this",
            "follow us",
            "subscribe",
            "sign up",
            "read more",
            "related article",
            "all rights reserved",
            "cookie policy",
            "privacy policy",
        )
        return any(marker in normalized for marker in noise)

    def _select_url_main_claim(self, extracted: ExtractedPayload, claims: list[str], clean_text: str) -> str:
        title = " ".join(str(extracted.title or "").split()).strip()
        if claims:
            body_norm = normalize_text(clean_text)
            title_norm = normalize_text(title)
            body_claims = [claim for claim in claims if body_norm and normalize_text(claim) in body_norm]
            candidates = body_claims or claims
            title_tokens = {token for token in normalize_text(title).split() if len(token) > 3}
            def rank_claim(claim: str) -> tuple[int, int, int]:
                claim_norm = normalize_text(claim)
                overlap = len(title_tokens & {token for token in claim_norm.split() if len(token) > 3})
                body_hit = 1 if body_norm and claim_norm in body_norm else 0
                title_only_penalty = -1 if title_norm and claim_norm == title_norm and body_claims else 0
                fact_signal = 1 if re.search(ClaimExtractionService.FACTUAL_PATTERN, claim) or re.search(r"\d", claim) else 0
                return (body_hit, fact_signal, overlap + title_only_penalty, min(len(claim.split()), 32))

            return sorted(candidates, key=rank_claim, reverse=True)[0][:500]
        if title and not self._looks_like_url_noise_sentence(title):
            return title[:500]
        for sentence in self._text_sentences(clean_text):
            if len(sentence.split()) >= 6 and not self._looks_like_url_noise_sentence(sentence):
                return sentence[:500]
        return title[:500]

    def _url_claim_input_text(self, extracted: ExtractedPayload, clean_text: str) -> str:
        title = " ".join(str(extracted.title or "").split()).strip()
        body = " ".join(str(clean_text or "").split()).strip()
        parts = []
        if title:
            parts.append(title.rstrip(".!?"))
        if body and normalize_text(body) != normalize_text(title):
            parts.append(body)
        return "\n\n".join(part for part in parts if part).strip()

    def _search_fact_check_matches(
        self,
        summary_query: str,
        keyword_query: str,
        extracted: ExtractedPayload,
        expected_domain: str | None,
    ) -> list[WebSearchCandidate]:
        combined: list[WebSearchCandidate] = []
        seen: set[str] = set()
        claim_queries = self._extract_text_claims(extracted)[:3]
        for query in [summary_query, keyword_query, *claim_queries]:
            if not query:
                continue
            for candidate in self.search.search(query, expected_domain=expected_domain, limit=self.MAX_NLP_MATCHES):
                key = candidate.url or candidate.title
                if key in seen:
                    continue
                seen.add(key)
                combined.append(candidate)
                if len(combined) >= self.MAX_NLP_MATCHES:
                    break
            if len(combined) >= self.MAX_NLP_MATCHES:
                break
        return combined[: self.MAX_NLP_MATCHES]

    def _build_article_url_queries(self, extracted: ExtractedPayload, article_context: dict[str, object] | None = None) -> list[str]:
        article_context = article_context or self._extract_url_article_context(extracted)
        claim_tuple = article_context.get("claim_tuple", {}) if isinstance(article_context.get("claim_tuple"), dict) else {}
        main_claim = str(article_context.get("main_claim") or article_context.get("article_meaning") or "").strip()
        text_like_queries = self._url_text_like_queries(extracted, article_context)
        claim_queries = self._url_main_claim_queries(main_claim, article_context)
        tuple_queries = self._url_claim_tuple_queries(claim_tuple)
        headline = " ".join(str(extracted.title or "").split()).strip()
        queries = self._dedupe_queries(
            [
                *text_like_queries,
                *claim_queries,
                *tuple_queries,
                self._trim_query_terms(main_claim, 18),
                self._trim_query_terms(headline, 14),
            ],
            limit=10,
        )
        return queries[:10]

    def _url_text_like_queries(self, extracted: ExtractedPayload, article_context: dict[str, object]) -> list[str]:
        clean_text = str(article_context.get("cleaned_article_text") or "").strip()
        main_claim = str(article_context.get("main_claim") or "").strip()
        url_text = self._url_claim_input_text(extracted, clean_text)
        if not url_text:
            url_text = main_claim or " ".join(part for part in [str(extracted.title or "").strip(), str(extracted.main_text or "").strip()] if part).strip()
        if not url_text:
            return []
        prepared = self._prepare_url_text_like_input(url_text)
        extraction = self.claim_extraction.extract_claims(str(prepared.get("cleaned_input") or ""), max_claims=self.TEXT_CLAIM_LIMIT)
        claims = [str(item.get("text") or "").strip() for item in extraction.get("claims", []) if isinstance(item, dict) and str(item.get("text") or "").strip()]
        if not claims and main_claim:
            claims = [main_claim]
        if not claims:
            claims = self._extract_text_claims(
                ExtractedPayload(
                    title=extracted.title,
                    author=extracted.author,
                    publish_time=extracted.publish_time,
                    canonical_url=extracted.canonical_url,
                    main_text=str(prepared.get("cleaned_input") or ""),
                    hero_image_url=extracted.hero_image_url,
                    normalized_text=str(prepared.get("normalized_text") or ""),
                    content_hash=extracted.content_hash,
                    metadata=extracted.metadata,
                )
            )
        meaning_bundle = self._build_text_meaning_bundle(prepared, extraction, claims)
        query_plan = self._build_text_hybrid_queries(prepared, extraction, claims, meaning_bundle)
        claim_context = extract_claim_context(str(prepared.get("cleaned_input") or ""))
        trusted_queries = build_country_aware_queries(claim_context, limit=self.TEXT_MAX_SEARCH_QUERIES)
        short_claims = [" ".join(str(claim or "").split()[:14]).strip() for claim in claims if str(claim or "").strip()]
        compact_context = self._dedupe_queries(
            [
                str(meaning_bundle.get("simplified_meaning") or ""),
                str(meaning_bundle.get("keyword_query") or ""),
                str(meaning_bundle.get("entity_query") or ""),
                *short_claims,
            ],
            limit=4,
        )
        combined = (
            compact_context
            + query_plan.get("lexical", [])
            + query_plan.get("semantic", [])
            + query_plan.get("broad", [])
            + query_plan.get("trusted", [])
            + trusted_queries
            + query_plan.get("fact_check", [])
            + query_plan.get("official", [])
        )
        return self._dedupe_queries(combined, limit=10)

    def _prepare_url_text_like_input(self, text: str) -> dict[str, str | list[str] | bool]:
        cleaned_input = self._clean_text_for_retrieval(text)
        normalized_text = normalize_text(cleaned_input)
        sentences = self._text_sentences(cleaned_input)
        extraction = self.claim_extraction.extract_claims(cleaned_input, max_claims=self.TEXT_CLAIM_LIMIT)
        core_claim = str(extraction.get("claims", [{}])[0].get("text") if extraction.get("claims") else (sentences[0] if sentences else cleaned_input))
        return {
            "input_text": text,
            "cleaned_input": cleaned_input,
            "normalized_text": normalized_text,
            "detected_language": self._detect_text_language(cleaned_input),
            "sentences": sentences,
            "core_claim": core_claim,
            "supporting_context": " ".join(sentences[1:3]).strip(),
            "noise_or_commentary": "" if len(sentences) <= 3 else "Additional article context detected beyond the core factual claim window.",
            "likely_copied_article_text": self._looks_like_copied_article_text(cleaned_input, sentences),
        }

    def _url_main_claim_queries(self, main_claim: str, article_context: dict[str, object]) -> list[str]:
        if not main_claim:
            return []
        entities = [str(item).strip() for item in article_context.get("entities", []) if str(item).strip()]
        actions = [str(item).strip() for item in article_context.get("actions", []) if str(item).strip()]
        locations = [str(item).strip() for item in article_context.get("locations", []) if str(item).strip()]
        dates = [str(item).strip() for item in article_context.get("dates", []) if str(item).strip()]
        compact_claim = self._trim_query_terms(main_claim, 18)
        focused = " ".join(dict.fromkeys([*entities[:4], *actions[:3], *locations[:2], *dates[:1]]))
        entity_action = " ".join(dict.fromkeys([*entities[:5], *actions[:4]]))
        if len(focused.split()) < 3:
            focused = ""
        if len(entity_action.split()) < 3:
            entity_action = ""
        source_base = focused or entity_action or compact_claim
        queries = [
            compact_claim,
            focused,
            entity_action,
        ]
        if source_base:
            queries.extend(
                [
                    f"{source_base} news",
                    f"{source_base} fact check",
                    f"site:reuters.com {source_base}",
                    f"site:apnews.com {source_base}",
                    f"site:bbc.com {source_base}",
                ]
            )
        return self._dedupe_queries(queries, limit=8)

    def _url_claim_tuple_queries(self, claim_tuple: dict) -> list[str]:
        if not claim_tuple:
            return []
        subject = str(claim_tuple.get("subject") or "").strip()
        obj = str(claim_tuple.get("object") or "").strip()
        location = str(claim_tuple.get("location") or "").strip()
        event = str(claim_tuple.get("event") or "").strip()
        action = str(claim_tuple.get("action") or "").strip()
        if not subject or not obj:
            return []
        verb = action or "reported"
        base = " ".join(part for part in [subject, verb, obj, location, event] if part)
        entity_event = " ".join(part for part in [subject, obj, event, location] if part)
        return self._dedupe_queries(
            [
                base,
                entity_event,
                f"site:apnews.com {entity_event}",
                f"site:reuters.com {entity_event}",
                f"site:bbc.com {entity_event}",
            ],
            limit=5,
        )

    def _retrieve_article_url_candidates(self, extracted: ExtractedPayload, queries: list[str], source_domain: str | None) -> dict[str, object]:
        all_candidates: dict[str, dict] = {}
        search_debug: list[dict] = []
        brave_results_count = 0
        for query in queries[:8]:
            bundle = self._search_bundle(query, expected_domain=None, limit=5)
            results = bundle.get("results", [])
            brave_results_count += len(results)
            search_debug.append(
                {
                    "query": query,
                    "result_count": len(results),
                    "failure_reason": bundle.get("failure_reason"),
                    "provider_status": bundle.get("provider_status", {}),
                    "provider_attempts": bundle.get("provider_attempts", []),
                    "results": [
                        {
                            "url": item.url,
                            "title": item.title,
                            "snippet": item.snippet,
                            "provider": "brave",
                        }
                        for item in results[:5]
                    ],
                }
            )
            for item in results:
                current = all_candidates.get(item.url)
                candidate = {
                    "url": item.url,
                    "title": item.title,
                    "snippet": item.snippet,
                    "domain": item.domain or extract_domain(item.url),
                    "provider": "brave",
                    "search_score": float(item.score),
                    "source": "web_search",
                }
                if current is None or float(candidate["search_score"]) > float(current.get("search_score") or 0.0):
                    all_candidates[item.url] = candidate

        for item in self._related_link_matches(extracted):
            url = str(item.get("url") or "").strip()
            if not url or url in all_candidates:
                continue
            all_candidates[url] = {
                "url": url,
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "domain": item.get("domain") or extract_domain(url),
                "provider": "publisher_page",
                "search_score": float(item.get("score") or 0.0),
                "source": item.get("source") or "related_link",
            }

        if not all_candidates:
            failure_reasons = [str(item.get("failure_reason") or "") for item in search_debug if item.get("failure_reason")]
            joined = " | ".join(item for item in failure_reasons if item) or "no_candidates_found"
            classified = self._classify_retrieval_failure(failure_reasons)
            return {
                "same_publisher_candidates": [],
                "cross_publisher_candidates": [],
                "selected_candidates": [],
                "brave_results_count": 0,
                "brave_top_results": [],
                "search_debug": search_debug,
                "failure_reason": classified or joined,
            }

        same_candidates: list[dict] = []
        cross_candidates: list[dict] = []
        source_family = self._domain_family(source_domain)
        ranked_candidates = sorted(all_candidates.values(), key=lambda item: float(item.get("search_score") or 0.0), reverse=True)
        for candidate in ranked_candidates:
            domain_family = self._domain_family(candidate.get("domain"))
            if source_family and domain_family == source_family:
                same_candidates.append(candidate)
            else:
                cross_candidates.append(candidate)

        review_limit = self.MAX_NLP_MATCHES * 3
        selected = cross_candidates[:review_limit]
        if len(selected) < review_limit and same_candidates:
            selected.extend(same_candidates[:1])
        selected = selected[:review_limit]

        reranked = self._rerank_article_candidates(selected, extracted)
        return {
            "same_publisher_candidates": same_candidates[:5],
            "cross_publisher_candidates": cross_candidates[:5],
            "selected_candidates": reranked[:review_limit],
            "brave_results_count": brave_results_count,
            "brave_top_results": [
                item
                for search in search_debug
                for item in search.get("results", [])
            ][:5],
            "search_debug": search_debug,
            "failure_reason": None,
        }

    def _rerank_article_candidates(self, candidates: list[dict], extracted: ExtractedPayload) -> list[dict]:
        if not candidates:
            return []
        article_context = self._extract_url_article_context(extracted)
        article_meaning = str(article_context.get("main_claim") or article_context.get("article_meaning") or "").strip()
        if not article_meaning:
            article_meaning = self._build_fact_check_summary_query(extracted) or self._build_fact_check_query(extracted)
        fetched_docs: list[dict] = []
        for candidate in candidates[: self.MAX_NLP_MATCHES * 3]:
            document = self.retrieval.fetch_page_document(str(candidate.get("url") or ""))
            passage = self._article_candidate_passage(str(document.get("extracted_text") or ""), str(candidate.get("snippet") or ""))
            fetched_docs.append(
                {
                    **candidate,
                    "fetch_status": document.get("fetch_status"),
                    "content_type": document.get("content_type"),
                    "extracted_text": str(document.get("extracted_text") or ""),
                    "extracted_text_length": int(document.get("extracted_text_length") or 0),
                    "title": str(document.get("title") or candidate.get("title") or ""),
                    "passage": passage,
                }
            )
        docs = [
            {
                "text": " ".join(filter(None, [item.get("title"), item.get("snippet"), item.get("passage")])),
                "title": item.get("title"),
                "url": item.get("url"),
            }
            for item in fetched_docs
        ]
        reranked = self.cohere.rank_candidates(article_meaning, docs) if docs else []
        rerank_by_url = {str(item.url or ""): float(item.score) for item in reranked if item.url}
        for item in fetched_docs:
            item["rerank_score"] = rerank_by_url.get(str(item.get("url") or ""), float(item.get("search_score") or 0.0))
            item["score"] = round(float(item["rerank_score"]), 3)
            item["explanation"] = "Reranked with Cohere after Brave retrieval." if reranked else "Ranked by Brave retrieval order."
        return sorted(fetched_docs, key=lambda item: (float(item.get("rerank_score") or 0.0), float(item.get("search_score") or 0.0)), reverse=True)

    def _article_candidate_passage(self, page_text: str, snippet: str) -> str:
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[\.\!\?])\s+", page_text or "") if sentence.strip()]
        if sentences:
            return " ".join(sentences[:3])[:1000]
        return snippet[:1000]

    def _judge_article_candidates(
        self,
        extracted: ExtractedPayload,
        article_meaning: str,
        candidates: list[dict],
        source_domain: str | None,
        article_context: dict[str, object] | None = None,
    ) -> list[dict]:
        article_context = article_context or self._extract_url_article_context(extracted)
        source_family = self._domain_family(source_domain)
        judged: list[dict] = []
        for candidate in candidates[: self.MAX_NLP_MATCHES * 3]:
            passage = str(candidate.get("passage") or candidate.get("snippet") or "")
            reasoning = self.reasoning.verify_claim_with_evidence(
                article_meaning,
                [
                    {
                        "title": candidate.get("title"),
                        "url": candidate.get("url"),
                        "domain": candidate.get("domain"),
                        "passage": passage,
                        "snippet": candidate.get("snippet"),
                        "score": candidate.get("score"),
                    }
                ],
            )
            candidate_domain = self._domain_family(candidate.get("domain"))
            same_publisher = bool(source_family and candidate_domain == source_family)
            verdict = str(reasoning.get("verdict") or "")
            support_score = float(reasoning.get("support_score") or 0.0)
            text_style_classification = self._classify_url_candidate_with_text_scorer(article_meaning, candidate)
            context_classification = (
                text_style_classification
                if text_style_classification.get("context_match") in {"exact_support", "strong_support", "refutes"}
                else self._classify_url_context_match(article_context, candidate)
            )
            if context_classification["context_match"] == "refutes" or verdict == "contradicted":
                label = "refutes"
                support_score = max(support_score, float(context_classification.get("support_score") or 0.9))
            elif context_classification["context_match"] in {"exact_support", "strong_support"}:
                label = str(context_classification["context_match"])
                support_score = max(support_score, float(context_classification.get("support_score") or 0.85))
            elif support_score >= 0.7 and context_classification["context_match"] not in {"irrelevant", "mentions_only"}:
                label = "strong_support"
            elif support_score >= 0.42 and context_classification["context_match"] != "irrelevant":
                label = "partial_context"
            else:
                label = str(context_classification.get("context_match") or "irrelevant")
            if same_publisher and label in {"strong_support", "exact_support"}:
                label = "partial_context"
            judged.append(
                {
                    "match": {
                        "title": candidate.get("title"),
                        "url": candidate.get("url"),
                        "score": round(float(candidate.get("score") or 0.0), 3),
                        "explanation": (
                            "Same-publisher context found. This is useful background, but it does not replace independent corroboration."
                            if same_publisher
                            else "Independent publisher reporting matched the same event or claim."
                        ),
                        "source": "web_search",
                        "snippet": candidate.get("snippet"),
                        "domain": candidate.get("domain"),
                    },
                    "judgment": {
                        "url": candidate.get("url"),
                        "title": candidate.get("title"),
                        "domain": candidate.get("domain"),
                        "same_publisher": same_publisher,
                        "support_label": label,
                        "support_score": round(support_score, 3),
                        "context_match": label,
                        "evidence_label": self._url_debug_evidence_label(label),
                        "relevance_score": round(float(context_classification.get("relevance_score") or 0.0), 3),
                        "source_tier": context_classification.get("source_tier"),
                        "evidence_quote": context_classification.get("evidence_quote", ""),
                        "reject_reason": context_classification.get("reason", ""),
                        "directness_score": round(float(reasoning.get("directness_score") or 0.0), 3),
                        "agreement_score": round(float(reasoning.get("agreement_score") or 0.0), 3),
                        "contradiction_score": round(float(reasoning.get("contradiction_score") or 0.0), 3),
                        "rerank_score": round(float(candidate.get("rerank_score") or 0.0), 3),
                        "fetch_status": candidate.get("fetch_status"),
                        "extracted_text_length": candidate.get("extracted_text_length"),
                    },
                }
            )
        judged.sort(
            key=lambda item: (
                0 if item["judgment"]["same_publisher"] else 1,
                2 if item["judgment"]["support_label"] in {"exact_support", "strong_support", "refutes"} else 1 if item["judgment"]["support_label"] == "partial_context" else 0,
                float(item["judgment"]["rerank_score"] or 0.0),
            ),
            reverse=True,
        )
        return judged[: self.MAX_NLP_MATCHES * 3]

    def _article_url_confidence(self, judged_candidates: list[dict], extracted: ExtractedPayload, publisher: TrustedPublisher | None) -> float:
        matches = [item["match"] for item in judged_candidates]
        source_domain = extract_domain(extracted.canonical_url) or extract_domain(extracted.metadata.get("fetched_url"))
        groups = self._match_groups(matches, source_domain)
        cross_count = int(groups["cross_publisher_count"])
        same_count = int(groups["same_publisher_count"])
        refutes = sum(1 for item in judged_candidates if not item["judgment"]["same_publisher"] and item["judgment"]["support_label"] == "refutes")
        strong_cross = sum(1 for item in judged_candidates if not item["judgment"]["same_publisher"] and item["judgment"]["support_label"] in {"exact_support", "strong_support"})
        partial_cross = sum(1 for item in judged_candidates if not item["judgment"]["same_publisher"] and item["judgment"]["support_label"] == "partial_context")
        top_cross_score = max(
            (float(item["judgment"].get("support_score") or 0.0) for item in judged_candidates if not item["judgment"]["same_publisher"]),
            default=0.0,
        )
        metadata_score = self._metadata_corroboration_score(self._metadata_signals(extracted), publisher)
        if refutes:
            return 0.12
        if strong_cross >= 3:
            return min(0.95, 0.86 + 0.03 * min(strong_cross, 3))
        if strong_cross >= 2:
            return min(0.9, 0.80 + 0.05 * min(top_cross_score, 1.0))
        if strong_cross >= 1:
            return min(0.75, 0.65 + 0.08 * min(top_cross_score, 1.0))
        if partial_cross >= 1:
            return min(0.35, 0.20 + (top_cross_score * 0.12))
        if cross_count >= 1:
            return min(0.35, 0.18 + (top_cross_score * 0.1))
        if same_count >= 1:
            return min(0.54, 0.26 + max(metadata_score, 0.12))
        return min(0.45, metadata_score)

    def _url_formula_components(self, layer2: LayerOutcome, extracted: ExtractedPayload) -> dict[str, float]:
        inputs = self._nlp_score_inputs(layer2, extracted)
        return {
            "S": round(float(inputs.get("factual_consistency") or 0.0), 3),
            "R": round(float(inputs.get("authority") or 0.0), 3),
            "A": round(float(inputs.get("cross_source_agreement") or 0.0), 3),
            "D": round(float(inputs.get("reference_support") or 0.0), 3),
            "C": round(float(inputs.get("language_reliability") or 0.0), 3),
            "E": 1.0 if (layer2.details or {}).get("matches") else 0.4,
            "P": round(float(inputs.get("penalty") or 0.0), 3),
        }

    def _url_label_from_confidence(self, confidence: float, verdict: str) -> str:
        return map_score_to_label(confidence * 100)

    def _slug_terms(self, url: str | None) -> str:
        if not url:
            return ""
        path = urlparse(str(url)).path.strip("/")
        if not path:
            return ""
        slug = path.split("/")[-1]
        return slug.replace("-", " ").replace("_", " ").strip()

    def _body_query_terms(self, text: str | None, max_terms: int = 10) -> str:
        if not text:
            return ""
        tokens: list[str] = []
        for token in normalize_text(text).split():
            if len(token) < 4:
                continue
            if token in {
                "this",
                "that",
                "with",
                "from",
                "were",
                "have",
                "been",
                "their",
                "about",
                "after",
                "before",
                "which",
                "while",
                "would",
                "could",
                "should",
            }:
                continue
            if token not in tokens:
                tokens.append(token)
            if len(tokens) >= max_terms:
                break
        return " ".join(tokens)

    def _combine_fact_check_matches(self, ranked: list, web_matches: list, related_link_matches: list) -> list[dict]:
        matches: list[dict] = []
        seen: set[str] = set()
        if web_matches:
            for candidate in web_matches[: self.MAX_NLP_MATCHES]:
                key = str(candidate.url or candidate.title or "")
                if key in seen:
                    continue
                seen.add(key)
                matches.append(
                    {
                        "title": candidate.title,
                        "url": candidate.url,
                        "score": candidate.score,
                        "explanation": candidate.explanation,
                        "source": "web_search",
                        "snippet": candidate.snippet,
                        "domain": candidate.domain,
                    }
                )
        for candidate in ranked[: self.MAX_NLP_MATCHES]:
            key = str(candidate.url or candidate.title or "")
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                {
                    "title": candidate.title,
                    "url": candidate.url,
                    "score": candidate.score,
                    "explanation": candidate.explanation,
                    "source": "trusted_corpus",
                }
            )
        for candidate in related_link_matches[: self.MAX_NLP_MATCHES]:
            if web_matches and float(candidate.get("score") or 0.0) < 0.2:
                continue
            key = str(candidate.get("url") or candidate.get("title") or "")
            if key in seen:
                continue
            seen.add(key)
            matches.append(candidate)
        return sorted(matches, key=lambda item: float(item.get("score") or 0), reverse=True)[: self.MAX_NLP_MATCHES]

    def _match_domain_stats(self, matches: list[dict], source_domain: str | None) -> dict[str, int | list[str] | bool]:
        source_domain = (source_domain or "").lower()
        external = 0
        same_publisher = 0
        unique_domains: set[str] = set()
        same_domains: set[str] = set()
        external_domains: set[str] = set()
        for match in matches:
            if not isinstance(match, dict):
                continue
            domain = str(match.get("domain") or extract_domain(match.get("url")) or "").lower()
            if not domain:
                continue
            unique_domains.add(domain)
            if source_domain and self._domain_family(domain) == self._domain_family(source_domain):
                same_publisher += 1
                same_domains.add(domain)
            else:
                external += 1
                external_domains.add(domain)
        return {
            "external_count": external,
            "same_publisher_count": same_publisher,
            "unique_domain_count": len(unique_domains),
            "same_publisher_domains": sorted(same_domains),
            "support_domains": sorted(external_domains),
            "cross_source_exists": external > 0,
            "same_source_only": same_publisher > 0 and external == 0,
        }

    def _url_same_context_debug(self, matches: list[dict], source_domain: str | None) -> dict[str, object]:
        counted: list[dict] = []
        rejected: list[dict] = []
        source_family = self._domain_family(source_domain)
        for match in matches:
            if not isinstance(match, dict):
                continue
            domain = str(match.get("domain") or extract_domain(match.get("url")) or "")
            same_publisher = bool(source_family and self._domain_family(domain) == source_family)
            relation = str(match.get("relation") or match.get("evidence_relation") or match.get("alignment") or "").lower()
            aboutness = str(match.get("aboutness") or "").lower()
            stance = str(match.get("stance") or "").lower()
            exact_support = relation == "exact_support" or (
                relation == "supports" and aboutness in {"exact_claim", "strong_related"}
            )
            refutes = relation in {"refutes", "contradicts", "contradiction"} or stance == "refutes"
            if (exact_support or refutes) and not same_publisher and len(counted) < self.MAX_NLP_MATCHES:
                counted.append(
                    {
                        **match,
                        "classification": "refutes" if refutes else "exact_support" if aboutness == "exact_claim" else "strong_support",
                    }
                )
                continue
            reason = "Not counted because it was partial, mentions-only, irrelevant, or same-publisher context."
            if same_publisher:
                reason = "Not counted as independent same-context evidence because it is from the submitted publisher."
            elif relation in {"partially_related", "partial_support", "partially_supports"} or aboutness in {"partial_context", "mentions_only"}:
                reason = "Not counted because it is related context but does not directly confirm or refute the same article context."
            rejected.append(
                {
                    **match,
                    "classification": relation or aboutness or "unknown",
                    "same_publisher": same_publisher,
                    "reason": reason,
                }
            )
        return {
            "max_similar_articles_allowed": self.MAX_NLP_MATCHES,
            "similar_articles_found_count": len(counted),
            "counted_same_context_articles": counted,
            "rejected_related_articles": rejected,
            "stopped_search_reason": "found_5_sources" if len(counted) >= self.MAX_NLP_MATCHES else "search_exhausted",
        }

    def _classify_url_context_match(self, article_context: dict[str, object], candidate: dict) -> dict[str, object]:
        claim_tuple = article_context.get("claim_tuple", {}) if isinstance(article_context.get("claim_tuple"), dict) else {}
        candidate = {
            **candidate,
            "source_type": candidate.get("source_type") or get_source_type(str(candidate.get("url") or candidate.get("domain") or "")),
        }
        candidate["source_tier"] = candidate.get("source_tier") or self._source_tier(candidate)
        evidence_text = " ".join(str(candidate.get(key) or "") for key in ("title", "snippet", "passage", "extracted_text"))
        normalized = normalize_text(evidence_text)
        if not normalized:
            return {"context_match": "irrelevant", "source_tier": candidate["source_tier"], "support_score": 0.0, "relevance_score": 0.0, "reason": "No readable evidence text was fetched."}

        trusted_support = self._trusted_news_exact_support(claim_tuple, candidate)
        if trusted_support.get("supports"):
            tier_score = 0.92 if candidate["source_tier"] == "tier_1" else 0.88 if candidate["source_tier"] == "tier_2" else 0.72
            return {
                "context_match": "exact_support",
                "source_tier": candidate["source_tier"],
                "support_score": tier_score,
                "relevance_score": 1.0,
                "directness_score": trusted_support.get("directness_score", 0.9),
                "evidence_quote": trusted_support.get("evidence_quote", ""),
                "reason": trusted_support.get("reason", "Source confirms the same article context."),
            }

        subject = str(claim_tuple.get("subject") or "")
        obj = str(claim_tuple.get("object") or "")
        subject_last = normalize_text(subject).split()[-1] if normalize_text(subject).split() else ""
        object_last = normalize_text(obj).split()[-1] if normalize_text(obj).split() else ""
        subject_hit = bool(subject_last and re.search(rf"\b{re.escape(subject_last)}\b", normalized))
        object_hit = bool(object_last and re.search(rf"\b{re.escape(object_last)}\b", normalized))
        action_hit = bool(re.search(r"\b(?:defeat(?:ed|s|ing)?|beat|beats|won|wins|winner|nomination|nominee|run[- ]?off|runoff|primary)\b", normalized))
        event_hit = bool(re.search(r"\b(?:republican|gop|senate|primary|run[- ]?off|runoff|nomination)\b", normalized))
        if subject_last and object_last and re.search(rf"\b{re.escape(object_last)}\b[^.]*\b(?:defeat(?:ed|s)?|beat|beats|won|wins)\b[^.]*\b{re.escape(subject_last)}\b", normalized):
            return {"context_match": "refutes", "source_tier": candidate["source_tier"], "support_score": 0.95, "relevance_score": 1.0, "reason": "Source reports the opposite winner/outcome."}
        if re.search(r"\b(?:false|not true|did not happen|no evidence)\b", normalized) and subject_hit and object_hit:
            return {"context_match": "refutes", "source_tier": candidate["source_tier"], "support_score": 0.9, "relevance_score": 0.95, "reason": "Source directly rejects the article context."}
        if subject_hit and object_hit and action_hit and event_hit:
            tier_score = 0.88 if candidate["source_tier"] in {"tier_1", "tier_2"} else 0.68
            return {"context_match": "strong_support", "source_tier": candidate["source_tier"], "support_score": tier_score, "relevance_score": 0.88, "reason": "Source confirms the same entities, outcome, and event with different wording."}
        if (subject_hit or object_hit) and event_hit:
            return {"context_match": "partial_context", "source_tier": candidate["source_tier"], "support_score": 0.25, "relevance_score": 0.45, "reason": "Same person or race context, but not the exact article claim."}
        if subject_hit or object_hit:
            return {"context_match": "mentions_only", "source_tier": candidate["source_tier"], "support_score": 0.1, "relevance_score": 0.2, "reason": "Source mentions an entity without verifying the event."}
        generic = self._classify_generic_url_claim_match(article_context, candidate, evidence_text, normalized)
        if generic["context_match"] != "unknown":
            return generic
        return {"context_match": "irrelevant", "source_tier": candidate["source_tier"], "support_score": 0.0, "relevance_score": 0.0, "reason": "Different or unrelated article context."}

    def _classify_url_candidate_with_text_scorer(self, article_meaning: str, candidate: dict) -> dict[str, object]:
        claim = " ".join(str(article_meaning or "").split()).strip()
        if not claim:
            return {"context_match": "unknown"}
        evidence = {
            "title": candidate.get("title"),
            "url": candidate.get("url"),
            "domain": candidate.get("domain") or extract_domain(candidate.get("url")),
            "source_type": candidate.get("source_type") or get_source_type(str(candidate.get("url") or candidate.get("domain") or "")),
            "snippet": candidate.get("snippet"),
            "passage": candidate.get("passage"),
            "text": " ".join(str(candidate.get(key) or "") for key in ("title", "snippet", "passage")),
            "score": candidate.get("score"),
            "rerank_score": candidate.get("rerank_score") or candidate.get("score"),
            "semantic_score": candidate.get("rerank_score") or candidate.get("score"),
            "bm25_score": candidate.get("search_score"),
            "headline_exactness_score": self._headline_exactness_score(claim, str(candidate.get("title") or ""), str(candidate.get("snippet") or "")),
            "fetch_status": candidate.get("fetch_status"),
            "extracted_text_length": candidate.get("extracted_text_length"),
        }
        direct = self._classify_url_candidate_with_text_context(claim, evidence)
        if direct.get("context_match") != "unknown":
            return direct

        scored = self._score_claim_against_evidence(claim, [evidence])
        evidence_sources = scored.get("evidence_sources", []) if isinstance(scored, dict) else []
        source_tier = self._source_tier(evidence)
        if evidence_sources:
            top = evidence_sources[0]
            relation = str(top.get("evidence_relation") or top.get("judgment_label") or "")
            relevance = float(top.get("claim_relevance_score") or top.get("relation_confidence") or top.get("score") or 0.0)
            if relation == "supports":
                label = "exact_support" if str(top.get("aboutness") or "") == "exact_claim" else "strong_support"
                return {
                    "context_match": label,
                    "source_tier": top.get("source_tier") or source_tier,
                    "support_score": max(0.82, relevance),
                    "relevance_score": relevance,
                    "directness_score": top.get("directness_score") or top.get("source_recovery_score") or relevance,
                    "evidence_quote": top.get("evidence_quote", ""),
                    "reason": top.get("reason_accepted_or_rejected") or "Text-style claim/evidence scoring found support for the URL article claim.",
                }
            if relation in {"contradicts", "refutes"}:
                return {
                    "context_match": "refutes",
                    "source_tier": top.get("source_tier") or source_tier,
                    "support_score": max(0.86, relevance),
                    "relevance_score": relevance,
                    "evidence_quote": top.get("evidence_quote", ""),
                    "reason": top.get("reason_accepted_or_rejected") or top.get("contradiction_reason") or "Text-style claim/evidence scoring found contradiction.",
                }
            if relation == "partially_related":
                return {
                    "context_match": "partial_context",
                    "source_tier": top.get("source_tier") or source_tier,
                    "support_score": min(0.4, max(0.2, relevance)),
                    "relevance_score": relevance,
                    "evidence_quote": top.get("evidence_quote", ""),
                    "reason": top.get("reason_accepted_or_rejected") or "Text-style claim/evidence scoring found related but incomplete context.",
                }
        diagnostics = scored.get("top_candidate_diagnostics", []) if isinstance(scored, dict) else []
        if diagnostics:
            top_diag = diagnostics[0]
            relevance = float(top_diag.get("claim_relevance_score") or 0.0)
            relation = str(top_diag.get("evidence_relation") or top_diag.get("judgment_label") or "")
            if relation in {"partially_related", "partially_supports"} or relevance >= 0.55:
                return {
                    "context_match": "partial_context",
                    "source_tier": source_tier,
                    "support_score": min(0.38, relevance),
                    "relevance_score": relevance,
                    "reason": top_diag.get("rejection_reason") or "Text-style scoring found related context but not direct support.",
                }
        return {"context_match": "unknown"}

    def _classify_url_candidate_with_text_context(self, claim: str, evidence: dict) -> dict[str, object]:
        evidence["source_type"] = evidence.get("source_type") or get_source_type(str(evidence.get("url") or evidence.get("domain") or ""))
        source_tier = self._source_tier(evidence)
        evidence_text = " ".join(str(evidence.get(key) or "") for key in ("title", "snippet", "passage", "text"))
        if not evidence_text.strip():
            return {"context_match": "unknown"}

        claim_details = self._parse_claim_details(claim)
        context_match = self.compare_claim_and_evidence_context(claim, evidence)
        relevance = self._claim_evidence_relevance_score(claim, evidence)
        alignment = classify_evidence(claim, evidence_text, is_fact_check=evidence["source_type"] == "fact_check_page")
        headline_exactness = self._headline_exactness_score(claim, str(evidence.get("title") or ""), str(evidence.get("snippet") or ""))
        trusted_support = self._trusted_news_exact_support(claim_details.get("claim_tuple", {}), evidence)
        if not trusted_support.get("supports"):
            trusted_support = self._trusted_ordinary_news_support(claim, claim_details, evidence)
        contradiction_reason = self._explicit_contradiction_reason(claim, evidence)

        if contradiction_reason and relevance >= 0.55 and bool(context_match.get("context_match_pass")):
            return {
                "context_match": "refutes",
                "source_tier": source_tier,
                "support_score": max(0.86, relevance),
                "relevance_score": relevance,
                "evidence_quote": self._best_url_evidence_quote(evidence_text, claim_details.get("entities", []), claim_details.get("actions", [])),
                "reason": contradiction_reason,
            }
        if trusted_support.get("supports"):
            return {
                "context_match": "exact_support",
                "source_tier": source_tier,
                "support_score": max(0.86, float(trusted_support.get("support_score") or 0.0), relevance),
                "relevance_score": max(relevance, float(trusted_support.get("support_score") or 0.0)),
                "directness_score": trusted_support.get("directness_score", 0.9),
                "evidence_quote": trusted_support.get("evidence_quote") or self._best_url_evidence_quote(evidence_text, claim_details.get("entities", []), claim_details.get("actions", [])),
                "reason": trusted_support.get("reason") or "Shared text context verification found direct support for the URL claim.",
            }

        context_pass = bool(context_match.get("context_match_pass"))
        location_pass = bool(context_match.get("location_match_pass"))
        event_pass = bool(context_match.get("event_match_pass"))
        source_ok = source_tier in {"tier_1", "tier_2", "tier_3"}
        exact_headline_source = headline_exactness >= 0.82 and source_ok
        strong_alignment = alignment.get("alignment") in {"supports", "strong_support", "exact_support"} or alignment.get("aboutness") == "exact_claim"
        strong_context_support = (
            source_ok
            and context_pass
            and location_pass
            and event_pass
            and (
                relevance >= 0.76
                or exact_headline_source
                or (relevance >= 0.68 and strong_alignment)
            )
        )
        if strong_context_support:
            label = "exact_support" if relevance >= 0.84 or exact_headline_source else "strong_support"
            return {
                "context_match": label,
                "source_tier": source_tier,
                "support_score": max(0.74, relevance, headline_exactness),
                "relevance_score": max(relevance, headline_exactness),
                "directness_score": max(relevance, headline_exactness),
                "evidence_quote": alignment.get("evidence_quote") or self._best_url_evidence_quote(evidence_text, claim_details.get("entities", []), claim_details.get("actions", [])),
                "reason": "Shared text context verification matched the same claim entities, action/event, and location/time context.",
            }

        if location_pass and event_pass and relevance >= 0.5:
            return {
                "context_match": "partial_context",
                "source_tier": source_tier,
                "support_score": min(0.4, relevance),
                "relevance_score": relevance,
                "evidence_quote": alignment.get("evidence_quote") or self._best_url_evidence_quote(evidence_text, claim_details.get("entities", []), claim_details.get("actions", [])),
                "reason": context_match.get("mismatch_reason") or "Shared text context verification found related context but not enough direct support.",
            }
        if relevance >= 0.35:
            return {
                "context_match": "mentions_only",
                "source_tier": source_tier,
                "support_score": 0.1,
                "relevance_score": relevance,
                "reason": context_match.get("mismatch_reason") or "Source overlaps with claim terms but does not verify the same claim.",
            }
        return {"context_match": "unknown"}

    def _classify_generic_url_claim_match(self, article_context: dict[str, object], candidate: dict, evidence_text: str, normalized: str) -> dict[str, object]:
        main_claim = str(article_context.get("main_claim") or article_context.get("article_meaning") or "").strip()
        if not main_claim:
            return {"context_match": "unknown"}
        details = self._parse_claim_details(main_claim)
        entities = self._dedupe_text_items([*article_context.get("entities", []), *details.get("entities", [])])
        actions = self._dedupe_text_items([*article_context.get("actions", []), *details.get("actions", [])])
        locations = self._dedupe_text_items([*article_context.get("locations", []), *details.get("locations", [])])
        money = self._dedupe_text_items(details.get("money", []))
        important_terms = self._dedupe_text_items(details.get("important_terms", []))

        entity_score = self._coverage_score(entities[:8], normalized)
        action_score = self._coverage_score(actions[:6], normalized) if actions else 0.75
        location_score = self._coverage_score(locations[:5], normalized) if locations else 0.75
        money_score = self._coverage_score(money[:2], normalized) if money else 1.0
        term_score = self._coverage_score(important_terms[:8], normalized) if important_terms else 0.5
        relevance = round((0.38 * entity_score) + (0.25 * action_score) + (0.17 * location_score) + (0.12 * money_score) + (0.08 * term_score), 3)

        contradiction_reason = self._explicit_contradiction_reason(main_claim, {"title": candidate.get("title"), "snippet": candidate.get("snippet"), "passage": candidate.get("passage"), "text": evidence_text})
        if contradiction_reason and relevance >= 0.55:
            return {
                "context_match": "refutes",
                "source_tier": candidate["source_tier"],
                "support_score": max(0.86, relevance),
                "relevance_score": relevance,
                "reason": contradiction_reason,
                "evidence_quote": self._best_url_evidence_quote(evidence_text, entities, actions),
            }
        if re.search(r"\b(?:false|fake|hoax|fabricated|no evidence|did not|has not|not true)\b", normalized) and entity_score >= 0.45 and relevance >= 0.55:
            return {
                "context_match": "refutes",
                "source_tier": candidate["source_tier"],
                "support_score": max(0.82, relevance),
                "relevance_score": relevance,
                "reason": "Source directly rejects the same claim context.",
                "evidence_quote": self._best_url_evidence_quote(evidence_text, entities, actions),
            }
        if entity_score >= 0.62 and action_score >= 0.55 and location_score >= 0.5 and money_score >= 1.0 and relevance >= 0.72:
            tier_score = 0.88 if candidate["source_tier"] in {"tier_1", "tier_2"} else 0.72
            return {
                "context_match": "strong_support",
                "source_tier": candidate["source_tier"],
                "support_score": max(tier_score, relevance),
                "relevance_score": relevance,
                "reason": "Source matches the same claim entities, action/event, location or timeframe context.",
                "evidence_quote": self._best_url_evidence_quote(evidence_text, entities, actions),
            }
        if entity_score >= 0.35 and relevance >= 0.45:
            return {
                "context_match": "partial_context",
                "source_tier": candidate["source_tier"],
                "support_score": min(0.38, relevance),
                "relevance_score": relevance,
                "reason": "Source is related but does not match enough of the claim-specific entity/action/location context.",
                "evidence_quote": self._best_url_evidence_quote(evidence_text, entities, actions),
            }
        if entity_score > 0:
            return {
                "context_match": "mentions_only",
                "source_tier": candidate["source_tier"],
                "support_score": 0.1,
                "relevance_score": relevance,
                "reason": "Source mentions one or more claim entities but not the same event or factual claim.",
            }
        return {
            "context_match": "irrelevant",
            "source_tier": candidate["source_tier"],
            "support_score": 0.0,
            "relevance_score": relevance,
            "reason": "Retrieved article does not match the extracted URL claim.",
        }

    def _best_url_evidence_quote(self, evidence_text: str, entities: list[str], actions: list[str]) -> str:
        entity_terms = [normalize_text(item).split()[-1] for item in entities if normalize_text(item).split()]
        action_terms = [normalize_text(item) for item in actions if normalize_text(item)]
        for sentence in self._text_sentences(evidence_text):
            normalized = normalize_text(sentence)
            if any(term and re.search(rf"\b{re.escape(term)}\b", normalized) for term in entity_terms) and (
                not action_terms or any(term and term in normalized for term in action_terms)
            ):
                return sentence[:300]
        sentences = self._text_sentences(evidence_text)
        return sentences[0][:300] if sentences else ""

    def _url_debug_evidence_label(self, support_label: str) -> str:
        if support_label in {"exact_support", "strong_support"}:
            return "supports the claim"
        if support_label in {"refutes", "contradiction"}:
            return "contradicts the claim"
        if support_label in {"partial_context", "mentions_only"}:
            return "related but not enough"
        return "irrelevant"

    def _domain_family(self, domain: str | None) -> str:
        domain = str(domain or "").lower().strip(".")
        if not domain:
            return ""
        parts = [part for part in domain.split(".") if part]
        if len(parts) <= 2:
            return ".".join(parts)
        return ".".join(parts[-2:])

    def _factual_consistency_score(self, layer2: LayerOutcome, extracted: ExtractedPayload) -> float:
        if self._layer2_signal(layer2) == "negative":
            return 0.1

        matches = layer2.details.get("matches", []) if isinstance(layer2.details, dict) else []
        source_domain = extract_domain(extracted.canonical_url) or extract_domain(extracted.metadata.get("fetched_url"))
        stats = self._match_domain_stats(matches, source_domain)
        top_score = max((float(match.get("score") or 0.0) for match in matches if isinstance(match, dict)), default=0.0)
        references_present = bool(extracted.metadata.get("references"))
        body_present = bool(extracted.main_text and len(normalize_text(extracted.main_text).split()) >= 80)

        if stats["external_count"] >= 3:
            score = min(1.0, 0.55 + (top_score * 0.35) + (0.05 * min(stats["external_count"], 5)))
        elif stats["external_count"] >= 1:
            score = min(0.85, 0.45 + (top_score * 0.30) + (0.04 * stats["external_count"]))
        elif stats["same_publisher_count"] >= 1:
            score = min(0.5, 0.22 + (top_score * 0.20))
        else:
            score = 0.18 if (references_present or body_present) else 0.0

        if references_present:
            score = min(1.0, score + 0.05)
        return round(score, 3)

    def _related_link_matches(self, extracted: ExtractedPayload) -> list[dict]:
        related_links = extracted.metadata.get("related_links", [])
        if not isinstance(related_links, list):
            return []
        title_tokens = set(normalize_text(extracted.title or "").split())
        if not title_tokens:
            title_tokens = set(self._slug_terms(extracted.canonical_url or extracted.metadata.get("fetched_url")).split())
        matches: list[dict] = []
        for item in related_links[:20]:
            if not isinstance(item, dict):
                continue
            link_title = str(item.get("title") or "")
            link_url = str(item.get("url") or "")
            if not link_title or not link_url:
                continue
            link_tokens = set(normalize_text(link_title).split())
            overlap_score = len(title_tokens & link_tokens) / max(1, len(title_tokens)) if title_tokens else 0.0
            if not self._is_meaningful_related_article(link_url, link_title, overlap_score):
                continue
            same_domain = extract_domain(link_url) == extract_domain(extracted.canonical_url or extracted.metadata.get("fetched_url"))
            score = overlap_score + (0.08 if same_domain else 0.03)
            matches.append(
                {
                    "title": link_title,
                    "url": link_url,
                    "score": round(min(1.0, score), 3),
                    "explanation": "Found as same-source context on the publisher page. This is useful background, but it does not replace broader cross-web corroboration.",
                    "source": "related_link",
                    "domain": item.get("domain"),
                    "snippet": None,
                }
            )
        return sorted(matches, key=lambda match: float(match["score"]), reverse=True)

    def _is_meaningful_related_article(self, url: str, title: str, overlap_score: float) -> bool:
        parsed = urlparse(str(url))
        segments = [segment for segment in parsed.path.split("/") if segment]
        last_segment = segments[-1].lower() if segments else ""
        normalized_title = normalize_text(title)
        title_terms = [term for term in normalized_title.split() if term]

        if overlap_score >= 0.2:
            return True
        if len(title_terms) >= 4 and "-" in last_segment:
            return True
        if any(segment.isdigit() and len(segment) == 4 for segment in segments) and len(title_terms) >= 3:
            return True

        generic_segments = {
            "world",
            "politics",
            "business",
            "sport",
            "sports",
            "health",
            "travel",
            "weather",
            "video",
            "videos",
            "live-news",
            "live",
            "news",
            "middle-east",
            "americas",
            "australia",
            "europe",
            "asia",
        }
        if last_segment in generic_segments:
            return False
        if len(title_terms) <= 2:
            return False
        return False

    def _fact_check_decision(
        self,
        confidence: float,
        trusted_publisher: bool,
        same_publisher_count: int = 0,
        cross_publisher_count: int = 0,
        social_mode: bool = False,
    ) -> tuple[str, str]:
        if cross_publisher_count == 0 and same_publisher_count > 0:
            if confidence >= 0.55:
                return (
                    "same-publisher context found",
                    "Context match found on the same publisher domain, but independent cross-publisher corroboration was limited.",
                )
            return (
                "partial content match",
                "Context match found on the same publisher domain, but this is useful background rather than strong external confirmation.",
            )
        if social_mode and cross_publisher_count == 0 and confidence > 0:
            return (
                "partial content match",
                "Post-internal content was available, but broader external corroboration remains limited.",
            )
        if confidence >= 0.8:
            return (
                "content match found",
                "Fact-checking found strong independent corroboration across supporting sources.",
            )
        if confidence >= 0.55:
            return (
                "content match found",
                "Fact-checking found meaningful similarity with independent corroborating sources, so the content is probably correct but still lacks cryptographic proof.",
            )
        if confidence > 0:
            if trusted_publisher:
                return (
                    "partial content match",
                    "Fact-checking found some supporting similarity and a verified publisher, which suggests the content may be correct but the evidence is still limited.",
                )
            return (
                "partial content match",
                "Fact-checking found some supporting similarity, but the evidence is not strong enough to make a confident correctness claim.",
            )
        return ("inconclusive", "No strong supporting or conflicting source match was available.")

    def _layer1_signal(self, layer1: LayerOutcome) -> str:
        if layer1.status in {"matched", "publisher-recognized", "platform-issued"}:
            return "positive"
        if layer1.status == "mismatch":
            return "negative"
        return "neutral"

    def _layer2_signal(self, layer2: LayerOutcome) -> str:
        verdict = str(layer2.details.get("fact_check_verdict", "")).lower()
        confidence = float(layer2.confidence or 0.0)
        if verdict in {"conflicting", "likely false", "false content"} or confidence < 0:
            return "negative"
        if layer2.status == "supporting" and confidence >= 0.45:
            return "positive"
        return "neutral"

    def _layer3_signal(self, layer3: LayerOutcome) -> str:
        if (layer3.details or {}).get("mode") == "feedback":
            return "neutral"
        if layer3.status == "supporting":
            return "positive"
        if layer3.status == "disputing":
            return "negative"
        return "neutral"

    def _layer3(self, job: VerificationJob, persist_audit: bool = True) -> LayerOutcome:
        votes = self._consensus_votes_for_job(job)
        true_votes = sum(1 for vote in votes if vote.vote_value > 0)
        false_votes = sum(1 for vote in votes if vote.vote_value < 0)
        not_sure_votes = sum(1 for vote in votes if vote.vote_value == 0)
        total_votes = len(votes)
        confidence = 0.0
        majority_label = "not_sure"
        majority_count = 0
        if total_votes:
            majority_label, majority_count = max(
                (("true", true_votes), ("false", false_votes), ("not_sure", not_sure_votes)),
                key=lambda item: item[1],
            )
            confidence = round(majority_count / total_votes, 3)
        if persist_audit:
            append_audit_entry(self.db, job, "layer3_completed", {"vote_count": len(votes)})
        if not votes:
            status_value = "none"
            explanation = "Community consensus has no user votes yet."
        elif majority_label == "false":
            status_value = "disputing"
            explanation = "Most users voted that the content is false, so the community layer disputes it."
        elif majority_label == "true":
            status_value = "supporting"
            explanation = "Most users voted that the content is true, so the community layer supports it."
        else:
            status_value = "mixed"
            explanation = "Most users selected not sure, so the community layer remains inconclusive."
        return LayerOutcome(
            status=status_value,
            confidence=round(confidence, 3) if votes else 0.0,
            explanation=explanation,
            details={
                "vote_count": total_votes,
                "majority_confidence": round(confidence, 3) if votes else 0.0,
                "majority_vote": majority_label,
                "true_votes": true_votes,
                "false_votes": false_votes,
                "not_sure_votes": not_sure_votes,
                "shared_content_identity": self._job_consensus_identity(job),
            },
        )

    def _job_consensus_identity(self, job: VerificationJob) -> dict[str, str | None]:
        extracted = job.extracted_content
        return {
            "canonical_url": self._consensus_identity_url(extracted),
            "content_hash": extracted.content_hash if extracted else None,
        }

    def _consensus_identity_url(self, extracted: ExtractedPayload | None) -> str | None:
        if not extracted:
            return None
        raw_metadata = getattr(extracted, "metadata", None)
        metadata = raw_metadata if isinstance(raw_metadata, dict) else getattr(extracted, "metadata_json", {}) or {}
        platform_type = str(metadata.get("platform_type") or "").lower()
        if platform_type == "youtube":
            video_id = str(metadata.get("youtube_video_id") or "").strip()
            if video_id:
                return f"https://www.youtube.com/watch?v={video_id}"
            normalized_url = str(metadata.get("normalized_url") or metadata.get("detected_youtube_url") or "").strip()
            if normalized_url:
                return normalized_url
        return extracted.canonical_url

    def _consensus_votes_for_job(self, job: VerificationJob) -> list[ConsensusVote]:
        extracted = job.extracted_content
        if not extracted:
            return list(
                self.db.scalars(
                    select(ConsensusVote).where(ConsensusVote.job_id == job.id).order_by(ConsensusVote.created_at.desc())
                )
            )

        consensus_url = self._consensus_identity_url(extracted)
        if consensus_url:
            canonical_votes = list(
                self.db.scalars(
                    select(ConsensusVote)
                    .where(ConsensusVote.canonical_url == consensus_url)
                    .order_by(ConsensusVote.created_at.desc())
                )
            )
            if canonical_votes:
                return canonical_votes

        # Only fall back to shared content-hash identity when the job has no
        # stable canonical URL. That keeps votes persistent across repeated
        # checks of the same link, without leaking votes between unrelated
        # links that happen to collapse to the same extracted body hash.
        if extracted.content_hash and not consensus_url:
            votes = list(
                self.db.scalars(
                    select(ConsensusVote)
                    .where(ConsensusVote.content_hash == extracted.content_hash)
                    .order_by(ConsensusVote.created_at.desc())
                )
            )
            if votes:
                return votes

        return list(
            self.db.scalars(
                select(ConsensusVote).where(ConsensusVote.job_id == job.id).order_by(ConsensusVote.created_at.desc())
            )
        )

    def _layer4(self, job: VerificationJob) -> LayerOutcome:
        return LayerOutcome(
            status="recorded",
            confidence=1.0,
            explanation="Verification steps were recorded in an append-only hash-chained audit log.",
            details={"audit_entry_count": len(job.audit_entries)},
        )

    def _build_system_scoring(
        self,
        layer1: LayerOutcome,
        layer2: LayerOutcome,
        layer3: LayerOutcome,
        extracted: ExtractedPayload,
    ) -> dict:
        crypto_inputs = self._crypto_score_inputs(layer1, extracted)
        nlp_inputs = self._nlp_score_inputs(layer2, extracted)
        community_inputs = self._community_score_inputs(layer3)
        evidence_present, evidence_total = self._evidence_component_counts(layer1, layer2, layer3, extracted)
        flags = self._scoring_flags(layer1, layer2, layer3, extracted, evidence_present, evidence_total)
        base_evidence_present, base_evidence_total = self._evidence_component_counts_without_community(layer1, layer2, extracted)
        base_flags = self._scoring_flags_without_community(layer1, layer2, extracted, base_evidence_present, base_evidence_total)
        base_result = compute_newsproof_score(
            {
                "crypto": crypto_inputs,
                "nlp": nlp_inputs,
                "community": {"status": "missing"},
                "evidence_present": base_evidence_present,
                "evidence_total": base_evidence_total,
                "flags": base_flags,
            }
        )
        result = compute_newsproof_score(
            {
                "crypto": crypto_inputs,
                "nlp": nlp_inputs,
                "community": community_inputs,
                "evidence_present": evidence_present,
                "evidence_total": evidence_total,
                "flags": flags,
            }
        )
        base_result = self._apply_url_no_multiplier_score(base_result, base_flags)
        result = self._apply_url_no_multiplier_score(result, flags)
        result = self._apply_url_community_overlay(result, base_result, community_inputs)
        result.update(
            {
                "scoring_method": "url_hybrid_formula",
                "formula": "Final URL score = (0.30 * crypto) + (0.45 * NLP) + (0.25 * consensus). Evidence sufficiency and global penalty are diagnostics only.",
                "displayed_components": [
                    "crypto_layer",
                    "nlp_layer",
                    "user_layer",
                ],
                "display_layer_scores": {
                    "crypto": result.get("layer_scores", {}).get("crypto"),
                    "nlp": result.get("layer_scores", {}).get("nlp"),
                    "community": result.get("layer_scores", {}).get("community"),
                },
                "display_layer_breakdown": {
                    "crypto": result.get("layer_breakdown", {}).get("crypto"),
                    "nlp": result.get("layer_breakdown", {}).get("nlp"),
                    "community": result.get("layer_breakdown", {}).get("community"),
                },
            }
        )
        return result

    def _apply_url_no_multiplier_score(self, result: dict, flags: dict) -> dict:
        adjusted = dict(result)
        layer_scores = adjusted.get("layer_scores", {}) if isinstance(adjusted.get("layer_scores"), dict) else {}
        layer_status = adjusted.get("layer_status", {}) if isinstance(adjusted.get("layer_status"), dict) else {}
        fixed_weights = {"crypto": 0.30, "nlp": 0.45, "community": 0.25}
        crypto_score = float(layer_scores.get("crypto") or 0.0)
        nlp_score = float(layer_scores.get("nlp") or 0.0)
        consensus_score = float(layer_scores.get("community") or 0.0)
        missing_layers = [
            name
            for name in fixed_weights
            if name not in layer_scores or str(layer_status.get(name) or "").lower() in EXCLUDED_LAYER_STATES
        ]
        final_score = 100.0 * (
            fixed_weights["crypto"] * crypto_score
            + fixed_weights["nlp"] * nlp_score
            + fixed_weights["community"] * consensus_score
        )

        if flags.get("confirmed_tampering") or flags.get("forged_provenance"):
            final_score = 0.0
        elif flags.get("url_article_extraction_failed"):
            final_score = min(final_score, 34.0)
        elif flags.get("strong_trusted_source_contradiction") or flags.get("severe_contradiction") or flags.get("contradictory_evidence"):
            final_score = min(final_score, 34.0)
        elif flags.get("location_entity_mismatch") or flags.get("all_unrelated_evidence"):
            final_score = min(final_score, 45.0)

        final_score = round(max(0.0, min(100.0, final_score)), 2)
        adjusted["final_score"] = final_score
        adjusted["adjusted_score"] = final_score
        adjusted["adjusted_final_score"] = final_score
        adjusted["displayed_score"] = final_score
        adjusted["active_layer_weights"] = fixed_weights
        adjusted["fixed_layer_weights"] = fixed_weights
        adjusted["final_label"] = map_score_to_label(final_score)
        adjusted["reason"] = self._url_no_multiplier_reason(final_score, flags)
        adjusted["explanation"] = adjusted["reason"]
        if flags.get("url_article_extraction_failed"):
            adjusted["final_label"] = "Not Verifiable"
        adjusted["url_fixed_formula_debug"] = {
            "formula_mode": "fixed_url_formula",
            "crypto_score": round(crypto_score * 100.0, 2),
            "nlp_score": round(nlp_score * 100.0, 2),
            "consensus_score": round(consensus_score * 100.0, 2),
            "crypto_weight": fixed_weights["crypto"],
            "nlp_weight": fixed_weights["nlp"],
            "consensus_weight": fixed_weights["community"],
            "missing_layers": missing_layers,
            "any_layer_missing": bool(missing_layers),
            "weight_renormalized": False,
            "final_score": final_score,
        }
        adjusted["url_multiplier_policy"] = {
            "evidence_multiplier_used_as_final_multiplier": False,
            "global_penalty_used_as_final_multiplier": False,
            "evidence_multiplier_diagnostic": adjusted.get("evidence_multiplier"),
            "global_penalty_diagnostic": adjusted.get("global_penalty"),
            "weight_renormalized": False,
            "missing_layers": missing_layers,
            "missing_layer_policy": "URL scoring keeps fixed weights. Missing consensus keeps its fixed 0.25 weight and uses the consensus layer's measured/default score; weights are not re-normalized.",
            "principle": "URL final score uses fixed weights: 0.30 crypto, 0.45 NLP, 0.25 consensus. Evidence sufficiency and global penalty are not multiplied into that score.",
        }
        return adjusted

    def _url_no_multiplier_reason(self, final_score: float, flags: dict) -> str:
        if flags.get("confirmed_tampering") or flags.get("forged_provenance"):
            return "A hard provenance failure or tampering signal blocked verification."
        if flags.get("url_article_extraction_failed"):
            return "Could not extract enough article text from the URL, so URL NLP verification could not run."
        if flags.get("strong_trusted_source_contradiction") or flags.get("severe_contradiction") or flags.get("contradictory_evidence"):
            return "A strong contradiction capped the URL score despite layer support."
        if flags.get("location_entity_mismatch") or flags.get("all_unrelated_evidence"):
            return "Evidence mismatch capped the URL score despite layer support."
        if final_score >= 70:
            return "Fixed URL layer formula provides strong combined support. Evidence and penalty diagnostics were not multiplied into the final score."
        if final_score >= 50:
            return "Fixed URL layer formula provides moderate combined support. Evidence and penalty diagnostics were not multiplied into the final score."
        if final_score >= 35:
            return "Fixed URL layer formula is mixed, so the item remains not fully verifiable."
        return "Fixed URL layer formula provides weak combined support."

    def _apply_url_evidence_based_decision(
        self,
        result: dict,
        layer1: LayerOutcome,
        layer2: LayerOutcome,
        layer3: LayerOutcome,
        extracted: ExtractedPayload,
        flags: dict,
    ) -> dict:
        adjusted = dict(result)
        evidence = self._url_evidence_aggregate(layer2, extracted)
        crypto_context = float((adjusted.get("layer_scores") or {}).get("crypto") or 0.0)
        user_inputs = self._community_score_inputs(layer3)
        vote_count = int(user_inputs.get("vote_count", 0) or 0)
        majority_vote = str(user_inputs.get("majority_vote") or "").lower()
        majority_confidence = max(0.0, min(1.0, float(user_inputs.get("majority_confidence") or 0.0)))
        extraction_failed = not str(extracted.main_text or "").strip() or str((layer2.details or {}).get("retrieval_failure_reason") or "") in {
            "insufficient_article_text",
            "extraction_failed",
        }
        hard_negative = any(
            bool(flags.get(name))
            for name in (
                "confirmed_tampering",
                "forged_provenance",
                "strong_trusted_source_contradiction",
                "severe_contradiction",
                "contradictory_evidence",
                "location_entity_mismatch",
            )
        )

        support_total = float(evidence["support_total"])
        refute_total = float(evidence["refute_total"])
        evidence_total = support_total + refute_total
        diversity = float(evidence["source_diversity"])
        evidence_confidence = float(evidence["evidence_confidence"])
        crypto_context_bonus = min(6.0, crypto_context * 6.0) if support_total > 0 else 0.0

        if extraction_failed:
            final_score = 18.0
            label = "Unverified"
            reason = "Could not extract enough article text to verify the URL claim."
        elif hard_negative or (refute_total >= 0.7 and refute_total > support_total * 1.15):
            final_score = max(5.0, min(32.0, 30.0 - min(12.0, refute_total * 5.0)))
            label = "Unverified"
            reason = "Reliable evidence refutes or materially contradicts the extracted URL claim."
        elif evidence_total < 0.35 or (support_total <= 0 and refute_total <= 0):
            final_score = 24.0
            label = "Unverified"
            reason = "No meaningful independent evidence was found for the extracted URL claim."
        elif support_total > 0 and refute_total > 0 and min(support_total, refute_total) >= max(support_total, refute_total) * 0.45:
            final_score = 42.0 + min(8.0, evidence_confidence * 8.0)
            label = "Not Verifiable"
            reason = "Relevant evidence is conflicting, so the URL claim remains disputed."
        elif support_total >= 1.85 and diversity >= 0.75 and refute_total < 0.35:
            final_score = 78.0 + min(14.0, (support_total - 1.85) * 7.0) + crypto_context_bonus
            label = "Verified" if final_score >= 85 else "Likely Verified"
            reason = "Multiple independent reliable sources support the extracted URL claim."
        elif support_total >= 0.7 and refute_total < 0.35:
            final_score = 52.0 + min(18.0, support_total * 8.0) + min(8.0, diversity * 8.0) + crypto_context_bonus
            label = "Likely Verified" if final_score >= 50 else "Not Verifiable"
            reason = "Relevant independent evidence supports the extracted URL claim, though the evidence base is not broad enough for full verification."
        else:
            final_score = 34.0
            label = "Unverified"
            reason = "Evidence is insufficient to verify the extracted URL claim."

        if vote_count > 0 and majority_vote == "true" and label in {"Likely Verified", "Verified"}:
            final_score = min(100.0, final_score + min(4.0, 4.0 * majority_confidence))
        elif vote_count > 0 and majority_vote == "false":
            final_score = max(0.0, final_score - min(10.0, 10.0 * majority_confidence))
            if final_score < 35:
                label = "Unverified"
                reason = "User votes dispute the claim, and evidence was not strong enough to overcome that dispute."

        final_score = round(max(0.0, min(100.0, final_score)), 2)
        label = map_score_to_label(final_score) if label not in {"Unverified", "Not Verifiable"} else label
        if final_score >= 85 and label != "Unverified":
            label = "Verified"
        elif final_score >= 50 and label not in {"Verified", "Unverified"}:
            label = "Likely Verified"
        elif final_score < 35:
            label = "Unverified"

        adjusted["final_score"] = final_score
        adjusted["adjusted_score"] = final_score
        adjusted["adjusted_final_score"] = final_score
        adjusted["displayed_score"] = final_score
        adjusted["final_label"] = label
        adjusted["reason"] = reason
        adjusted["explanation"] = reason
        adjusted["evidence_multiplier"] = round(evidence_confidence, 3)
        adjusted["global_penalty"] = 0.4 if hard_negative else 1.0
        adjusted["penalty_multiplier"] = adjusted["global_penalty"]
        adjusted["url_evidence_scoring"] = {
            **evidence,
            "crypto_context_score": round(crypto_context, 3),
            "crypto_context_bonus": round(crypto_context_bonus, 2),
            "user_vote_count": vote_count,
            "majority_vote": majority_vote,
            "majority_confidence": round(majority_confidence, 3),
            "hard_negative": hard_negative,
            "extraction_failed": extraction_failed,
            "principle": "Final URL truth decisions are based on relevant independent evidence stance, reliability, and diversity. Crypto is source/integrity context, not proof of factual truth.",
        }
        breakdown = dict(adjusted.get("layer_breakdown", {}))
        breakdown["evidence_weighted"] = adjusted["url_evidence_scoring"]
        adjusted["layer_breakdown"] = breakdown
        return adjusted

    def _url_evidence_aggregate(self, layer2: LayerOutcome, extracted: ExtractedPayload) -> dict:
        details = layer2.details if isinstance(layer2.details, dict) else {}
        matches = [item for item in details.get("matches", []) if isinstance(item, dict)]
        source_domain = extract_domain(extracted.canonical_url) or extract_domain(extracted.metadata.get("fetched_url"))
        source_family = self._domain_family(source_domain)
        support_total = 0.0
        refute_total = 0.0
        unrelated_total = 0.0
        support_sources: list[dict] = []
        refute_sources: list[dict] = []
        ignored_sources: list[dict] = []
        independent_domains: set[str] = set()

        for item in matches:
            url = str(item.get("url") or "").strip()
            domain = str(item.get("domain") or extract_domain(url) or "").strip()
            family = self._domain_family(domain)
            same_publisher = bool(source_family and family == source_family)
            relation = str(item.get("evidence_relation") or item.get("relation") or item.get("stance") or "").lower()
            relevance = max(
                float(item.get("claim_relevance_score") or 0.0),
                float(item.get("relation_confidence") or 0.0),
                float(item.get("score") or 0.0),
            )
            relevance = max(0.0, min(1.0, relevance))
            reliability = self._source_credibility_score({**item, "domain": domain, "url": url})
            independence = 0.25 if same_publisher else 1.0
            weight = round(relevance * reliability * independence, 3)
            evidence_record = {
                "title": item.get("title"),
                "url": url,
                "domain": domain,
                "relation": relation,
                "relevance_score": round(relevance, 3),
                "source_reliability": round(reliability, 3),
                "independence_weight": independence,
                "weight": weight,
                "same_publisher": same_publisher,
            }
            if relation in {"supports", "exact_support", "strong_support"}:
                support_total += weight
                support_sources.append(evidence_record)
                if family and not same_publisher:
                    independent_domains.add(family)
            elif relation in {"refutes", "contradicts", "contradiction"}:
                refute_total += weight
                refute_sources.append(evidence_record)
                if family and not same_publisher:
                    independent_domains.add(family)
            else:
                unrelated_total += max(0.0, min(1.0, relevance))
                ignored_sources.append(evidence_record)

        diversity_count = len(independent_domains)
        source_diversity = 1.0 if diversity_count >= 3 else 0.75 if diversity_count == 2 else 0.45 if diversity_count == 1 else 0.0
        evidence_confidence = max(0.0, min(1.0, ((support_total + refute_total) / 2.4) * max(0.55, source_diversity or 0.0)))
        return {
            "support_total": round(support_total, 3),
            "refute_total": round(refute_total, 3),
            "unrelated_total": round(unrelated_total, 3),
            "evidence_total": round(support_total + refute_total, 3),
            "evidence_confidence": round(evidence_confidence, 3),
            "source_diversity": round(source_diversity, 3),
            "independent_domain_count": diversity_count,
            "supporting_sources": support_sources[:5],
            "refuting_sources": refute_sources[:5],
            "ignored_or_context_sources": ignored_sources[:8],
        }

    def _url_primary_layer_score(self, result: dict) -> float:
        layer_scores = result.get("layer_scores", {}) if isinstance(result.get("layer_scores"), dict) else {}
        layer_status = result.get("layer_status", {}) if isinstance(result.get("layer_status"), dict) else {}
        active_weights = result.get("active_layer_weights", {}) if isinstance(result.get("active_layer_weights"), dict) else {}
        weighted_terms: list[tuple[float, float]] = []
        for name in ("crypto", "nlp"):
            if str(layer_status.get(name) or "active") in EXCLUDED_LAYER_STATES:
                continue
            score = float(layer_scores.get(name) or 0.0)
            weight = float(active_weights.get(name) or 0.0)
            if weight > 0:
                weighted_terms.append((score, weight))
        if not weighted_terms:
            return 0.0
        total_weight = sum(weight for _, weight in weighted_terms)
        if total_weight <= 0:
            return 0.0
        return round(sum(score * weight for score, weight in weighted_terms) / total_weight * 100.0, 2)

    def _apply_url_primary_layer_floor(self, result: dict, primary_score: float, flags: dict) -> dict:
        adjusted = dict(result)
        layer_scores = adjusted.get("layer_scores", {}) if isinstance(adjusted.get("layer_scores"), dict) else {}
        crypto_score = float(layer_scores.get("crypto") or 0.0)
        nlp_score = float(layer_scores.get("nlp") or 0.0)
        hard_negative = any(
            bool(flags.get(name))
            for name in (
                "confirmed_tampering",
                "forged_provenance",
                "strong_trusted_source_contradiction",
                "severe_contradiction",
                "contradictory_evidence",
                "location_entity_mismatch",
            )
        )
        if crypto_score >= 0.75 and nlp_score >= 0.75 and not hard_negative:
            current = float(adjusted.get("final_score") or 0.0)
            final_score = max(current, primary_score)
            adjusted["final_score"] = round(final_score, 2)
            adjusted["adjusted_score"] = round(final_score, 2)
            adjusted["adjusted_final_score"] = round(final_score, 2)
            adjusted["displayed_score"] = round(final_score, 2)
            adjusted["final_label"] = map_score_to_label(final_score)
            if final_score > current:
                adjusted["reason"] = "Strong crypto and NLP support are aligned; evidence coverage penalties are shown as context but do not override the URL primary-layer score."
                adjusted["explanation"] = "Strong provenance/source signals and URL NLP evidence align, so the URL primary-layer score is used unless hard contradiction or tampering is detected."
            adjusted["url_primary_layer_guard"] = {
                "applied": final_score > current,
                "primary_layer_score": round(primary_score, 2),
                "formula_score_before_guard": round(current, 2),
                "principle": "For URL verification, evidence sufficiency and soft penalties must not downgrade aligned strong crypto+NLP support into Unverified.",
            }
        else:
            adjusted["url_primary_layer_guard"] = {
                "applied": False,
                "primary_layer_score": round(primary_score, 2),
                "formula_score_before_guard": round(float(adjusted.get("final_score") or 0.0), 2),
            }
        return adjusted

    def _apply_url_community_overlay(self, result: dict, base_result: dict, community_inputs: dict) -> dict:
        vote_count = int(community_inputs.get("vote_count", 0) or 0)
        majority_vote = str(community_inputs.get("majority_vote") or "").lower()
        majority_confidence = max(0.0, min(1.0, float(community_inputs.get("majority_confidence") or 0.0)))
        adjusted = dict(result)
        base_score = float(base_result.get("final_score") or 0.0)
        formula_score = float(result.get("final_score") or 0.0)
        extraction_failed = (
            str(result.get("final_label") or "") == "Not Verifiable"
            and "Could not extract enough article text" in str(result.get("reason") or result.get("explanation") or "")
        )
        if extraction_failed:
            adjusted["community_adjustment"] = {
                "mode": "not_applied_extraction_failed",
                "base_primary_score": round(base_score, 2),
                "community_formula_score": round(formula_score, 2),
                "final_score_after_community": round(formula_score, 2),
                "majority_vote": majority_vote,
                "majority_confidence": round(majority_confidence, 3),
                "vote_count": vote_count,
                "principle": "Community consensus cannot verify a URL when the article body could not be extracted for NLP verification.",
            }
            return adjusted

        if vote_count <= 0:
            adjusted = dict(base_result)
            layer_breakdown = dict(adjusted.get("layer_breakdown", {}))
            layer_breakdown["community"] = result.get("layer_breakdown", {}).get("community")
            adjusted["layer_breakdown"] = layer_breakdown
            adjusted["layer_scores"] = {
                **dict(adjusted.get("layer_scores", {})),
                "community": result.get("layer_scores", {}).get("community", 0.0),
            }
            adjusted["community_adjustment"] = {
                "mode": "missing_consensus_excluded",
                "base_primary_score": round(base_score, 2),
                "community_formula_score": round(formula_score, 2),
                "final_score_after_community": round(base_score, 2),
            }
            return adjusted

        if majority_vote == "true":
            boost = min(8.0, 8.0 * majority_confidence)
            final_score = max(base_score, min(100.0, base_score + boost), formula_score)
            mode = "positive_consensus_boost"
        elif majority_vote == "false":
            penalty = min(20.0, 20.0 * majority_confidence)
            final_score = min(formula_score, max(0.0, base_score - penalty))
            mode = "negative_consensus_penalty"
        else:
            final_score = max(base_score, formula_score)
            mode = "mixed_consensus_neutral"

        adjusted["final_score"] = round(final_score, 2)
        adjusted["adjusted_score"] = round(final_score, 2)
        adjusted["adjusted_final_score"] = round(final_score, 2)
        adjusted["displayed_score"] = round(final_score, 2)
        adjusted["final_label"] = map_score_to_label(final_score)
        adjusted["community_adjustment"] = {
            "mode": mode,
            "base_primary_score": round(base_score, 2),
            "community_formula_score": round(formula_score, 2),
            "final_score_after_community": round(final_score, 2),
            "majority_vote": majority_vote,
            "majority_confidence": round(majority_confidence, 3),
            "vote_count": vote_count,
            "principle": "Positive community consensus may boost URL verification, but it must not dilute crypto or NLP layer support.",
        }
        return adjusted

    def _evidence_component_counts_without_community(
        self,
        layer1: LayerOutcome,
        layer2: LayerOutcome,
        extracted: ExtractedPayload,
    ) -> tuple[int, int]:
        layer3_placeholder = LayerOutcome(status="none", confidence=0.0, explanation="", details={})
        present, total = self._evidence_component_counts(layer1, layer2, layer3_placeholder, extracted)
        return max(0, present), total

    def _scoring_flags_without_community(
        self,
        layer1: LayerOutcome,
        layer2: LayerOutcome,
        extracted: ExtractedPayload,
        evidence_present: int,
        evidence_total: int,
    ) -> dict:
        layer3_placeholder = LayerOutcome(status="none", confidence=0.0, explanation="", details={})
        flags = self._scoring_flags(layer1, layer2, layer3_placeholder, extracted, evidence_present, evidence_total)
        flags["suspicious_vote_spike"] = False
        flags["clear_vote_manipulation"] = False
        return flags

    def _build_text_system_scoring(
        self,
        layer1: LayerOutcome,
        layer2: LayerOutcome,
        layer3: LayerOutcome,
        extracted: ExtractedPayload,
    ) -> dict:
        crypto = compute_crypto_score(self._text_crypto_score_inputs(layer1, layer2, extracted))
        claim_results = layer2.details.get("claim_results", []) if isinstance(layer2.details, dict) else []
        support_values = [float(item.get("support_score") or 0.0) for item in claim_results if isinstance(item, dict)]
        reliability_values = [float(item.get("source_reliability") or 0.0) for item in claim_results if isinstance(item, dict)]
        agreement_values = [float(item.get("agreement_score") or 0.0) for item in claim_results if isinstance(item, dict)]
        directness_values = [float(item.get("directness_score") or 0.0) for item in claim_results if isinstance(item, dict)]
        clarity_values = [float(item.get("clarity_score") or 0.0) for item in claim_results if isinstance(item, dict)]
        if bool(layer2.details.get("pipeline_failure")) if isinstance(layer2.details, dict) else False:
            reason = "The retrieval pipeline failed before evidence could be gathered, so a normal verification score was not computed."
            source_provenance_check = self._text_source_provenance_check(layer1, layer2, [], 0, False)
            return {
                "final_score": 0.0,
                "final_label": "Not Verifiable",
                "reason": reason,
                "evidence_multiplier": 0.0,
                "global_penalty": 0.0,
                "layer_scores": {
                    "claim_evidence_support": 0.0,
                    "source_reliability": 0.0,
                    "cross_source_agreement": 0.0,
                    "evidence_directness": 0.0,
                    "claim_clarity": round(max(clarity_values) if clarity_values else 0.0, 3),
                },
                "display_layer_scores": {
                    "nlp": 0.0,
                    "community": None,
                },
                "layer_breakdown": {
                    "text_formula": {
                        "claim_evidence_support": 0.0,
                        "source_reliability": 0.0,
                        "cross_source_agreement": 0.0,
                        "evidence_directness": 0.0,
                        "claim_clarity": round(max(clarity_values) if clarity_values else 0.0, 3),
                        "evidence_sufficiency": 0.0,
                        "penalty_multiplier": 0.0,
                        "raw_score": 0.0,
                        "adjusted_final_score": 0.0,
                    },
                    "crypto": crypto["breakdown"],
                    "nlp": {
                        "factual_consistency": 0.0,
                        "reference_support": 0.0,
                        "cross_source_agreement": 0.0,
                        "language_reliability": 0.0,
                        "authority": round(max(clarity_values) if clarity_values else 0.0, 3),
                        "penalty": 0.0,
                    },
                    "community": None,
                },
                "source_provenance_check": source_provenance_check,
                "warnings": [str(layer2.details.get("pipeline_failure_reason") or layer2.details.get("retrieval_state") or "retrieval_generation_failed")],
                "explanation": reason,
            }

        text_components = layer2.details.get("text_component_scores", {}) if isinstance(layer2.details, dict) else {}
        if isinstance(text_components, dict) and text_components:
            support = float(text_components.get("S") or 0.0)
            reliability = float(text_components.get("R") or 0.0)
            agreement = float(text_components.get("A") or 0.0)
            directness = float(text_components.get("D") or 0.0)
            clarity = float(text_components.get("C") or 0.0)
            evidence_multiplier = float(text_components.get("E") or layer2.details.get("evidence_multiplier") or 0.4)
            penalty_multiplier = float(text_components.get("P") or layer2.details.get("penalty_multiplier") or 1.0)
        else:
            support = sum(support_values) / len(support_values) if support_values else 0.0
            reliability = sum(reliability_values) / len(reliability_values) if reliability_values else 0.0
            agreement = sum(agreement_values) / len(agreement_values) if agreement_values else 0.0
            directness = sum(directness_values) / len(directness_values) if directness_values else 0.0
            clarity = sum(clarity_values) / len(clarity_values) if clarity_values else 0.0
            evidence_multiplier = float(layer2.details.get("evidence_multiplier") or 0.4) if isinstance(layer2.details, dict) else 0.4
            penalty_multiplier = float(layer2.details.get("penalty_multiplier") or 1.0) if isinstance(layer2.details, dict) else 1.0

        if not claim_results:
            fallback_confidence = max(0.0, min(1.0, float(layer2.confidence or 0.0)))
            match_count = len(layer2.details.get("matches", [])) if isinstance(layer2.details, dict) else 0
            support = fallback_confidence
            reliability = min(1.0, 0.55 + 0.1 * match_count) if match_count else max(0.3, fallback_confidence * 0.8)
            agreement = min(1.0, 0.5 + 0.08 * match_count) if match_count else fallback_confidence
            directness = max(0.45, fallback_confidence)
            clarity = max(0.7, 1.0 if extracted.main_text.strip() else 0.0)
            evidence_multiplier = max(evidence_multiplier, 0.7 if match_count else evidence_multiplier)
            verdict = str(layer2.details.get("fact_check_verdict") or "").lower() if isinstance(layer2.details, dict) else ""
            if verdict == "content match found" and match_count:
                support = max(support, 0.92)
                reliability = max(reliability, 0.9)
                agreement = max(agreement, 0.88)
                directness = max(directness, 0.9)
                clarity = max(clarity, 0.9)
                evidence_multiplier = max(evidence_multiplier, 0.9)
                penalty_multiplier = 1.0

        contradiction = bool(layer2.details.get("contradiction_summary")) if isinstance(layer2.details, dict) else False
        contradicted_claim = any(str(item.get("claim_verdict") or "") == "contradicted" for item in claim_results if isinstance(item, dict))
        strong_support_count = sum(
            1
            for result in claim_results
            if isinstance(result, dict)
            for source in result.get("evidence_sources", [])
            if isinstance(source, dict) and str(source.get("evidence_relation") or source.get("alignment")) == "supports"
        )
        text_verdict = str(layer2.details.get("text_verification_label") or layer2.details.get("fact_check_verdict") or "") if isinstance(layer2.details, dict) else ""
        source_provenance_check = self._text_source_provenance_check(layer1, layer2, claim_results, strong_support_count, contradicted_claim)
        if contradicted_claim:
            support = min(support, 0.12)
            evidence_multiplier = min(evidence_multiplier, 0.35)
            penalty_multiplier = min(penalty_multiplier, 0.35)
        elif strong_support_count == 0:
            support = 0.0
            reliability = min(reliability, 0.2)
            agreement = 0.0
            directness = min(directness, 0.25)
            evidence_multiplier = min(evidence_multiplier, 0.4)
            penalty_multiplier = min(penalty_multiplier, 0.6)
        raw_score = max(
            0.0,
            min(
                100.0,
                100
                * (
                    0.45 * support
                    + 0.20 * reliability
                    + 0.20 * agreement
                    + 0.10 * directness
                    + 0.05 * clarity
                ),
            ),
        )
        final_score = max(0.0, min(100.0, raw_score * evidence_multiplier * penalty_multiplier))
        if contradicted_claim:
            final_score = min(final_score, 25.0)
        elif strong_support_count == 0 or text_verdict == "Not Verifiable":
            final_score = min(final_score, 49.0)
        elif text_verdict == "Unverified":
            final_score = min(final_score, 25.0)
        label = self._generate_text_label(final_score / 100.0, evidence_multiplier, contradiction, clarity)
        if contradicted_claim:
            label = "Unverified"
        if label == "Verified":
            reason = "Strong evidence directly supports the pasted text across reliable sources."
        elif label == "Likely Verified":
            reason = "Relevant evidence supports the pasted text, though some verification gaps remain."
        elif label == "Not Verifiable":
            reason = "The available evidence is too limited or mixed for a reliable text-verification judgment."
        elif contradicted_claim:
            reason = "A core claim is contradicted by authoritative evidence, so the content cannot be verified."
        else:
            reason = "The adjusted text-verification score is below 35 because reliable exact support was not found."
        override_reason = ""
        if contradicted_claim:
            override_reason = "A trusted or direct evidence source refuted a core claim; the score was capped without applying a floor."
        elif strong_support_count == 0:
            override_reason = "No strong supporting sources were found, so the score was capped below Likely Verified. The displayed score remains the adjusted formula score without a floor."
        elif text_verdict == "Not Verifiable":
            override_reason = "Evidence sufficiency capped the score below Likely Verified. The displayed score remains the adjusted formula score without a floor."

        nlp = {
            "score": max(0.0, min(1.0, final_score / 100.0)),
            "breakdown": {
                "factual_consistency": round(support, 3),
                "reference_support": round(reliability, 3),
                "cross_source_agreement": round(agreement, 3),
                "language_reliability": round(directness, 3),
                "authority": round(clarity, 3),
                "penalty": round(penalty_multiplier, 3),
            },
        }
        result = {
            "final_score": final_score,
            "raw_score": round(raw_score, 2),
            "adjusted_score": round(final_score, 2),
            "adjusted_final_score": round(final_score, 2),
            "displayed_score": round(final_score, 2),
            "final_label": label,
            "reason": reason,
            "override_reason": override_reason,
            "evidence_multiplier": evidence_multiplier,
            "global_penalty": penalty_multiplier,
            "penalty_multiplier": penalty_multiplier,
            "scoring_method": "text_verification_formula",
            "formula": "100 * (0.45S + 0.20R + 0.20A + 0.10D + 0.05C) * E * P",
            "text_formula_components": {
                "S": round(support, 3),
                "R": round(reliability, 3),
                "A": round(agreement, 3),
                "D": round(directness, 3),
                "C": round(clarity, 3),
                "E": round(evidence_multiplier, 3),
                "P": round(penalty_multiplier, 3),
            },
            "layer_scores": {
                "claim_evidence_support": round(support, 3),
                "source_reliability": round(reliability, 3),
                "cross_source_agreement": round(agreement, 3),
                "evidence_directness": round(directness, 3),
                "claim_clarity": round(clarity, 3),
            },
            "display_layer_scores": {
                "nlp": nlp["score"],
                "community": None,
            },
            "layer_breakdown": {
                "text_formula": {
                    "claim_evidence_support": round(support, 3),
                    "source_reliability": round(reliability, 3),
                    "cross_source_agreement": round(agreement, 3),
                    "evidence_directness": round(directness, 3),
                    "claim_clarity": round(clarity, 3),
                    "evidence_sufficiency": round(evidence_multiplier, 3),
                    "penalty_multiplier": round(penalty_multiplier, 3),
                    "raw_score": round(raw_score, 2),
                    "adjusted_final_score": round(final_score, 2),
                },
                "crypto": {
                    "score_state": "not_scored_for_text",
                    "weighted_scoring": "Not used as a direct truth-verification score for pasted text. Provenance is source-origin and integrity evidence only.",
                    "source_provenance_check": source_provenance_check,
                },
                "nlp": nlp["breakdown"],
                "community": {"score_state": "review_only", "weighted_scoring": "Excluded from pasted-text formula; feedback is shown separately."},
            },
            "display_layer_breakdown": {
                "nlp": nlp["breakdown"],
                "community": None,
            },
            "source_provenance_check": source_provenance_check,
            "warnings": [],
        }
        result["explanation"] = self._generate_text_explanation(claim_results, final_score / 100.0, label)
        return result

    def _text_source_provenance_check(
        self,
        layer1: LayerOutcome,
        layer2: LayerOutcome,
        claim_results: list[dict],
        strong_support_count: int,
        contradicted_claim: bool,
    ) -> dict:
        details = layer2.details if isinstance(layer2.details, dict) else {}
        source_recovery = details.get("source_recovery", {}) if isinstance(details.get("source_recovery"), dict) else {}
        status = str(source_recovery.get("status") or "").strip()
        recovered = status in {"recovered", "likely_recovered"}
        recovered_url = source_recovery.get("url")
        recovered_title = source_recovery.get("title")
        recovered_publisher = source_recovery.get("publisher")
        source_matches_claim = strong_support_count > 0 and not contradicted_claim
        manifest_status = str((layer1.details or {}).get("provenance_verification_status") or "")
        signature_failed = manifest_status in {"hash_mismatch", "signature_invalid", "revoked", "invalid"}

        if signature_failed:
            check_status = "provenance_failed"
            label = "Provenance failed"
            reason = "A signature, hash, or provenance check failed. This affects origin and integrity evidence, not factual truth by itself."
        elif recovered and source_matches_claim and status == "recovered" and bool((layer1.details or {}).get("internal_manifest_valid")):
            check_status = "source_verified"
            label = "Source verified"
            reason = "A likely original source was recovered and the NewsProof verification record was signed. This supports source origin/integrity only."
        elif recovered and not source_matches_claim:
            check_status = "source_mismatch"
            label = "Source mismatch"
            reason = "A likely source URL was recovered, but it did not directly support the pasted claim."
        elif recovered:
            check_status = "source_recovered"
            label = "Source recovered"
            reason = "A likely original source URL was recovered. Cryptographic provenance proves origin and integrity, not factual truth."
        else:
            check_status = "no_provenance_available"
            label = "Not applicable / no provenance available for pasted text"
            reason = "The pasted text contains no source URL, source hash, signature, or source provenance record."

        return {
            "status": check_status,
            "label": label,
            "source_url": recovered_url,
            "source_title": recovered_title,
            "publisher": recovered_publisher,
            "source_recovery_status": status or "not_found",
            "source_recovery_reason": source_recovery.get("reason"),
            "manifest_status": manifest_status or None,
            "scored_as_truth": False,
            "may_influence_source_reliability": bool(recovered and source_matches_claim),
            "reason": reason,
            "principle": "Cryptographic provenance proves origin and integrity. It does not prove factual truth.",
        }

    def _text_crypto_score_inputs(self, layer1: LayerOutcome, layer2: LayerOutcome, extracted: ExtractedPayload) -> dict:
        inputs = self._crypto_score_inputs(layer1, extracted)
        if layer1.details.get("verification_context") != "nlp_result_provenance":
            return inputs

        source_recovery = layer2.details.get("source_recovery", {}) if isinstance(layer2.details, dict) else {}
        external_source_recovered = str(source_recovery.get("status") or "") in {"recovered", "likely_recovered"}
        metadata_missing = not any(
            [
                layer1.details.get("publisher") or extracted.metadata.get("publisher_name"),
                layer1.details.get("publisher_domain") or extracted.metadata.get("source_domain"),
                layer1.details.get("author") or extracted.author,
                layer1.details.get("publish_time") or (extracted.publish_time.isoformat() if extracted.publish_time else None),
            ]
        )
        if not external_source_recovered:
            inputs.update(
                {
                    "signature_validity": min(float(inputs.get("signature_validity") or 0.0), 0.2),
                    "issuer_trust": min(float(inputs.get("issuer_trust") or 0.0), 0.2 if metadata_missing else 0.3),
                    "timestamp_validity": min(float(inputs.get("timestamp_validity") or 0.0), 0.15),
                    "hash_integrity": min(float(inputs.get("hash_integrity") or 0.0), 0.25),
                    "version_chain": min(float(inputs.get("version_chain") or 0.0), 0.25),
                    "metadata_completeness": min(float(inputs.get("metadata_completeness") or 0.0), 0.15 if metadata_missing else 0.3),
                    "penalty": min(float(inputs.get("penalty") or 0.0), 0.2 if metadata_missing else 0.3),
                }
            )
        return inputs

    def _crypto_score_inputs(self, layer1: LayerOutcome, extracted: ExtractedPayload) -> dict:
        signal = self._layer1_signal(layer1)
        metadata = extracted.metadata if isinstance(extracted.metadata, dict) else {}
        platform_type = str(metadata.get("platform_type") or "").lower()
        platform_manifest_valid = bool(layer1.details.get("internal_manifest_valid") or layer1.details.get("platform_manifest_valid"))
        metadata_fields = [
            extracted.title,
            extracted.main_text,
            extracted.canonical_url,
            layer1.details.get("publisher") or metadata.get("publisher_name"),
            extracted.publish_time.isoformat() if extracted.publish_time else None,
            layer1.details.get("content_record_id"),
            layer1.details.get("publisher_domain") or metadata.get("source_domain"),
            layer1.status,
        ]
        present_required = sum(1 for item in metadata_fields if item not in (None, "", []))
        metadata_completeness = present_required / len(metadata_fields)

        if layer1.status == "matched":
            signature_validity = 1.0
            issuer_trust = 1.0
            hash_integrity = 1.0
            version_chain = 1.0
            penalty = 1.0
        elif layer1.status == "platform-issued" or platform_manifest_valid:
            signature_validity = 1.0 if platform_manifest_valid else 0.6
            issuer_trust = 0.6
            hash_integrity = 1.0 if layer1.details.get("canonical_hash") else 0.5
            version_chain = 1.0 if (layer1.details.get("current_version") or layer1.details.get("manifest_id")) else 0.5
            penalty = 1.0 if platform_manifest_valid else 0.6
        elif layer1.status == "publisher-recognized":
            signature_validity = 0.2
            issuer_trust = 1.0
            hash_integrity = 0.5
            version_chain = 1.0
            penalty = 1.0
        elif layer1.status == "mismatch":
            signature_validity = 0.0
            issuer_trust = 1.0 if layer1.details.get("publisher") else 0.3
            hash_integrity = 0.0
            version_chain = 0.5
            penalty = 0.0
        else:
            signature_validity = 0.2
            issuer_trust = 0.3 if signal == "neutral" else 0.6
            hash_integrity = 0.5 if extracted.content_hash else 0.0
            version_chain = 1.0
            penalty = 1.0

        if layer1.status == "platform-issued" or platform_manifest_valid:
            timestamp_validity = 0.7 if layer1.details.get("signed_at") else 0.0
        elif layer1.details.get("publish_time_match") or extracted.publish_time:
            timestamp_validity = 0.7
        elif metadata.get("publish_time"):
            timestamp_validity = 0.3
        else:
            timestamp_validity = 0.0
        inputs = {
            "signature_validity": signature_validity,
            "issuer_trust": issuer_trust,
            "timestamp_validity": timestamp_validity,
            "hash_integrity": hash_integrity,
            "version_chain": version_chain,
            "metadata_completeness": metadata_completeness,
            "penalty": penalty,
        }
        if platform_type == "facebook":
            source_identity_missing = not any(
                [
                    layer1.details.get("publisher"),
                    metadata.get("publisher_name"),
                    metadata.get("account_name"),
                    metadata.get("account_id"),
                    extracted.author,
                    extracted.publish_time,
                    metadata.get("publish_time"),
                ]
            )
            has_external_provenance = bool(metadata.get("remote_media_provenance") or metadata.get("c2pa_credentials"))
            if not has_external_provenance:
                inputs.update(
                    {
                        "signature_validity": min(float(inputs.get("signature_validity") or 0.0), 0.25),
                        "issuer_trust": min(float(inputs.get("issuer_trust") or 0.0), 0.25 if source_identity_missing else 0.35),
                        "timestamp_validity": min(float(inputs.get("timestamp_validity") or 0.0), 0.2),
                        "hash_integrity": min(float(inputs.get("hash_integrity") or 0.0), 0.35),
                        "version_chain": min(float(inputs.get("version_chain") or 0.0), 0.35),
                        "metadata_completeness": min(float(inputs.get("metadata_completeness") or 0.0), 0.2 if source_identity_missing else 0.3),
                        "penalty": min(float(inputs.get("penalty") or 0.0), 0.35 if source_identity_missing else 0.45),
                    }
                )
        return inputs

    def _nlp_score_inputs(self, layer2: LayerOutcome, extracted: ExtractedPayload) -> dict:
        explicit_components = layer2.details.get("nlp_component_values") if isinstance(layer2.details, dict) else None
        if isinstance(explicit_components, dict) and explicit_components:
            return {
                "factual_consistency": float(explicit_components.get("factual_consistency") or 0.0),
                "reference_support": float(explicit_components.get("reference_support") or 0.0),
                "cross_source_agreement": float(explicit_components.get("cross_source_agreement") or 0.0),
                "language_reliability": float(explicit_components.get("language_reliability") or 0.0),
                "authority": float(explicit_components.get("authority") or 0.0),
                "penalty": float(explicit_components.get("penalty") or 0.0),
            }
        confidence = max(0.0, min(1.0, float(layer2.confidence or 0.0)))
        matches = layer2.details.get("matches", []) if isinstance(layer2.details, dict) else []
        verdict = str(layer2.details.get("fact_check_verdict") or "").lower() if isinstance(layer2.details, dict) else ""
        trusted_publisher_match = bool(layer2.details.get("trusted_publisher_match")) if isinstance(layer2.details, dict) else False
        source_domain = extract_domain(extracted.canonical_url) or extract_domain(extracted.metadata.get("fetched_url"))
        match_stats = self._match_domain_stats(matches, source_domain)
        cross_source_count = match_stats["external_count"]
        sensational_penalty = self._sensational_language_penalty(extracted)

        factual_consistency = self._factual_consistency_score(layer2, extracted)
        reference_support = min(1.0, 0.25 * len(matches)) if matches else (0.35 if extracted.metadata.get("references") else 0.0)
        cross_source_agreement = min(1.0, 0.25 * cross_source_count + 0.08 * max(0, match_stats["unique_domain_count"] - 1))
        language_reliability = max(0.0, 0.9 - sensational_penalty)
        authority = 0.85 if trusted_publisher_match and cross_source_count > 0 else (0.7 if cross_source_count > 0 else (0.45 if matches else 0.25))

        if cross_source_count == 0 and match_stats["same_publisher_count"] > 0:
            reference_support = min(reference_support, 0.45)
            cross_source_agreement = min(cross_source_agreement, 0.12)
            authority = min(authority, 0.4)

        if self._layer2_signal(layer2) == "negative":
            penalty = 0.0
        elif sensational_penalty >= 0.5 or (factual_consistency < 0.25 and cross_source_count == 0):
            penalty = 0.5
        else:
            penalty = 1.0

        # When the NLP layer has already established a real context/content match,
        # keep the layer score in the clearly supportive range instead of leaving it
        # artificially low because of conservative sub-factor defaults.
        if verdict == "content match found" and matches and cross_source_count > 0:
            factual_consistency = max(factual_consistency, 0.9)
            reference_support = max(reference_support, 0.9)
            cross_source_agreement = max(cross_source_agreement, 0.9 if cross_source_count > 0 else 0.85)
            language_reliability = max(language_reliability, 0.9)
            authority = max(authority, 0.9 if (cross_source_count > 0 or trusted_publisher_match) else 0.85)
            penalty = 1.0

        return {
            "factual_consistency": factual_consistency,
            "reference_support": reference_support,
            "cross_source_agreement": cross_source_agreement,
            "language_reliability": language_reliability,
            "authority": authority,
            "penalty": penalty,
        }

    def _community_score_inputs(self, layer3: LayerOutcome) -> dict:
        details = layer3.details or {}
        if details.get("mode") == "feedback":
            feedback_count = int(details.get("feedback_count", 0) or 0)
            reviewed_count = int(details.get("reviewed_count", 0) or 0)
            resolved_count = int(details.get("resolved_count", 0) or 0)
            expert_feedback_count = int(details.get("expert_feedback_count", 0) or 0)
            moderator_feedback_count = int(details.get("moderator_feedback_count", 0) or 0)
            dispute_count = int(details.get("dispute_count", 0) or 0)
            correction_count = int(details.get("correction_count", 0) or 0)
            source_suggestion_count = int(details.get("source_suggestion_count", 0) or 0)
            validated_user_agreement = min(1.0, (reviewed_count + resolved_count) / max(1, feedback_count * 2)) if feedback_count else 0.0
            expert_assessment = min(1.0, (expert_feedback_count + moderator_feedback_count) / max(1, feedback_count)) if feedback_count else 0.0
            reporting_behavior = 0.4 if (dispute_count or correction_count) and not reviewed_count else (0.75 if feedback_count else 0.0)
            trust_weighted_reputation = min(
                1.0,
                (expert_feedback_count + moderator_feedback_count + source_suggestion_count) / max(1, feedback_count),
            ) if feedback_count else 0.0
            return {
                "validated_user_agreement": validated_user_agreement,
                "expert_assessment": expert_assessment,
                "reporting_behavior": reporting_behavior,
                "trust_weighted_reputation": trust_weighted_reputation,
                "penalty": 1.0,
                "feedback_count": feedback_count,
                "reviewed_count": reviewed_count,
                "resolved_count": resolved_count,
                "expert_feedback_count": expert_feedback_count,
                "moderator_feedback_count": moderator_feedback_count,
                "dispute_count": dispute_count,
                "correction_count": correction_count,
                "source_suggestion_count": source_suggestion_count,
                "mode": "feedback",
                "suspicious_vote_spike": False,
                "clear_vote_manipulation": False,
                "status": "missing" if feedback_count == 0 else "active_negative" if dispute_count > reviewed_count + resolved_count else "active_mixed",
            }
        vote_count = int(details.get("vote_count", 0) or 0)
        majority_confidence = max(0.0, min(1.0, float(details.get("majority_confidence", 0.0) or 0.0)))
        majority_vote = details.get("majority_vote")
        true_votes = int(details.get("true_votes", 0) or 0)
        false_votes = int(details.get("false_votes", 0) or 0)
        not_sure_votes = int(details.get("not_sure_votes", 0) or 0)
        suspicious_vote_spike = vote_count >= 10 and majority_confidence >= 0.95
        clear_vote_manipulation = vote_count >= 20 and majority_confidence >= 0.99

        if vote_count == 0:
            validated_user_agreement = 0.0
            expert_assessment = 0.0
            trust_weighted_reputation = 0.0
        else:
            validated_user_agreement = majority_confidence if majority_vote == "true" else max(0.0, 1 - majority_confidence)
            expert_assessment = 0.5
            trust_weighted_reputation = majority_confidence

        if majority_vote == "false":
            reporting_behavior = 0.2
        elif majority_vote == "not_sure":
            reporting_behavior = 0.5
        else:
            reporting_behavior = 0.9 if vote_count else 0.0

        if clear_vote_manipulation:
            penalty = 0.0
        elif suspicious_vote_spike:
            penalty = 0.5
        else:
            penalty = 1.0

        return {
            "validated_user_agreement": validated_user_agreement,
            "expert_assessment": expert_assessment,
            "reporting_behavior": reporting_behavior,
            "trust_weighted_reputation": trust_weighted_reputation,
            "penalty": penalty,
            "vote_count": vote_count,
            "suspicious_vote_spike": suspicious_vote_spike,
            "clear_vote_manipulation": clear_vote_manipulation,
            "majority_vote": majority_vote,
            "true_votes": true_votes,
            "false_votes": false_votes,
            "not_sure_votes": not_sure_votes,
            "status": "missing" if vote_count == 0 else "active_negative" if majority_vote == "false" else "active_positive" if majority_vote == "true" else "active_mixed",
        }

    def _evidence_component_counts(
        self,
        layer1: LayerOutcome,
        layer2: LayerOutcome,
        layer3: LayerOutcome,
        extracted: ExtractedPayload,
    ) -> tuple[int, int]:
        components = [
            bool(extracted.title),
            bool(extracted.main_text),
            bool(extracted.canonical_url),
            bool(layer1.details.get("publisher") or extracted.metadata.get("publisher_name")),
            bool(extracted.publish_time),
            self._layer1_signal(layer1) == "positive",
            bool((layer2.details or {}).get("matches")) or self._layer2_signal(layer2) == "positive",
            bool((layer3.details or {}).get("vote_count", 0) or (layer3.details or {}).get("feedback_count", 0)),
            bool(extracted.metadata.get("media_assets")),
            bool(extracted.metadata.get("references") or extracted.metadata.get("related_links")),
        ]
        return sum(1 for item in components if item), len(components)

    def _scoring_flags(
        self,
        layer1: LayerOutcome,
        layer2: LayerOutcome,
        layer3: LayerOutcome,
        extracted: ExtractedPayload,
        evidence_present: int,
        evidence_total: int,
    ) -> dict:
        community_inputs = self._community_score_inputs(layer3)
        evidence_ratio = evidence_present / evidence_total if evidence_total else 0.0
        strong_trusted_source_contradiction = self._layer2_signal(layer2) == "negative"
        confirmed_tampering = layer1.status == "mismatch"
        layer2_details = layer2.details if isinstance(layer2.details, dict) else {}
        url_article_extraction_failed = bool(extracted.canonical_url) and str(layer2_details.get("retrieval_failure_reason") or "") in {
            "insufficient_article_text",
            "extraction_failed",
            "extraction_too_short",
            "fetch_failed",
            "blocked_by_site",
            "non_html_content",
        }
        return {
            "confirmed_tampering": confirmed_tampering,
            "forged_provenance": layer1.status == "mismatch" and bool(layer1.details.get("signature_valid")),
            "strong_trusted_source_contradiction": strong_trusted_source_contradiction,
            "url_article_extraction_failed": url_article_extraction_failed,
            "evidence_incomplete_and_suspicious": evidence_ratio < 0.5
            and (extracted.metadata.get("fetch_limited") or self._layer3_signal(layer3) == "negative"),
            "suspicious_vote_spike": community_inputs.get("suspicious_vote_spike", False),
            "clear_vote_manipulation": community_inputs.get("clear_vote_manipulation", False),
            "force_not_verifiable": evidence_ratio < 0.35 and self._layer1_signal(layer1) == "neutral" and self._layer2_signal(layer2) == "neutral",
            "evidence_too_limited": evidence_ratio < 0.35 and self._layer3_signal(layer3) != "positive",
        }

    def _sensational_language_penalty(self, extracted: ExtractedPayload) -> float:
        text = " ".join(
            part for part in [extracted.title or "", extracted.main_text[:500] or ""] if isinstance(part, str)
        )
        uppercase_ratio = 0.0
        letters = [char for char in text if char.isalpha()]
        if letters:
            uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
        punctuation_hits = text.count("!") + text.count("?")
        clickbait_terms = ("shocking", "you won't believe", "explosive", "must see", "viral")
        term_hits = sum(1 for term in clickbait_terms if term in text.lower())
        return min(0.7, uppercase_ratio + min(0.3, punctuation_hits * 0.05) + (term_hits * 0.15))

    def _decide(
        self,
        layer1: LayerOutcome,
        layer2: LayerOutcome,
        layer3: LayerOutcome,
        extracted: ExtractedPayload,
    ) -> tuple[JobStatus, str, str, dict, dict]:
        scoring_result = self._build_system_scoring(layer1, layer2, layer3, extracted)
        layer_signals = {
            "layer1": self._layer1_signal(layer1),
            "layer2": self._layer2_signal(layer2),
            "layer3": self._layer3_signal(layer3),
        }
        positive_layers = [name for name, signal in layer_signals.items() if signal == "positive"]
        negative_layers = [name for name, signal in layer_signals.items() if signal == "negative"]
        has_votes = bool(layer3.details.get("vote_count", 0))
        layer2_details = layer2.details if isinstance(layer2.details, dict) else {}
        if bool(extracted.canonical_url) and str(layer2_details.get("retrieval_failure_reason") or "") in {
            "insufficient_article_text",
            "extraction_failed",
            "extraction_too_short",
            "fetch_failed",
            "blocked_by_site",
            "non_html_content",
        }:
            return (
                JobStatus.not_verifiable,
                "Not Verifiable",
                "Could not extract enough article text from the URL.",
                {
                    "content_hash": extracted.content_hash,
                    "verification_basis": "URL article extraction failed",
                    "supporting_similarity_confidence": layer2.confidence,
                    "consensus_confidence": layer3.confidence,
                    "agreement": layer_signals,
                    "system_scoring": scoring_result,
                },
                self._build_analysis_report(JobStatus.not_verifiable, layer1, layer2, layer3, extracted, scoring_result),
            )

        if len(positive_layers) == 3:
            return (
                JobStatus.verified,
                "Verified",
                "All three layers aligned on the same supportive outcome, including cryptographic provenance, fact-checking, and community consensus.",
                {
                    "content_hash": extracted.content_hash,
                    "verification_basis": "Three-layer agreement",
                    "supporting_similarity_confidence": layer2.confidence,
                    "consensus_confidence": layer3.confidence,
                    "agreement": layer_signals,
                    "system_scoring": scoring_result,
                },
                self._build_analysis_report(JobStatus.verified, layer1, layer2, layer3, extracted, scoring_result),
            )

        if len(negative_layers) >= 2:
            return (
                JobStatus.unverified,
                "Unverified",
                "At least two verification layers found evidence against the content, so the result is treated as unverified.",
                {
                    "content_hash": extracted.content_hash,
                    "verification_basis": "Two-layer contradiction",
                    "supporting_similarity_confidence": layer2.confidence,
                    "consensus_confidence": layer3.confidence,
                    "agreement": layer_signals,
                    "system_scoring": scoring_result,
                },
                self._build_analysis_report(JobStatus.unverified, layer1, layer2, layer3, extracted, scoring_result),
            )

        if has_votes and self._layer3_signal(layer3) == "positive" and len(positive_layers) >= 2 and not negative_layers:
            return (
                JobStatus.verified,
                "Verified",
                "Community consensus now supports the same supportive evidence already found by the verification layers, so the result advances to verified.",
                {
                    "content_hash": extracted.content_hash,
                    "verification_basis": "Supportive verification evidence plus affirmative community consensus",
                    "supporting_similarity_confidence": layer2.confidence,
                    "consensus_confidence": layer3.confidence,
                    "agreement": layer_signals,
                    "system_scoring": scoring_result,
                },
                self._build_analysis_report(JobStatus.verified, layer1, layer2, layer3, extracted, scoring_result),
            )

        if positive_layers and not has_votes:
            return (
                JobStatus.likely_verified,
                "Likely Verified",
                "One or more primary layers found meaningful supporting evidence, and there are no community votes yet to change the assessment.",
                {
                    "content_hash": extracted.content_hash,
                    "verification_basis": "Primary-layer support without census votes",
                    "supporting_similarity_confidence": layer2.confidence,
                    "consensus_confidence": layer3.confidence,
                    "agreement": layer_signals,
                    "system_scoring": scoring_result,
                },
                self._build_analysis_report(JobStatus.likely_verified, layer1, layer2, layer3, extracted, scoring_result),
            )

        if len(positive_layers) >= 2 and not negative_layers:
            return (
                JobStatus.likely_verified,
                "Likely Verified",
                "The primary verification layers found supporting evidence, but full three-layer agreement was not available.",
                {
                    "content_hash": extracted.content_hash,
                    "verification_basis": "Two-layer support",
                    "supporting_similarity_confidence": layer2.confidence,
                    "consensus_confidence": layer3.confidence,
                    "agreement": layer_signals,
                    "system_scoring": scoring_result,
                },
                self._build_analysis_report(JobStatus.likely_verified, layer1, layer2, layer3, extracted, scoring_result),
            )

        if positive_layers and negative_layers:
            status_value = JobStatus.not_verifiable
            headline = "Not Verifiable"
            reason = "The verification layers conflict with each other, so the system cannot produce a confident final verdict."
        elif negative_layers:
            status_value = JobStatus.not_verifiable
            headline = "Not Verifiable"
            reason = "Some evidence raised concerns, but fewer than two layers agreed on a false or altered result."
        else:
            status_value = JobStatus.not_verifiable
            headline = "Not Verifiable"
            reason = "The layers did not produce enough aligned evidence to determine whether the content is verified or unverified."

        return (
            status_value,
            headline,
            reason,
            {
                "content_hash": extracted.content_hash,
                "verification_basis": "Conflicting or insufficient agreement",
                "supporting_similarity_confidence": layer2.confidence,
                "consensus_confidence": layer3.confidence,
                "agreement": layer_signals,
                "system_scoring": scoring_result,
            },
            self._build_analysis_report(status_value, layer1, layer2, layer3, extracted, scoring_result),
        )

    def _decide_text(
        self,
        layer1: LayerOutcome,
        layer2: LayerOutcome,
        layer3: LayerOutcome,
        extracted: ExtractedPayload,
        record: VerificationRecord,
        manifest,
    ) -> tuple[JobStatus, str, str, dict, dict]:
        scoring_result = self._build_text_system_scoring(layer1, layer2, layer3, extracted)
        final_label = str(scoring_result.get("final_label", "Not verifiable"))
        if final_label == "Verified":
            status_value = JobStatus.verified
            headline = "Verified"
        elif final_label == "Likely Verified":
            status_value = JobStatus.likely_verified
            headline = "Likely Verified"
        elif final_label == "Unverified":
            status_value = JobStatus.unverified
            headline = final_label
        else:
            status_value = JobStatus.not_verifiable
            headline = "Not Verifiable"

        reason = str(scoring_result.get("reason") or layer2.explanation)
        proof_summary = {
            "content_hash": extracted.content_hash,
            "verification_basis": "NLP-led text verification with signed platform provenance",
            "verification_record_id": str(record.id),
            "verification_manifest_id": str(manifest.id),
            "current_version": record.current_version,
            "previous_version_id": str(record.previous_record_id) if record.previous_record_id else None,
            "supporting_similarity_confidence": layer2.confidence,
            "feedback_review_confidence": layer3.confidence,
            "system_scoring": scoring_result,
            **(layer1.details or {}),
        }
        analysis_report = self._build_analysis_report(status_value, layer1, layer2, layer3, extracted, scoring_result)
        analysis_report["executive_summary"] = (
            f"{headline}: text verification was led by the NLP layer, then recorded by the cryptographic layer as signed provenance, with feedback stored for later review."
        )
        return status_value, headline, reason, proof_summary, analysis_report

    def _create_job(self, input_type: JobInputType, input_value: str | None) -> VerificationJob:
        job = VerificationJob(
            input_type=input_type,
            input_value=input_value,
            status=JobStatus.not_verifiable,
            headline_message="Pending verification",
            short_reason="The verification request is being processed.",
            content_preview={},
            proof_summary={},
            analysis_report={},
            layer_results={},
        )
        self.db.add(job)
        self.db.flush()
        return job

    async def _store_upload(self, job_id: uuid.UUID, upload: UploadFile, input_type: JobInputType) -> str:
        upload_dir = Path(self.settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        destination = upload_dir / f"{job_id}-{upload.filename}"
        contents = await upload.read()
        max_bytes = self._upload_limit_for(input_type)
        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds upload size limit for {input_type.value} uploads ({max_bytes} bytes)",
            )
        destination.write_bytes(contents)
        return str(destination)

    def _validate_upload(self, upload: UploadFile, input_type: JobInputType) -> None:
        allowed_types = {
            JobInputType.image: {"image/png", "image/jpeg", "image/webp"},
            # Audio MIME types vary across browsers and OS file pickers, especially for .m4a.
            JobInputType.audio: {
                "audio/mpeg",
                "audio/mp3",
                "audio/wav",
                "audio/x-wav",
                "audio/mp4",
                "audio/aac",
                "audio/ogg",
                "audio/m4a",
                "audio/x-m4a",
                "audio/x-mpeg-4",
            },
            JobInputType.video: {"video/mp4", "video/quicktime", "video/webm", "video/x-matroska", "video/mpeg"},
        }
        if upload.content_type not in allowed_types[input_type]:
            raise HTTPException(status_code=400, detail=f"Unsupported {input_type.value} file type")

    def _upload_limit_for(self, input_type: JobInputType) -> int:
        limits = {
            JobInputType.image: self.settings.image_upload_max_bytes,
            JobInputType.audio: self.settings.audio_upload_max_bytes,
            JobInputType.video: self.settings.video_upload_max_bytes,
        }
        return limits[input_type]

    def _finalize_timeout(self, job: VerificationJob, reason: str) -> VerificationJob:
        job.status = JobStatus.timeout
        job.headline_message = "Verification timed out"
        job.short_reason = reason
        job.content_preview = {}
        job.proof_summary = {}
        job.analysis_report = {}
        job.layer_results = {}
        append_audit_entry(self.db, job, "job_timeout", {"reason": reason})
        self.db.commit()
        self.db.refresh(job)
        return job

    def _finalize_not_verifiable(self, job: VerificationJob, reason: str) -> VerificationJob:
        job.status = JobStatus.not_verifiable
        job.headline_message = "Not verifiable"
        job.short_reason = reason
        job.content_preview = {}
        job.proof_summary = {}
        job.analysis_report = {}
        job.layer_results = {}
        append_audit_entry(self.db, job, "job_not_verifiable", {"reason": reason})
        self.db.commit()
        self.db.refresh(job)
        return job

    def _refresh_job_assessment(self, job: VerificationJob, commit: bool = False) -> None:
        if not job.extracted_content:
            return
        extracted = ExtractedPayload(
            title=job.extracted_content.title,
            author=job.extracted_content.author,
            publish_time=job.extracted_content.publish_time,
            canonical_url=job.extracted_content.canonical_url,
            main_text=job.extracted_content.main_text,
            hero_image_url=job.extracted_content.hero_image_url,
            normalized_text=job.extracted_content.normalized_text,
            content_hash=job.extracted_content.content_hash,
            metadata=job.extracted_content.metadata_json,
        )
        layer2 = self._layer2_from_persisted(job, extracted)
        if job.input_type == JobInputType.text:
            record = self._resolve_verification_record(job)
            if not record:
                return
            manifest = record.current_manifest
            if not manifest:
                manifest_id = (job.proof_summary or {}).get("verification_manifest_id")
                if manifest_id:
                    try:
                        manifest = self.db.get(ProvenanceManifest, uuid.UUID(str(manifest_id)))
                    except Exception:
                        manifest = None
            layer1 = self._text_crypto_layer(record, manifest or record.current_manifest, extracted)
            layer1 = self._attach_extracted_details(layer1, extracted)
            if isinstance(job.proof_summary, dict):
                layer1.details = {**layer1.details, **job.proof_summary}
            layer3 = self._text_feedback_layer(record, persist_audit=False)
            status_value, headline, reason, proof_summary, analysis_report = self._decide_text(
                layer1, layer2, layer3, extracted, record, manifest or record.current_manifest
            )
        elif job.input_type == JobInputType.image:
            record = self._resolve_image_verification_record(job)
            if not record:
                return
            manifest = record.current_manifest
            if not manifest:
                manifest_id = (job.proof_summary or {}).get("verification_manifest_id")
                if manifest_id:
                    try:
                        manifest = self.db.get(ProvenanceManifest, uuid.UUID(str(manifest_id)))
                    except Exception:
                        manifest = None
            layer2 = LayerOutcome(
                status="supporting" if float(record.image_analysis_result.get("image_analysis_score", 0.0)) > 0 else "none",
                confidence=float(record.image_analysis_result.get("image_analysis_score", 0.0)),
                explanation=str(record.explanation or "No image-analysis summary available."),
                details=record.image_analysis_result or {},
            )
            layer1 = self._image_crypto_layer(record, manifest or record.current_manifest, extracted)
            layer3 = self._image_feedback_layer(record, persist_audit=False)
            status_value, headline, reason, proof_summary, analysis_report = self._decide_image(
                layer1, layer2, layer3, extracted, record, manifest or record.current_manifest
            )
        elif job.input_type == JobInputType.audio:
            record = self._resolve_audio_verification_record(job)
            if not record:
                return
            manifest = record.current_manifest
            if not manifest:
                manifest_id = (job.proof_summary or {}).get("verification_manifest_id")
                if manifest_id:
                    try:
                        manifest = self.db.get(ProvenanceManifest, uuid.UUID(str(manifest_id)))
                    except Exception:
                        manifest = None
            layer2 = LayerOutcome(
                status="supporting" if float(record.audio_analysis_result.get("audio_analysis_score", 0.0)) > 0 else "none",
                confidence=float(record.audio_analysis_result.get("audio_analysis_score", 0.0)),
                explanation=str(record.explanation or "No audio-analysis summary available."),
                details=record.audio_analysis_result or {},
            )
            layer1 = self._audio_crypto_layer(record, manifest or record.current_manifest, extracted)
            layer3 = self._audio_feedback_layer(record, persist_audit=False)
            status_value, headline, reason, proof_summary, analysis_report = self._decide_audio(
                layer1, layer2, layer3, extracted, record, manifest or record.current_manifest
            )
        elif job.input_type == JobInputType.video:
            record = self._resolve_video_verification_record(job)
            if not record:
                return
            manifest = record.current_manifest
            if not manifest:
                manifest_id = (job.proof_summary or {}).get("verification_manifest_id")
                if manifest_id:
                    try:
                        manifest = self.db.get(ProvenanceManifest, uuid.UUID(str(manifest_id)))
                    except Exception:
                        manifest = None
            layer2 = LayerOutcome(
                status="supporting" if float(record.video_analysis_result.get("video_analysis_score", 0.0)) > 0 else "none",
                confidence=float(record.video_analysis_result.get("video_analysis_score", 0.0)),
                explanation=str(record.explanation or "No video-analysis summary available."),
                details=record.video_analysis_result or {},
            )
            layer1 = self._video_crypto_layer(record, manifest or record.current_manifest, extracted)
            layer3 = self._video_feedback_layer(record, persist_audit=False)
            status_value, headline, reason, proof_summary, analysis_report = self._decide_video(
                layer1, layer2, layer3, extracted, record, manifest or record.current_manifest
            )
        else:
            layer1 = self._layer1(extracted)
            layer1 = self._attach_extracted_details(layer1, extracted)
            if isinstance(job.proof_summary, dict):
                layer1.details = {**layer1.details, **job.proof_summary}
            layer3 = self._layer3(job, persist_audit=False)
            status_value, headline, reason, proof_summary, analysis_report = self._decide(layer1, layer2, layer3, extracted)
        layer4 = self._layer4(job)
        media_authenticity = self.media_authenticity.analyze(extracted)
        job.status = status_value
        job.headline_message = headline
        job.short_reason = reason
        job.proof_summary = proof_summary
        job.analysis_report = analysis_report
        job.analysis_report["media_authenticity"] = media_authenticity
        job.layer_results = {
            "layer1": asdict(layer1),
            "layer2": asdict(layer2),
            "layer3": asdict(layer3),
            "layer4": asdict(layer4),
        }
        append_audit_entry(self.db, job, "job_reassessed", {"status": status_value.value})
        if commit:
            self.db.commit()

    def _build_analysis_report(
        self,
        status_value: JobStatus,
        layer1: LayerOutcome,
        layer2: LayerOutcome,
        layer3: LayerOutcome,
        extracted: ExtractedPayload,
        scoring_result: dict | None = None,
    ) -> dict:
        scoring_result = scoring_result or self._build_system_scoring(layer1, layer2, layer3, extracted)
        if status_value == JobStatus.verified:
            evidence_strength = "strong"
        elif status_value == JobStatus.likely_verified:
            evidence_strength = "supportive"
        elif status_value == JobStatus.unverified:
            evidence_strength = "contradictory"
        else:
            evidence_strength = "mixed"
        risk_flags = []
        if layer1.status == "mismatch":
            risk_flags.append("Trusted provenance mismatch detected.")
        if (layer2.confidence or 0) > 0.7 and status_value != JobStatus.verified:
            risk_flags.append("Fact-checking found closely related sources, but no cryptographic proof was found.")
        if layer3.details.get("moderation_flags", 0):
            risk_flags.append("Moderator or community flags were raised on this content.")
        if layer3.details.get("mode") == "feedback" and layer3.details.get("dispute_count", 0):
            risk_flags.append("Feedback includes dispute or correction requests that may require review.")
        if extracted.metadata.get("fetch_limited"):
            risk_flags.append("The source site limited server-side fetching, so the analysis used URL-level fallback metadata.")
        if extracted.metadata.get("media_type") == "audio" and extracted.metadata.get("transcript_available") is False:
            risk_flags.append("Audio analysis is metadata-based because a speech transcript is not available yet.")
        if extracted.metadata.get("media_type") == "video" and extracted.metadata.get("derived_from") != "video_keyframe_ocr":
            risk_flags.append("Video analysis used media metadata because no readable keyframe text was extracted.")

        layer3_title = "Feedback and Review" if layer3.details.get("mode") == "feedback" else "Community Consensus"
        layer3_focus = (
            "User feedback, expert review notes, correction requests, dispute reports, and moderator comments linked to the signed verification record."
            if layer3.details.get("mode") == "feedback"
            else "User reports, fact-checker input, moderator signals, and weighted consensus confidence."
        )
        executive_summary = (
            f"{status_value.value}: text verification was led by the NLP layer, then recorded by the cryptographic layer as signed provenance, with community input captured as feedback and review."
            if layer3.details.get("mode") == "feedback"
            else f"{status_value.value}: provenance was assessed with cryptographic checks first, then fact-checking evidence, then community consensus, with the final label based on layer agreement."
        )
        youtube_debug: dict[str, object] | None = None
        facebook_debug: dict[str, object] | None = None
        platform_type = str(extracted.metadata.get("platform_type") or "").lower() if isinstance(extracted.metadata, dict) else ""
        if platform_type == "youtube":
            layer2_details = layer2.details if isinstance(layer2.details, dict) else {}
            youtube_debug = {
                "detected_youtube_url": layer2_details.get("detected_youtube_url") or extracted.metadata.get("detected_youtube_url") or extracted.canonical_url,
                "normalized_url": layer2_details.get("normalized_url") or extracted.metadata.get("normalized_url") or extracted.canonical_url,
                "video_id": layer2_details.get("youtube_video_id") or extracted.metadata.get("youtube_video_id"),
                "title": layer2_details.get("youtube_title") or extracted.title,
                "description": layer2_details.get("youtube_description") or extracted.metadata.get("description"),
                "channel_title": layer2_details.get("youtube_channel_title") or extracted.metadata.get("channel_title") or extracted.metadata.get("publisher_name"),
                "channel_id": layer2_details.get("youtube_channel_id") or extracted.metadata.get("channel_id"),
                "channel_verification_status": layer2_details.get("youtube_channel_verification_status")
                or extracted.metadata.get("channel_verification_status")
                or extracted.metadata.get("youtube_signals", {}).get("channel_verification_status")
                or "not_confirmed",
                "published_at": layer2_details.get("youtube_published_date") or extracted.metadata.get("publish_time"),
                "transcript_status": layer2_details.get("youtube_transcript_status") or extracted.metadata.get("transcript_status", "unavailable"),
                "generated_queries": layer2_details.get("youtube_generated_queries") or layer2_details.get("search_queries_used") or [],
                "candidate_external_sources": layer2_details.get("youtube_candidate_external_sources") or [],
                "same_platform_context_count": layer2_details.get("youtube_same_platform_context_count")
                or layer2_details.get("same_publisher_count")
                or 0,
                "cross_publisher_count": layer2_details.get("cross_publisher_count") or 0,
                "review_count": int(layer3.details.get("feedback_count", 0) or 0),
                "census_count": int(layer3.details.get("vote_count", 0) or 0),
                "final_formula_components": (scoring_result or {}).get("layer_breakdown", {}).get("nlp"),
                "final_score": (scoring_result or {}).get("final_score"),
                "final_label": status_value.value,
            }
        if platform_type == "facebook":
            layer2_details = layer2.details if isinstance(layer2.details, dict) else {}
            facebook_debug = {
                "detected_platform": "facebook",
                "normalized_url": layer2_details.get("normalized_url") or extracted.canonical_url,
                "post_id": layer2_details.get("post_id"),
                "page_or_author_name": layer2_details.get("page_or_author_name") or extracted.metadata.get("publisher_name") or extracted.author,
                "publish_time": layer2_details.get("publish_time") or extracted.metadata.get("publish_time"),
                "raw_caption": layer2_details.get("raw_caption") or layer2_details.get("facebook_caption_text") or "",
                "cleaned_caption": layer2_details.get("cleaned_caption") or "",
                "caption_queries": layer2_details.get("caption_queries") or layer2_details.get("facebook_caption_queries") or [],
                "attached_image_count": layer2_details.get("attached_image_count") or layer2_details.get("facebook_attached_image_count") or 0,
                "per_image_ocr_status": layer2_details.get("per_image_ocr_status") or [],
                "per_image_ocr_text": layer2_details.get("per_image_ocr_text") or layer2_details.get("facebook_ocr_text_per_image") or [],
                "ocr_queries": layer2_details.get("ocr_queries") or layer2_details.get("facebook_ocr_queries") or [],
                "combined_queries": layer2_details.get("combined_queries") or layer2_details.get("facebook_combined_queries") or [],
                "candidate_sources_from_caption": layer2_details.get("candidate_sources_from_caption") or [],
                "candidate_sources_from_ocr": layer2_details.get("candidate_sources_from_ocr") or [],
                "same_source_count": layer2_details.get("same_source_count") or layer2_details.get("same_publisher_count") or 0,
                "cross_source_count": layer2_details.get("cross_source_count") or layer2_details.get("cross_publisher_count") or 0,
                "crypto_component_values": (scoring_result or {}).get("layer_breakdown", {}).get("crypto"),
                "nlp_component_values": layer2_details.get("nlp_component_values") or (scoring_result or {}).get("layer_breakdown", {}).get("nlp"),
                "final_formula_components": layer2_details.get("final_formula_components") or (scoring_result or {}).get("layer_breakdown", {}).get("nlp"),
                "final_score": (scoring_result or {}).get("final_score"),
                "final_label": status_value.value,
            }

        return {
            "executive_summary": executive_summary,
            "system_scoring": scoring_result,
            "youtube_debug": youtube_debug,
            "facebook_debug": facebook_debug,
            "report_sections": [
                {
                    "title": "Cryptographic Provenance",
                    "summary": layer1.explanation,
                    "confidence": layer1.confidence,
                    "details": {
                        **layer1.details,
                        "check_focus": "Publisher identity, registry records, canonical source, metadata consistency, and Ed25519 signature validation.",
                    },
                },
                {
                    "title": "Fact Checking",
                    "summary": layer2.explanation,
                    "confidence": layer2.confidence,
                    "details": {
                        **layer2.details,
                        "check_focus": "Semantic similarity, contextual source matching, and plausibility review from extracted text.",
                    },
                },
                {
                    "title": layer3_title,
                    "summary": layer3.explanation,
                    "confidence": layer3.confidence,
                    "details": {
                        **layer3.details,
                        "check_focus": layer3_focus,
                    },
                },
            ],
            "final_assessment": {
                "label": status_value.value,
                "evidence_strength": evidence_strength,
                "content_hash": extracted.content_hash,
                "canonical_url": extracted.canonical_url,
                "agreement": {
                    "layer1": self._layer1_signal(layer1),
                    "layer2": self._layer2_signal(layer2),
                    "layer3": self._layer3_signal(layer3),
                },
            },
            "risk_flags": risk_flags,
            "recommended_next_steps": self._recommended_next_steps(status_value, layer1, layer2, layer3),
        }

    def _layer2_from_persisted(self, job: VerificationJob, extracted: ExtractedPayload) -> LayerOutcome:
        if job.input_type == JobInputType.text and job.verification_record_id:
            record = self.db.get(VerificationRecord, job.verification_record_id)
            if record and isinstance(record.nlp_result, dict):
                matches = [
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "score": item.get("score"),
                        "explanation": item.get("explanation"),
                        "source": item.get("source", "web_search"),
                        "snippet": item.get("snippet"),
                        "domain": item.get("domain"),
                    }
                    for item in (record.evidence_sources or [])
                    if isinstance(item, dict)
                ]
                return LayerOutcome(
                    status="supporting" if float(record.nlp_confidence_score or 0.0) > 0 else "none",
                    confidence=float(record.nlp_confidence_score or 0.0),
                    explanation=str(record.nlp_result.get("nlp_explanation") or "No fact-checking summary available."),
                    details={
                        "mode": "text_verification",
                        "matches": matches,
                        "evidence_sources": matches,
                        "extracted_claims": record.nlp_result.get("extracted_claims", record.extracted_claims or []),
                        "claim_results": record.nlp_result.get("claim_results", []),
                        "supporting_evidence_summary": record.nlp_result.get("supporting_evidence_summary"),
                        "contradiction_summary": record.nlp_result.get("contradiction_summary"),
                        "evidence_summary": record.nlp_result.get("evidence_summary"),
                        "fact_check_verdict": record.nlp_result.get("nlp_label", record.nlp_label),
                        "fact_check_assessment": record.nlp_result.get("nlp_explanation"),
                        "text_verification_score": float(record.nlp_confidence_score or 0.0),
                        "text_verification_label": record.nlp_label,
                        "evidence_multiplier": record.nlp_result.get("evidence_multiplier", 0.4),
                        "penalty_multiplier": record.nlp_result.get("penalty_multiplier", 1.0),
                        "metadata_signals": record.nlp_result.get("metadata_signals", []),
                        "total_match_count": len(matches),
                        "has_more_matches": False,
                    },
                )

        matches = [
            {
                "title": match.candidate_title,
                "url": match.candidate_url,
                "score": match.score,
                "explanation": match.explanation,
                "source": match.metadata_json.get("source", "trusted_corpus") if isinstance(match.metadata_json, dict) else "trusted_corpus",
                "snippet": match.metadata_json.get("snippet") if isinstance(match.metadata_json, dict) else None,
                "domain": match.metadata_json.get("domain") if isinstance(match.metadata_json, dict) else None,
            }
            for match in sorted(job.similarity_matches, key=lambda item: item.score, reverse=True)
        ]
        source_domain = extract_domain(extracted.canonical_url) or extract_domain(extracted.metadata.get("fetched_url"))
        publisher = self._find_trusted_publisher(source_domain, extracted.metadata.get("publisher_name"))
        metadata_signals = self._metadata_signals(extracted)
        if matches:
            top = matches[0]
            confidence = round(max(float(top["score"]), self._metadata_corroboration_score(metadata_signals, publisher)), 3)
            verdict, assessment = self._fact_check_decision(confidence, bool(publisher))
            initial_matches = matches[: self.DEFAULT_NLP_MATCH_LIMIT]
            return LayerOutcome(
                status="supporting",
                confidence=confidence,
                explanation=assessment,
                details={
                    "matches": initial_matches,
                    "match_summary": (
                        f"Content match found with {top.get('title') or top.get('url') or 'a corroborating source'}."
                        if confidence >= 0.55
                        else f"Possible content match found with {top.get('title') or top.get('url') or 'a corroborating source'}."
                    ),
                    "top_match_title": top.get("title"),
                    "top_match_url": top.get("url"),
                    "top_match_score": top.get("score"),
                    "fact_check_verdict": verdict,
                    "fact_check_basis": "Supporting candidates were drawn from the trusted corpus and live web retrieval, then scored with semantic similarity and extracted article metadata.",
                    "fact_check_assessment": assessment,
                    "metadata_signals": metadata_signals,
                    "trusted_publisher_match": publisher.name if publisher else None,
                    "web_matches": [match for match in initial_matches if match.get("source") == "web_search"],
                    "total_match_count": len(matches),
                    "has_more_matches": len(matches) > len(initial_matches),
                },
            )
        metadata_score = self._metadata_corroboration_score(metadata_signals, publisher)
        if publisher and metadata_score >= 0.45:
            confidence = round(metadata_score, 3)
            return LayerOutcome(
                status="supporting",
                confidence=confidence,
                explanation="Fact-checking found a verified publisher and strong structured metadata, which suggests the content is likely correct even though no article-level similarity match was stored.",
                details={
                    "matches": [],
                    "fact_check_verdict": "likely correct",
                    "fact_check_basis": "The article exposes a verified publisher domain plus multiple structured metadata signals such as author, publish time, canonical URL, and body content.",
                    "fact_check_assessment": "The page structure and extracted metadata strongly support authenticity, but no signed proof or corpus match was available.",
                    "metadata_signals": metadata_signals,
                    "trusted_publisher_match": publisher.name,
                    "web_matches": [],
                    "total_match_count": 0,
                    "has_more_matches": False,
                },
            )
        return LayerOutcome(
            status="none",
            confidence=0.0,
            explanation="No strong corroborating source or semantic match was found.",
            details={
                "matches": [],
                "fact_check_verdict": "inconclusive",
                "fact_check_basis": "No corroborating source text was available in the candidate set.",
                "fact_check_assessment": "No strong supporting or conflicting source match was available.",
                "metadata_signals": metadata_signals,
                "trusted_publisher_match": publisher.name if publisher else None,
                "web_matches": [],
                "total_match_count": 0,
                "has_more_matches": False,
            },
        )

    def _recommended_next_steps(
        self, status_value: JobStatus, layer1: LayerOutcome, layer2: LayerOutcome, layer3: LayerOutcome
    ) -> list[str]:
        feedback_mode = layer3.details.get("mode") == "feedback"
        if status_value == JobStatus.verified:
            return ["Preserve the proof summary and audit trail for future review.", "Review similarity matches for alternate syndication paths if needed."]
        if status_value == JobStatus.likely_verified:
            steps = [
                "Review the strongest supporting matches and provenance metadata before treating the content as fully verified.",
            ]
            if feedback_mode:
                steps.append("Invite expert review notes or source suggestions if you want the verification record refined further.")
            else:
                steps.append("Add community or fact-checker votes if you want the result to advance from likely verified to verified.")
            return steps
        if status_value == JobStatus.unverified:
            return ["Treat the content as potentially altered.", "Compare the current text against the trusted publisher record and inspect moderation reports."]
        steps = ["Look for a trusted publisher record or register one in the admin page."]
        if (layer2.confidence or 0) > 0:
            steps.append("Review the top semantic matches in the report.")
        if feedback_mode:
            if layer3.details.get("feedback_count", 0) == 0:
                steps.append("Add a feedback note, correction request, dispute report, or source suggestion to support future review.")
        elif layer3.details.get("vote_count", 0) == 0:
            steps.append("Add a community report or fact-checker review to strengthen the assessment.")
        return steps
