from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    ArticleSnapshot,
    AudioVerificationRecord,
    CryptoObjectType,
    EvidenceBundle,
    ImageVerificationRecord,
    IssuerCredential,
    ProvenanceManifest,
    ProvenanceVerificationStatus,
    ReviewEvent,
    ReviewEventType,
    TimestampLog,
    VerificationRecord,
    VerificationLog,
    VideoVerificationRecord,
)
from app.services.crypto.canonicalizer import (
    canonicalize_article_snapshot,
    canonicalize_audio_verification_record,
    canonicalize_evidence_bundle,
    canonicalize_image_verification_record,
    canonicalize_review_event,
    canonicalize_verification_record,
    canonicalize_video_verification_record,
    normalize_whitespace,
)
from app.services.crypto.hasher import sha256_hex
from app.services.crypto.issuer_registry import ensure_default_issuers
from app.services.crypto.signer import get_managed_public_key_b64, sign_digest
from app.services.crypto.timestamp_provider import get_timestamp_provider
from app.services.crypto.verifier import compare_canonical_json, latest_manifest_for_object, verify_manifest


class CryptoProvenanceService:
    def __init__(self, db: Session):
        self.db = db

    def ensure_default_issuers(self) -> None:
        ensure_default_issuers(self.db)

    def list_issuers(self) -> list[IssuerCredential]:
        return list(self.db.scalars(select(IssuerCredential).order_by(IssuerCredential.issuer_name.asc())))

    def get_issuer(self, issuer_id: UUID) -> IssuerCredential:
        issuer = self.db.get(IssuerCredential, issuer_id)
        if not issuer:
            raise ValueError("Issuer not found")
        return issuer

    def create_issuer(
        self,
        issuer_type: str,
        issuer_name: str,
        public_key: str,
        issued_at: datetime | None = None,
        expires_at: datetime | None = None,
    ) -> IssuerCredential:
        issuer = IssuerCredential(
            issuer_type=issuer_type,
            issuer_name=issuer_name,
            public_key=public_key,
            status="active",
            issued_at=issued_at or datetime.utcnow(),
            expires_at=expires_at,
            metadata_json={},
        )
        self.db.add(issuer)
        self.db.commit()
        self.db.refresh(issuer)
        return issuer

    def revoke_issuer(self, issuer_id: UUID) -> IssuerCredential:
        issuer = self.get_issuer(issuer_id)
        issuer.status = "revoked"
        self.db.add(issuer)
        self.db.commit()
        self.db.refresh(issuer)
        return issuer

    def create_article_snapshot(self, payload: dict[str, Any]) -> ArticleSnapshot:
        article = ArticleSnapshot(
            source_url=str(payload["source_url"]),
            publisher_name=normalize_whitespace(payload["publisher_name"]),
            author_name=normalize_whitespace(payload.get("author_name")),
            title=normalize_whitespace(payload["title"]),
            published_at=payload.get("published_at"),
            captured_at=payload.get("captured_at") or datetime.utcnow(),
            normalized_article_text=normalize_whitespace(payload["normalized_article_text"]),
            media_urls=payload.get("media_urls", []),
            media_hashes=self._hash_media_urls(payload.get("media_urls", [])),
            metadata_json=payload.get("metadata", {}),
            previous_version_id=payload.get("previous_version_id"),
        )
        self.db.add(article)
        self.db.commit()
        self.db.refresh(article)
        return article

    def create_evidence_bundle(self, payload: dict[str, Any]) -> EvidenceBundle:
        bundle = EvidenceBundle(
            linked_article_id=payload["linked_article_id"],
            source_links=payload.get("source_links", []),
            extracted_claims=[normalize_whitespace(item) for item in payload.get("extracted_claims", [])],
            supporting_notes=normalize_whitespace(payload.get("supporting_notes")),
            created_by=normalize_whitespace(payload["created_by"]),
            created_at=payload.get("created_at") or datetime.utcnow(),
            metadata_json=payload.get("metadata", {}),
            previous_version_id=payload.get("previous_version_id"),
        )
        self.db.add(bundle)
        self.db.commit()
        self.db.refresh(bundle)
        return bundle

    def create_review_event(self, payload: dict[str, Any]) -> ReviewEvent:
        event = ReviewEvent(
            linked_article_id=payload["linked_article_id"],
            event_type=ReviewEventType(payload["event_type"]),
            event_payload=payload.get("event_payload", {}),
            created_by=normalize_whitespace(payload["created_by"]),
            created_at=payload.get("created_at") or datetime.utcnow(),
            previous_version_id=payload.get("previous_version_id"),
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def sign_object(self, object_type: CryptoObjectType, object_id: UUID, signer_id: UUID | None = None) -> ProvenanceManifest:
        self.ensure_default_issuers()
        signer = self._resolve_signer(signer_id)
        canonical_json = self._canonical_json_for_object(object_type, object_id)
        sha256_hash = sha256_hex(canonical_json.encode("utf-8"))
        signature = sign_digest(sha256_hash)
        timestamp_token = get_timestamp_provider().issue(sha256_hash)
        previous_manifest_id = self._previous_manifest_id(object_type, object_id)
        manifest = ProvenanceManifest(
            object_type=object_type,
            object_id=object_id,
            canonical_json=canonical_json,
            sha256_hash=sha256_hash,
            signature=signature,
            signing_algorithm="Ed25519",
            signer_id=signer.id,
            signer_certificate_or_public_key_reference=signer.public_key,
            timestamp_token=timestamp_token,
            version_number=self._version_number_for_object(object_type, object_id),
            previous_manifest_id=previous_manifest_id,
            verification_status=ProvenanceVerificationStatus.valid,
        )
        self.db.add(manifest)
        self.db.flush()
        self.db.add(TimestampLog(manifest_id=manifest.id, provider_type=timestamp_token.get("provider", "internal"), token=timestamp_token))
        verification_result = verify_manifest(self.db, manifest)
        manifest.verification_status = ProvenanceVerificationStatus(verification_result["status"])
        self.db.add(
            VerificationLog(
                manifest_id=manifest.id,
                verified_by=signer.issuer_name,
                verification_status=manifest.verification_status,
                result_json=verification_result,
            )
        )
        self.db.commit()
        self.db.refresh(manifest)
        return manifest

    def verify_manifest(self, manifest_id: UUID, verified_by: str | None = None) -> dict[str, Any]:
        manifest = self.get_manifest(manifest_id)
        result = verify_manifest(self.db, manifest)
        manifest.verification_status = ProvenanceVerificationStatus(result["status"])
        self.db.add(manifest)
        self.db.add(
            VerificationLog(
                manifest_id=manifest.id,
                verified_by=verified_by,
                verification_status=manifest.verification_status,
                result_json=result,
            )
        )
        self.db.commit()
        self.db.refresh(manifest)
        return result

    def get_manifest(self, manifest_id: UUID) -> ProvenanceManifest:
        manifest = self.db.get(ProvenanceManifest, manifest_id)
        if not manifest:
            raise ValueError("Manifest not found")
        return manifest

    def serialize_manifest(self, manifest: ProvenanceManifest) -> dict[str, Any]:
        verification_result = verify_manifest(self.db, manifest)
        payload = json.loads(manifest.canonical_json)
        return {
            "manifest_id": manifest.id,
            "object_type": manifest.object_type,
            "object_id": manifest.object_id,
            "article": payload.get("article", {}),
            "source": payload.get("source", {}),
            "assets": payload.get("assets", []),
            "references": payload.get("references", []),
            "transport_evidence": payload.get("transport_evidence", {}),
            "version_chain": payload.get("version_chain", {}),
            "canonicalization": payload.get("canonicalization", {}),
            "crypto": verification_result.get("crypto", {}),
            "verification": {
                "status": verification_result.get("status"),
                "valid": verification_result.get("valid"),
                "checks": verification_result.get("checks", {}),
            },
            "confidence": verification_result.get("confidence", {}),
            "explanation": verification_result.get("explanation", {}),
            "canonical_json": manifest.canonical_json,
            "created_at": manifest.created_at,
            "previous_manifest_id": manifest.previous_manifest_id,
            "verification_status": manifest.verification_status,
        }

    def record_history(self, object_type: CryptoObjectType, record_id: UUID) -> dict[str, Any]:
        chain = self._record_chain(object_type, record_id)
        return {
            "object_type": object_type.value,
            "record_id": str(record_id),
            "versions": [self._serialize_record_summary(object_type, item) for item in chain],
            "manifests": [self._serialize_manifest_summary(latest_manifest_for_object(self.db, object_type, item.id)) for item in chain if latest_manifest_for_object(self.db, object_type, item.id)],
        }

    def compare_versions(self, object_type: CryptoObjectType, left_id: UUID, right_id: UUID) -> dict[str, Any]:
        left_json = self._canonical_json_for_object(object_type, left_id)
        right_json = self._canonical_json_for_object(object_type, right_id)
        diff = compare_canonical_json(left_json, right_json)
        return {
            "object_type": object_type.value,
            "left_version_id": str(left_id),
            "right_version_id": str(right_id),
            **diff,
        }

    def _resolve_signer(self, signer_id: UUID | None) -> IssuerCredential:
        managed_public_key = get_managed_public_key_b64()
        if signer_id:
            signer = self.get_issuer(signer_id)
            if signer.status == "revoked":
                raise ValueError("Signer is revoked")
            if signer.public_key != managed_public_key:
                raise ValueError("Signer does not have a managed server-side signing key")
            return signer
        signer = self.db.scalar(select(IssuerCredential).where(IssuerCredential.public_key == managed_public_key).limit(1))
        if not signer:
            raise ValueError("Managed platform issuer not found")
        return signer

    def _canonical_json_for_object(self, object_type: CryptoObjectType, object_id: UUID) -> str:
        if object_type == CryptoObjectType.article_snapshot:
            record = self.db.get(ArticleSnapshot, object_id)
            if not record:
                raise ValueError("Article snapshot not found")
            return canonicalize_article_snapshot(record)
        if object_type == CryptoObjectType.evidence_bundle:
            record = self.db.get(EvidenceBundle, object_id)
            if not record:
                raise ValueError("Evidence bundle not found")
            return canonicalize_evidence_bundle(record)
        if object_type == CryptoObjectType.review_event:
            record = self.db.get(ReviewEvent, object_id)
            if not record:
                raise ValueError("Review event not found")
            return canonicalize_review_event(record)
        if object_type == CryptoObjectType.verification_record:
            record = self.db.get(VerificationRecord, object_id)
            if not record:
                raise ValueError("Verification record not found")
            return canonicalize_verification_record(record)
        if object_type == CryptoObjectType.image_verification_record:
            record = self.db.get(ImageVerificationRecord, object_id)
            if not record:
                raise ValueError("Image verification record not found")
            return canonicalize_image_verification_record(record)
        if object_type == CryptoObjectType.audio_verification_record:
            record = self.db.get(AudioVerificationRecord, object_id)
            if not record:
                raise ValueError("Audio verification record not found")
            return canonicalize_audio_verification_record(record)
        if object_type == CryptoObjectType.video_verification_record:
            record = self.db.get(VideoVerificationRecord, object_id)
            if not record:
                raise ValueError("Video verification record not found")
            return canonicalize_video_verification_record(record)
        raise ValueError("Unsupported object type")

    def _previous_manifest_id(self, object_type: CryptoObjectType, object_id: UUID) -> UUID | None:
        previous_object_id = None
        if object_type == CryptoObjectType.article_snapshot:
            record = self.db.get(ArticleSnapshot, object_id)
            previous_object_id = record.previous_version_id if record else None
        elif object_type == CryptoObjectType.evidence_bundle:
            record = self.db.get(EvidenceBundle, object_id)
            previous_object_id = record.previous_version_id if record else None
        elif object_type == CryptoObjectType.review_event:
            record = self.db.get(ReviewEvent, object_id)
            previous_object_id = record.previous_version_id if record else None
        elif object_type == CryptoObjectType.verification_record:
            record = self.db.get(VerificationRecord, object_id)
            previous_object_id = record.previous_record_id if record else None
        elif object_type == CryptoObjectType.image_verification_record:
            record = self.db.get(ImageVerificationRecord, object_id)
            previous_object_id = record.previous_record_id if record else None
        elif object_type == CryptoObjectType.audio_verification_record:
            record = self.db.get(AudioVerificationRecord, object_id)
            previous_object_id = record.previous_record_id if record else None
        elif object_type == CryptoObjectType.video_verification_record:
            record = self.db.get(VideoVerificationRecord, object_id)
            previous_object_id = record.previous_record_id if record else None
        if not previous_object_id:
            return None
        previous_manifest = latest_manifest_for_object(self.db, object_type, previous_object_id)
        return previous_manifest.id if previous_manifest else None

    def _record_chain(self, object_type: CryptoObjectType, record_id: UUID) -> list[Any]:
        chain: list[Any] = []
        current_id: UUID | None = record_id
        while current_id:
            record = self._get_record(object_type, current_id)
            if not record:
                break
            chain.append(record)
            current_id = getattr(record, "previous_version_id", None) or getattr(record, "previous_record_id", None)
        chain.reverse()
        return chain

    def _get_record(self, object_type: CryptoObjectType, record_id: UUID) -> Any:
        model_map = {
            CryptoObjectType.article_snapshot: ArticleSnapshot,
            CryptoObjectType.evidence_bundle: EvidenceBundle,
            CryptoObjectType.review_event: ReviewEvent,
            CryptoObjectType.verification_record: VerificationRecord,
            CryptoObjectType.image_verification_record: ImageVerificationRecord,
            CryptoObjectType.audio_verification_record: AudioVerificationRecord,
            CryptoObjectType.video_verification_record: VideoVerificationRecord,
        }
        model = model_map[object_type]
        return self.db.get(model, record_id)

    def _version_number_for_object(self, object_type: CryptoObjectType, object_id: UUID) -> int:
        if object_type == CryptoObjectType.verification_record:
            record = self.db.get(VerificationRecord, object_id)
            return int(record.current_version) if record else 1
        if object_type == CryptoObjectType.image_verification_record:
            record = self.db.get(ImageVerificationRecord, object_id)
            return int(record.current_version) if record else 1
        if object_type == CryptoObjectType.audio_verification_record:
            record = self.db.get(AudioVerificationRecord, object_id)
            return int(record.current_version) if record else 1
        if object_type == CryptoObjectType.video_verification_record:
            record = self.db.get(VideoVerificationRecord, object_id)
            return int(record.current_version) if record else 1
        previous_manifest_id = self._previous_manifest_id(object_type, object_id)
        if not previous_manifest_id:
            return 1
        previous_manifest = self.db.get(ProvenanceManifest, previous_manifest_id)
        return int(previous_manifest.version_number) + 1 if previous_manifest else 1

    def _serialize_record_summary(self, object_type: CryptoObjectType, record: Any) -> dict[str, Any]:
        base = {
            "id": str(record.id),
            "created_at": record.created_at.isoformat() if getattr(record, "created_at", None) else None,
            "previous_version_id": str(
                getattr(record, "previous_version_id", None) or getattr(record, "previous_record_id", None)
            )
            if (getattr(record, "previous_version_id", None) or getattr(record, "previous_record_id", None))
            else None,
        }
        if object_type == CryptoObjectType.article_snapshot:
            return {
                **base,
                "title": record.title,
                "publisher_name": record.publisher_name,
                "source_url": record.source_url,
                "captured_at": record.captured_at.isoformat() if record.captured_at else None,
            }
        if object_type == CryptoObjectType.evidence_bundle:
            return {**base, "linked_article_id": str(record.linked_article_id), "created_by": record.created_by}
        if object_type == CryptoObjectType.verification_record:
            return {
                **base,
                "input_text_hash": record.input_text_hash,
                "nlp_label": record.nlp_label,
                "nlp_confidence_score": record.nlp_confidence_score,
                "current_version": record.current_version,
                "current_manifest_id": str(record.current_manifest_id) if record.current_manifest_id else None,
            }
        if object_type == CryptoObjectType.image_verification_record:
            return {
                **base,
                "image_hash": record.image_hash,
                "perceptual_hash": record.perceptual_hash,
                "final_label": record.final_label,
                "final_score": record.final_score,
                "current_version": record.current_version,
                "current_manifest_id": str(record.current_manifest_id) if record.current_manifest_id else None,
            }
        if object_type == CryptoObjectType.audio_verification_record:
            return {
                **base,
                "audio_hash": record.audio_hash,
                "audio_fingerprint": record.audio_fingerprint,
                "final_label": record.final_label,
                "final_score": record.final_score,
                "current_version": record.current_version,
                "current_manifest_id": str(record.current_manifest_id) if record.current_manifest_id else None,
            }
        if object_type == CryptoObjectType.video_verification_record:
            return {
                **base,
                "video_hash": record.video_hash,
                "video_fingerprint": record.video_fingerprint,
                "final_label": record.final_label,
                "final_score": record.final_score,
                "current_version": record.current_version,
                "current_manifest_id": str(record.current_manifest_id) if record.current_manifest_id else None,
            }
        return {
            **base,
            "linked_article_id": str(record.linked_article_id),
            "event_type": record.event_type.value,
            "created_by": record.created_by,
        }

    def _serialize_manifest_summary(self, manifest: ProvenanceManifest | None) -> dict[str, Any] | None:
        if not manifest:
            return None
        verification_result = verify_manifest(self.db, manifest)
        payload = json.loads(manifest.canonical_json)
        return {
            "manifest_id": str(manifest.id),
            "object_id": str(manifest.object_id),
            "sha256_hash": manifest.sha256_hash,
            "created_at": manifest.created_at.isoformat(),
            "verification_status": manifest.verification_status.value,
            "signer_id": str(manifest.signer_id),
            "version_number": payload.get("version_chain", {}).get("version_number"),
            "change_type": payload.get("version_chain", {}).get("change_type"),
            "confidence_score": verification_result.get("confidence", {}).get("score"),
        }

    def _hash_media_urls(self, media_urls: list[str]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for url in media_urls:
            cleaned = str(url).strip()
            if not cleaned:
                continue
            try:
                with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                    response = client.get(cleaned)
                    response.raise_for_status()
                    digest = sha256_hex(response.content)
                    entries.append(
                        {
                            "url": cleaned,
                            "sha256_hash": digest,
                            "content_type": response.headers.get("content-type"),
                            "content_length": response.headers.get("content-length"),
                            "status": "hashed",
                        }
                    )
            except Exception as exc:
                entries.append({"url": cleaned, "sha256_hash": None, "status": "unavailable", "error": str(exc)})
        return entries
