from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from app.models import ArticleSnapshot, AudioVerificationRecord, EvidenceBundle, ImageVerificationRecord, ReviewEvent, VerificationRecord, VideoVerificationRecord
from app.services.crypto.hasher import sha256_hex
from app.utils import canonicalize_url

CANONICALIZATION_VERSION = "2026-04-18.crypto.v2"

STABLE_COMPONENT_FIELDS = (
    "headings",
    "paragraphs",
    "pull_quotes",
    "captions",
    "footnotes",
    "embedded_references",
    "tables",
    "chart_descriptions",
    "linked_documents",
)

STABLE_REFERENCE_FIELDS = (
    "hyperlinks",
    "quoted_sources",
    "report_references",
    "dataset_references",
    "law_or_policy_references",
    "social_post_references",
)


def normalize_unicode(value: str | None) -> str:
    if not value:
        return ""
    return unicodedata.normalize("NFKC", value)


def normalize_whitespace(value: str | None) -> str:
    return re.sub(r"\s+", " ", normalize_unicode(value)).strip()


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(_normalize_jsonish(data), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonicalize_article_snapshot(snapshot: ArticleSnapshot) -> str:
    return canonical_json(build_article_payload(snapshot))


def canonicalize_evidence_bundle(bundle: EvidenceBundle) -> str:
    return canonical_json(build_evidence_payload(bundle))


def canonicalize_review_event(event: ReviewEvent) -> str:
    return canonical_json(build_review_event_payload(event))


def canonicalize_verification_record(record: VerificationRecord) -> str:
    return canonical_json(build_verification_record_payload(record))


def canonicalize_image_verification_record(record: ImageVerificationRecord) -> str:
    return canonical_json(build_image_verification_record_payload(record))


def canonicalize_audio_verification_record(record: AudioVerificationRecord) -> str:
    return canonical_json(build_audio_verification_record_payload(record))


def canonicalize_video_verification_record(record: VideoVerificationRecord) -> str:
    return canonical_json(build_video_verification_record_payload(record))


def build_article_payload(snapshot: ArticleSnapshot) -> dict[str, Any]:
    metadata = snapshot.metadata_json or {}
    canonical_url = canonicalize_url(str(metadata.get("canonical_url") or snapshot.source_url))
    components = _build_article_components(metadata, snapshot.normalized_article_text)
    assets = _build_article_assets(snapshot, metadata)
    references = _build_article_references(metadata)
    transport_evidence = _build_transport_evidence(metadata.get("transport_evidence", {}))
    version_number = int(metadata.get("version_number") or (2 if snapshot.previous_version_id else 1))
    payload_hash = sha256_hex(
        canonical_json(
            {
                "title": normalize_whitespace(snapshot.title),
                "subtitle": normalize_whitespace(metadata.get("subtitle")),
                "summary": normalize_whitespace(metadata.get("summary")),
                "body": normalize_whitespace(snapshot.normalized_article_text),
                "canonical_url": canonical_url,
                "components": components,
                "assets": assets,
                "references": references,
            }
        ).encode("utf-8")
    )

    return {
        "article": {
            "article_id": str(metadata.get("article_id") or snapshot.id),
            "title": normalize_whitespace(snapshot.title),
            "subtitle": normalize_whitespace(metadata.get("subtitle")),
            "summary": normalize_whitespace(metadata.get("summary")),
            "lede": normalize_whitespace(metadata.get("summary") or metadata.get("lede")),
            "full_body_text": normalize_whitespace(snapshot.normalized_article_text),
            "canonical_url": canonical_url,
            "article_slug": normalize_whitespace(metadata.get("article_slug") or _slug_from_url(canonical_url)),
            "content_type": normalize_whitespace(metadata.get("content_type") or "news"),
            "language": normalize_whitespace(metadata.get("language")),
            "category": normalize_whitespace(metadata.get("category")),
            "tags": _normalize_text_list(metadata.get("tags") or metadata.get("keywords")),
            "status": normalize_whitespace(metadata.get("article_status") or "published"),
            "created_at": _serialize_datetime(metadata.get("created_at")),
            "published_at": _serialize_datetime(snapshot.published_at or metadata.get("published_at")),
            "updated_at": _serialize_datetime(metadata.get("updated_at")),
            "timezone": normalize_whitespace(metadata.get("timezone")) or "UTC",
            "version_number": version_number,
            "previous_version_id": str(snapshot.previous_version_id) if snapshot.previous_version_id else None,
            "previous_payload_hash": normalize_whitespace(metadata.get("previous_payload_hash")),
            "change_reason": normalize_whitespace(metadata.get("change_reason")),
            "payload_hash": payload_hash,
            "components": components,
        },
        "source": {
            "publisher_name": normalize_whitespace(snapshot.publisher_name),
            "publisher_id": normalize_whitespace(metadata.get("publisher_id")),
            "publisher_domain": normalize_whitespace(metadata.get("publisher_domain") or _domain_from_url(canonical_url)),
            "section_name": normalize_whitespace(metadata.get("section_name") or metadata.get("section")),
            "author_names": _normalize_text_list(metadata.get("author_names")) or _normalize_text_list([snapshot.author_name]),
            "author_ids": _normalize_text_list(metadata.get("author_ids")),
            "editor_names": _normalize_text_list(metadata.get("editor_names")),
            "editor_ids": _normalize_text_list(metadata.get("editor_ids")),
            "source_newsroom": normalize_whitespace(metadata.get("source_newsroom")),
            "source_bureau": normalize_whitespace(metadata.get("source_newsroom") or metadata.get("bureau")),
            "syndication_source": normalize_whitespace(metadata.get("syndication_source")),
        },
        "assets": assets,
        "references": references,
        "transport_evidence": transport_evidence,
        "version_chain": {
            "current_version_number": version_number,
            "version_number": version_number,
            "previous_version_id": str(snapshot.previous_version_id) if snapshot.previous_version_id else None,
            "previous_payload_hash": normalize_whitespace(metadata.get("previous_payload_hash")),
            "chain_position": version_number,
            "change_type": normalize_whitespace(metadata.get("change_type") or ("update" if snapshot.previous_version_id else "first_publish")),
            "change_reason": normalize_whitespace(metadata.get("change_reason")),
            "diff_summary": normalize_whitespace(metadata.get("diff_summary")),
        },
        "canonicalization": {
            "canonicalization_version": normalize_whitespace(metadata.get("canonicalization_version") or CANONICALIZATION_VERSION),
            "unicode_normalization": "NFKC",
            "timestamp_normalization": "ISO8601_UTC",
            "component_order_preserved": True,
            "excluded_dynamic_elements": [
                "ad_blocks",
                "impression_counters",
                "comment_counts",
                "related_story_widgets",
                "recommendation_widgets",
                "session_specific_ui",
                "login_state",
                "dynamic_formatting_classes",
                "personalized_content",
                "analytics_scripts",
            ],
        },
        "crypto": {
            "payload_hash": payload_hash,
            "manifest_hash": None,
            "issuer_id": normalize_whitespace(metadata.get("issuer_id")),
            "key_id": normalize_whitespace(metadata.get("key_id")),
            "key_version": normalize_whitespace(metadata.get("key_version") or "v1"),
            "signature_algorithm": "Ed25519",
            "signature_value": None,
            "signature_created_at": None,
        },
    }


def build_evidence_payload(bundle: EvidenceBundle) -> dict[str, Any]:
    metadata = bundle.metadata_json or {}
    extracted_claims = [normalize_whitespace(item) for item in (bundle.extracted_claims or []) if normalize_whitespace(item)]
    claim_entries = [
        {
            "component_id": f"claim-{index + 1}",
            "component_type": "extracted_claim",
            "component_order": index,
            "raw_value": claim,
            "normalized_value": claim,
            "component_hash": sha256_hex(claim.encode("utf-8")),
        }
        for index, claim in enumerate(extracted_claims)
    ]
    references = [
        {
            "reference_text": normalize_whitespace(link),
            "target_url": canonicalize_url(str(link)),
            "source_name": None,
            "position_in_article": index,
            "normalized_reference_hash": sha256_hex(normalize_whitespace(link).encode("utf-8")),
        }
        for index, link in enumerate(bundle.source_links or [])
        if normalize_whitespace(str(link))
    ]

    return {
        "article": {
            "linked_article_id": str(bundle.linked_article_id),
            "bundle_id": str(bundle.id),
        },
        "source": {
            "created_by": normalize_whitespace(bundle.created_by),
        },
        "assets": [],
        "references": references,
        "transport_evidence": {},
        "version_chain": {
            "current_version_number": int(metadata.get("version_number") or (2 if bundle.previous_version_id else 1)),
            "previous_version_id": str(bundle.previous_version_id) if bundle.previous_version_id else None,
            "previous_payload_hash": normalize_whitespace(metadata.get("previous_payload_hash")),
            "chain_position": int(metadata.get("version_number") or (2 if bundle.previous_version_id else 1)),
            "change_type": normalize_whitespace(metadata.get("change_type") or ("update" if bundle.previous_version_id else "first_publish")),
            "change_reason": normalize_whitespace(metadata.get("change_reason")),
            "diff_summary": normalize_whitespace(metadata.get("diff_summary")),
        },
        "canonicalization": {
            "canonicalization_version": normalize_whitespace(metadata.get("canonicalization_version") or CANONICALIZATION_VERSION),
            "unicode_normalization": "NFKC",
            "timestamp_normalization": "ISO8601_UTC",
            "component_order_preserved": True,
        },
        "evidence_bundle": {
            "evidence_id": str(bundle.id),
            "linked_article_id": str(bundle.linked_article_id),
            "source_links": sorted(canonicalize_url(str(item)) for item in (bundle.source_links or []) if str(item).strip()),
            "extracted_claims": claim_entries,
            "supporting_notes": normalize_whitespace(bundle.supporting_notes),
            "created_by": normalize_whitespace(bundle.created_by),
            "created_at": _serialize_datetime(bundle.created_at),
            "metadata": _normalize_jsonish(metadata),
        },
    }


def build_review_event_payload(event: ReviewEvent) -> dict[str, Any]:
    payload = event.event_payload or {}
    version_number = int(payload.get("version_number") or (2 if event.previous_version_id else 1))
    normalized_event_payload = _normalize_jsonish(payload)

    return {
        "article": {
            "linked_article_id": str(event.linked_article_id),
            "event_id": str(event.id),
        },
        "source": {
            "created_by": normalize_whitespace(event.created_by),
        },
        "assets": [],
        "references": [],
        "transport_evidence": {},
        "version_chain": {
            "current_version_number": version_number,
            "previous_version_id": str(event.previous_version_id) if event.previous_version_id else None,
            "previous_payload_hash": normalize_whitespace(payload.get("previous_payload_hash")),
            "chain_position": version_number,
            "change_type": normalize_whitespace(payload.get("change_type") or event.event_type.value),
            "change_reason": normalize_whitespace(payload.get("change_reason")),
            "diff_summary": normalize_whitespace(payload.get("diff_summary")),
        },
        "canonicalization": {
            "canonicalization_version": normalize_whitespace(payload.get("canonicalization_version") or CANONICALIZATION_VERSION),
            "unicode_normalization": "NFKC",
            "timestamp_normalization": "ISO8601_UTC",
            "component_order_preserved": True,
        },
        "review_event": {
            "event_id": str(event.id),
            "linked_article_id": str(event.linked_article_id),
            "event_type": event.event_type.value,
            "event_payload": normalized_event_payload,
            "created_by": normalize_whitespace(event.created_by),
            "created_at": _serialize_datetime(event.created_at),
        },
    }


def build_verification_record_payload(record: VerificationRecord) -> dict[str, Any]:
    nlp_result = _normalize_jsonish(record.nlp_result or {})
    extracted_claims = [
        {
            "claim_id": f"claim-{index + 1}",
            "raw_value": normalize_whitespace(str(claim)),
            "claim_hash": sha256_hex(normalize_whitespace(str(claim)).encode("utf-8")),
        }
        for index, claim in enumerate(record.extracted_claims or [])
        if normalize_whitespace(str(claim))
    ]
    evidence_sources = [
        {
            "source_id": f"evidence-{index + 1}",
            "source_name": normalize_whitespace(str(item.get("title") or item.get("source_name") or "")) if isinstance(item, dict) else "",
            "target_url": canonicalize_url(str(item.get("url") or item.get("target_url") or "")) if isinstance(item, dict) else None,
            "reference_hash": sha256_hex(
                canonical_json(_normalize_jsonish(item)).encode("utf-8") if isinstance(item, dict) else normalize_whitespace(str(item)).encode("utf-8")
            ),
        }
        for index, item in enumerate(record.evidence_sources or [])
    ]
    canonical_payload = canonical_json(
        {
            "input_text_hash": record.input_text_hash,
            "normalized_text": normalize_whitespace(record.normalized_text),
            "claims": extracted_claims,
            "evidence_sources": evidence_sources,
            "nlp_result": nlp_result,
            "current_version": record.current_version,
            "previous_record_id": str(record.previous_record_id) if record.previous_record_id else None,
        }
    )
    payload_hash = sha256_hex(canonical_payload.encode("utf-8"))

    return {
        "article": {
            "verification_record_id": str(record.id),
            "input_text": normalize_whitespace(record.input_text),
            "normalized_text": normalize_whitespace(record.normalized_text),
            "input_text_hash": record.input_text_hash,
            "extracted_claims": extracted_claims,
            "evidence_sources": evidence_sources,
            "nlp_result": nlp_result,
            "nlp_confidence_score": record.nlp_confidence_score,
            "nlp_label": normalize_whitespace(record.nlp_label),
            "created_at": _serialize_datetime(record.created_at),
            "updated_at": _serialize_datetime(record.updated_at),
        },
        "source": {
            "issuer_name": "NewsProof system",
            "issuer_type": "platform",
            "verification_record_id": str(record.id),
        },
        "assets": [],
        "references": evidence_sources,
        "transport_evidence": {},
        "version_chain": {
            "current_version_number": record.current_version,
            "version_number": record.current_version,
            "previous_version_id": str(record.previous_record_id) if record.previous_record_id else None,
            "previous_payload_hash": None,
            "chain_position": record.current_version,
            "change_type": "verification_update" if record.previous_record_id else "first_publish",
            "change_reason": "NLP verification result recorded",
            "diff_summary": normalize_whitespace(nlp_result.get("revision_summary")),
        },
        "canonicalization": {
            "canonicalization_version": CANONICALIZATION_VERSION,
            "unicode_normalization": "NFKC",
            "timestamp_normalization": "ISO8601_UTC",
            "component_order_preserved": True,
        },
        "crypto": {
            "payload_hash": payload_hash,
            "manifest_hash": None,
            "issuer_id": normalize_whitespace(nlp_result.get("issuer_id") or "newsproof-platform"),
            "key_id": normalize_whitespace(nlp_result.get("key_id") or "platform-managed"),
            "key_version": normalize_whitespace(nlp_result.get("key_version") or "v1"),
            "signature_algorithm": "Ed25519",
            "signature_value": None,
            "signature_created_at": None,
        },
    }


def build_image_verification_record_payload(record: ImageVerificationRecord) -> dict[str, Any]:
    ocr_result = _normalize_jsonish(record.ocr_result or {})
    analysis = _normalize_jsonish(record.image_analysis_result or {})
    crypto_result = _normalize_jsonish(record.crypto_result or {})
    canonical_payload = canonical_json(
        {
            "image_hash": normalize_whitespace(record.image_hash),
            "perceptual_hash": normalize_whitespace(record.perceptual_hash),
            "caption_or_claim": normalize_whitespace(record.caption_or_claim),
            "ocr_result": ocr_result,
            "image_analysis_result": analysis,
            "crypto_result": crypto_result,
            "current_version": record.current_version,
            "previous_record_id": str(record.previous_record_id) if record.previous_record_id else None,
        }
    )
    payload_hash = sha256_hex(canonical_payload.encode("utf-8"))

    return {
        "article": {
            "image_verification_record_id": str(record.id),
            "uploaded_filename": normalize_whitespace(record.uploaded_filename),
            "caption_or_claim": normalize_whitespace(record.caption_or_claim),
            "image_hash": normalize_whitespace(record.image_hash),
            "perceptual_hash": normalize_whitespace(record.perceptual_hash),
            "ocr_result": ocr_result,
            "analysis_snapshot": analysis,
            "crypto_result": crypto_result,
            "final_score": record.final_score,
            "final_label": normalize_whitespace(record.final_label),
            "explanation": normalize_whitespace(record.explanation),
            "created_at": _serialize_datetime(record.created_at),
            "updated_at": _serialize_datetime(record.updated_at),
            "payload_hash": payload_hash,
        },
        "source": {
            "issuer_name": "NewsProof system",
            "issuer_type": "platform",
            "verification_record_id": str(record.id),
        },
        "assets": [
            {
                "asset_id": "uploaded-image",
                "asset_type": "image",
                "asset_url": None,
                "asset_hash": normalize_whitespace(record.image_hash),
                "caption_hash": sha256_hex(normalize_whitespace(record.caption_or_claim).encode("utf-8")) if normalize_whitespace(record.caption_or_claim) else None,
                "credit": None,
                "mime_type": normalize_whitespace((analysis.get("metadata_snapshot") or {}).get("mime_type")),
            }
        ],
        "references": _normalize_jsonish(analysis.get("evidence_sources") or []),
        "transport_evidence": {},
        "version_chain": {
            "current_version_number": record.current_version,
            "version_number": record.current_version,
            "previous_version_id": str(record.previous_record_id) if record.previous_record_id else None,
            "previous_payload_hash": normalize_whitespace(crypto_result.get("previous_payload_hash")),
            "chain_position": record.current_version,
            "change_type": "verification_update" if record.previous_record_id else "first_publish",
            "change_reason": "Image verification result recorded",
            "diff_summary": normalize_whitespace(analysis.get("revision_summary")),
        },
        "canonicalization": {
            "canonicalization_version": CANONICALIZATION_VERSION,
            "unicode_normalization": "NFKC",
            "timestamp_normalization": "ISO8601_UTC",
            "component_order_preserved": True,
        },
        "crypto": {
            "payload_hash": payload_hash,
            "manifest_hash": None,
            "issuer_id": normalize_whitespace(crypto_result.get("issuer_id") or "newsproof-platform"),
            "key_id": normalize_whitespace(crypto_result.get("key_id") or "platform-managed"),
            "key_version": normalize_whitespace(crypto_result.get("key_version") or "v1"),
            "signature_algorithm": "Ed25519",
            "signature_value": None,
            "signature_created_at": None,
        },
    }


def build_audio_verification_record_payload(record: AudioVerificationRecord) -> dict[str, Any]:
    analysis = _normalize_jsonish(record.audio_analysis_result or {})
    crypto_result = _normalize_jsonish(record.crypto_result or {})
    transcript_result = _normalize_jsonish(analysis.get("transcription_result") or {})
    canonical_payload = canonical_json(
        {
            "audio_hash": normalize_whitespace(record.audio_hash),
            "audio_fingerprint": normalize_whitespace(record.audio_fingerprint),
            "caption_or_claim": normalize_whitespace(record.caption_or_claim),
            "claimed_speaker": normalize_whitespace(record.claimed_speaker),
            "transcript_text": normalize_whitespace(record.transcript_text),
            "transcript_summary": normalize_whitespace(record.transcript_summary),
            "analysis_result": analysis,
            "crypto_result": crypto_result,
            "current_version": record.current_version,
            "previous_record_id": str(record.previous_record_id) if record.previous_record_id else None,
        }
    )
    payload_hash = sha256_hex(canonical_payload.encode("utf-8"))

    return {
        "article": {
            "audio_verification_record_id": str(record.id),
            "uploaded_filename": normalize_whitespace(record.uploaded_filename),
            "caption_or_claim": normalize_whitespace(record.caption_or_claim),
            "claimed_speaker": normalize_whitespace(record.claimed_speaker),
            "audio_hash": normalize_whitespace(record.audio_hash),
            "audio_fingerprint": normalize_whitespace(record.audio_fingerprint),
            "transcript_text": normalize_whitespace(record.transcript_text),
            "transcript_summary": normalize_whitespace(record.transcript_summary),
            "analysis_snapshot": analysis,
            "transcription_result": transcript_result,
            "crypto_result": crypto_result,
            "final_score": record.final_score,
            "final_label": normalize_whitespace(record.final_label),
            "explanation": normalize_whitespace(record.explanation),
            "created_at": _serialize_datetime(record.created_at),
            "updated_at": _serialize_datetime(record.updated_at),
            "payload_hash": payload_hash,
        },
        "source": {
            "issuer_name": "NewsProof system",
            "issuer_type": "platform",
            "verification_record_id": str(record.id),
        },
        "assets": [
            {
                "asset_id": "uploaded-audio",
                "asset_type": "audio",
                "asset_url": None,
                "asset_hash": normalize_whitespace(record.audio_hash),
                "caption_hash": sha256_hex(normalize_whitespace(record.caption_or_claim).encode("utf-8")) if normalize_whitespace(record.caption_or_claim) else None,
                "credit": normalize_whitespace(record.claimed_speaker),
                "mime_type": normalize_whitespace((analysis.get("metadata_snapshot") or {}).get("mime_type")),
            }
        ],
        "references": _normalize_jsonish(analysis.get("evidence_sources") or []),
        "transport_evidence": {},
        "version_chain": {
            "current_version_number": record.current_version,
            "version_number": record.current_version,
            "previous_version_id": str(record.previous_record_id) if record.previous_record_id else None,
            "previous_payload_hash": normalize_whitespace(crypto_result.get("previous_payload_hash")),
            "chain_position": record.current_version,
            "change_type": "verification_update" if record.previous_record_id else "first_publish",
            "change_reason": "Audio verification result recorded",
            "diff_summary": normalize_whitespace(analysis.get("revision_summary")),
        },
        "canonicalization": {
            "canonicalization_version": CANONICALIZATION_VERSION,
            "unicode_normalization": "NFKC",
            "timestamp_normalization": "ISO8601_UTC",
            "component_order_preserved": True,
        },
        "crypto": {
            "payload_hash": payload_hash,
            "manifest_hash": None,
            "issuer_id": normalize_whitespace(crypto_result.get("issuer_id") or "newsproof-platform"),
            "key_id": normalize_whitespace(crypto_result.get("key_id") or "platform-managed"),
            "key_version": normalize_whitespace(crypto_result.get("key_version") or "v1"),
            "signature_algorithm": "Ed25519",
            "signature_value": None,
            "signature_created_at": None,
        },
    }


def build_video_verification_record_payload(record: VideoVerificationRecord) -> dict[str, Any]:
    ocr_result = _normalize_jsonish(record.ocr_result or {})
    audio_result = _normalize_jsonish(record.audio_result or {})
    analysis = _normalize_jsonish(record.video_analysis_result or {})
    crypto_result = _normalize_jsonish(record.crypto_result or {})
    canonical_payload = canonical_json(
        {
            "video_hash": normalize_whitespace(record.video_hash),
            "video_fingerprint": normalize_whitespace(record.video_fingerprint),
            "caption_or_claim": normalize_whitespace(record.caption_or_claim),
            "ocr_result": ocr_result,
            "audio_result": audio_result,
            "analysis_result": analysis,
            "crypto_result": crypto_result,
            "current_version": record.current_version,
            "previous_record_id": str(record.previous_record_id) if record.previous_record_id else None,
        }
    )
    payload_hash = sha256_hex(canonical_payload.encode("utf-8"))

    return {
        "article": {
            "video_verification_record_id": str(record.id),
            "uploaded_filename": normalize_whitespace(record.uploaded_filename),
            "caption_or_claim": normalize_whitespace(record.caption_or_claim),
            "video_hash": normalize_whitespace(record.video_hash),
            "video_fingerprint": normalize_whitespace(record.video_fingerprint),
            "ocr_result": ocr_result,
            "audio_result": audio_result,
            "analysis_snapshot": analysis,
            "crypto_result": crypto_result,
            "final_score": record.final_score,
            "final_label": normalize_whitespace(record.final_label),
            "explanation": normalize_whitespace(record.explanation),
            "created_at": _serialize_datetime(record.created_at),
            "updated_at": _serialize_datetime(record.updated_at),
            "payload_hash": payload_hash,
        },
        "source": {
            "issuer_name": "NewsProof system",
            "issuer_type": "platform",
            "verification_record_id": str(record.id),
        },
        "assets": [
            {
                "asset_id": "uploaded-video",
                "asset_type": "video",
                "asset_url": None,
                "asset_hash": normalize_whitespace(record.video_hash),
                "caption_hash": sha256_hex(normalize_whitespace(record.caption_or_claim).encode("utf-8")) if normalize_whitespace(record.caption_or_claim) else None,
                "credit": None,
                "mime_type": normalize_whitespace((analysis.get("metadata_snapshot") or {}).get("mime_type")),
            }
        ],
        "references": _normalize_jsonish(analysis.get("evidence_sources") or []),
        "transport_evidence": {},
        "version_chain": {
            "current_version_number": record.current_version,
            "version_number": record.current_version,
            "previous_version_id": str(record.previous_record_id) if record.previous_record_id else None,
            "previous_payload_hash": normalize_whitespace(crypto_result.get("previous_payload_hash")),
            "chain_position": record.current_version,
            "change_type": "verification_update" if record.previous_record_id else "first_publish",
            "change_reason": "Video verification result recorded",
            "diff_summary": normalize_whitespace(analysis.get("revision_summary")),
        },
        "canonicalization": {
            "canonicalization_version": CANONICALIZATION_VERSION,
            "unicode_normalization": "NFKC",
            "timestamp_normalization": "ISO8601_UTC",
            "component_order_preserved": True,
        },
        "crypto": {
            "payload_hash": payload_hash,
            "manifest_hash": None,
            "issuer_id": normalize_whitespace(crypto_result.get("issuer_id") or "newsproof-platform"),
            "key_id": normalize_whitespace(crypto_result.get("key_id") or "platform-managed"),
            "key_version": normalize_whitespace(crypto_result.get("key_version") or "v1"),
            "signature_algorithm": "Ed25519",
            "signature_value": None,
            "signature_created_at": None,
        },
    }


def _build_article_components(metadata: dict[str, Any], body_text: str) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    explicit_components = metadata.get("components")
    if isinstance(explicit_components, list) and explicit_components:
        for index, component in enumerate(explicit_components):
            if not isinstance(component, dict):
                continue
            normalized_value = normalize_whitespace(component.get("normalized_value") or component.get("raw_value"))
            components.append(
                {
                    "component_id": normalize_whitespace(component.get("component_id")) or f"component-{index + 1}",
                    "component_type": normalize_whitespace(component.get("component_type")) or "component",
                    "component_order": int(component.get("component_order", index)),
                    "raw_value": normalize_whitespace(component.get("raw_value")),
                    "normalized_value": normalized_value,
                    "component_hash": sha256_hex(normalized_value.encode("utf-8")),
                }
            )
    else:
        order = 0
        for field_name in STABLE_COMPONENT_FIELDS:
            values = metadata.get(field_name, [])
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                continue
            component_type = field_name.rstrip("s")
            for value in values:
                normalized_value = normalize_whitespace(str(value))
                if not normalized_value:
                    continue
                components.append(
                    {
                        "component_id": f"{component_type}-{order + 1}",
                        "component_type": component_type,
                        "component_order": order,
                        "raw_value": normalize_whitespace(str(value)),
                        "normalized_value": normalized_value,
                        "component_hash": sha256_hex(normalized_value.encode("utf-8")),
                    }
                )
                order += 1
        if not components and normalize_whitespace(body_text):
            for index, paragraph in enumerate(_split_paragraphs(body_text)):
                components.append(
                    {
                        "component_id": f"paragraph-{index + 1}",
                        "component_type": "paragraph",
                        "component_order": index,
                        "raw_value": paragraph,
                        "normalized_value": paragraph,
                        "component_hash": sha256_hex(paragraph.encode("utf-8")),
                    }
                )

    return sorted(components, key=lambda item: (item["component_order"], item["component_id"]))


def _build_article_assets(snapshot: ArticleSnapshot, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    explicit_assets = metadata.get("assets", [])
    media_hash_map = {}
    for item in snapshot.media_hashes or []:
        if isinstance(item, dict) and item.get("url"):
            media_hash_map[str(item.get("url"))] = item

    assets: list[dict[str, Any]] = []
    if isinstance(explicit_assets, list) and explicit_assets:
        iterable = explicit_assets
    else:
        iterable = [
            {
                "asset_type": "media",
                "asset_url": url,
            }
            for url in snapshot.media_urls or []
        ]

    for index, asset in enumerate(iterable):
        if not isinstance(asset, dict):
            asset = {"asset_url": asset, "asset_type": "media"}
        asset_url = canonicalize_url(str(asset.get("asset_url") or asset.get("original_asset_url") or asset.get("url") or ""))
        if not asset_url:
            continue
        caption = normalize_whitespace(asset.get("caption"))
        credit = normalize_whitespace(asset.get("credit"))
        media_hash_entry = media_hash_map.get(asset_url, {})
        asset_hash = (
            normalize_whitespace(asset.get("asset_hash"))
            or normalize_whitespace(media_hash_entry.get("sha256_hash"))
            or sha256_hex(asset_url.encode("utf-8"))
        )
        assets.append(
            {
                "asset_id": normalize_whitespace(asset.get("asset_id")) or f"asset-{index + 1}",
                "asset_type": normalize_whitespace(asset.get("asset_type") or asset.get("type") or "media"),
                "asset_url": asset_url,
                "original_asset_url": canonicalize_url(str(asset.get("original_asset_url") or asset_url)),
                "asset_hash": asset_hash,
                "caption_hash": sha256_hex(caption.encode("utf-8")) if caption else None,
                "caption": caption,
                "alt_text": normalize_whitespace(asset.get("alt_text") or asset.get("alt")),
                "credit": credit,
                "mime_type": normalize_whitespace(asset.get("mime_type") or media_hash_entry.get("content_type")),
                "asset_size": _coerce_int(asset.get("asset_size") or media_hash_entry.get("content_length")),
            }
        )
    return assets


def _build_article_references(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    explicit_references = metadata.get("references")
    references: list[dict[str, Any]] = []
    if isinstance(explicit_references, list) and explicit_references:
        iterable = explicit_references
    else:
        iterable = []
        for field_name in STABLE_REFERENCE_FIELDS:
            values = metadata.get(field_name, [])
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                continue
            for index, value in enumerate(values):
                iterable.append(
                    {
                        "reference_text": value,
                        "target_url": value if "http" in str(value) else None,
                        "source_name": None if "http" in str(value) else value,
                        "position_in_article": index,
                    }
                )
    for index, reference in enumerate(iterable):
        if not isinstance(reference, dict):
            reference = {"reference_text": reference}
        reference_text = normalize_whitespace(reference.get("reference_text"))
        target_url = canonicalize_url(str(reference.get("target_url"))) if reference.get("target_url") else None
        source_name = normalize_whitespace(reference.get("source_name"))
        hash_source = reference_text or target_url or source_name
        if not hash_source:
            continue
        references.append(
            {
                "reference_text": reference_text,
                "target_url": target_url,
                "source_name": source_name,
                "position_in_article": _coerce_int(reference.get("position_in_article")) or index,
                "normalized_reference_hash": sha256_hex(normalize_whitespace(hash_source).encode("utf-8")),
            }
        )
    return references


def _build_transport_evidence(raw: dict[str, Any]) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    return {
        "https_enabled": bool(raw.get("https_enabled")),
        "certificate_subject": normalize_whitespace(raw.get("certificate_subject")),
        "certificate_issuer": normalize_whitespace(raw.get("certificate_issuer")),
        "certificate_valid_from": _serialize_datetime(raw.get("certificate_valid_from")),
        "certificate_valid_to": _serialize_datetime(raw.get("certificate_valid_to")),
        "certificate_hostname_match": bool(raw.get("certificate_hostname_match")) if raw.get("certificate_hostname_match") is not None else None,
        "weighting_note": "Transport evidence is informational only and is not treated as article-level signature proof.",
    }


def _split_paragraphs(body_text: str) -> list[str]:
    body = normalize_whitespace(body_text)
    if not body:
        return []
    paragraphs = [normalize_whitespace(part) for part in re.split(r"\n{2,}|(?<=\.)\s{2,}", body) if normalize_whitespace(part)]
    return paragraphs or [body]


def _normalize_text_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return [normalize_whitespace(str(value)) for value in values if normalize_whitespace(str(value))]


def _serialize_datetime(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, str):
        return normalize_whitespace(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return normalize_whitespace(str(value))


def _normalize_jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_jsonish(value[key]) for key in sorted(value.keys(), key=lambda item: str(item))}
    if isinstance(value, list):
        return [_normalize_jsonish(item) for item in value]
    if isinstance(value, str):
        return normalize_whitespace(value)
    if isinstance(value, datetime):
        return _serialize_datetime(value)
    return value


def _coerce_int(value: Any) -> int | None:
    if value in (None, "", False):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _slug_from_url(url: str | None) -> str | None:
    if not url:
        return None
    path = urlparse(url).path.strip("/")
    return normalize_whitespace(path.split("/")[-1]) if path else None


def _domain_from_url(url: str | None) -> str | None:
    if not url:
        return None
    return normalize_whitespace(urlparse(url).netloc)
