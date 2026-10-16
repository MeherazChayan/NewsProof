from __future__ import annotations

from typing import Any

from app.models import ProvenanceVerificationStatus


REQUIRED_MANIFEST_FIELDS = (
    "title",
    "body",
    "canonical_url",
    "publisher",
    "published_at",
    "version_number",
    "payload_hash",
    "issuer_id",
    "key_id",
    "signature_or_no_signature_status",
)


def compute_metadata_completeness(manifest_payload: dict[str, Any], verification: dict[str, Any]) -> float:
    article = manifest_payload.get("article", {})
    source = manifest_payload.get("source", {})
    version_chain = manifest_payload.get("version_chain", {})
    crypto = verification.get("crypto", {})
    score_map = {
        "title": bool(article.get("title")),
        "body": bool(article.get("full_body_text")),
        "canonical_url": bool(article.get("canonical_url")),
        "publisher": bool(source.get("publisher_id") or source.get("publisher_name")),
        "published_at": bool(
            article.get("published_at")
            or article.get("created_at")
            or article.get("updated_at")
            or verification.get("timestamps", {}).get("signed_at")
        ),
        "version_number": version_chain.get("version_number") is not None,
        "payload_hash": bool(article.get("payload_hash")),
        "issuer_id": bool(crypto.get("issuer_id")),
        "key_id": bool(crypto.get("key_id")),
        "signature_or_no_signature_status": bool(crypto.get("signature_value")) or verification.get("status") == ProvenanceVerificationStatus.no_provenance.value,
    }
    present = sum(1 for field in REQUIRED_MANIFEST_FIELDS if score_map[field])
    return present / len(REQUIRED_MANIFEST_FIELDS)


def compute_crypto_confidence(verification: dict[str, Any], manifest_payload: dict[str, Any]) -> dict[str, Any]:
    status = verification["status"]
    signature_score = verification["factors"]["signature_validity"]
    issuer_score = verification["factors"]["issuer_trust"]
    timestamp_score = verification["factors"]["timestamp_validity"]
    hash_score = verification["factors"]["hash_integrity"]
    version_score = verification["factors"]["version_chain"]
    metadata_score = compute_metadata_completeness(manifest_payload, verification)
    penalty = verification["factors"]["penalty_multiplier"]

    weighted = (
        0.35 * signature_score
        + 0.20 * issuer_score
        + 0.15 * timestamp_score
        + 0.15 * hash_score
        + 0.10 * version_score
        + 0.05 * metadata_score
    )
    score = round(100 * weighted * penalty, 2)

    if status in {
        ProvenanceVerificationStatus.hash_mismatch.value,
        ProvenanceVerificationStatus.signature_invalid.value,
    }:
        score = 0.0

    band = _interpretation_band(score)
    return {
        "score": score,
        "band": band,
        "breakdown": {
            "signature_validity": signature_score,
            "issuer_trust": issuer_score,
            "timestamp_validity": timestamp_score,
            "hash_integrity": hash_score,
            "version_chain": version_score,
            "metadata_completeness": round(metadata_score, 4),
            "penalty_multiplier": penalty,
        },
        "helper_text": "This score reflects strength of provenance evidence, not whether the article claim is true.",
    }


def _interpretation_band(score: float) -> str:
    if score >= 90:
        return "strong provenance"
    if score >= 70:
        return "good provenance with minor gaps"
    if score >= 40:
        return "partial provenance"
    if score > 0:
        return "weak provenance"
    return "failed provenance"
