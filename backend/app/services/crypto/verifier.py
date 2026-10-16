from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    ArticleSnapshot,
    AudioVerificationRecord,
    CryptoObjectType,
    EvidenceBundle,
    ImageVerificationRecord,
    IssuerCredential,
    IssuerType,
    ProvenanceManifest,
    ProvenanceVerificationStatus,
    ReviewEvent,
    VerificationRecord,
    VideoVerificationRecord,
)
from app.security import verify_ed25519_signature
from app.services.crypto.canonicalizer import (
    canonicalize_article_snapshot,
    canonicalize_audio_verification_record,
    canonicalize_evidence_bundle,
    canonicalize_image_verification_record,
    canonicalize_review_event,
    canonicalize_verification_record,
    canonicalize_video_verification_record,
)
from app.services.crypto.confidence import compute_crypto_confidence
from app.services.crypto.hasher import sha256_hex
from app.services.crypto.timestamp_provider import get_timestamp_provider


def rebuild_canonical_json(db: Session, object_type: CryptoObjectType, object_id: UUID) -> str:
    if object_type == CryptoObjectType.article_snapshot:
        record = db.get(ArticleSnapshot, object_id)
        if not record:
            raise ValueError("Article snapshot not found")
        return canonicalize_article_snapshot(record)
    if object_type == CryptoObjectType.evidence_bundle:
        record = db.get(EvidenceBundle, object_id)
        if not record:
            raise ValueError("Evidence bundle not found")
        return canonicalize_evidence_bundle(record)
    if object_type == CryptoObjectType.review_event:
        record = db.get(ReviewEvent, object_id)
        if not record:
            raise ValueError("Review event not found")
        return canonicalize_review_event(record)
    if object_type == CryptoObjectType.verification_record:
        record = db.get(VerificationRecord, object_id)
        if not record:
            raise ValueError("Verification record not found")
        return canonicalize_verification_record(record)
    if object_type == CryptoObjectType.image_verification_record:
        record = db.get(ImageVerificationRecord, object_id)
        if not record:
            raise ValueError("Image verification record not found")
        return canonicalize_image_verification_record(record)
    if object_type == CryptoObjectType.audio_verification_record:
        record = db.get(AudioVerificationRecord, object_id)
        if not record:
            raise ValueError("Audio verification record not found")
        return canonicalize_audio_verification_record(record)
    if object_type == CryptoObjectType.video_verification_record:
        record = db.get(VideoVerificationRecord, object_id)
        if not record:
            raise ValueError("Video verification record not found")
        return canonicalize_video_verification_record(record)
    raise ValueError("Unsupported object type")


def verify_manifest(db: Session, manifest: ProvenanceManifest) -> dict[str, Any]:
    canonical_json = rebuild_canonical_json(db, manifest.object_type, manifest.object_id)
    manifest_payload = json.loads(canonical_json)
    rebuilt_hash = sha256_hex(canonical_json.encode("utf-8"))
    manifest_hash = sha256_hex(manifest.canonical_json.encode("utf-8"))
    issuer = db.get(IssuerCredential, manifest.signer_id)
    timestamp_result = get_timestamp_provider().verify(manifest.timestamp_token, manifest.sha256_hash)
    chain_check = _verify_version_chain(db, manifest, manifest_payload)
    warnings: list[str] = []

    canonical_json_matches = canonical_json == manifest.canonical_json
    stored_hash_matches = rebuilt_hash == manifest.sha256_hash
    signature_present = bool(manifest.signature)
    signature_valid = False
    if issuer and issuer.status != "revoked" and signature_present:
        signature_valid = verify_ed25519_signature(issuer.public_key, manifest.signature, manifest.sha256_hash.encode("utf-8"))

    status = _resolve_status(
        issuer=issuer,
        canonical_json_matches=canonical_json_matches,
        stored_hash_matches=stored_hash_matches,
        signature_present=signature_present,
        signature_valid=signature_valid,
        timestamp_valid=timestamp_result.valid,
        version_chain_valid=chain_check["valid"],
    )
    factors = _build_factor_scores(
        issuer=issuer,
        status=status,
        signature_present=signature_present,
        signature_valid=signature_valid,
        timestamp_valid=timestamp_result.valid,
        timestamp_provider=timestamp_result.provider,
        stored_hash_matches=stored_hash_matches and canonical_json_matches,
        version_chain_valid=chain_check["valid"],
        version_chain_complete=chain_check["complete"],
    )

    if not signature_present:
        warnings.append("No external content signature was present; provenance may be platform-issued or partial.")
    if issuer is None:
        warnings.append("Signer could not be matched to a known issuer.")
    elif issuer.status == "revoked":
        warnings.append("The issuer key has been revoked.")
    if not timestamp_result.valid:
        warnings.append(timestamp_result.reason or "Timestamp verification failed.")
    if not chain_check["valid"]:
        warnings.append(chain_check["reason"])
    elif not chain_check["complete"]:
        warnings.append("Version chain metadata is incomplete.")

    checks: dict[str, Any] = {
        "canonical_json_matches": canonical_json_matches,
        "rebuilt_sha256_hash": rebuilt_hash,
        "stored_sha256_hash": manifest.sha256_hash,
        "manifest_hash": manifest_hash,
        "stored_manifest_hash": manifest_hash,
        "signature_present": signature_present,
        "signature_valid": signature_valid,
        "issuer_found": issuer is not None,
        "issuer_status": issuer.status if issuer else None,
        "version_chain_valid": chain_check["valid"],
        "version_chain_complete": chain_check["complete"],
        "version_chain_reason": chain_check["reason"],
    }
    if timestamp_result.provider:
        checks["timestamp_provider"] = timestamp_result.provider
    checks["timestamp_valid"] = timestamp_result.valid
    if timestamp_result.reason:
        checks["timestamp_reason"] = timestamp_result.reason

    verification = {
        "manifest_id": str(manifest.id),
        "object_type": manifest.object_type.value,
        "object_id": str(manifest.object_id),
        "status": status.value,
        "valid": status == ProvenanceVerificationStatus.valid,
        "checks": checks,
        "crypto": {
            "payload_hash": manifest_payload.get("article", {}).get("payload_hash"),
            "manifest_hash": manifest_hash,
            "issuer_id": str(manifest.signer_id),
            "key_id": manifest.timestamp_token.get("key_id") or manifest_payload.get("crypto", {}).get("key_id") or _derive_key_id(issuer),
            "public_key_id": _derive_key_id(issuer),
            "key_version": manifest_payload.get("crypto", {}).get("key_version") or "v1",
            "signature_algorithm": manifest.signing_algorithm,
            "signature_value": manifest.signature,
            "signature_created_at": manifest.created_at.isoformat(),
        },
        "timestamps": {
            "signed_at": manifest.created_at.isoformat(),
            "timestamp": manifest.timestamp_token.get("issued_at") or manifest.created_at.isoformat(),
            "timestamp_source": manifest.timestamp_token.get("mode") or timestamp_result.provider,
            "timestamp_authority_id": manifest.timestamp_token.get("tsa_url") or manifest.timestamp_token.get("provider"),
            "timestamp_verification_result": "valid" if timestamp_result.valid else "invalid",
        },
        "version_chain": {
            "current_manifest_id": str(manifest.id),
            "current_version_number": manifest_payload.get("version_chain", {}).get("current_version_number"),
            "previous_manifest_id": str(manifest.previous_manifest_id) if manifest.previous_manifest_id else None,
            "previous_payload_hash": manifest_payload.get("version_chain", {}).get("previous_payload_hash"),
            "chain_position": manifest_payload.get("version_chain", {}).get("chain_position"),
            "change_type": manifest_payload.get("version_chain", {}).get("change_type"),
            "diff_summary": manifest_payload.get("version_chain", {}).get("diff_summary"),
            "valid": chain_check["valid"],
            "complete": chain_check["complete"],
        },
        "factors": factors,
        "warnings": warnings,
    }
    confidence = compute_crypto_confidence(verification, manifest_payload)
    explanation = explain_crypto_result(verification, confidence)

    return {
        **verification,
        "confidence": confidence,
        "explanation": explanation,
    }


def explain_crypto_result(verification: dict[str, Any], confidence: dict[str, Any]) -> dict[str, Any]:
    status = verification["status"]
    warnings = verification.get("warnings", [])
    checks = verification["checks"]
    timestamp_valid = checks.get("timestamp_valid")
    version_chain_valid = checks.get("version_chain_valid")
    issuer_status = checks.get("issuer_status")
    signature_valid = checks.get("signature_valid")

    if status == ProvenanceVerificationStatus.valid.value:
        summary = "Valid signature from trusted issuer, timestamp verified, and version chain intact."
    elif status == ProvenanceVerificationStatus.partial_provenance.value:
        summary = "No external content signature found. Platform-issued provenance created at ingestion."
    elif status == ProvenanceVerificationStatus.no_provenance.value:
        summary = "No external content signature found and only weak provenance evidence is available."
    elif status == ProvenanceVerificationStatus.hash_mismatch.value:
        summary = "Hash mismatch detected. The current content does not match the recorded manifest."
    elif status == ProvenanceVerificationStatus.signature_invalid.value:
        summary = "A signature exists, but cryptographic verification did not succeed."
    elif status == ProvenanceVerificationStatus.issuer_unknown.value:
        summary = "A signature or provenance reference exists, but the issuer is not known to the registry."
    elif status == ProvenanceVerificationStatus.issuer_revoked.value:
        summary = "Issuer key has been revoked. Provenance confidence sharply reduced."
    elif status == ProvenanceVerificationStatus.timestamp_invalid.value:
        summary = "Core provenance exists, but the timestamp could not be validated."
    else:
        summary = "Version chain continuity could not be verified for this record."

    reasons = []
    if signature_valid:
        reasons.append("signature verified")
    if issuer_status == "active":
        reasons.append("issuer active")
    if timestamp_valid:
        reasons.append("timestamp verified")
    if version_chain_valid:
        reasons.append("version chain intact")
    if not reasons:
        reasons.append("no strong cryptographic proof was confirmed")

    return {
        "summary": summary,
        "human_explanation": summary,
        "reasoning": ", ".join(reasons),
        "confidence_band": confidence["band"],
        "warnings": warnings,
    }


def _resolve_status(
    *,
    issuer: IssuerCredential | None,
    canonical_json_matches: bool,
    stored_hash_matches: bool,
    signature_present: bool,
    signature_valid: bool,
    timestamp_valid: bool,
    version_chain_valid: bool,
) -> ProvenanceVerificationStatus:
    if not canonical_json_matches or not stored_hash_matches:
        return ProvenanceVerificationStatus.hash_mismatch
    if issuer and issuer.status == "revoked":
        return ProvenanceVerificationStatus.issuer_revoked
    if issuer is None:
        if signature_present:
            return ProvenanceVerificationStatus.issuer_unknown
        return ProvenanceVerificationStatus.no_provenance
    if signature_present and not signature_valid:
        return ProvenanceVerificationStatus.signature_invalid
    if not timestamp_valid:
        return ProvenanceVerificationStatus.timestamp_invalid
    if not version_chain_valid:
        return ProvenanceVerificationStatus.version_chain_broken
    if signature_valid:
        return ProvenanceVerificationStatus.valid
    if issuer.issuer_type == IssuerType.platform:
        return ProvenanceVerificationStatus.partial_provenance
    return ProvenanceVerificationStatus.no_provenance


def _build_factor_scores(
    *,
    issuer: IssuerCredential | None,
    status: ProvenanceVerificationStatus,
    signature_present: bool,
    signature_valid: bool,
    timestamp_valid: bool,
    timestamp_provider: str,
    stored_hash_matches: bool,
    version_chain_valid: bool,
    version_chain_complete: bool,
) -> dict[str, float]:
    if status == ProvenanceVerificationStatus.signature_invalid:
        signature_score = 0.0
    elif signature_valid:
        signature_score = 1.0
    elif signature_present:
        signature_score = 0.6
    else:
        signature_score = 0.2

    if issuer is None:
        issuer_score = 0.3
    elif issuer.status == "revoked":
        issuer_score = 0.0
    elif issuer.issuer_type == IssuerType.publisher:
        issuer_score = 1.0
    elif issuer.issuer_type == IssuerType.platform:
        issuer_score = 0.6
    else:
        issuer_score = 0.7

    if timestamp_valid and timestamp_provider == "rfc3161":
        timestamp_score = 1.0
    elif timestamp_valid:
        timestamp_score = 0.7
    elif timestamp_provider:
        timestamp_score = 0.3
    else:
        timestamp_score = 0.0

    hash_score = 1.0 if stored_hash_matches else 0.0

    if version_chain_valid and version_chain_complete:
        version_score = 1.0
    elif version_chain_valid:
        version_score = 0.5
    else:
        version_score = 0.0

    if status in {ProvenanceVerificationStatus.signature_invalid, ProvenanceVerificationStatus.hash_mismatch}:
        penalty = 0.0
    elif status == ProvenanceVerificationStatus.issuer_revoked:
        penalty = 0.3
    elif status == ProvenanceVerificationStatus.timestamp_invalid:
        penalty = 0.6
    else:
        penalty = 1.0

    return {
        "signature_validity": signature_score,
        "issuer_trust": issuer_score,
        "timestamp_validity": timestamp_score,
        "hash_integrity": hash_score,
        "version_chain": version_score,
        "penalty_multiplier": penalty,
    }


def _verify_version_chain(db: Session, manifest: ProvenanceManifest, manifest_payload: dict[str, Any]) -> dict[str, Any]:
    if not manifest.previous_manifest_id:
        return {"valid": True, "complete": True, "reason": "first version or root manifest"}

    previous_manifest = db.get(ProvenanceManifest, manifest.previous_manifest_id)
    if not previous_manifest:
        return {"valid": False, "complete": False, "reason": "previous manifest missing"}

    previous_object_id = _previous_object_id(db, manifest.object_type, manifest.object_id)
    if not previous_object_id:
        return {"valid": False, "complete": False, "reason": "previous object reference missing"}

    if previous_manifest.object_type != manifest.object_type or previous_manifest.object_id != previous_object_id:
        return {"valid": False, "complete": False, "reason": "previous manifest does not match previous object version"}

    expected_previous_hash = manifest_payload.get("version_chain", {}).get("previous_payload_hash")
    previous_payload = json.loads(previous_manifest.canonical_json)
    previous_payload_hash = previous_payload.get("article", {}).get("payload_hash") or previous_manifest.sha256_hash
    if expected_previous_hash and expected_previous_hash != previous_payload_hash:
        return {"valid": False, "complete": True, "reason": "previous payload hash does not match recorded chain hash"}

    return {"valid": True, "complete": bool(expected_previous_hash), "reason": "version chain verified"}


def _previous_object_id(db: Session, object_type: CryptoObjectType, object_id: UUID) -> UUID | None:
    if object_type == CryptoObjectType.article_snapshot:
        record = db.get(ArticleSnapshot, object_id)
        return record.previous_version_id if record else None
    if object_type == CryptoObjectType.evidence_bundle:
        record = db.get(EvidenceBundle, object_id)
        return record.previous_version_id if record else None
    if object_type == CryptoObjectType.review_event:
        record = db.get(ReviewEvent, object_id)
        return record.previous_version_id if record else None
    if object_type == CryptoObjectType.verification_record:
        record = db.get(VerificationRecord, object_id)
        return record.previous_record_id if record else None
    if object_type == CryptoObjectType.image_verification_record:
        record = db.get(ImageVerificationRecord, object_id)
        return record.previous_record_id if record else None
    if object_type == CryptoObjectType.audio_verification_record:
        record = db.get(AudioVerificationRecord, object_id)
        return record.previous_record_id if record else None
    if object_type == CryptoObjectType.video_verification_record:
        record = db.get(VideoVerificationRecord, object_id)
        return record.previous_record_id if record else None
    return None


def latest_manifest_for_object(db: Session, object_type: CryptoObjectType, object_id: UUID) -> ProvenanceManifest | None:
    return db.scalar(
        select(ProvenanceManifest)
        .where(ProvenanceManifest.object_type == object_type, ProvenanceManifest.object_id == object_id)
        .order_by(ProvenanceManifest.created_at.desc())
    )


def compare_canonical_json(left_json: str, right_json: str) -> dict[str, Any]:
    left = json.loads(left_json)
    right = json.loads(right_json)
    differences: list[dict[str, Any]] = []
    _collect_differences("", left, right, differences)
    return {"differences": differences, "changed_fields": [item["field"] for item in differences]}


def _derive_key_id(issuer: IssuerCredential | None) -> str | None:
    if not issuer:
        return None
    return f"{issuer.issuer_name.lower().replace(' ', '-')}-key"


def _collect_differences(path: str, left: Any, right: Any, differences: list[dict[str, Any]]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left.keys()) | set(right.keys())):
            next_path = f"{path}.{key}" if path else str(key)
            _collect_differences(next_path, left.get(key), right.get(key), differences)
        return
    if isinstance(left, list) and isinstance(right, list):
        max_len = max(len(left), len(right))
        for index in range(max_len):
            next_path = f"{path}[{index}]"
            left_value = left[index] if index < len(left) else None
            right_value = right[index] if index < len(right) else None
            _collect_differences(next_path, left_value, right_value, differences)
        return
    if left != right:
        differences.append({"field": path, "left": left, "right": right})
