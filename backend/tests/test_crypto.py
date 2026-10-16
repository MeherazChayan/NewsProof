import json
import uuid

from app.models import CryptoObjectType
from app.services.crypto.canonicalizer import canonicalize_article_snapshot
from app.services.crypto.service import CryptoProvenanceService


def _create_article(service: CryptoProvenanceService, previous_version_id=None):
    return service.create_article_snapshot(
        {
            "source_url": "https://example.com/story",
            "publisher_name": "Example News",
            "author_name": " Reporter Name ",
            "title": " Example Story ",
            "published_at": None,
            "normalized_article_text": "A   story   body with   extra whitespace.",
            "media_urls": ["https://example.com/media/story.jpg"],
            "metadata": {"section": "World", "subtitle": " Subtitle ", "summary": " Summary text. "},
            "previous_version_id": previous_version_id,
        }
    )


def test_canonicalization_consistency(db_session):
    service = CryptoProvenanceService(db_session)
    article = _create_article(service)

    first = canonicalize_article_snapshot(article)
    second = canonicalize_article_snapshot(article)
    payload = json.loads(first)

    assert first == second
    assert payload["source"]["author_names"] == ["Reporter Name"]
    assert payload["article"]["full_body_text"] == "A story body with extra whitespace."
    assert payload["canonicalization"]["canonicalization_version"]
    assert payload["article"]["components"]
    assert payload["assets"][0]["asset_hash"]


def test_signature_verification_happy_path(db_session):
    service = CryptoProvenanceService(db_session)
    service.ensure_default_issuers()
    article = _create_article(service)

    manifest = service.sign_object(CryptoObjectType.article_snapshot, article.id)
    result = service.verify_manifest(manifest.id)

    assert result["valid"] is True
    assert result["status"] == "valid"
    assert result["confidence"]["score"] >= 80
    assert result["explanation"]["summary"].startswith("Valid signature")


def test_tamper_detection_fails_verification(db_session):
    service = CryptoProvenanceService(db_session)
    service.ensure_default_issuers()
    article = _create_article(service)
    manifest = service.sign_object(CryptoObjectType.article_snapshot, article.id)

    article.title = "Tampered Story"
    db_session.add(article)
    db_session.commit()

    result = service.verify_manifest(manifest.id)

    assert result["valid"] is False
    assert result["status"] == "hash_mismatch"
    assert result["confidence"]["score"] == 0.0


def test_version_chain_validation_detects_broken_link(db_session):
    service = CryptoProvenanceService(db_session)
    service.ensure_default_issuers()
    v1 = _create_article(service)
    m1 = service.sign_object(CryptoObjectType.article_snapshot, v1.id)
    v2 = _create_article(service, previous_version_id=v1.id)
    m2 = service.sign_object(CryptoObjectType.article_snapshot, v2.id)

    m2.previous_manifest_id = uuid.uuid4()
    db_session.add(m2)
    db_session.commit()

    result = service.verify_manifest(m2.id)

    assert result["valid"] is False
    assert result["status"] == "version_chain_broken"
    assert m1.id != m2.previous_manifest_id


def test_revoked_issuer_fails_verification(db_session):
    service = CryptoProvenanceService(db_session)
    service.ensure_default_issuers()
    article = _create_article(service)
    manifest = service.sign_object(CryptoObjectType.article_snapshot, article.id)

    service.revoke_issuer(manifest.signer_id)
    result = service.verify_manifest(manifest.id)

    assert result["valid"] is False
    assert result["status"] == "issuer_revoked"
    assert result["confidence"]["breakdown"]["penalty_multiplier"] == 0.3


def test_timestamp_validation_fallback_detects_invalid_token(db_session):
    service = CryptoProvenanceService(db_session)
    service.ensure_default_issuers()
    article = _create_article(service)
    manifest = service.sign_object(CryptoObjectType.article_snapshot, article.id)

    manifest.timestamp_token = {**manifest.timestamp_token, "hashed_message": "bad"}
    db_session.add(manifest)
    db_session.commit()

    result = service.verify_manifest(manifest.id)

    assert result["valid"] is False
    assert result["status"] == "timestamp_invalid"
    assert result["confidence"]["breakdown"]["penalty_multiplier"] == 0.6


def test_history_and_compare_expose_version_chain(db_session):
    service = CryptoProvenanceService(db_session)
    service.ensure_default_issuers()
    first = _create_article(service)
    service.sign_object(CryptoObjectType.article_snapshot, first.id)
    second = service.create_article_snapshot(
        {
            "source_url": "https://example.com/story",
            "publisher_name": "Example News",
            "author_name": "Reporter Name",
            "title": "Example Story Updated",
            "published_at": None,
            "normalized_article_text": "A story body with extra whitespace and an update.",
            "media_urls": [],
            "metadata": {"section": "World"},
            "previous_version_id": first.id,
        }
    )
    service.sign_object(CryptoObjectType.article_snapshot, second.id)

    history = service.record_history(CryptoObjectType.article_snapshot, second.id)
    comparison = service.compare_versions(CryptoObjectType.article_snapshot, first.id, second.id)

    assert len(history["versions"]) == 2
    assert "article.title" in comparison["changed_fields"]


def test_manifest_serialization_exposes_crypto_identity_fields(db_session):
    service = CryptoProvenanceService(db_session)
    service.ensure_default_issuers()
    article = _create_article(service)
    manifest = service.sign_object(CryptoObjectType.article_snapshot, article.id)
    serialized = service.serialize_manifest(manifest)

    assert serialized["crypto"]["signature_algorithm"] == "Ed25519"
    assert serialized["confidence"]["breakdown"]["issuer_trust"] >= 0.6
