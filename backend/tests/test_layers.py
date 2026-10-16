import base64
import asyncio
from datetime import datetime
from io import BytesIO

import pytest
from bs4 import BeautifulSoup
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PIL import Image
from sqlalchemy import select

from app.models import ConsensusRole, ContentRecord, ConsensusVote, ExtractedContent, JobInputType, JobStatus, PublicKey, SimilarityMatch, TrustedPublisher, VerificationJob
from app.schemas import ConsensusVoteRequest
from app.services.audit import append_audit_entry
from app.services.bootstrap import ensure_default_publishers
from app.services.content_extraction import (
    ExtractedPayload,
    _candidate_sidecar_manifest_urls,
    _build_payload_from_html,
    _byte_c2pa_indicators,
    _detect_linkedin_page_signal,
    _describe_remote_provenance,
    _format_certificate_name,
    _header_c2pa_indicators,
    _header_manifest_url,
    _provenance_note,
    _response_kind_from_content_type,
    _verify_remote_manifest_reference,
    build_text_payload,
)
from app.services.search_service import SearchService, WebSearchCandidate
from app.services.soft_binding import SoftBindingRecoveryService
from app.services.media import extract_image_forensics_from_bytes
from app.services.media_authenticity import MediaAuthenticityService
from app.services.claim_extraction_service import ClaimExtractionService
from app.services.claim_evidence_verifier import ClaimEvidenceVerifier
from app.services.context_match_service import ContextMatchService
from app.services.ocr import extract_text_details_from_image_bytes
from app.services.reasoning_service import ReasoningService
from app.services import retrieval_service as retrieval_module
from app.services.retrieval_service import RetrievalService
from app.services.scoring import compute_nlp_score, map_score_to_label
from app.services.source_adapter_service import SourceAdapterService
from app.services.verification import LayerOutcome, VerificationService
from app.services.vision_reasoning_service import VisionReasoningService


def test_retrieval_html_parser_fallback_when_lxml_rejects_markup(monkeypatch):
    real_beautiful_soup = BeautifulSoup

    def flaky_beautiful_soup(html, parser):
        if parser == "lxml":
            raise ValueError("not enough values to unpack")
        return real_beautiful_soup(html, parser)

    monkeypatch.setattr(retrieval_module, "BeautifulSoup", flaky_beautiful_soup)
    service = RetrievalService(None, None, None, None)

    soup = service._parse_html("<html><title>Fallback works</title><p>Body</p></html>")

    assert soup is not None
    assert soup.title.get_text(" ", strip=True) == "Fallback works"


def test_url_extraction_uses_strong_beautifulsoup_fallback():
    html = """
    <html>
      <head>
        <title>Example News</title>
        <meta property="og:title" content="City council approves transit expansion">
      </head>
      <body>
        <nav>Home Subscribe Share</nav>
        <div class="story-content">
          <p>The city council approved a 2.5 billion dollar transit expansion after a late-night vote on Tuesday.</p>
          <p>The plan will add three rail stations and expand bus service across the northern district.</p>
          <p>Officials said construction will begin next year after the transportation agency finishes procurement.</p>
          <p>Opposition members said the budget should include more money for road repairs.</p>
        </div>
        <aside>Related articles and newsletter signup</aside>
      </body>
    </html>
    """

    payload = _build_payload_from_html("https://example.com/news/transit", html)

    assert "city council approved" in payload.main_text.lower()
    assert "related articles" not in payload.main_text.lower()
    assert payload.metadata["extraction_method"] in {
        "beautifulsoup_article_containers",
        "beautifulsoup_paragraph_fallback",
        "readability_lxml",
    }
    assert payload.metadata["cleaned_article_length"] >= 200


def test_wayback_url_extraction_removes_archive_toolbar_and_keeps_article_body():
    html = """
    <html>
      <body>
        <div id="wm-ipp">Wayback Machine toolbar calendar captures</div>
        <article>
          <h1>Archived claim headline</h1>
          <p>Former officials denied the viral claim after it circulated widely on social media.</p>
          <p>The archived story alleged that a secret government account funded private travel for years.</p>
          <p>Fact checkers later found no public records supporting the alleged travel ledger or payments.</p>
          <p>The article text remains available in the archived page body for verification.</p>
        </article>
      </body>
    </html>
    """

    payload = _build_payload_from_html("https://web.archive.org/web/20200101000000/https://fake.example/story", html)

    assert "wayback machine toolbar" not in payload.main_text.lower()
    assert "secret government account" in payload.main_text.lower()
    assert payload.metadata["original_archived_url"] == "https://fake.example/story"


def test_layer1_verified(db_session):
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    publisher = TrustedPublisher(name="Daily Ledger", domain="ledger.test", metadata_json={})
    db_session.add(publisher)
    db_session.flush()

    payload = build_text_payload("Important verified article body")
    signature = private_key.sign(payload.content_hash.encode("utf-8"))
    db_session.add(
        PublicKey(
            publisher_id=publisher.id,
            key_id="primary",
            public_key=base64.b64encode(public_key.public_bytes_raw()).decode("utf-8"),
            is_active=True,
        )
    )
    db_session.add(
        ContentRecord(
            publisher_id=publisher.id,
            canonical_url=None,
            title=payload.title,
            content_hash=payload.content_hash,
            signature=base64.b64encode(signature).decode("utf-8"),
            key_id="primary",
            source_metadata={"normalized_text": payload.normalized_text},
        )
    )
    db_session.commit()

    outcome = VerificationService(db_session)._layer1(payload)
    assert outcome.status == "matched"


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (85, "Verified"),
        (62, "Likely Verified"),
        (40, "Not Verifiable"),
        (6.2, "Unverified"),
        (0, "Unverified"),
        (34.9, "Unverified"),
        (35, "Not Verifiable"),
        (49.9, "Not Verifiable"),
        (50, "Likely Verified"),
        (70, "Verified"),
    ],
)
def test_map_score_to_label_uses_global_thresholds(score, expected):
    assert map_score_to_label(score) == expected


def test_verification_type_label_helpers_use_global_thresholds(db_session):
    service = VerificationService(db_session)

    assert service._generate_text_label(0.062, 0.2, False, 0.0) == "Unverified"
    assert service._url_label_from_confidence(0.495, "partial content match") == "Not Verifiable"
    assert service._generate_image_label(72, 0.9, 1.0, {}) == "Verified"
    assert service._generate_audio_label(55, 0.7, 1.0, {}) == "Likely Verified"
    assert service._generate_video_label(34, 0.7, 1.0, {}) == "Unverified"
    assert service._facebook_layer2_label(0.062) == "Unverified"


def test_layer1_recognizes_trusted_publisher_from_domain_and_metadata(db_session):
    publisher = TrustedPublisher(
        name="Example News",
        domain="example.com",
        metadata_json={"domains": ["www.example.com"]},
    )
    db_session.add(publisher)
    db_session.flush()

    db_session.add(
        ContentRecord(
            publisher_id=publisher.id,
            canonical_url="https://www.example.com/story",
            title="Trusted article",
            content_hash="stored-hash",
            signature=None,
            key_id=None,
            source_metadata={"author": "Reporter", "publish_time": "2026-03-17T10:30:00", "normalized_text": "trusted article body"},
        )
    )
    db_session.commit()

    payload = ExtractedPayload(
        title="Trusted article",
        author="Reporter",
        publish_time=datetime.fromisoformat("2026-03-17T10:30:00"),
        canonical_url="https://www.example.com/story",
        main_text="Updated body text that does not hash-match the stored record",
        hero_image_url=None,
        normalized_text="updated body text that does not hash match the stored record",
        content_hash="different-hash",
        metadata={"fetched_url": "https://www.example.com/story", "source_domain": "www.example.com"},
    )

    outcome = VerificationService(db_session)._layer1(payload)
    assert outcome.status == "publisher-recognized"
    assert outcome.details["publisher"] == "Example News"


def test_layer1_uses_seeded_major_publishers(db_session):
    ensure_default_publishers(db_session)

    payload = ExtractedPayload(
        title="Original report",
        author="Reporter",
        publish_time=datetime.fromisoformat("2026-03-17T10:30:00"),
        canonical_url="https://www.nytimes.com/2026/03/17/world/example-story.html",
        main_text="Please enable JS and disable any ad blocker",
        hero_image_url=None,
        normalized_text="please enable js and disable any ad blocker",
        content_hash="nyt-different-hash",
        metadata={
            "fetched_url": "https://www.nytimes.com/2026/03/17/world/example-story.html",
            "source_domain": "www.nytimes.com",
            "publisher_name": "The New York Times",
        },
    )

    outcome = VerificationService(db_session)._layer1(payload)
    assert outcome.status == "publisher-recognized"
    assert outcome.details["publisher"] == "The New York Times"


def test_layer2_does_not_treat_verified_publisher_metadata_as_url_nlp_support(db_session):
    ensure_default_publishers(db_session)
    service = VerificationService(db_session)
    job = VerificationJob(
        input_type=JobInputType.url,
        input_value="https://www.cnn.com/2026/03/17/politics/example-story",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()

    payload = ExtractedPayload(
        title="Example CNN story",
        author="Reporter Name",
        publish_time=datetime.fromisoformat("2026-03-17T14:02:02"),
        canonical_url="https://www.cnn.com/2026/03/17/politics/example-story",
        main_text=" ".join(["This is a full article body from CNN."] * 40),
        hero_image_url="https://media.cnn.com/example.jpg",
        normalized_text=" ".join(["this is a full article body from cnn"] * 40),
        content_hash="cnn-test-hash",
        metadata={
            "fetched_url": "https://www.cnn.com/2026/03/17/politics/example-story",
            "publisher_name": "CNN",
            "source_domain": "www.cnn.com",
            "section": "Politics",
            "keywords": ["politics", "iran"],
            "media_assets": [{"type": "image", "url": "https://media.cnn.com/example.jpg"}],
        },
    )

    outcome = service._layer2(job, payload)

    assert outcome.status == "none"
    assert outcome.details["trusted_publisher_match"] == "CNN"
    assert outcome.details["matches"] == []
    assert outcome.details["fact_check_verdict"] == "inconclusive"


def test_layer2_similarity_does_not_remain_inconclusive_when_match_exists(db_session):
    publisher = TrustedPublisher(name="Example News", domain="example.com", metadata_json={})
    db_session.add(publisher)
    db_session.flush()
    db_session.add(
        ContentRecord(
            publisher_id=publisher.id,
            canonical_url="https://example.com/story",
            title="Example story",
            content_hash="stored-hash",
            signature=None,
            key_id=None,
            source_metadata={"normalized_text": "alpha beta gamma delta epsilon zeta eta theta"},
        )
    )
    job = VerificationJob(
        input_type=JobInputType.text,
        input_value="alpha beta gamma delta epsilon",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()

    outcome = VerificationService(db_session)._layer2(job, build_text_payload("alpha beta gamma delta epsilon"))

    assert outcome.status == "supporting"
    assert outcome.details["fact_check_verdict"] != "inconclusive"
    assert outcome.details["matches"]
    assert len(outcome.details["matches"]) <= 5


def test_layer2_uses_web_retrieval_matches(db_session):
    service = VerificationService(db_session)
    service.search.search = lambda query, expected_domain=None, limit=5: [
        WebSearchCandidate(
            title="Matched web article",
            url="https://example.com/matched-article",
            snippet="This result closely matches the article headline and summary.",
            domain="example.com",
            score=0.81,
            explanation="Found through web retrieval and scored with title/snippet overlap plus domain consistency.",
        )
    ]
    job = VerificationJob(
        input_type=JobInputType.url,
        input_value="https://example.com/story",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()

    payload = ExtractedPayload(
        title="Matched web article",
        author="Reporter Name",
        publish_time=datetime.fromisoformat("2026-03-17T14:02:02"),
        canonical_url="https://example.com/story",
        main_text="A matched story body",
        hero_image_url=None,
        normalized_text="matched web article a matched story body",
        content_hash="web-search-test",
        metadata={
            "fetched_url": "https://example.com/story",
            "publisher_name": "Example News",
            "source_domain": "example.com",
        },
    )

    outcome = service._layer2(job, payload)

    assert outcome.status == "none"
    assert outcome.details["matches"] == []
    assert outcome.details["retrieval_failure_reason"] == "extraction_too_short"


def test_layer2_social_posts_use_caption_and_image_ocr_for_text_reasoning(db_session):
    service = VerificationService(db_session)
    captured: dict[str, object] = {}

    def fake_extract_remote_social_image_ocr(_payload):
        return {
            "text_detected": True,
            "ocr_text": "8 children killed in Louisiana",
            "ocr_status": "completed",
            "ocr_summary": "Text extracted from social post image.",
        }

    def fake_retrieve_text_evidence(input_text, _extracted, claims):
        captured["input_text"] = input_text
        return {
            claim: [
                {
                    "title": "CNN reports 8 children killed in Louisiana shootings",
                    "url": "https://www.cnn.com/example",
                    "snippet": "CNN reported that 8 children were killed across multiple homes in Louisiana.",
                    "passage": "CNN reported that 8 children were killed across multiple homes in Louisiana.",
                    "domain": "cnn.com",
                    "score": 0.91,
                    "retrieval_score": 0.91,
                    "source": "web_search",
                }
            ]
            for claim in claims
        }

    service._extract_remote_social_image_ocr = fake_extract_remote_social_image_ocr
    service._retrieve_text_evidence = fake_retrieve_text_evidence

    job = VerificationJob(
        input_type=JobInputType.url,
        input_value="https://www.facebook.com/example/posts/123",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()

    payload = ExtractedPayload(
        title="8 children killed in domestic shootings across multiple homes in Louisiana, police say",
        author=None,
        publish_time=None,
        canonical_url="https://www.facebook.com/example/posts/123",
        main_text="",
        hero_image_url="https://cdn.example.com/post.jpg",
        normalized_text="",
        content_hash="fb-social-hash",
        metadata={
            "fetched_url": "https://www.facebook.com/example/posts/123",
            "source_domain": "www.facebook.com",
            "source_type": "social_post",
            "platform_type": "facebook",
            "description": "Facebook caption describing the Louisiana shootings.",
            "media_assets": [{"type": "image", "url": "https://cdn.example.com/post.jpg", "caption": "Louisiana shootings post image"}],
        },
    )

    outcome = service._layer2(job, payload)

    assert outcome.status == "supporting"
    assert "Facebook caption describing the Louisiana shootings." in str(captured["input_text"])
    assert "8 children killed in Louisiana" in str(captured["input_text"])
    assert outcome.details["social_image_ocr"] == "8 children killed in Louisiana"
    assert outcome.details["social_ocr_result"]["text_detected"] is True


def test_text_direct_source_recovery_recovers_original_article_url(db_session):
    service = VerificationService(db_session)

    def fake_search(query, expected_domain=None, limit=5):
        return [
            WebSearchCandidate(
                title="Eight children killed in domestic shootings across Louisiana homes",
                url="https://www.cnn.com/2026/04/27/us/louisiana-shootings-children-killed/index.html",
                snippet="Eight children were killed in domestic shootings across multiple homes in Louisiana, police said.",
                domain="www.cnn.com",
                score=0.92,
                explanation="exact sentence match",
            )
        ]

    service.search.search = fake_search
    service.retrieval.fetch_page_document = lambda url: {
        "url": url,
        "fetch_status": "success",
        "content_type": "text/html",
        "title": "Eight children killed in domestic shootings across Louisiana homes",
        "extracted_text": (
            "Eight children were killed in domestic shootings across multiple homes in Louisiana, police said. "
            "Authorities said the killings took place at several homes and left the community in shock."
        ),
        "extracted_text_length": 170,
        "extract_status": "extracted",
    }

    payload = build_text_payload(
        "Eight children were killed in domestic shootings across multiple homes in Louisiana, police said. "
        "Authorities said the killings took place at several homes and left the community in shock."
    )
    prepared = service.preprocess_text(payload.main_text)
    extraction = service.claim_extraction.extract_claims(prepared["cleaned_input"], max_claims=service.TEXT_CLAIM_LIMIT)

    recovery = service._direct_source_recovery(prepared, extraction, payload)

    assert recovery["status"] in {"direct_source_recovered", "likely_source_recovered"}
    assert "cnn.com" in str(recovery["best_candidate"]["url"])
    assert float(recovery["best_score"]) >= 0.62


def test_text_nlp_layer_uses_direct_source_recovery_before_formula_scoring(db_session):
    service = VerificationService(db_session)

    def fake_search_with_debug(query, expected_domain=None, limit=5):
        return {
            "results": [
                WebSearchCandidate(
                    title="Eight children killed in domestic shootings across Louisiana homes",
                    url="https://www.cnn.com/2026/04/27/us/louisiana-shootings-children-killed/index.html",
                    snippet="Eight children were killed in domestic shootings across multiple homes in Louisiana, police said.",
                    domain="www.cnn.com",
                    score=0.92,
                    explanation=f"match for {query}",
                )
            ],
            "failure_reason": None,
            "provider_attempts": [{"provider": "bing", "query": query, "result_count": 1, "failure_reason": None}],
            "stopped_by_budget": False,
        }

    service.search.search_with_debug = fake_search_with_debug
    service.search.search = lambda query, expected_domain=None, limit=5: fake_search_with_debug(query, expected_domain, limit)["results"]
    service.retrieval.fetch_page_document = lambda url: {
        "url": url,
        "fetch_status": "success",
        "content_type": "text/html",
        "title": "Eight children killed in domestic shootings across Louisiana homes",
        "extracted_text": (
            "Eight children were killed in domestic shootings across multiple homes in Louisiana, police said. "
            "Authorities said the killings took place at several homes and left the community in shock."
        ),
        "extracted_text_length": 170,
        "extract_status": "extracted",
    }

    job = VerificationJob(
        input_type=JobInputType.text,
        input_value="sample",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()

    payload = build_text_payload(
        "Eight children were killed in domestic shootings across multiple homes in Louisiana, police said. "
        "Authorities said the killings took place at several homes and left the community in shock."
    )

    layer = service._text_nlp_layer(job, payload, payload.main_text)

    assert layer.details["matches"]
    assert layer.details["text_debug"]["final_source_recovery"]["status"] == "recovered"
    assert layer.details["text_debug"]["top_5_urls"]
    assert float(layer.details["text_component_scores"]["E"]) >= 0.6


def test_text_headline_recovers_exact_trusted_article_not_fake_low_score(db_session, monkeypatch):
    service = VerificationService(db_session)
    headline = "Trump completes his month of GOP revenge, and other takeaways from the Texas runoffs"
    cnn_url = "https://www.cnn.com/2024/05/29/politics/texas-runoffs-trump-gop-revenge-takeaways/index.html"

    def fake_search(query, expected_domain=None, limit=5):
        return [
            WebSearchCandidate(
                title="Texas runoff election calendar and voting information",
                url="https://example.com/texas-runoff-calendar",
                snippet="A general page about voting dates in Texas runoff elections.",
                domain="example.com",
                score=0.88,
                explanation="background result",
            ),
            WebSearchCandidate(
                title=headline,
                url=cnn_url,
                snippet="CNN Politics reports on Trump-backed challengers, GOP revenge, and takeaways from the Texas runoffs.",
                domain="www.cnn.com",
                score=0.86,
                explanation="exact headline match",
            ),
        ][:limit]

    monkeypatch.setattr(service.search, "search", fake_search)
    monkeypatch.setattr(
        service.retrieval,
        "fetch_page_document",
        lambda url: {
            "url": url,
            "fetch_status": "success",
            "content_type": "text/html",
            "title": headline if url == cnn_url else "Texas runoff election calendar and voting information",
            "extracted_text": (
                f"{headline}. CNN Politics reported that the Texas runoffs completed a month of GOP revenge for Donald Trump."
                if url == cnn_url
                else "This page lists Texas runoff election dates and polling place information."
            ),
            "extracted_text_length": 160,
            "extract_status": "extracted",
        },
    )

    payload = build_text_payload(headline)
    job = service._create_job(JobInputType.text, headline[:100])

    outcome = service._text_nlp_layer(job, payload, headline)

    assert outcome.details["text_debug"]["final_source_recovery"]["status"] == "recovered"
    assert outcome.details["matches"]
    assert outcome.details["matches"][0]["url"] == cnn_url
    assert outcome.details["matches"][0]["evidence_relation"] == "supports"
    assert outcome.details["text_verification_label"] in {"Likely Verified", "Verified"}
    assert outcome.details["text_verification_score"] >= 0.5


def test_verified_election_news_uses_trusted_news_exact_support(db_session, monkeypatch):
    service = VerificationService(db_session)
    text = (
        "Texas Attorney General Ken Paxton easily defeated Senator John Cornyn in a Republican run-off primary "
        "on Tuesday night, winning the party's nomination for U.S. Senate."
    )

    def fake_search(query, expected_domain=None, limit=5):
        return [
            WebSearchCandidate(
                title="Ken Paxton defeats John Cornyn in Texas Republican Senate runoff",
                url="https://apnews.com/article/texas-paxton-cornyn-senate-runoff",
                snippet="Ken Paxton won the Republican nomination for U.S. Senate by defeating John Cornyn in Texas.",
                domain="apnews.com",
                score=0.94,
                explanation=f"trusted exact support for {query}",
            ),
            WebSearchCandidate(
                title="Paxton beats Cornyn in Texas GOP Senate primary runoff",
                url="https://www.reuters.com/world/us/paxton-cornyn-texas-senate-runoff",
                snippet="Texas Attorney General Ken Paxton defeated Senator John Cornyn in the Republican U.S. Senate runoff.",
                domain="www.reuters.com",
                score=0.91,
                explanation="trusted exact support",
            ),
            WebSearchCandidate(
                title="Texas runoff calendar and voting information",
                url="https://example.com/texas-runoff-calendar",
                snippet="General information about Texas runoff election dates.",
                domain="example.com",
                score=0.72,
                explanation="background",
            ),
        ][:limit]

    monkeypatch.setattr(service.search, "search", fake_search)
    monkeypatch.setattr(
        service.retrieval,
        "fetch_page_document",
        lambda url: {
            "url": url,
            "fetch_status": "success",
            "content_type": "text/html",
            "title": (
                "Ken Paxton defeats John Cornyn in Texas Republican Senate runoff"
                if "apnews" in url
                else "Paxton beats Cornyn in Texas GOP Senate primary runoff"
                if "reuters" in url
                else "Texas runoff calendar and voting information"
            ),
            "extracted_text": (
                "Ken Paxton won the Republican nomination for U.S. Senate by defeating John Cornyn in Texas on Tuesday night."
                if "apnews" in url
                else "Texas Attorney General Ken Paxton defeated Senator John Cornyn in the Republican U.S. Senate runoff."
                if "reuters" in url
                else "General information about Texas runoff election dates."
            ),
            "extracted_text_length": 180,
            "extract_status": "extracted",
        },
    )

    payload = build_text_payload(text)
    job = service._create_job(JobInputType.text, text[:100])

    outcome = service._text_nlp_layer(job, payload, text)

    components = outcome.details["text_component_scores"]
    debug = outcome.details["text_debug"]
    assert outcome.details["text_verification_label"] in {"Verified", "Likely Verified"}
    assert outcome.details["text_verification_score"] >= 0.5
    assert components["S"] >= 0.8
    assert components["D"] >= 0.8
    assert components["R"] >= 0.9
    assert components["A"] >= 0.75
    assert components["E"] >= 0.85
    assert debug["extracted_claim_tuple"]["subject"] == "Ken Paxton"
    assert debug["extracted_claim_tuple"]["object"] == "John Cornyn"
    assert debug["extracted_claim_tuple"]["claim_type"] == "election result"
    assert debug["broad_searches"]
    assert debug["trusted_searches"]
    assert debug["fact_check_searches"]
    assert debug["official_searches"]
    assert any("site:apnews.com" in query for query in outcome.details["search_queries_used"])
    assert len(debug["exact_supporting_sources"]) >= 2
    assert not debug["refuting_sources"]
    assert all(item.get("reason") for item in debug["exact_supporting_sources"])


def test_paxton_cornyn_short_claim_counts_ap_support_as_exact(db_session, monkeypatch):
    service = VerificationService(db_session)
    claim = "Texas Attorney General Ken Paxton easily defeated Senator John Cornyn in a Republican run-off primary on Tuesday night."
    evidence = [
        {
            "title": "Ken Paxton wins Senate primary runoff, defeats incumbent Sen. John Cornyn.",
            "url": "https://apnews.com/article/texas-paxton-cornyn-senate-runoff",
            "domain": "apnews.com",
            "snippet": "Ken Paxton wins Senate primary runoff, defeats incumbent Sen. John Cornyn.",
            "passage": "Ken Paxton wins Senate primary runoff, defeats incumbent Sen. John Cornyn.",
            "score": 0.94,
        }
    ]
    monkeypatch.setattr(
        service.evidence_judgment,
        "judge",
        lambda claim_text, sources: ([{**item, "judgment_label": "unrelated", "judgment_confidence": 0.0} for item in sources], None),
    )

    result = service._score_claim_against_evidence(claim, evidence)
    source = result["evidence_sources"][0]
    aggregate = service._finalize_text_verification(
        [result],
        result["evidence_sources"],
        {
            "simplified_meaning": claim,
            "claim_tuple": service._extract_structured_claim_tuple(claim),
            "exact_phrase_queries": [],
        },
    )

    assert source["aboutness"] == "exact_claim"
    assert source["stance"] == "supports"
    assert source["relation"] == "exact_support"
    assert result["support_score"] >= 0.8
    assert result["directness_score"] >= 0.8
    assert result["source_reliability"] >= 0.9
    assert aggregate["final_label"] in {"Likely Verified", "Verified"}


def test_snopes_false_hospital_claim_refutes_exact_claim(db_session, monkeypatch):
    service = VerificationService(db_session)
    claim = "Stephen Colbert and his wife opened a free hospital for homeless people."
    evidence = [
        {
            "title": "No, Stephen Colbert and his wife did not open a free hospital for homeless people",
            "url": "https://www.snopes.com/fact-check/colbert-free-hospital/",
            "domain": "snopes.com",
            "snippet": "Rating: False. Stephen Colbert and his wife did not open a free hospital for homeless people.",
            "passage": "Rating: False. Stephen Colbert and his wife did not open a free hospital for homeless people.",
            "score": 0.96,
        }
    ]
    monkeypatch.setattr(
        service.evidence_judgment,
        "judge",
        lambda claim_text, sources: ([{**item, "judgment_label": "unrelated", "judgment_confidence": 0.0} for item in sources], None),
    )

    result = service._score_claim_against_evidence(claim, evidence)
    source = result["evidence_sources"][0]
    aggregate = service._finalize_text_verification(
        [result],
        result["evidence_sources"],
        {
            "simplified_meaning": claim,
            "claim_tuple": service._extract_structured_claim_tuple(claim),
            "exact_phrase_queries": [],
        },
    )

    assert source["aboutness"] == "exact_claim"
    assert source["stance"] == "refutes"
    assert source["evidence_relation"] == "refutes"
    assert aggregate["final_label"] == "Unverified"


def test_ordinary_trusted_news_confirmation_can_be_likely_verified(db_session, monkeypatch):
    service = VerificationService(db_session)
    claim = "Authorities are reviewing writings of a California teacher suspected of shooting."
    evidence = [
        {
            "title": "Authorities review writings of California teacher suspected of shooting",
            "url": "https://www.cnn.com/2026/05/31/us/california-teacher-shooting-writings",
            "domain": "cnn.com",
            "snippet": "Authorities are reviewing writings linked to a California teacher suspected in the shooting, officials said.",
            "passage": "Authorities are reviewing writings linked to a California teacher suspected in the shooting, officials said.",
            "score": 0.93,
        }
    ]
    monkeypatch.setattr(
        service.evidence_judgment,
        "judge",
        lambda claim_text, sources: ([{**item, "judgment_label": "unrelated", "judgment_confidence": 0.0} for item in sources], None),
    )

    result = service._score_claim_against_evidence(claim, evidence)
    source = result["evidence_sources"][0]
    aggregate = service._finalize_text_verification(
        [result],
        result["evidence_sources"],
        {
            "simplified_meaning": claim,
            "claim_tuple": service._extract_structured_claim_tuple(claim),
            "exact_phrase_queries": [],
        },
    )

    assert source["aboutness"] in {"exact_claim", "strong_related"}
    assert source["stance"] == "supports"
    assert source["evidence_relation"] == "supports"
    assert result["support_score"] >= 0.75
    assert result["directness_score"] >= 0.75
    assert aggregate["final_label"] in {"Likely Verified", "Verified"}


def test_missing_evidence_is_not_labeled_false_at_claim_level(db_session):
    service = VerificationService(db_session)
    claim = "A city council approved a new transit plan."

    result = service._score_claim_against_evidence(claim, [])
    aggregate = service._finalize_text_verification(
        [result],
        [],
        {
            "simplified_meaning": claim,
            "claim_tuple": service._extract_structured_claim_tuple(claim),
            "exact_phrase_queries": [],
        },
    )

    assert result["claim_verdict"] == "insufficient_evidence"
    assert result["claim_label"] == "Not Verifiable"
    assert aggregate["final_label"] == "Not Verifiable"


def test_text_meaning_bundle_builds_short_queries(db_session):
    service = VerificationService(db_session)
    input_text = (
        "The White House clarified the timeline of family warnings before the Connecticut shooting, "
        "officials said on Tuesday."
    )
    prepared = service.preprocess_text(input_text)
    extraction = service.claim_extraction.extract_claims(prepared["cleaned_input"], max_claims=service.TEXT_CLAIM_LIMIT)

    bundle = service._build_text_meaning_bundle(prepared, extraction, [item["text"] for item in extraction.get("claims", [])])

    assert bundle["simplified_meaning"]
    assert len(bundle["search_queries"]) >= 2
    assert len(bundle["search_queries"]) <= 5
    assert any("White House" in query or "white house" in query.lower() for query in bundle["search_queries"])


def test_text_top_article_retrieval_uses_small_candidate_pool(db_session):
    service = VerificationService(db_session)
    seen_queries = []

    def fake_search(query, expected_domain=None, limit=5):
        seen_queries.append(query)
        return [
            WebSearchCandidate(
                title="White House clarifies warning timeline before Connecticut shooting",
                url="https://example.com/story-1",
                snippet="The White House clarified the timeline of family warnings before the Connecticut shooting.",
                domain="example.com",
                score=0.91,
                explanation="strong title and snippet match",
            ),
            WebSearchCandidate(
                title="AP report on Connecticut shooting timeline",
                url="https://apnews.com/story-2",
                snippet="Officials discussed when family warnings were raised before the shooting.",
                domain="apnews.com",
                score=0.76,
                explanation="partial support",
            ),
        ]

    service.search.search = fake_search
    service.retrieval.fetch_page_document = lambda url: {
        "url": url,
        "fetch_status": "success",
        "content_type": "text/html",
        "title": "White House clarifies warning timeline before Connecticut shooting",
        "extracted_text": "The White House clarified the timeline of family warnings before the Connecticut shooting, officials said on Tuesday.",
        "extracted_text_length": 120,
        "extract_status": "extracted",
    }

    prepared = service.preprocess_text(
        "The White House clarified the timeline of family warnings before the Connecticut shooting, officials said on Tuesday."
    )
    extraction = service.claim_extraction.extract_claims(prepared["cleaned_input"], max_claims=service.TEXT_CLAIM_LIMIT)
    claims = [item["text"] for item in extraction.get("claims", [])]
    bundle = service._build_text_meaning_bundle(prepared, extraction, claims)

    top_articles, debug = service._retrieve_text_top_articles(prepared, extraction, bundle)

    assert 1 <= len(seen_queries) <= 5
    assert 1 <= len(top_articles) <= 5
    assert debug["search_queries"]
    assert debug["top_articles"]
    assert top_articles[0]["support_judgment"] in {"strong support", "partial support"}


def test_text_nlp_layer_exposes_retrieval_failure_reason(db_session):
    service = VerificationService(db_session)

    service.search.search_with_debug = lambda query, expected_domain=None, limit=5: {
        "results": [],
        "failure_reason": "bing_request_failed:TimeoutException",
        "provider_attempts": [{"provider": "bing", "query": query, "result_count": 0, "failure_reason": "bing_request_failed:TimeoutException"}],
        "providers_used": [],
        "stopped_by_budget": False,
    }
    service.search.search = lambda query, expected_domain=None, limit=5: []

    job = VerificationJob(
        input_type=JobInputType.text,
        input_value="sample",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()

    text = "Officials said the White House clarified the timeline of family warnings before the Connecticut shooting on Tuesday."
    payload = build_text_payload(text)
    layer = service._text_nlp_layer(job, payload, text)

    assert layer.details["pipeline_failure"] is True
    assert layer.details["pipeline_failure_reason"] == "retrieval_backend_failure:bing_request_failed:TimeoutException"
    assert layer.details["text_debug"]["search_failure_reason"] == "retrieval_backend_failure:bing_request_failed:TimeoutException"


def test_text_query_generation_requires_exact_phrase_queries(db_session):
    service = VerificationService(db_session)
    prepared = {
        "input_text": "Sample",
        "cleaned_input": "",
        "normalized_text": "",
        "detected_language": "en",
        "sentences": [],
        "core_claim": "",
        "likely_copied_article_text": False,
    }
    extraction = {"claims": [], "entities": [], "locations": [], "dates": [], "numbers": []}
    claims: list[str] = []

    meaning_bundle = service._build_text_meaning_bundle(prepared, extraction, claims)
    query_plan = {"lexical": [], "semantic": []}

    assert service._validate_text_query_plan(meaning_bundle, query_plan) == "retrieval_generation_failed:exact_phrase_queries_missing"


def test_text_query_generation_produces_exact_phrase_queries_for_article_like_text(db_session):
    service = VerificationService(db_session)
    text = (
        "I don't expect forgiveness: Authorities review writings of California teacher suspected of shooting. "
        "Investigators are reviewing writings linked to a California teacher suspected in the shooting, authorities said. "
        "Officials said the writings may help explain the suspect's motive."
    )
    prepared = service.preprocess_text(text)
    extraction = service.claim_extraction.extract_claims(prepared["cleaned_input"], max_claims=service.TEXT_CLAIM_LIMIT)
    claims = [item["text"] for item in extraction.get("claims", [])]
    meaning_bundle = service._build_text_meaning_bundle(prepared, extraction, claims)
    query_plan = service._build_text_hybrid_queries(prepared, extraction, claims, meaning_bundle)

    assert meaning_bundle["exact_phrase_queries"]
    assert query_plan["lexical"]
    assert query_plan["lexical"][0].startswith('"')


def test_text_claims_with_no_retrieved_evidence_remain_not_verifiable(db_session):
    service = VerificationService(db_session)
    claim_results = [
        {
            "claim_text": "Example factual claim",
            "support_score": 0.0,
            "source_reliability": 0.0,
            "agreement_score": 0.0,
            "directness_score": 0.0,
            "clarity_score": 0.8,
            "claim_score": 0.0,
            "claim_label": "Not Verifiable",
            "retrieval_outcome": "no_relevant_evidence_found",
            "evidence_summary": "No relevant evidence was found for this claim.",
            "evidence_sources": [],
        }
    ]

    aggregate = service._aggregate_text_verification_score(claim_results, ["Example factual claim"])

    assert aggregate["final_label"] == "Not Verifiable"
    assert aggregate["final_score"] <= 0.49


def test_dhaka_claim_rejects_china_incident_as_unrelated(db_session):
    service = VerificationService(db_session)
    claim = "ঢাকায় শিক্ষার্থীদের আন্দোলনে পুলিশের সঙ্গে সংঘর্ষ হয়েছে"
    evidence = [
        {
            "title": "Students clash with police during protest in Beijing",
            "url": "https://example.com/china-protest",
            "domain": "example.com",
            "snippet": "Police in Beijing, China clashed with students during a protest.",
            "passage": "Police in Beijing, China clashed with students during a protest.",
            "rerank_score": 0.94,
            "semantic_score": 0.92,
            "bm25_score": 0.88,
        }
    ]

    result = service._score_claim_against_evidence(claim, evidence)
    aggregate = service._aggregate_text_verification_score([result], [claim])

    assert result["evidence_sources"] == []
    assert result["top_candidate_diagnostics"][0]["evidence_relation"] == "unrelated"
    assert result["top_candidate_diagnostics"][0]["location_match_pass"] is False
    assert aggregate["final_label"] != "Likely Verified"
    assert aggregate["final_score"] <= 0.45


def test_dhaka_fire_rejects_beijing_fire_location_mismatch(db_session):
    service = VerificationService(db_session)
    claim = "A major fire happened in Dhaka"
    evidence = [
        {
            "title": "A major fire happened in Beijing",
            "url": "https://example.com/beijing-fire",
            "domain": "example.com",
            "snippet": "A major fire happened in Beijing, China, officials said.",
            "passage": "A major fire happened in Beijing, China, officials said.",
            "rerank_score": 0.96,
            "semantic_score": 0.95,
            "bm25_score": 0.9,
        }
    ]

    result = service._score_claim_against_evidence(claim, evidence)

    assert result["evidence_sources"] == []
    assert result["top_candidate_diagnostics"][0]["evidence_relation"] == "unrelated"
    assert "does not match" in result["top_candidate_diagnostics"][0]["mismatch_reason"]


def test_sundarbans_fake_claim_rejects_unrelated_china_real_estate_source(db_session):
    service = VerificationService(db_session)
    claim = "Bangladesh government secretly sold the Sundarbans to a Chinese company"
    evidence = [
        {
            "title": "Chinese real estate firms invest in urban apartments",
            "url": "https://example.com/china-real-estate",
            "domain": "example.com",
            "snippet": "Chinese companies invested in commercial property projects in China and other cities.",
            "passage": "Chinese companies invested in commercial property projects in China and other cities.",
            "rerank_score": 0.91,
            "semantic_score": 0.89,
            "bm25_score": 0.86,
        }
    ]

    result = service._score_claim_against_evidence(claim, evidence)
    aggregate = service._aggregate_text_verification_score([result], [claim])

    assert result["evidence_sources"] == []
    assert result["top_candidate_diagnostics"][0]["evidence_relation"] in {"unrelated", "insufficient", "background_only"}
    assert aggregate["final_label"] in {"Not Verifiable", "Unverified"}
    assert aggregate["final_score"] <= 0.45


def test_bangla_dhaka_claim_accepts_matching_bangla_evidence(db_session):
    service = VerificationService(db_session)
    claim = "ঢাকায় শিক্ষার্থীদের আন্দোলনে পুলিশের সঙ্গে সংঘর্ষ হয়েছে"
    evidence = [
        {
            "title": "ঢাকায় শিক্ষার্থীদের আন্দোলনে পুলিশের সঙ্গে সংঘর্ষ",
            "url": "https://example.com/dhaka-student-clash",
            "domain": "example.com",
            "snippet": "ঢাকায় শিক্ষার্থীদের আন্দোলনে পুলিশের সঙ্গে সংঘর্ষ হয়েছে বলে স্থানীয় সূত্র জানিয়েছে।",
            "passage": "ঢাকায় শিক্ষার্থীদের আন্দোলনে পুলিশের সঙ্গে সংঘর্ষ হয়েছে বলে স্থানীয় সূত্র জানিয়েছে।",
            "rerank_score": 0.95,
            "semantic_score": 0.94,
            "bm25_score": 0.92,
        }
    ]

    result = service._score_claim_against_evidence(claim, evidence)

    assert result["evidence_sources"]
    assert result["evidence_sources"][0]["evidence_relation"] == "supports"
    assert result["evidence_sources"][0]["location_match_pass"] is True
    assert result["evidence_sources"][0]["event_match_pass"] is True


def test_fake_claim_without_direct_support_is_capped_below_likely_verified(db_session):
    service = VerificationService(db_session)
    claim = "A secret moon base opened under Dhaka city last night"
    evidence = [
        {
            "title": "Scientists discuss future moon bases",
            "url": "https://example.com/moon-base-background",
            "domain": "example.com",
            "snippet": "Researchers discussed future plans for lunar habitats and space exploration.",
            "passage": "Researchers discussed future plans for lunar habitats and space exploration.",
            "rerank_score": 0.72,
            "semantic_score": 0.82,
            "bm25_score": 0.6,
        }
    ]

    result = service._score_claim_against_evidence(claim, evidence)
    aggregate = service._aggregate_text_verification_score([result], [claim])

    assert aggregate["final_label"] != "Likely Verified"
    assert aggregate["final_label"] != "Verified"
    assert aggregate["final_score"] <= 0.55


def test_claim_evidence_verifier_fallback_rejects_wrong_location():
    verifier = ClaimEvidenceVerifier(ContextMatchService())
    result = verifier.verify(
        "Students clashed with police in Dhaka.",
        {
            "title": "Students clash with police in Beijing",
            "snippet": "Police in Beijing, China clashed with students during a protest.",
        },
        relevance_score=0.94,
        fallback_label="supports",
    )

    assert result.relation_label == "unrelated"
    assert result.confidence >= 0.85
    assert result.context_match["location_match_pass"] is False


def test_claim_evidence_verifier_fallback_accepts_matching_bangla_support():
    verifier = ClaimEvidenceVerifier(ContextMatchService())
    result = verifier.verify(
        "ঢাকায় শিক্ষার্থীদের আন্দোলনে পুলিশের সঙ্গে সংঘর্ষ হয়েছে",
        {
            "title": "ঢাকায় শিক্ষার্থীদের আন্দোলনে পুলিশের সঙ্গে সংঘর্ষ",
            "snippet": "ঢাকায় শিক্ষার্থীদের আন্দোলনে পুলিশের সঙ্গে সংঘর্ষ হয়েছে বলে স্থানীয় সূত্র জানিয়েছে।",
        },
        relevance_score=0.94,
        fallback_label="supports",
    )

    assert result.relation_label == "supports"
    assert result.context_match["location_match_pass"] is True
    assert result.context_match["event_match_pass"] is True


def test_source_adapter_normalizes_youtube_metadata_without_api_key():
    payload = ExtractedPayload(
        title="Original title",
        author=None,
        publish_time=None,
        canonical_url="https://www.youtube.com/watch?v=abc123xyz89",
        main_text="Original body",
        hero_image_url=None,
        normalized_text="original body",
        content_hash="hash",
        metadata={
            "publisher_name": "Fallback Channel",
            "youtube_signals": {"channel_verified": True},
        },
    )

    adapted = asyncio.run(SourceAdapterService().adapt_url_payload("https://www.youtube.com/watch?v=abc123xyz89", payload))

    normalized = adapted.metadata["normalized_source"]
    assert normalized["platform_type"] == "youtube"
    assert normalized["source_type"] == "social_post"
    assert adapted.canonical_url == "https://www.youtube.com/watch?v=abc123xyz89"
    assert adapted.metadata["normalized_url"] == "https://www.youtube.com/watch?v=abc123xyz89"
    assert adapted.metadata["youtube_video_id"] == "abc123xyz89"
    assert adapted.metadata["publisher_name"] == "Fallback Channel"
    assert adapted.metadata["channel_verification_status"] == "verified"
    assert adapted.metadata["transcript_status"] == "unavailable"


def test_source_adapter_rejects_blocked_youtube_page_and_uses_real_metadata(monkeypatch):
    service = SourceAdapterService()

    async def fake_oembed(_url):
        return {
            "title": "Real Video Title",
            "author_name": "Real Channel",
            "thumbnail_url": "https://img.youtube.com/example.jpg",
        }

    async def fake_api(_video_id):
        return {
            "snippet": {
                "title": "Real Video Title",
                "description": "Real description from video metadata.",
                "channelTitle": "Real Channel",
                "channelId": "channel-123",
                "publishedAt": "2026-04-27T10:00:00Z",
                "thumbnails": {"high": {"url": "https://img.youtube.com/example-high.jpg"}},
            },
            "channel": {"title": "Real Channel"},
        }

    monkeypatch.setattr(service, "_fetch_youtube_oembed_metadata", fake_oembed)
    monkeypatch.setattr(service, "_fetch_youtube_api_metadata", fake_api)

    payload = ExtractedPayload(
        title="Our systems have detected unusual traffic from your computer network",
        author=None,
        publish_time=None,
        canonical_url="https://www.youtube.com/watch?v=abc123xyz89",
        main_text="Our systems have detected unusual traffic from your computer network. Please verify you are human.",
        hero_image_url=None,
        normalized_text="blocked page",
        content_hash="hash",
        metadata={"description": "captcha page", "youtube_signals": {}},
    )

    adapted = asyncio.run(service.adapt_url_payload("https://www.youtube.com/watch?v=abc123xyz89", payload))

    assert adapted.title == "Real Video Title"
    assert "unusual traffic" not in adapted.main_text.lower()
    assert adapted.metadata["blocked_page_detected"] is True
    assert adapted.metadata["metadata_fetch_status"] == "api_metadata_loaded"
    assert adapted.metadata["channel_title"] == "Real Channel"
    assert adapted.metadata["channel_id"] == "channel-123"
    assert adapted.metadata["publish_time"] == "2026-04-27T10:00:00+00:00"


def test_youtube_consensus_votes_do_not_leak_from_generic_watch_identity(db_session):
    service = VerificationService(db_session)

    other_job = service._create_job(JobInputType.url, "https://www.youtube.com/watch")
    db_session.add(
        ConsensusVote(
            job_id=other_job.id,
            canonical_url="https://www.youtube.com/watch",
            content_hash="generic-watch",
            reporter_name="Existing voter",
            role=ConsensusRole.user,
            vote_value=1,
            confidence=1.0,
            weight=1.0,
            moderation_flag=False,
        )
    )

    youtube_job = service._create_job(JobInputType.url, "https://www.youtube.com/watch?v=abc123xyz89")
    db_session.add(
        ExtractedContent(
            job_id=youtube_job.id,
            title="Example YouTube video",
            author=None,
            publish_time=None,
            canonical_url="https://www.youtube.com/watch",
            main_text="Video description text",
            hero_image_url=None,
            normalized_text="video description text",
            content_hash="youtube-video-hash",
            metadata_json={
                "platform_type": "youtube",
                "youtube_video_id": "abc123xyz89",
                "normalized_url": "https://www.youtube.com/watch?v=abc123xyz89",
                "channel_title": "Example Channel",
            },
        )
    )
    db_session.flush()

    outcome = service._layer3(youtube_job, persist_audit=False)

    assert outcome.details["shared_content_identity"]["canonical_url"] == "https://www.youtube.com/watch?v=abc123xyz89"
    assert outcome.details["vote_count"] == 0
    assert outcome.status == "none"


def test_youtube_social_input_ignores_blocked_page_text(db_session):
    service = VerificationService(db_session)
    payload = ExtractedPayload(
        title="Video metadata could not be retrieved",
        author=None,
        publish_time=None,
        canonical_url="https://www.youtube.com/watch?v=abc123xyz89",
        main_text="Our systems have detected unusual traffic from your computer network",
        hero_image_url=None,
        normalized_text="blocked page",
        content_hash="blocked-youtube-hash",
        metadata={
            "platform_type": "youtube",
            "description": "",
            "channel_title": None,
            "transcript_text": "",
            "blocked_page_detected": True,
            "blocked_page_reason": "our systems have detected unusual traffic",
            "metadata_fetch_status": "blocked_page_detected_metadata_unavailable",
        },
    )

    assert service._build_social_text_input(payload) == ""


def test_source_adapter_repairs_social_error_payload_with_description_and_image():
    payload = ExtractedPayload(
        title="Sorry, something went wrong.",
        author=None,
        publish_time=None,
        canonical_url="https://www.facebook.com/example/posts/123",
        main_text="Sorry, something went wrong. We're working on getting this fixed as soon as we can. Go back",
        hero_image_url="https://cdn.example.com/post.jpg",
        normalized_text="sorry something went wrong",
        content_hash="hash",
        metadata={
            "description": "Facebook post caption about a developing event.",
            "hero_image_url": "https://cdn.example.com/post.jpg",
            "source_domain": "www.facebook.com",
        },
    )

    adapted = asyncio.run(SourceAdapterService().adapt_url_payload("https://www.facebook.com/example/posts/123", payload))

    assert adapted.title == "Facebook post caption about a developing event."
    assert adapted.main_text == "Facebook post caption about a developing event."
    assert adapted.metadata["social_payload_repaired"] is True
    assert adapted.metadata["media_assets"][0]["url"] == "https://cdn.example.com/post.jpg"


def test_ocr_reports_invalid_image_bytes():
    result = extract_text_details_from_image_bytes(b"not-an-image", source="https://example.com/bad")

    assert result["ocr_status"] == "invalid_image_format"
    assert "invalid_image_format" in result["failure_reasons"]
    assert result["text_detected"] is False


def test_ocr_reports_too_small_images():
    image = Image.new("RGB", (20, 20), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    result = extract_text_details_from_image_bytes(buffer.getvalue(), source="https://example.com/tiny.png")

    assert result["ocr_status"] == "image_too_small_for_ocr"
    assert "image_too_small_for_ocr" in result["failure_reasons"]


def test_image_system_scoring_caps_when_only_ocr_and_no_corroboration(db_session):
    service = VerificationService(db_session)
    layer1 = LayerOutcome(
        status="platform-issued",
        confidence=0.78,
        explanation="ok",
        details={
            "manifest_id": "m1",
            "provenance_verification_status": "valid",
            "crypto_breakdown": {},
        },
    )
    layer2 = LayerOutcome(
        status="warning",
        confidence=0.72,
        explanation="ocr only",
        details={
            "image_analysis_score": 0.72,
            "verification_confidence": 0.2,
            "retrieval_confidence": 1.0,
            "ocr_result": {"text_detected": True},
            "no_external_corroboration": True,
            "no_provenance_support": True,
            "manipulation_detected": False,
            "misleading_reuse": False,
            "evidence_elements_found": {
                "metadata_found": True,
                "ocr_checked": True,
                "ocr_text_detected": True,
                "reverse_image_checked": True,
                "context_checked": True,
                "manipulation_analysis_completed": True,
                "original_source_match_found": False,
                "provenance_record_found": False,
            },
            "image_analysis_breakdown": {},
            "image_branches": {},
            "evidence_sources": [],
        },
    )

    result = service._build_image_system_scoring(layer1, layer2)

    assert result["final_score"] <= 49.0
    assert result["final_label"] == "Not Verifiable"


def test_layer1_matches_verified_publisher_with_flexible_name_normalization(db_session):
    ensure_default_publishers(db_session)
    payload = ExtractedPayload(
        title="Example CNN story",
        author="Reporter Name",
        publish_time=datetime.fromisoformat("2026-03-17T14:02:02"),
        canonical_url="https://www.cnn.com/2026/03/17/politics/example-story",
        main_text="Example body",
        hero_image_url=None,
        normalized_text="example body",
        content_hash="cnn-normalization-test",
        metadata={
            "fetched_url": "https://www.cnn.com/2026/03/17/politics/example-story",
            "publisher_name": "CNN News",
            "source_domain": "www.cnn.com",
        },
    )

    outcome = VerificationService(db_session)._layer1(payload)

    assert outcome.status == "publisher-recognized"
    assert outcome.details["publisher"] == "CNN"


def test_refresh_job_reassesses_layer2_with_metadata_signals(db_session):
    ensure_default_publishers(db_session)
    job = VerificationJob(
        input_type=JobInputType.url,
        input_value="https://www.cnn.com/2026/03/17/politics/example-story",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()
    job.extracted_content = ExtractedContent(
        job_id=job.id,
        title="Example CNN story",
        author="Reporter Name",
        publish_time=datetime.fromisoformat("2026-03-17T14:02:02"),
        canonical_url="https://www.cnn.com/2026/03/17/politics/example-story",
        main_text=" ".join(["This is a full article body from CNN."] * 40),
        hero_image_url=None,
        normalized_text=" ".join(["this is a full article body from cnn"] * 40),
        content_hash="cnn-refresh-test",
        metadata_json={
            "fetched_url": "https://www.cnn.com/2026/03/17/politics/example-story",
            "publisher_name": "CNN",
            "source_domain": "www.cnn.com",
            "section": "Politics",
            "keywords": ["politics", "iran"],
            "media_assets": [{"type": "image", "url": "https://media.cnn.com/example.jpg"}],
        },
    )
    db_session.commit()

    service = VerificationService(db_session)
    service._refresh_job_assessment(job, commit=False)

    assert job.layer_results["layer1"]["status"] == "publisher-recognized"
    assert job.layer_results["layer2"]["status"] == "supporting"
    assert job.layer_results["layer2"]["details"]["trusted_publisher_match"] == "CNN"


def test_layer3_supporting_consensus(db_session):
    job = VerificationJob(
        input_type=JobInputType.text,
        input_value="sample",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()
    db_session.add(ConsensusVote(job_id=job.id, vote_value=1, weight=2.0, moderation_flag=False))
    db_session.add(ConsensusVote(job_id=job.id, vote_value=-1, weight=1.0, moderation_flag=True))
    db_session.commit()

    outcome = VerificationService(db_session)._layer3(job)
    assert outcome.status == "supporting"
    assert outcome.details["vote_count"] == 2


def test_audit_log_hash_chain(db_session):
    job = VerificationJob(
        input_type=JobInputType.text,
        input_value="sample",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()

    first = append_audit_entry(db_session, job, "first", {"step": 1})
    second = append_audit_entry(db_session, job, "second", {"step": 2})

    assert second.previous_hash == first.entry_hash
    assert second.entry_hash != first.entry_hash


def test_analysis_report_uses_three_named_sections(db_session):
    payload = build_text_payload("Layer naming regression check")
    service = VerificationService(db_session)

    report = service._build_analysis_report(
        JobStatus.not_verifiable,
        service._layer1(payload),
        LayerOutcome(
            status="none",
            confidence=0.0,
            explanation="No strong corroborating source or semantic match was found.",
            details={"matches": [], "fact_check_verdict": "inconclusive"},
        ),
        LayerOutcome(
            status="none",
            confidence=0.0,
            explanation="Community consensus combines user reports, weighted roles, and moderation flags as supporting evidence only.",
            details={"vote_count": 0, "moderation_flags": 0},
        ),
        payload,
    )
    titles = [section["title"] for section in report["report_sections"]]

    assert titles == ["Cryptographic Provenance", "Fact Checking", "Community Consensus"]
    assert "system_scoring" in report
    assert "final_score" in report["system_scoring"]


@pytest.mark.asyncio
async def test_remote_manifest_positive_case_uses_verified_state():
    class StubResponse:
        def __init__(self, content: bytes, content_type: str = "application/json"):
            self.content = content
            self.headers = {"content-type": content_type}

        def raise_for_status(self):
            return None

    class StubClient:
        async def get(self, _url: str, headers: dict | None = None):
            return StubResponse(b'{"manifest":"c2pa"}')

    ok, details = await _verify_remote_manifest_reference(StubClient(), "https://example.com/asset.c2pa")
    assert ok is True
    assert details["status"] == "verified"


def test_linkedin_page_signal_present_but_manifest_not_retrievable_wording():
    html = """
    <html>
      <head><meta property="og:title" content="LinkedIn post with Content Credentials" /></head>
      <body><div>Content Credentials</div></body>
    </html>
    """
    signal = _detect_linkedin_page_signal(BeautifulSoup(html, "lxml"), "https://www.linkedin.com/posts/example")
    finding = {
        "type": "image",
        "verification_state": "platform_signal_detected_manifest_not_retrievable",
        "platform": "linkedin",
        "page_level_platform_signal_found": signal["found"],
    }
    assert signal["found"] is True
    assert "LinkedIn appears to signal Content Credentials" in _describe_remote_provenance(finding)


def test_non_media_content_type_from_linkedin_cdn_gets_clear_note():
    finding = {
        "platform": "linkedin",
        "response_kind": "json",
        "content_type": "application/json",
        "verification_state": "could_not_evaluate_asset",
    }
    note = _provenance_note(finding)
    assert "JSON" in note


def test_sidecar_manifest_url_candidates_are_generated():
    candidates = _candidate_sidecar_manifest_urls("https://media.licdn.com/dms/image/v2/example.jpg")
    assert any(candidate.endswith(".c2pa") for candidate in candidates)


def test_checked_successfully_no_credentials_wording_is_not_buggy():
    finding = {"type": "image", "verification_state": "checked_no_retrievable_credentials"}
    assert _describe_remote_provenance(finding) == "The asset was checked successfully, but no retrievable credentials were found."


def test_soft_binding_path_can_be_reported_without_repository():
    service = SoftBindingRecoveryService()
    result = service.recoverDecoupledManifest(b"binary-data", "https://media.licdn.com/example.jpg", "image/jpeg")
    assert result.status == "soft_binding_recovery_not_configured"
    assert result.repository_configured is False


def test_decide_returns_likely_verified_when_primary_layers_support_without_votes(db_session):
    service = VerificationService(db_session)
    payload = build_text_payload("Primary layers support this article with strong matching evidence.")
    layer1 = LayerOutcome(
        status="publisher-recognized",
        confidence=0.62,
        explanation="Trusted publisher recognized.",
        details={},
    )
    layer2 = LayerOutcome(
        status="supporting",
        confidence=0.64,
        explanation="Corroborating similarity found.",
        details={"fact_check_verdict": "content match found"},
    )
    layer3 = LayerOutcome(
        status="none",
        confidence=0.0,
        explanation="No community votes yet.",
        details={"vote_count": 0},
    )

    status_value, headline, _, _, _ = service._decide(layer1, layer2, layer3, payload)

    assert status_value == JobStatus.likely_verified
    assert headline == "Likely Verified"


def test_decide_returns_verified_when_all_three_layers_align(db_session):
    service = VerificationService(db_session)
    payload = build_text_payload("All three layers align on this content.")
    layer1 = LayerOutcome(status="matched", confidence=0.99, explanation="Signed proof matched.", details={})
    layer2 = LayerOutcome(
        status="supporting",
        confidence=0.83,
        explanation="Strong corroborating source found.",
        details={"fact_check_verdict": "content match found"},
    )
    layer3 = LayerOutcome(
        status="supporting",
        confidence=0.6,
        explanation="Consensus supports the content.",
        details={"vote_count": 3},
    )

    status_value, headline, _, _, _ = service._decide(layer1, layer2, layer3, payload)

    assert status_value == JobStatus.verified
    assert headline == "Verified"


def test_consensus_vote_upgrades_likely_verified_job_to_verified(db_session):
    ensure_default_publishers(db_session)
    job = VerificationJob(
        input_type=JobInputType.url,
        input_value="https://www.cnn.com/2026/03/17/politics/example-story",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()
    job.extracted_content = ExtractedContent(
        job_id=job.id,
        title="Example CNN story",
        author="Reporter Name",
        publish_time=datetime.fromisoformat("2026-03-17T14:02:02"),
        canonical_url="https://www.cnn.com/2026/03/17/politics/example-story",
        main_text=" ".join(["This is a full article body from CNN."] * 40),
        hero_image_url=None,
        normalized_text=" ".join(["this is a full article body from cnn"] * 40),
        content_hash="cnn-consensus-upgrade-test",
        metadata_json={
            "fetched_url": "https://www.cnn.com/2026/03/17/politics/example-story",
            "publisher_name": "CNN",
            "source_domain": "www.cnn.com",
            "section": "Politics",
            "keywords": ["politics", "iran"],
            "media_assets": [{"type": "image", "url": "https://media.cnn.com/example.jpg"}],
        },
    )
    db_session.commit()

    service = VerificationService(db_session)
    service._refresh_job_assessment(job, commit=True)
    assert job.status == JobStatus.likely_verified

    service.add_consensus_vote(job.id, ConsensusVoteRequest(reporter_name="Tester", vote_value=1))
    refreshed = service.get_result(job.id)

    assert refreshed.status == JobStatus.verified
    assert refreshed.layer_results["layer3"]["status"] == "supporting"


def test_consensus_votes_apply_to_new_job_with_same_content_identity(db_session):
    ensure_default_publishers(db_session)
    service = VerificationService(db_session)
    extracted = ExtractedPayload(
        title="Example CNN story",
        author="Reporter Name",
        publish_time=datetime.fromisoformat("2026-03-17T14:02:02"),
        canonical_url="https://www.cnn.com/2026/03/17/politics/example-story",
        main_text=" ".join(["This is a full article body from CNN."] * 40),
        hero_image_url=None,
        normalized_text=" ".join(["this is a full article body from cnn"] * 40),
        content_hash="cnn-consensus-shared-identity",
        metadata={
            "fetched_url": "https://www.cnn.com/2026/03/17/politics/example-story",
            "publisher_name": "CNN",
            "source_domain": "www.cnn.com",
            "section": "Politics",
            "keywords": ["politics", "iran"],
            "media_assets": [{"type": "image", "url": "https://media.cnn.com/example.jpg"}],
        },
    )

    first_job = service._create_job(JobInputType.url, extracted.canonical_url)
    first_result = service._run_pipeline(first_job, extracted)
    assert first_result.status == JobStatus.likely_verified

    service.add_consensus_vote(first_job.id, ConsensusVoteRequest(reporter_name="Tester", vote_value=1))

    second_job = service._create_job(JobInputType.url, extracted.canonical_url)
    second_result = service._run_pipeline(second_job, extracted)

    assert second_result.status == JobStatus.verified
    assert second_result.layer_results["layer3"]["status"] == "supporting"
    assert second_result.layer_results["layer3"]["details"]["vote_count"] == 1


def test_consensus_votes_do_not_leak_across_different_urls_with_same_hash(db_session):
    ensure_default_publishers(db_session)
    service = VerificationService(db_session)
    first_payload = ExtractedPayload(
        title="First story",
        author="Reporter",
        publish_time=datetime.fromisoformat("2026-03-17T14:02:02"),
        canonical_url="https://www.cnn.com/2026/03/17/politics/first-story",
        main_text="shared body hash",
        hero_image_url=None,
        normalized_text="shared body hash",
        content_hash="shared-consensus-hash",
        metadata={"publisher_name": "CNN", "source_domain": "www.cnn.com"},
    )
    second_payload = ExtractedPayload(
        title="Second story",
        author="Reporter",
        publish_time=datetime.fromisoformat("2026-03-17T15:02:02"),
        canonical_url="https://www.cnn.com/2026/03/17/politics/second-story",
        main_text="shared body hash",
        hero_image_url=None,
        normalized_text="shared body hash",
        content_hash="shared-consensus-hash",
        metadata={"publisher_name": "CNN", "source_domain": "www.cnn.com"},
    )

    first_job = service._create_job(JobInputType.url, first_payload.canonical_url)
    service._run_pipeline(first_job, first_payload)
    service.add_consensus_vote(first_job.id, ConsensusVoteRequest(reporter_name="Tester", vote_value=1))

    second_job = service._create_job(JobInputType.url, second_payload.canonical_url)
    second_result = service._run_pipeline(second_job, second_payload)

    assert second_result.layer_results["layer3"]["details"]["vote_count"] == 0
    assert second_result.layer_results["layer3"]["status"] == "none"


def test_supportive_consensus_upgrades_likely_verified_when_two_layers_align(db_session):
    service = VerificationService(db_session)
    payload = build_text_payload("A supportive publisher signal plus community confirmation should verify this.")
    layer1 = LayerOutcome(
        status="publisher-recognized",
        confidence=0.62,
        explanation="Trusted publisher recognized.",
        details={},
    )
    layer2 = LayerOutcome(
        status="none",
        confidence=0.0,
        explanation="No strong NLP support.",
        details={"fact_check_verdict": "inconclusive"},
    )
    layer3 = LayerOutcome(
        status="supporting",
        confidence=0.75,
        explanation="Consensus supports the content.",
        details={"vote_count": 4, "majority_vote": "true"},
    )

    status_value, headline, _, _, _ = service._decide(layer1, layer2, layer3, payload)

    assert status_value == JobStatus.verified
    assert headline == "Verified"


def test_decide_returns_unverified_when_two_layers_are_negative(db_session):
    service = VerificationService(db_session)
    payload = build_text_payload("Two layers disagree with the content.")
    layer1 = LayerOutcome(status="mismatch", confidence=0.98, explanation="Trusted proof mismatch.", details={})
    layer2 = LayerOutcome(status="none", confidence=0.0, explanation="No NLP support.", details={"fact_check_verdict": "inconclusive"})
    layer3 = LayerOutcome(
        status="disputing",
        confidence=-0.7,
        explanation="Consensus disputes the content.",
        details={"vote_count": 2},
    )

    status_value, headline, _, _, _ = service._decide(layer1, layer2, layer3, payload)

    assert status_value == JobStatus.unverified
    assert headline == "Unverified"


def test_audio_analysis_uses_transcript_text_pipeline(db_session, monkeypatch):
    service = VerificationService(db_session)
    extracted = ExtractedPayload(
        title="Audio upload: speech.wav",
        author=None,
        publish_time=None,
        canonical_url=None,
        main_text="Uploaded audio summary",
        hero_image_url=None,
        normalized_text="uploaded audio summary",
        content_hash="audio-hash",
        metadata={
            "media_type": "audio",
            "caption_or_claim": "",
            "claimed_speaker": "Anchor",
            "probe": {"format": {"format_name": "wav", "duration": "14.2"}},
            "forensics": {"fields": {}},
        },
    )

    monkeypatch.setattr(
        service.transcription,
        "transcribe_audio",
        lambda _path: {
            "transcript_text": "The governor said inflation fell sharply this month.",
            "cleaned_transcript": "The governor said inflation fell sharply this month.",
            "transcript_status": "completed",
            "transcript_confidence": 0.91,
            "transcript_summary": "Transcript created locally.",
            "timestamps": [{"start": 0.0, "end": 2.0, "text": "The governor said inflation fell sharply this month."}],
            "detected_language": "en",
            "language_probability": 0.98,
            "chunk_count": 1,
            "engine": "faster-whisper",
            "model": "small",
            "file_format": "wav",
            "duration_seconds": 14.2,
            "preprocessing_status": "completed",
            "preprocessing_details": {"status": "completed"},
            "failure_reason": None,
        },
    )
    monkeypatch.setattr(
        service,
        "_gather_text_candidate_documents",
        lambda prepared, extraction, query_plan, source_recovery_stage: (
            [
                {
                    "title": "Reuters on inflation",
                    "url": "https://www.reuters.com/example",
                    "domain": "www.reuters.com",
                    "snippet": "Reuters reported inflation fell sharply this month.",
                    "passage": "Reuters reported inflation fell sharply this month.",
                    "source": "web_search",
                }
            ],
            {"lexical_searches": [], "semantic_searches": [], "candidate_fetches": [], "retrieval_failure_reason": None},
        ),
    )
    monkeypatch.setattr(
        service,
        "_hybrid_rank_text_candidates",
        lambda prepared, claims, query_plan, candidates: {
            "bm25_top_results": candidates,
            "semantic_top_results": candidates,
            "reranked_top_results": candidates,
            "rerank_failure_reason": None,
        },
    )
    monkeypatch.setattr(
        service,
        "_score_claim_against_evidence",
        lambda claim, evidence: {
            "claim_text": claim,
            "claim_score": 0.82,
            "claim_label": "Verified",
            "evidence_summary": "Supporting evidence found.",
            "evidence_sources": evidence,
        },
    )
    monkeypatch.setattr(
        service,
        "_finalize_text_verification",
        lambda claim_results, top_articles, meaning_bundle: {
            "final_score": 0.82,
            "final_label": "Verified",
            "explanation": "Transcript evidence aligned with external reporting.",
            "supporting_evidence_summary": "Supporting evidence found.",
            "contradiction_summary": "",
            "evidence_summary": "Supporting evidence found.",
            "evidence_sources": top_articles,
            "component_scores": {"S": 0.9, "R": 0.8, "A": 0.8, "D": 0.8, "C": 0.9, "E": 0.8, "P": 1.0},
        },
    )

    outcome = service._audio_analysis_layer(extracted, "sample.wav", "speech.wav")

    assert outcome.details["transcription_result"]["engine"] == "faster-whisper"
    assert outcome.details["transcription_result"]["transcript_status"] == "completed"
    assert outcome.details["text_verification_label"] == "Verified"
    assert outcome.details["matches"][0]["url"] == "https://www.reuters.com/example"
    assert outcome.details["text_debug"]["cleaned_input"] == "The governor said inflation fell sharply this month."


def test_audio_analysis_handles_empty_transcript_without_fake_verification(db_session, monkeypatch):
    service = VerificationService(db_session)
    extracted = ExtractedPayload(
        title="Audio upload: silence.wav",
        author=None,
        publish_time=None,
        canonical_url=None,
        main_text="Uploaded audio summary",
        hero_image_url=None,
        normalized_text="uploaded audio summary",
        content_hash="audio-empty-hash",
        metadata={"media_type": "audio", "probe": {}, "forensics": {"fields": {}}},
    )
    monkeypatch.setattr(
        service.transcription,
        "transcribe_audio",
        lambda _path: {
            "transcript_text": "",
            "cleaned_transcript": "",
            "transcript_status": "speech_not_detected",
            "transcript_confidence": 0.0,
            "transcript_summary": "No meaningful speech detected.",
            "timestamps": [],
            "detected_language": None,
            "language_probability": None,
            "chunk_count": 1,
            "engine": "faster-whisper",
            "model": "small",
            "file_format": "wav",
            "duration_seconds": 8.0,
            "preprocessing_status": "completed",
            "preprocessing_details": {"status": "completed"},
            "failure_reason": None,
        },
    )

    outcome = service._audio_analysis_layer(extracted, "silence.wav", "silence.wav")

    assert outcome.details["transcription_result"]["transcript_status"] == "speech_not_detected"
    assert outcome.details["matches"] == []
    assert outcome.details["text_verification_score"] == 0.0
    assert outcome.details["evidence_elements_found"]["transcript_extracted"] is False


def test_text_nlp_layer_extracts_claims_and_limits_web_evidence_to_five(db_session, monkeypatch):
    service = VerificationService(db_session)
    extracted = build_text_payload(
        "President Ada Lovelace said the central bank cut rates on Monday. "
        "The government then announced a new emergency fuel subsidy."
    )

    def fake_search(query, expected_domain=None, limit=5):
        return [
            WebSearchCandidate(
                title=f"Match {index} for {query[:20]}",
                url=f"https://example{index}.com/story-{index}",
                snippet="Relevant evidence snippet",
                domain=f"example{index}.com",
                score=0.72 - (index * 0.03),
                explanation="mocked",
            )
            for index in range(7)
        ]

    monkeypatch.setattr(service.search, "search", fake_search)
    monkeypatch.setattr(
        service.retrieval,
        "fetch_page_document",
        lambda url: {
            "url": url,
            "fetch_status": "success",
            "content_type": "text/html",
            "title": "President Ada Lovelace said the central bank cut rates on Monday",
            "extracted_text": (
                "President Ada Lovelace said the central bank cut rates on Monday. "
                "The government then announced a new emergency fuel subsidy."
            ),
            "extracted_text_length": 122,
            "extract_status": "extracted",
        },
    )

    job = service._create_job(JobInputType.text, extracted.main_text[:100])
    outcome = service._text_nlp_layer(job, extracted, extracted.main_text)

    assert outcome.details["mode"] == "text_verification"
    assert len(outcome.details["extracted_claims"]) >= 1
    assert 0 < len(outcome.details["evidence_sources"]) <= 5
    assert all(isinstance(item, dict) for item in outcome.details["claim_results"])
    assert any(item.get("claim_verdict") in {"supported", "insufficient_evidence", "contradicted"} for item in outcome.details["claim_results"])


def test_sundarbans_extraordinary_claim_rejects_keyword_only_sources(db_session, monkeypatch):
    service = VerificationService(db_session)
    text = (
        "The government of Bangladesh secretly signed a deal to sell the Sundarbans to a Chinese real estate conglomerate "
        "for $50 billion. Officials allegedly planned to convert the mangrove forest into luxury resort islands."
    )
    extracted = build_text_payload(text)

    def fake_search(query, expected_domain=None, limit=5):
        return [
            WebSearchCandidate(
                title="Bangladesh and China discuss investment cooperation",
                url="https://news.example.com/bangladesh-china-investment",
                snippet="Officials discussed infrastructure and trade cooperation between Bangladesh and China.",
                domain="news.example.com",
                score=0.91,
                explanation="mocked",
            ),
            WebSearchCandidate(
                title="Sundarbans tourism operators promote resort visits",
                url="https://travel.example.com/sundarbans-resorts",
                snippet="A tourism article describes resort stays near the Sundarbans mangrove forest.",
                domain="travel.example.com",
                score=0.86,
                explanation="mocked",
            ),
            WebSearchCandidate(
                title="UNESCO Sundarbans World Heritage overview",
                url="https://whc.unesco.org/en/list/798/",
                snippet="The Sundarbans is a mangrove forest and World Heritage site in Bangladesh.",
                domain="whc.unesco.org",
                score=0.82,
                explanation="mocked",
            ),
        ][:limit]

    monkeypatch.setattr(service.search, "search", fake_search)
    monkeypatch.setattr(
        service.retrieval,
        "fetch_page_document",
        lambda url: {
            "url": url,
            "fetch_status": "success",
            "content_type": "text/html",
            "title": "Bangladesh and China discuss investment cooperation",
            "extracted_text": (
                "Bangladesh and China discussed infrastructure investment, trade, and regional cooperation. "
                "The article covers ports, power generation, and commercial partnerships."
            ),
            "extracted_text_length": 236,
            "extract_status": "extracted",
        },
    )

    job = service._create_job(JobInputType.text, text[:100])
    outcome = service._text_nlp_layer(job, extracted, text)

    assert outcome.details["text_verification_label"] not in {"Verified", "Likely Verified"}
    assert outcome.details["text_verification_score"] < 0.3
    assert "No credible direct evidence" in outcome.explanation
    assert all(source.get("stance") != "supports" for source in outcome.details["evidence_sources"])
    assert all(result.get("claim_verdict") != "supported" for result in outcome.details["claim_results"])


def test_audio_transcript_fact_check_matches_text_for_same_claim(db_session, monkeypatch):
    service = VerificationService(db_session)
    text = (
        "The government of Bangladesh secretly signed a deal to sell the Sundarbans to a Chinese real estate conglomerate "
        "for $50 billion. Officials allegedly planned to convert the mangrove forest into luxury resort islands."
    )

    def fake_search(query, expected_domain=None, limit=5):
        return [
            WebSearchCandidate(
                title="Bangladesh and China discuss investment cooperation",
                url="https://news.example.com/bangladesh-china-investment",
                snippet="Officials discussed infrastructure and trade cooperation between Bangladesh and China.",
                domain="news.example.com",
                score=0.91,
                explanation="mocked",
            ),
            WebSearchCandidate(
                title="Sundarbans tourism operators promote resort visits",
                url="https://travel.example.com/sundarbans-resorts",
                snippet="A tourism article describes resort stays near the Sundarbans mangrove forest.",
                domain="travel.example.com",
                score=0.86,
                explanation="mocked",
            ),
        ][:limit]

    monkeypatch.setattr(service.search, "search", fake_search)
    monkeypatch.setattr(
        service.retrieval,
        "fetch_page_document",
        lambda url: {
            "url": url,
            "fetch_status": "success",
            "content_type": "text/html",
            "title": "Bangladesh and China discuss investment cooperation",
            "extracted_text": "Bangladesh and China discussed infrastructure investment, trade, and regional cooperation.",
            "extracted_text_length": 88,
            "extract_status": "extracted",
        },
    )
    monkeypatch.setattr(
        service.transcription,
        "transcribe_audio",
        lambda _path: {
            "transcript_text": text,
            "cleaned_transcript": text,
            "transcript_status": "completed",
            "transcript_confidence": 0.96,
            "transcript_summary": "Transcript created locally.",
            "timestamps": [{"start": 0.0, "end": 9.0, "text": text}],
            "detected_language": "en",
            "language_probability": 0.99,
            "chunk_count": 1,
            "engine": "faster-whisper",
            "model": "small",
            "file_format": "wav",
            "duration_seconds": 9.0,
            "preprocessing_status": "completed",
            "preprocessing_details": {"status": "completed"},
            "failure_reason": None,
        },
    )

    text_payload = build_text_payload(text)
    text_job = service._create_job(JobInputType.text, text[:100])
    text_outcome = service._text_nlp_layer(text_job, text_payload, text)

    audio_payload = ExtractedPayload(
        title="Audio upload: parity.wav",
        author=None,
        publish_time=None,
        canonical_url=None,
        main_text="Uploaded audio summary",
        hero_image_url=None,
        normalized_text="uploaded audio summary",
        content_hash="audio-parity-hash",
        metadata={"media_type": "audio", "probe": {"format": {"duration": "9.0"}}, "forensics": {"fields": {}}},
    )
    audio_job = service._create_job(JobInputType.audio, "parity.wav")
    audio_outcome = service._audio_analysis_layer(audio_payload, "parity-test.wav", "parity.wav", job=audio_job)

    assert audio_outcome.details["transcription_result"]["transcript_confidence"] == 0.96
    assert audio_outcome.details["extracted_claims"] == text_outcome.details["extracted_claims"]
    assert audio_outcome.details["search_queries_used"] == text_outcome.details["search_queries_used"]
    assert audio_outcome.details["text_verification_label"] == text_outcome.details["text_verification_label"]
    assert abs(audio_outcome.details["text_verification_score"] - text_outcome.details["text_verification_score"]) <= 0.001
    for key in ("S", "E", "P", "relevant_count"):
        assert audio_outcome.details["text_component_scores"].get(key) == text_outcome.details["text_component_scores"].get(key)

    layer1 = LayerOutcome(status="none", confidence=0.0, explanation="No provenance.", details={})
    layer3 = LayerOutcome(status="none", confidence=0.0, explanation="No votes.", details={})
    scoring = service._build_audio_system_scoring(layer1, audio_outcome, layer3)
    assert scoring["formula_components"]["N"] == audio_outcome.details["text_verification_score"]
    assert scoring["formula_components"]["audio_authenticity"] == audio_outcome.details["audio_analysis_score"]
    assert scoring["final_score"] <= 25.0


def test_musk_presidential_claim_detects_eligibility_contradiction(db_session, monkeypatch):
    service = VerificationService(db_session)
    text = (
        "Elon Musk announced he will run for president of the United States in 2028 after renouncing his South African citizenship. "
        "He reportedly plans to fund his entire campaign using Bitcoin from his personal wallet."
    )

    def fake_search(query, expected_domain=None, limit=5):
        return [
            WebSearchCandidate(
                title="U.S. presidential eligibility requires natural-born citizenship",
                url="https://www.usa.gov/requirements-for-presidential-candidates",
                snippet="The U.S. Constitution requires the president to be a natural-born citizen of the United States.",
                domain="www.usa.gov",
                score=0.95,
                explanation="mocked",
            ),
            WebSearchCandidate(
                title="Elon Musk biography",
                url="https://www.britannica.com/money/Elon-Musk",
                snippet="Elon Musk was born in Pretoria, South Africa, and later became a U.S. citizen.",
                domain="www.britannica.com",
                score=0.9,
                explanation="mocked",
            ),
            WebSearchCandidate(
                title="Musk discusses politics and Bitcoin",
                url="https://news.example.com/musk-politics-bitcoin",
                snippet="Elon Musk has commented on U.S. politics and cryptocurrency, including Bitcoin.",
                domain="news.example.com",
                score=0.84,
                explanation="mocked",
            ),
        ][:limit]

    monkeypatch.setattr(service.search, "search", fake_search)
    monkeypatch.setattr(
        service.retrieval,
        "fetch_page_document",
        lambda url: {
            "url": url,
            "fetch_status": "success",
            "content_type": "text/html",
            "title": "U.S. presidential eligibility and Elon Musk biography",
            "extracted_text": (
                "The U.S. Constitution requires the president to be a natural-born citizen of the United States. "
                "Elon Musk was born in Pretoria, South Africa, and later became a U.S. citizen. "
                "Renouncing South African citizenship does not change a person's birthplace or natural-born status."
            ),
            "extracted_text_length": 280,
            "extract_status": "extracted",
        },
    )

    payload = build_text_payload(text)
    job = service._create_job(JobInputType.text, text[:100])
    outcome = service._text_nlp_layer(job, payload, text)

    assert "Likely Verified" != outcome.details["text_verification_label"]
    assert "Verified" != outcome.details["text_verification_label"]
    assert outcome.details["text_verification_score"] <= 0.3
    assert any(result.get("claim_verdict") == "contradicted" for result in outcome.details["claim_results"])
    assert "natural-born" in outcome.explanation or "natural-born" in outcome.details.get("contradiction_summary", "")
    assert all(source.get("stance") != "supports" for source in outcome.details["evidence_sources"])


def test_claim_extraction_service_splits_atomic_claims():
    service = ClaimExtractionService()
    result = service.extract_claims(
        "President Ada Lovelace said the central bank cut rates on Monday. "
        "The government then announced a new emergency fuel subsidy."
    )

    assert result["checkable"] is True
    assert len(result["claims"]) >= 2


def test_reasoning_service_prefers_full_passage_support():
    service = ReasoningService()
    result = service.verify_claim_with_evidence(
        "8 children killed in domestic shootings across multiple homes in Louisiana, police say",
        [
            {
                "title": "Louisiana police say 8 children killed across multiple homes",
                "url": "https://example.com/report",
                "domain": "example.com",
                "passage": "Louisiana police said eight children were killed in domestic shootings across multiple homes.",
                "snippet": "",
                "score": 0.92,
            }
        ],
    )

    assert result["verdict"] == "supported"
    assert result["support_score"] >= 0.6


def test_vision_reasoning_compares_context_without_ocr_only(tmp_path):
    image_path = tmp_path / "scene.jpg"
    from PIL import Image

    Image.new("RGB", (1280, 720), color="white").save(image_path)
    service = VisionReasoningService()
    description = service.describe_image(str(image_path), context="Police at a Louisiana crime scene", ocr_text="")
    comparison = service.compare_visual_context(description, "Police at a Louisiana crime scene")

    assert "scene_description" in description
    assert comparison["verdict"] in {"supports_context", "mixed_context", "neutral"}


def test_build_text_system_scoring_uses_text_formula_and_not_review_score(db_session):
    service = VerificationService(db_session)
    extracted = build_text_payload("The health ministry confirmed a measles outbreak in the capital on Tuesday.")
    layer1 = LayerOutcome(status="platform-issued", confidence=0.82, explanation="Platform manifest valid.", details={"internal_manifest_valid": True, "manifest_id": "m1"})
    layer2 = LayerOutcome(
        status="supporting",
        confidence=0.74,
        explanation="The pasted text is supported by relevant evidence.",
        details={
            "claim_results": [
                {
                    "support_score": 0.9,
                    "source_reliability": 0.85,
                    "agreement_score": 0.8,
                    "directness_score": 0.75,
                    "clarity_score": 0.9,
                    "evidence_sources": [
                        {
                            "url": "https://www.reuters.com/world/measles-outbreak",
                            "domain": "www.reuters.com",
                            "source_type": "trusted_news_page",
                            "evidence_relation": "supports",
                            "alignment": "supports",
                        }
                    ],
                }
            ],
            "evidence_multiplier": 0.88,
            "penalty_multiplier": 1.0,
            "contradiction_summary": "",
        },
    )
    layer3 = LayerOutcome(status="reviewed_feedback", confidence=1.0, explanation="Feedback reviewed.", details={"mode": "feedback", "feedback_count": 3})

    scoring = service._build_text_system_scoring(layer1, layer2, layer3, extracted)

    assert "community" not in scoring["layer_scores"]
    assert scoring["display_layer_scores"]["community"] is None
    assert scoring["source_provenance_check"]["scored_as_truth"] is False
    assert scoring["final_label"] in {"Verified", "Likely Verified"}
    assert scoring["final_score"] >= 50


def test_layer3_uses_majority_vote_counts(db_session):
    job = VerificationJob(
        input_type=JobInputType.text,
        input_value="sample",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()
    db_session.add(ConsensusVote(job_id=job.id, vote_value=1, weight=1.0, confidence=1.0, moderation_flag=False, role=ConsensusRole.user))
    db_session.add(ConsensusVote(job_id=job.id, vote_value=1, weight=1.0, confidence=1.0, moderation_flag=False, role=ConsensusRole.user))
    db_session.add(ConsensusVote(job_id=job.id, vote_value=0, weight=1.0, confidence=1.0, moderation_flag=False, role=ConsensusRole.user))
    db_session.commit()

    outcome = VerificationService(db_session)._layer3(job, persist_audit=False)

    assert outcome.status == "supporting"
    assert outcome.details["true_votes"] == 2
    assert outcome.details["false_votes"] == 0
    assert outcome.details["not_sure_votes"] == 1
    assert outcome.details["majority_vote"] == "true"


def test_html_extraction_collects_article_metadata():
    html = """
    <html>
      <head>
        <title>Test Story</title>
        <meta property="og:title" content="Original Story Title" />
        <meta property="og:site_name" content="The Example Times" />
        <meta property="article:published_time" content="2026-03-17T10:30:00Z" />
        <meta name="keywords" content="politics, world" />
        <script type="application/ld+json">
          {
            "@context": "https://schema.org",
            "@type": "NewsArticle",
            "headline": "Original Story Title",
            "author": {"@type": "Person", "name": "Jane Reporter"},
            "publisher": {"@type": "Organization", "name": "The Example Times"},
            "datePublished": "2026-03-17T10:30:00Z",
            "keywords": ["politics", "world"]
          }
        </script>
      </head>
      <body>
        <article>
          <h1>Original Story Title</h1>
          <p>Body paragraph one.</p>
          <p>Body paragraph two.</p>
        </article>
      </body>
    </html>
    """

    payload = _build_payload_from_html("https://www.example.com/story", html)

    assert payload.title == "Original Story Title"
    assert payload.author == "Jane Reporter"
    assert payload.metadata["publisher_name"] == "The Example Times"
    assert payload.metadata["source_domain"] == "www.example.com"
    assert payload.metadata["keywords"] == ["politics", "world"]


def test_html_extraction_collects_article_media_metadata():
    html = """
    <html>
      <head>
        <meta property="og:image" content="https://cdn.example.com/hero.jpg" />
        <meta property="og:video" content="https://cdn.example.com/clip.mp4" />
        <meta name="c2pa.manifest" content="https://cdn.example.com/newsproof.manifest" />
        <link rel="c2pa-manifest" href="/manifests/article.c2pa" type="application/c2pa" />
      </head>
      <body>
        <article>
          <figure>
            <img src="/images/story.jpg" alt="Smoke over the skyline" width="1200" height="800" />
            <figcaption>Smoke rises above the skyline after the strike.</figcaption>
          </figure>
          <figure>
            <video poster="/images/video-poster.jpg" width="1280" height="720">
              <source src="/videos/field-report.mp4" type="video/mp4" />
            </video>
            <figcaption>Field footage from the scene.</figcaption>
          </figure>
        </article>
      </body>
    </html>
    """

    payload = _build_payload_from_html("https://www.example.com/story", html)

    assert payload.hero_image_url == "https://cdn.example.com/hero.jpg"
    media_assets = payload.metadata["media_assets"]
    assert len(media_assets) >= 4
    assert payload.metadata["c2pa_credentials"] == [
        {
            "source": "meta",
            "type": "manifest",
            "url": "https://cdn.example.com/newsproof.manifest",
            "description": "Content credentials manifest advertised in page metadata.",
        },
        {
            "source": "link",
            "type": "application/c2pa",
            "url": "https://www.example.com/manifests/article.c2pa",
            "description": "Content credentials manifest linked from the article page.",
        },
    ]
    assert any(asset["type"] == "image" and asset["url"] == "https://cdn.example.com/hero.jpg" for asset in media_assets)
    assert any(
        asset["type"] == "image"
        and asset["url"] == "https://www.example.com/images/story.jpg"
        and asset["alt"] == "Smoke over the skyline"
        and asset["caption"] == "Smoke rises above the skyline after the strike."
        for asset in media_assets
    )
    assert any(
        asset["type"] == "video"
        and asset["url"] == "https://www.example.com/videos/field-report.mp4"
        and asset["poster"] == "https://www.example.com/images/video-poster.jpg"
        and asset["mime_type"] == "video/mp4"
        for asset in media_assets
    )


def test_html_extraction_collects_related_article_links():
    html = """
    <html>
      <body>
        <main>
          <a href="/2026/03/20/world/example-related-story">Related Example Story About Global Tensions</a>
          <a href="https://www.example.com/politics/another-related-piece" title="Another Related Politics Piece">More</a>
        </main>
      </body>
    </html>
    """

    payload = _build_payload_from_html("https://www.example.com/world/source-story", html)

    related_links = payload.metadata["related_links"]
    assert any(link["url"] == "https://www.example.com/2026/03/20/world/example-related-story" for link in related_links)
    assert any(link["url"] == "https://www.example.com/politics/another-related-piece" for link in related_links)


def test_search_service_resolves_duckduckgo_redirect_urls():
    service = SearchService()
    redirected = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.nytimes.com%2F2026%2F03%2F20%2Fworld%2Fstory.html"

    assert service._resolve_result_url(redirected) == "https://www.nytimes.com/2026/03/20/world/story.html"


def test_remote_media_provenance_detectors_find_c2pa_signals():
    headers = {
        "Content-Type": "image/jpeg",
        "Link": '<https://cdn.example.com/manifests/story.c2pa>; rel="c2pa-manifest"',
        "Content-Credentials": "present",
    }
    sample = b"\x00\x01c2pa\x00jumb\x00manifest"

    assert "header:content-credentials" in _header_c2pa_indicators(headers)
    assert _header_manifest_url(headers, "https://cdn.example.com/story.jpg") == "https://cdn.example.com/manifests/story.c2pa"
    assert "bytes:c2pa" in _byte_c2pa_indicators(sample)


def test_remote_media_provenance_description_reports_checked_without_evidence():
    description = _describe_remote_provenance(
        {
            "type": "image",
            "signals": [],
            "manifest_url": None,
        }
    )

    assert "was checked for content credentials" in description
    assert "no cryptographic provenance evidence was detected" in description


def test_image_forensics_extracts_basic_dimensions_from_bytes():
    from io import BytesIO
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (16, 12), color="red").save(buffer, format="JPEG")
    forensic = extract_image_forensics_from_bytes(buffer.getvalue())

    assert forensic["media_kind"] == "image"
    assert forensic["width"] == 16
    assert forensic["height"] == 12
    assert forensic["format"] == "JPEG"


def test_media_authenticity_flags_ai_image_software():
    result = MediaAuthenticityService().analyze(
        ExtractedPayload(
            title="Image upload",
            author=None,
            publish_time=None,
            canonical_url=None,
            main_text="summary",
            hero_image_url=None,
            normalized_text="summary",
            content_hash="hash",
            metadata={
                "media_type": "image",
                "forensics": {
                    "media_kind": "image",
                    "fields": {"software": "MidJourney v6"},
                },
            },
        )
    )

    assert result["status"] == "high_deepfake_risk"


def test_media_authenticity_likely_authentic_for_camera_metadata():
    result = MediaAuthenticityService().analyze(
        ExtractedPayload(
            title="Image upload",
            author=None,
            publish_time=None,
            canonical_url=None,
            main_text="summary",
            hero_image_url=None,
            normalized_text="summary",
            content_hash="hash",
            metadata={
                "media_type": "image",
                "forensics": {
                    "media_kind": "image",
                    "fields": {"device_model": "Canon EOS R5", "captured_at": "2026:03:22 10:00:00"},
                },
            },
        )
    )

    assert result["status"] == "likely_authentic"


def test_media_authenticity_reports_not_applicable_for_url_content():
    result = MediaAuthenticityService().analyze(build_text_payload("This is just text content for a URL-like workflow."))

    assert result["status"] == "not_applicable"


def test_remote_media_forensics_are_attached_to_layer1_details(db_session):
    payload = ExtractedPayload(
        title="Remote media forensic check",
        author=None,
        publish_time=None,
        canonical_url="https://example.com/story",
        main_text="Body",
        hero_image_url=None,
        normalized_text="body",
        content_hash="forensic-check",
        metadata={
            "source_domain": "example.com",
            "remote_media_provenance": [
                {
                    "url": "https://cdn.example.com/story.jpg",
                    "type": "image",
                    "forensics": {
                        "media_kind": "image",
                        "metadata_status": "present",
                        "width": 1200,
                        "height": 800,
                    },
                }
            ],
            "forensics": {
                "media_kind": "audio",
                "metadata_status": "limited",
                "format_name": "mp3",
            },
        },
    )

    outcome = VerificationService(db_session)._attach_extracted_details(
        LayerOutcome(status="no-proof", confidence=0.0, explanation="none", details={}),
        payload,
    )

    assert outcome.details["forensics"]["media_kind"] == "audio"
    assert outcome.details["remote_media_provenance"][0]["forensics"]["width"] == 1200


def test_c2pa_extractors_ignore_generic_web_app_manifests():
    html = """
    <html>
      <head>
        <link rel="manifest" href="/themes/mit/assets/img/favicon/manifest.webmanifest" type="application/manifest+json" />
      </head>
      <body></body>
    </html>
    """

    payload = _build_payload_from_html("https://news.mit.edu/story", html)

    assert payload.metadata["c2pa_credentials"] == []
    assert (
        _header_manifest_url(
            {"Link": '<https://news.mit.edu/themes/mit/assets/img/favicon/manifest.webmanifest>; rel="manifest"'},
            "https://news.mit.edu/story",
        )
        is None
    )


def test_search_service_builds_query_variants():
    service = SearchService()

    variants = service._query_variants("CNBC Trump Japan trade politics story", "cnbc.com")

    assert variants[0] == "CNBC Trump Japan trade politics story"
    assert any(variant.startswith("site:cnbc.com ") for variant in variants)


def test_layer2_uses_same_source_related_links_as_matches(db_session):
    service = VerificationService(db_session)
    service.search.search = lambda query, expected_domain=None, limit=5: []
    job = VerificationJob(
        input_type=JobInputType.url,
        input_value="https://www.cnbc.com/2026/03/20/politics/source-story",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()

    payload = ExtractedPayload(
        title="Trump weighs new trade steps with Japan",
        author="@CNBC",
        publish_time=datetime.fromisoformat("2026-03-20T01:03:20"),
        canonical_url="https://www.cnbc.com/2026/03/20/politics/source-story",
        main_text="CNBC article body text about trade and Japan policy.",
        hero_image_url=None,
        normalized_text="cnbc article body text about trade and japan policy",
        content_hash="cnbc-related-links-test",
        metadata={
            "publisher_name": "CNBC",
            "source_domain": "www.cnbc.com",
            "related_links": [
                {
                    "title": "Trump weighs new trade steps with Japan as tensions grow",
                    "url": "https://www.cnbc.com/2026/03/20/politics/trump-japan-trade-steps.html",
                    "domain": "www.cnbc.com",
                    "source": "related_link",
                }
            ],
        },
    )

    outcome = service._layer2(job, payload)

    assert outcome.status == "supporting"
    assert outcome.details["matches"]
    assert outcome.details["matches"][0]["source"] == "related_link"


def test_layer2_still_runs_cross_web_search_when_related_links_exist(db_session):
    service = VerificationService(db_session)
    calls: list[tuple[str, str | None, int]] = []

    def fake_search(query: str, expected_domain: str | None = None, limit: int = 10):
        calls.append((query, expected_domain, limit))
        return [
            WebSearchCandidate(
                title="Independent fact check",
                url="https://independent.example.org/fact-check",
                snippet="Independent reporting that matches the same core event.",
                domain="independent.example.org",
                score=0.76,
                explanation="Found through cross-web retrieval outside the original publisher domain.",
            )
        ]

    service.search.search = fake_search
    job = VerificationJob(
        input_type=JobInputType.url,
        input_value="https://www.cnbc.com/2026/03/20/politics/source-story",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()

    payload = ExtractedPayload(
        title="Trump weighs new trade steps with Japan",
        author="@CNBC",
        publish_time=datetime.fromisoformat("2026-03-20T01:03:20"),
        canonical_url="https://www.cnbc.com/2026/03/20/politics/source-story",
        main_text="CNBC article body text about trade and Japan policy.",
        hero_image_url=None,
        normalized_text="cnbc article body text about trade and japan policy",
        content_hash="cnbc-cross-web-search-test",
        metadata={
            "publisher_name": "CNBC",
            "source_domain": "www.cnbc.com",
            "related_links": [
                {
                    "title": "Trump weighs new trade steps with Japan as tensions grow",
                    "url": "https://www.cnbc.com/2026/03/20/politics/trump-japan-trade-steps.html",
                    "domain": "www.cnbc.com",
                    "source": "related_link",
                }
            ],
        },
    )

    outcome = service._layer2(job, payload)

    assert calls
    assert outcome.details["matches"] == []
    assert not any(match.get("source") == "web_search" for match in outcome.details["matches"])
    assert outcome.details["similar_articles_found_count"] == 0
    assert outcome.details["stopped_search_reason"] == "search_exhausted"


def test_layer2_does_not_force_weak_many_matches_for_url(db_session):
    service = VerificationService(db_session)
    service.search.search = lambda query, expected_domain=None, limit=5: [
        WebSearchCandidate(
            title=f"Cross-web corroboration {index}",
            url=f"https://example{index}.org/story",
            snippet="Matched reporting from a separate source.",
            domain=f"example{index}.org",
            score=round(0.95 - (index * 0.01), 3),
            explanation="Found through broad web retrieval.",
        )
        for index in range(15)
    ]
    job = VerificationJob(
        input_type=JobInputType.url,
        input_value="https://www.cnbc.com/2026/03/20/politics/source-story",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()

    payload = ExtractedPayload(
        title="Trump weighs new trade steps with Japan",
        author="@CNBC",
        publish_time=datetime.fromisoformat("2026-03-20T01:03:20"),
        canonical_url="https://www.cnbc.com/2026/03/20/politics/source-story",
        main_text="CNBC article body text about trade and Japan policy.",
        hero_image_url=None,
        normalized_text="cnbc article body text about trade and japan policy",
        content_hash="cnbc-many-matches-test",
        metadata={
            "publisher_name": "CNBC",
            "source_domain": "www.cnbc.com",
        },
    )

    outcome = service._layer2(job, payload)
    db_session.flush()

    assert outcome.details["matches"] == []
    assert outcome.details["has_more_matches"] is False
    assert outcome.details["total_match_count"] == 0
    assert outcome.details["similar_articles_found_count"] == 0
    assert outcome.details["stopped_search_reason"] == "search_exhausted"


def test_url_article_claim_verification_counts_max_five_same_context_sources(db_session, monkeypatch):
    service = VerificationService(db_session)
    matches = [
        {
            "title": f"Independent confirmation {index}",
            "url": f"https://publisher{index}.example/story",
            "domain": f"publisher{index}.example",
            "source": "web_search",
            "score": 0.9 - index * 0.01,
            "evidence_relation": "supports",
            "relation": "exact_support",
            "aboutness": "exact_claim",
            "stance": "supports",
        }
        for index in range(7)
    ]
    matches.extend(
        [
            {
                "title": "Related but weak context",
                "url": "https://weak.example/story",
                "domain": "weak.example",
                "source": "web_search",
                "score": 0.7,
                "evidence_relation": "partially_related",
                "aboutness": "partial_context",
                "stance": "neutral",
            },
            {
                "title": "Same publisher context",
                "url": "https://www.cnbc.com/2026/03/20/politics/related",
                "domain": "www.cnbc.com",
                "source": "web_search",
                "score": 0.88,
                "evidence_relation": "supports",
                "relation": "exact_support",
                "aboutness": "exact_claim",
                "stance": "supports",
            },
        ]
    )

    debug = service._url_same_context_debug(matches, "www.cnbc.com")

    assert debug["max_similar_articles_allowed"] == 5
    assert debug["similar_articles_found_count"] == 5
    assert len(debug["counted_same_context_articles"]) == 5
    assert debug["stopped_search_reason"] == "found_5_sources"
    assert all(match["domain"] != "www.cnbc.com" for match in debug["counted_same_context_articles"])
    assert any(item["classification"] == "partially_related" for item in debug["rejected_related_articles"])
    assert any(item["same_publisher"] is True for item in debug["rejected_related_articles"])


def test_url_article_claim_verification_does_not_count_partial_sources(db_session, monkeypatch):
    service = VerificationService(db_session)
    partial_matches = [
        {
            "title": "Related policy background",
            "url": "https://related.example/story",
            "domain": "related.example",
            "source": "web_search",
            "score": 0.77,
            "evidence_relation": "partially_related",
            "aboutness": "partial_context",
            "stance": "neutral",
        }
    ]
    debug = service._url_same_context_debug(partial_matches, "www.cnbc.com")

    assert debug["similar_articles_found_count"] == 0
    assert debug["counted_same_context_articles"] == []
    assert debug["stopped_search_reason"] == "search_exhausted"
    assert debug["rejected_related_articles"][0]["reason"]


def test_url_context_search_verifies_paxton_cornyn_from_cross_publishers(db_session, monkeypatch):
    service = VerificationService(db_session)
    source_url = "https://www.nbcnews.com/politics/2026-primary-elections/texas-senate-runoff-results"
    search_results = [
        WebSearchCandidate(
            title="Ken Paxton wins Republican Senate runoff against John Cornyn",
            url="https://apnews.com/article/paxton-cornyn-texas-runoff",
            snippet="Ken Paxton wins Republican Senate runoff against John Cornyn.",
            domain="apnews.com",
            score=0.96,
            explanation="AP exact same context",
        ),
        WebSearchCandidate(
            title="Paxton defeats Cornyn in Texas GOP Senate nomination race",
            url="https://www.reuters.com/world/us/paxton-cornyn-texas-senate-runoff",
            snippet="Paxton defeats Cornyn in Texas GOP Senate nomination race.",
            domain="www.reuters.com",
            score=0.94,
            explanation="Reuters exact same context",
        ),
        WebSearchCandidate(
            title="Paxton beats Cornyn in Republican runoff",
            url="https://www.texastribune.org/2026/05/31/paxton-cornyn-runoff/",
            snippet="Paxton beats Cornyn in Republican runoff.",
            domain="www.texastribune.org",
            score=0.9,
            explanation="Texas Tribune exact same context",
        ),
    ]
    monkeypatch.setattr(
        service,
        "_search_bundle",
        lambda query, expected_domain=None, limit=5: {"results": search_results[:limit], "failure_reason": None, "provider_status": {}, "provider_attempts": []},
    )
    monkeypatch.setattr(
        service.retrieval,
        "fetch_page_document",
        lambda url: {
            "fetch_status": "success",
            "content_type": "text/html",
            "title": next((item.title for item in search_results if item.url == url), ""),
            "extracted_text": next((item.snippet for item in search_results if item.url == url), ""),
            "extracted_text_length": 120,
        },
    )
    monkeypatch.setattr(service.cohere, "rank_candidates", lambda query, docs: [])
    job = VerificationJob(
        input_type=JobInputType.url,
        input_value=source_url,
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()
    payload = ExtractedPayload(
        title="Ken Paxton defeated John Cornyn in the Texas Republican U.S. Senate runoff",
        author="Reporter",
        publish_time=datetime.fromisoformat("2026-05-31T01:03:20"),
        canonical_url=source_url,
        main_text="Ken Paxton defeated John Cornyn in the Texas Republican U.S. Senate runoff.",
        hero_image_url=None,
        normalized_text="ken paxton defeated john cornyn in the texas republican u s senate runoff",
        content_hash="url-paxton-cornyn-context-test",
        metadata={"publisher_name": "NBC News", "source_domain": "www.nbcnews.com"},
    )

    outcome = service._layer2_article_url(job, payload, None, [], persist_matches=False)

    assert outcome.details["url_context_debug"]["claim_tuple"]["subject"] == "Ken Paxton"
    assert outcome.details["url_context_debug"]["search_api_called"] is True
    assert outcome.details["same_context_articles_found_count"] >= 2
    assert outcome.details["url_context_debug"]["nlp_context_score"] >= 0.8
    assert outcome.details["final_label"] in {"Likely Verified", "Verified"}


def test_search_service_prefers_cross_publisher_results_when_scores_tie():
    service = SearchService()
    candidates = [
        WebSearchCandidate(
            title="Same publisher 1",
            url="https://www.cnn.com/story-a",
            snippet="",
            domain="www.cnn.com",
            score=0.8,
            explanation="same publisher",
        ),
        WebSearchCandidate(
            title="Same publisher 2",
            url="https://edition.cnn.com/story-b",
            snippet="",
            domain="edition.cnn.com",
            score=0.79,
            explanation="same publisher",
        ),
        WebSearchCandidate(
            title="Independent source",
            url="https://www.reuters.com/world/story",
            snippet="",
            domain="www.reuters.com",
            score=0.8,
            explanation="independent",
        ),
    ]

    results = service._diversify_results(candidates, expected_domain="cnn.com", limit=2)

    assert len(results) == 2
    assert results[0].domain == "www.reuters.com"


def test_factual_consistency_is_capped_for_same_publisher_only_matches(db_session):
    service = VerificationService(db_session)
    payload = ExtractedPayload(
        title="Example story",
        author="Reporter",
        publish_time=datetime.fromisoformat("2026-04-19T10:30:00"),
        canonical_url="https://www.cnn.com/2026/04/19/world/example-story",
        main_text=" ".join(["Detailed article body text"] * 60),
        hero_image_url=None,
        normalized_text=" ".join(["detailed article body text"] * 60),
        content_hash="same-publisher-only",
        metadata={"fetched_url": "https://www.cnn.com/2026/04/19/world/example-story"},
    )
    layer2 = LayerOutcome(
        status="supporting",
        confidence=0.82,
        explanation="Matches found.",
        details={
            "matches": [
                {"url": "https://www.cnn.com/related-a", "domain": "www.cnn.com", "score": 0.91},
                {"url": "https://edition.cnn.com/related-b", "domain": "edition.cnn.com", "score": 0.87},
            ]
        },
    )

    score = service._factual_consistency_score(layer2, payload)

    assert score <= 0.5


def test_nlp_score_inputs_reward_cross_publisher_corroboration_more_than_same_publisher(db_session):
    service = VerificationService(db_session)
    payload = ExtractedPayload(
        title="Example story",
        author="Reporter",
        publish_time=datetime.fromisoformat("2026-04-19T10:30:00"),
        canonical_url="https://www.cnn.com/2026/04/19/world/example-story",
        main_text=" ".join(["Detailed article body text"] * 60),
        hero_image_url=None,
        normalized_text=" ".join(["detailed article body text"] * 60),
        content_hash="cross-publisher-better",
        metadata={"fetched_url": "https://www.cnn.com/2026/04/19/world/example-story"},
    )
    same_publisher = LayerOutcome(
        status="supporting",
        confidence=0.82,
        explanation="Matches found.",
        details={
            "matches": [
                {"url": "https://www.cnn.com/related-a", "domain": "www.cnn.com", "score": 0.91},
                {"url": "https://edition.cnn.com/related-b", "domain": "edition.cnn.com", "score": 0.87},
            ]
        },
    )
    cross_publisher = LayerOutcome(
        status="supporting",
        confidence=0.82,
        explanation="Matches found.",
        details={
            "matches": [
                {"url": "https://www.reuters.com/world/example", "domain": "www.reuters.com", "score": 0.81},
                {"url": "https://apnews.com/article/example", "domain": "apnews.com", "score": 0.79},
                {"url": "https://www.bbc.com/news/example", "domain": "www.bbc.com", "score": 0.77},
            ]
        },
    )

    same_inputs = service._nlp_score_inputs(same_publisher, payload)
    cross_inputs = service._nlp_score_inputs(cross_publisher, payload)

    assert cross_inputs["factual_consistency"] > same_inputs["factual_consistency"]
    assert cross_inputs["cross_source_agreement"] > same_inputs["cross_source_agreement"]


def test_layer2_separates_same_publisher_context_from_cross_publisher_corroboration(db_session, monkeypatch):
    service = VerificationService(db_session)

    def fake_search(query: str, expected_domain=None, limit=5):
        return [
            WebSearchCandidate(
                title="Reuters: Paxton defeats Cornyn in Texas Senate runoff",
                url="https://www.reuters.com/world/example",
                snippet="Ken Paxton defeated John Cornyn in the Texas Republican U.S. Senate runoff.",
                domain="www.reuters.com",
                score=0.82,
                explanation="cross publisher",
            ),
            WebSearchCandidate(
                title="CNN related Paxton Cornyn context",
                url="https://www.cnn.com/2026/04/19/world/example-context",
                snippet="Ken Paxton and John Cornyn campaigned in the Texas Republican U.S. Senate runoff.",
                domain="www.cnn.com",
                score=0.79,
                explanation="same publisher",
            ),
        ]

    service.search.search = fake_search
    monkeypatch.setattr(
        service.retrieval,
        "fetch_page_document",
        lambda url: {
            "fetch_status": "success",
            "content_type": "text/html",
            "title": "Reuters: Paxton defeats Cornyn in Texas Senate runoff"
            if "reuters" in url
            else "CNN related Paxton Cornyn context",
            "extracted_text": "Ken Paxton defeated John Cornyn in the Texas Republican U.S. Senate runoff."
            if "reuters" in url
            else "Ken Paxton and John Cornyn campaigned in the Texas Republican U.S. Senate runoff.",
            "extracted_text_length": 140,
        },
    )
    job = VerificationJob(
        input_type=JobInputType.url,
        input_value="https://www.cnn.com/2026/04/19/world/example-story",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()

    payload = ExtractedPayload(
        title="Ken Paxton defeated John Cornyn in the Texas Republican U.S. Senate runoff",
        author="Reporter",
        publish_time=datetime.fromisoformat("2026-04-19T10:30:00"),
        canonical_url="https://www.cnn.com/2026/04/19/world/example-story",
        main_text="Ken Paxton defeated John Cornyn in the Texas Republican U.S. Senate runoff.",
        hero_image_url=None,
        normalized_text="ken paxton defeated john cornyn in the texas republican u s senate runoff",
        content_hash="cross-publisher-split",
        metadata={
            "fetched_url": "https://www.cnn.com/2026/04/19/world/example-story",
            "related_links": [
                {
                    "title": "CNN related Paxton Cornyn context",
                    "url": "https://www.cnn.com/2026/04/19/world/example-context",
                    "domain": "www.cnn.com",
                    "source": "related_link",
                }
            ],
        },
    )

    outcome = service._layer2(job, payload)

    assert outcome.details["same_publisher_count"] >= 1
    assert outcome.details["cross_publisher_count"] >= 1
    assert outcome.details["cross_publisher_agreement_exists"] is True
    assert "www.reuters.com" in outcome.details["support_domains"]


def test_layer2_same_publisher_only_is_context_not_full_corroboration(db_session):
    service = VerificationService(db_session)
    service.search.search = lambda query, expected_domain=None, limit=5: [
        WebSearchCandidate(
            title="CNN context",
            url="https://www.cnn.com/2026/04/19/world/example-context",
            snippet="Same publisher follow-up.",
            domain="www.cnn.com",
            score=0.88,
            explanation="same publisher only",
        )
    ]
    job = VerificationJob(
        input_type=JobInputType.url,
        input_value="https://www.cnn.com/2026/04/19/world/example-story",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()

    payload = ExtractedPayload(
        title="Example story headline",
        author="Reporter",
        publish_time=datetime.fromisoformat("2026-04-19T10:30:00"),
        canonical_url="https://www.cnn.com/2026/04/19/world/example-story",
        main_text="Example article body discussing the same event in detail.",
        hero_image_url=None,
        normalized_text="example article body discussing the same event in detail",
        content_hash="same-publisher-context-only",
        metadata={"fetched_url": "https://www.cnn.com/2026/04/19/world/example-story"},
    )

    outcome = service._layer2(job, payload)

    assert outcome.confidence <= 0.54
    assert outcome.details["corroboration_scope"] == "same_source_only"
    assert "same publisher domain" in outcome.explanation.lower()


def test_source_adapter_youtube_marks_transcript_when_present():
    payload = ExtractedPayload(
        title="Original title",
        author=None,
        publish_time=None,
        canonical_url="https://www.youtube.com/watch?v=abc123xyz89",
        main_text="Original body",
        hero_image_url=None,
        normalized_text="original body",
        content_hash="hash",
        metadata={
            "publisher_name": "Fallback Channel",
            "transcript_text": "Transcript line one. Transcript line two.",
        },
    )

    adapted = asyncio.run(SourceAdapterService().adapt_url_payload("https://www.youtube.com/watch?v=abc123xyz89", payload))

    assert adapted.metadata["youtube_video_id"] == "abc123xyz89"
    assert adapted.metadata["transcript_status"] == "available"
    assert adapted.metadata["transcript_length"] > 0


def test_layer2_facebook_debug_includes_caption_and_ocr_queries(db_session):
    service = VerificationService(db_session)

    service._direct_source_recovery = lambda prepared, extraction, payload: {
        "status": "not_attempted",
        "overlap_scores": [],
        "final_source_recovery": {"status": "not_found", "reason": "not attempted"},
    }
    service._gather_text_candidate_documents = lambda prepared, extraction, query_plan, source_recovery_stage: (
        [{"url": "https://www.reuters.com/world/example", "title": "Reuters corroboration", "domain": "www.reuters.com"}],
        {
            "lexical_searches": [{"provider_status": {"brave": "configured"}, "result_count": 1, "results": [{"url": "https://www.reuters.com/world/example", "title": "Reuters corroboration"}]}],
            "semantic_searches": [],
            "candidate_fetches": [{"url": "https://www.reuters.com/world/example", "fetch_status": "success"}],
        },
    )
    service._hybrid_rank_text_candidates = lambda prepared, claims, query_plan, candidates: {"reranked_top_results": candidates}
    service._build_claim_evidence_from_top_articles = lambda claim, top_articles: top_articles
    service._score_claim_against_evidence = lambda claim, evidence: {
        "claim_text": claim,
        "claim_score": 0.76,
        "claim_label": "Likely Verified",
        "support_score": 0.8,
        "directness_score": 0.7,
        "clarity_score": 0.8,
        "evidence_summary": "Caption and OCR both aligned with external reporting.",
        "evidence_sources": evidence,
    }
    service._finalize_text_verification = lambda claim_results, top_articles, meaning_bundle: {
        "final_score": 0.76,
        "final_label": "Likely Verified",
        "explanation": "Caption text matched related reporting and OCR text was searched separately.",
        "evidence_summary": "External support was found.",
        "supporting_evidence_summary": "External support was found.",
        "contradiction_summary": "",
        "evidence_sources": [
            {
                "title": "Reuters corroboration",
                "url": "https://www.reuters.com/world/example",
                "domain": "www.reuters.com",
                "score": 0.81,
                "source": "web_search",
            }
        ],
        "evidence_multiplier": 0.8,
        "penalty_multiplier": 1.0,
        "component_scores": {"S": 0.8, "R": 0.7, "A": 0.7, "D": 0.8, "C": 0.8, "E": 0.8, "P": 1.0},
    }

    job = VerificationJob(
        input_type=JobInputType.url,
        input_value="https://www.facebook.com/example/posts/123",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()

    payload = ExtractedPayload(
        title="Facebook post about a developing event",
        author=None,
        publish_time=None,
        canonical_url="https://www.facebook.com/example/posts/123",
        main_text="",
        hero_image_url="https://cdn.example.com/post.jpg",
        normalized_text="",
        content_hash="fb-query-debug",
        metadata={
            "fetched_url": "https://www.facebook.com/example/posts/123",
            "platform_type": "facebook",
            "source_type": "social_post",
            "description": "Facebook caption describing the Louisiana shootings.",
            "social_image_ocr_result": {"ocr_text": "8 children killed in Louisiana", "text_detected": True},
            "social_image_ocr_results": [{"image_url": "https://cdn.example.com/post.jpg", "ocr_text": "8 children killed in Louisiana", "ocr_status": "completed"}],
            "social_image_ocr_texts": ["8 children killed in Louisiana"],
            "media_assets": [{"type": "image", "url": "https://cdn.example.com/post.jpg"}],
        },
    )

    outcome = service._layer2(job, payload)

    assert outcome.details["facebook_caption_text"] == "Facebook caption describing the Louisiana shootings."
    assert outcome.details["facebook_ocr_queries"]
    assert outcome.details["facebook_combined_queries"]
    assert outcome.details["cross_publisher_count"] == 1


def test_layer2_facebook_recomputes_fresh_caption_and_ocr_scores(db_session):
    service = VerificationService(db_session)

    def fake_stream(_extracted, input_text, stream_name):
        if stream_name == "caption":
            score = 0.32
            match = {"url": "https://example.com/caption", "title": "Caption source", "domain": "example.com", "score": 0.32}
        elif stream_name == "ocr":
            score = 0.68
            match = {"url": "https://reuters.com/ocr", "title": "OCR source", "domain": "www.reuters.com", "score": 0.68}
        else:
            score = 0.54
            match = {"url": "https://apnews.com/combined", "title": "Combined source", "domain": "apnews.com", "score": 0.54}
        return {
            "name": stream_name,
            "prepared": {"cleaned_input": input_text.lower()},
            "query_plan": {"lexical": [f"{stream_name}-q"], "semantic": []},
            "aggregate": {
                "final_score": score,
                "final_label": "Likely Verified",
                "component_scores": {"S": score, "R": 0.6, "A": 0.5, "D": 0.7, "C": 0.8},
                "evidence_multiplier": 0.8,
                "penalty_multiplier": 1.0,
            },
            "matches": [match],
            "debug": {"raw_input": input_text},
        }

    service._evaluate_facebook_text_stream = fake_stream
    service._extract_remote_social_image_ocr = lambda _payload: {"ocr_text": "OCR evidence text", "text_detected": True, "ocr_status": "completed"}

    job = VerificationJob(
        input_type=JobInputType.url,
        input_value="https://www.facebook.com/example/posts/abc",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()

    payload = ExtractedPayload(
        title="Example Facebook post",
        author=None,
        publish_time=None,
        canonical_url="https://www.facebook.com/example/posts/abc",
        main_text="",
        hero_image_url="https://cdn.example.com/fb.jpg",
        normalized_text="",
        content_hash="facebook-fresh-scores",
        metadata={
            "fetched_url": "https://www.facebook.com/example/posts/abc",
            "platform_type": "facebook",
            "source_type": "social_post",
            "description": "Caption evidence text",
            "social_image_ocr_results": [{"image_url": "https://cdn.example.com/fb.jpg", "ocr_text": "OCR evidence text", "ocr_status": "completed", "text_detected": True}],
            "social_image_ocr_texts": ["OCR evidence text"],
            "media_assets": [{"type": "image", "url": "https://cdn.example.com/fb.jpg"}],
        },
    )

    outcome = service._layer2(job, payload, persist_matches=False)

    assert outcome.details["candidate_sources_from_caption"] == ["https://example.com/caption"]
    assert outcome.details["candidate_sources_from_ocr"] == ["https://reuters.com/ocr"]
    assert outcome.details["cross_source_count"] >= 2
    assert outcome.details["nlp_component_values"]["factual_consistency"] != 0.54
    assert outcome.details["final_score"] > 0


def test_crypto_score_inputs_cap_internal_facebook_provenance(db_session):
    service = VerificationService(db_session)
    layer1 = LayerOutcome(
        status="platform-issued",
        confidence=0.8,
        explanation="Internal platform manifest exists.",
        details={
            "internal_manifest_valid": True,
            "platform_manifest_valid": True,
            "canonical_hash": "abc",
            "current_version": 2,
            "signed_at": "2026-04-28T11:00:00Z",
        },
    )
    extracted = ExtractedPayload(
        title="Facebook post",
        author=None,
        publish_time=None,
        canonical_url="https://www.facebook.com/example/posts/123",
        main_text="Some post text",
        hero_image_url=None,
        normalized_text="some post text",
        content_hash="fb-crypto-cap",
        metadata={
            "platform_type": "facebook",
            "source_type": "social_post",
            "source_domain": "www.facebook.com",
        },
    )

    inputs = service._crypto_score_inputs(layer1, extracted)

    assert inputs["signature_validity"] <= 0.25
    assert inputs["issuer_trust"] <= 0.25
    assert inputs["penalty"] <= 0.35


def test_related_link_matches_filter_generic_section_links(db_session):
    service = VerificationService(db_session)
    payload = ExtractedPayload(
        title="Trump weighs new steps in Middle East policy",
        author="Reporter",
        publish_time=datetime.fromisoformat("2026-04-19T10:30:00"),
        canonical_url="https://www.cnn.com/2026/04/19/world/trump-middle-east-policy/index.html",
        main_text="Body",
        hero_image_url=None,
        normalized_text="body",
        content_hash="filter-generic-related-links",
        metadata={
            "fetched_url": "https://www.cnn.com/2026/04/19/world/trump-middle-east-policy/index.html",
            "related_links": [
                {"title": "Americas", "url": "https://www.cnn.com/world/americas", "domain": "www.cnn.com", "source": "related_link"},
                {"title": "Australia", "url": "https://www.cnn.com/world/australia", "domain": "www.cnn.com", "source": "related_link"},
                {
                    "title": "Trump weighs new steps in Middle East policy as tensions rise",
                    "url": "https://www.cnn.com/2026/04/18/world/trump-middle-east-policy-tensions/index.html",
                    "domain": "www.cnn.com",
                    "source": "related_link",
                },
            ],
        },
    )

    matches = service._related_link_matches(payload)

    assert len(matches) == 1
    assert "tensions" in matches[0]["url"]


def test_build_fact_check_summary_and_keyword_queries(db_session):
    service = VerificationService(db_session)
    payload = ExtractedPayload(
        title="Iran says it will review Strait of Hormuz policy",
        author="Reporter",
        publish_time=datetime.fromisoformat("2026-04-19T10:30:00"),
        canonical_url="https://example.com/world/iran-strait-of-hormuz-policy-review",
        main_text=(
            "Iran said officials would review policy around the Strait of Hormuz after new regional tensions. "
            "Shipping markets and regional governments responded quickly to the announcement."
        ),
        hero_image_url=None,
        normalized_text="iran said officials would review policy around the strait of hormuz after new regional tensions shipping markets and regional governments responded quickly to the announcement",
        content_hash="summary-query-test",
        metadata={
            "summary": "Officials are reviewing policy around the Strait of Hormuz after rising tensions.",
            "keywords": ["Iran", "Strait of Hormuz", "shipping markets"],
            "section": "World",
        },
    )

    summary_query = service._build_fact_check_summary_query(payload)
    keyword_query = service._build_fact_check_keyword_query(payload)

    assert "Iran says it will review Strait of Hormuz policy" in summary_query
    assert "Strait of Hormuz" in keyword_query
    assert "shipping markets" in keyword_query


def test_combine_fact_check_matches_prefers_web_articles_over_related_links(db_session):
    service = VerificationService(db_session)
    web_matches = [
        WebSearchCandidate(
            title="Reuters corroboration",
            url="https://www.reuters.com/world/example",
            snippet="Reuters reports the same event.",
            domain="www.reuters.com",
            score=0.81,
            explanation="web",
        )
    ]
    related_links = [
        {
            "title": "Middle East",
            "url": "https://www.cnn.com/world/middle-east",
            "score": 0.08,
            "explanation": "related",
            "source": "related_link",
            "domain": "www.cnn.com",
            "snippet": None,
        }
    ]

    combined = service._combine_fact_check_matches([], web_matches, related_links)

    assert len(combined) == 1
    assert combined[0]["source"] == "web_search"
    assert "reuters" in combined[0]["url"]


def test_nlp_score_inputs_promote_content_match_found_into_high_confidence_range(db_session):
    service = VerificationService(db_session)
    payload = ExtractedPayload(
        title="Iran says it will review Strait of Hormuz policy",
        author="Reporter",
        publish_time=datetime.fromisoformat("2026-04-19T10:30:00"),
        canonical_url="https://example.com/world/iran-strait-of-hormuz-policy-review",
        main_text=(
            "Iran said officials would review policy around the Strait of Hormuz after new regional tensions. "
            "Shipping markets and regional governments responded quickly to the announcement."
        ),
        hero_image_url=None,
        normalized_text="iran said officials would review policy around the strait of hormuz after new regional tensions shipping markets and regional governments responded quickly to the announcement",
        content_hash="high-confidence-content-match",
        metadata={"fetched_url": "https://example.com/world/iran-strait-of-hormuz-policy-review"},
    )
    layer2 = LayerOutcome(
        status="supporting",
        confidence=0.62,
        explanation="Fact-checking found meaningful similarity with corroborating sources.",
        details={
            "fact_check_verdict": "content match found",
            "trusted_publisher_match": "Reuters",
            "matches": [
                {"url": "https://www.reuters.com/world/example", "domain": "www.reuters.com", "score": 0.81},
                {"url": "https://apnews.com/article/example", "domain": "apnews.com", "score": 0.79},
            ],
        },
    )

    score_inputs = service._nlp_score_inputs(layer2, payload)
    score = compute_nlp_score(score_inputs)

    assert score["score"] >= 0.9


def test_text_system_scoring_excludes_review_layer_from_verification_score(db_session):
    service = VerificationService(db_session)
    extracted = build_text_payload(
        "President Donald Trump this month threatened to strike Iranian infrastructure in ways that might well be war crimes."
    )
    layer1 = LayerOutcome(status="platform-issued", confidence=0.84, explanation="Platform provenance valid.", details={"internal_manifest_valid": True, "canonical_hash": "abc", "manifest_id": "m1", "signed_at": "2026-04-19T00:00:00Z"})
    layer2 = LayerOutcome(
        status="supporting",
        confidence=0.62,
        explanation="Content match found.",
        details={
            "claim_results": [
                {
                    "support_score": 0.92,
                    "source_reliability": 0.88,
                    "agreement_score": 0.82,
                    "directness_score": 0.86,
                    "clarity_score": 0.9,
                    "claim_verdict": "supported",
                    "evidence_sources": [
                        {
                            "url": "https://www.reuters.com/world/example",
                            "domain": "www.reuters.com",
                            "source_type": "trusted_news_page",
                            "evidence_relation": "supports",
                            "alignment": "supports",
                        }
                    ],
                }
            ],
            "evidence_multiplier": 0.9,
            "penalty_multiplier": 1.0,
        },
    )
    layer3 = LayerOutcome(
        status="review_attention",
        confidence=0.0,
        explanation="Feedback exists.",
        details={"mode": "feedback", "feedback_count": 2, "dispute_count": 1},
    )

    scoring = service._build_text_system_scoring(layer1, layer2, layer3, extracted)

    assert "community" not in scoring["layer_scores"]
    assert scoring["display_layer_scores"]["community"] is None
    assert scoring["source_provenance_check"]["scored_as_truth"] is False
    assert scoring["final_score"] > 70


def test_url_system_scoring_uses_url_hybrid_components(db_session):
    service = VerificationService(db_session)
    extracted = ExtractedPayload(
        title="Example URL story",
        author="Reporter",
        publish_time=datetime.fromisoformat("2026-04-19T10:30:00"),
        canonical_url="https://example.com/news/story",
        main_text="A URL article body with enough content for normal URL verification.",
        hero_image_url=None,
        normalized_text="a url article body with enough content for normal url verification",
        content_hash="url-hybrid-safety",
        metadata={"fetched_url": "https://example.com/news/story", "source_domain": "example.com"},
    )
    layer1 = LayerOutcome(status="matched", confidence=0.95, explanation="Signed proof matched.", details={"internal_manifest_valid": True})
    layer2 = LayerOutcome(
        status="supporting",
        confidence=0.82,
        explanation="Contextual corroboration found.",
        details={
            "fact_check_verdict": "content match found",
            "matches": [{"url": "https://www.reuters.com/world/example", "domain": "www.reuters.com", "evidence_relation": "supports", "score": 0.9}],
        },
    )
    layer3 = LayerOutcome(status="none", confidence=0.0, explanation="No votes yet.", details={"vote_count": 0})

    scoring = service._build_system_scoring(layer1, layer2, layer3, extracted)

    assert scoring["scoring_method"] == "url_hybrid_formula"
    assert scoring["formula"] == "Final URL score = (0.30 * crypto) + (0.45 * NLP) + (0.25 * consensus). Evidence sufficiency and global penalty are diagnostics only."
    assert scoring["displayed_components"] == [
        "crypto_layer",
        "nlp_layer",
        "user_layer",
    ]
    assert set(scoring["layer_scores"]) == {"crypto", "nlp", "community"}
    assert "text_formula_components" not in scoring
    assert "claim_evidence_support" not in scoring["layer_scores"]
    assert "source_reliability" not in scoring["layer_scores"]
    assert "cross_source_agreement" not in scoring["layer_scores"]
    assert "evidence_directness" not in scoring["layer_scores"]
    assert "claim_clarity" not in scoring["layer_scores"]
    assert scoring["active_layer_weights"] == {"crypto": 0.30, "nlp": 0.45, "community": 0.25}
    assert scoring["url_fixed_formula_debug"]["weight_renormalized"] is False


def test_url_final_score_uses_fixed_weights_without_renormalizing_missing_consensus(db_session):
    service = VerificationService(db_session)
    adjusted = service._apply_url_no_multiplier_score(
        {
            "layer_scores": {"crypto": 0.85, "nlp": 0.068, "community": 0.10},
            "layer_status": {"crypto": "active", "nlp": "active", "community": "missing"},
            "active_layer_weights": {"crypto": 0.40, "nlp": 0.60},
            "evidence_multiplier": 1.0,
            "global_penalty": 1.0,
        },
        {},
    )

    assert adjusted["final_score"] == 31.06
    assert adjusted["active_layer_weights"] == {"crypto": 0.30, "nlp": 0.45, "community": 0.25}
    assert adjusted["url_fixed_formula_debug"] == {
        "formula_mode": "fixed_url_formula",
        "crypto_score": 85.0,
        "nlp_score": 6.8,
        "consensus_score": 10.0,
        "crypto_weight": 0.30,
        "nlp_weight": 0.45,
        "consensus_weight": 0.25,
        "missing_layers": ["community"],
        "any_layer_missing": True,
        "weight_renormalized": False,
        "final_score": 31.06,
    }


def test_url_nlp_wrapper_passes_clean_article_text_to_text_pipeline(db_session, monkeypatch):
    service = VerificationService(db_session)
    body = " ".join(
        [
            "Germany failed to win a temporary United Nations Security Council seat after member states voted for Austria and Portugal instead.",
            "The vote happened at the United Nations General Assembly during the 2026 election for non-permanent council members.",
            "Diplomats said the result was a setback for Germany's campaign because Austria and Portugal received stronger backing.",
            "The article claimed Germany lost the election despite lobbying other member states for support.",
            "It also said Trinidad and Tobago and Zimbabwe won seats for their respective regional groups.",
            "The report focused on the same vote outcome and the countries elected to the council.",
        ]
    )
    extracted = ExtractedPayload(
        title="Germany fails to win UN Security Council seat",
        author="Reporter",
        publish_time=datetime.fromisoformat("2026-06-03T10:00:00"),
        canonical_url="https://example.com/germany-un-seat",
        main_text=body,
        hero_image_url=None,
        normalized_text=body.lower(),
        content_hash="url-text-wrapper",
        metadata={"fetched_url": "https://example.com/germany-un-seat", "source_domain": "example.com", "extraction_method": "readability_article_text"},
    )
    captured = {}

    def fake_text_nlp_layer(job, payload, input_text):
        captured["payload"] = payload
        captured["input_text"] = input_text
        return LayerOutcome(
            status="supporting",
            confidence=0.82,
            explanation="External evidence supports the claim.",
            details={
                "mode": "text_verification",
                "extracted_claims": ["Germany failed to win a temporary United Nations Security Council seat."],
                "trusted_source_queries": ["Germany failed win UN Security Council seat Austria Portugal"],
                "lexical_queries": ["Germany Security Council seat vote"],
                "semantic_queries": ["Germany loses UN Security Council election"],
                "fact_check_queries": [],
                "official_source_queries": [],
                "exact_phrase_queries": ["Germany failed to win a temporary United Nations Security Council seat"],
                "evidence_sources": [
                    {
                        "title": "Austria and Portugal elected to UN Security Council",
                        "url": "https://www.reuters.com/world/example",
                        "domain": "www.reuters.com",
                        "source": "web_search",
                        "evidence_relation": "supports",
                        "relation_confidence": 0.86,
                        "claim_relevance_score": 0.9,
                        "score": 0.86,
                    }
                ],
                "text_component_scores": {"S": 0.82, "R": 0.82, "A": 0.82, "D": 0.82, "C": 0.82, "E": 1.0, "P": 1.0},
                "fact_check_verdict": "Likely Verified",
                "fact_check_assessment": "The same claim was supported by independent reporting.",
                "retrieval_debug": {"trusted_searches": []},
            },
        )

    monkeypatch.setattr(service, "_text_nlp_layer", fake_text_nlp_layer)

    outcome = service._layer2_article_url_via_text_pipeline(
        job=None,
        extracted=extracted,
        publisher=None,
        metadata_signals=[],
        source_domain="example.com",
        persist_matches=False,
    )

    expected_article_text = f"{extracted.title}\n\n{body}"
    assert captured["input_text"] == expected_article_text
    assert captured["payload"].metadata["source_type"] == "url_article_text"
    assert outcome.confidence == 0.82
    assert outcome.details["final_score"] == 0.82
    assert outcome.details["nlp_component_values"] == {
        "factual_consistency": 0.82,
        "reference_support": 0.82,
        "cross_source_agreement": 0.82,
        "language_reliability": 0.82,
        "authority": 0.82,
        "penalty": 1.0,
    }
    assert outcome.details["url_context_debug"]["final_text_passed_to_text_pipeline"].startswith(extracted.title)
    assert outcome.details["cross_publisher_count"] == 1


def test_url_nlp_extraction_failure_stays_low(db_session):
    service = VerificationService(db_session)
    extracted = ExtractedPayload(
        title="Fallback title",
        author=None,
        publish_time=None,
        canonical_url="https://web.archive.org/example",
        main_text="Submitted URL from web.archive.org. Full article text could not be fetched server-side.",
        hero_image_url=None,
        normalized_text="fallback title",
        content_hash="url-fallback",
        metadata={"fetched_url": "https://web.archive.org/example", "fetch_limited": True, "source_type": "url_fallback"},
    )

    outcome = service._layer2_article_url_via_text_pipeline(
        job=None,
        extracted=extracted,
        publisher=None,
        metadata_signals=[],
        source_domain="web.archive.org",
        persist_matches=False,
    )

    assert outcome.confidence == 0.0
    assert outcome.details["final_score"] == 0.0
    assert outcome.details["retrieval_failure_reason"] == "extraction_too_short"
    assert outcome.details["nlp_component_values"] == {
        "factual_consistency": 0.0,
        "reference_support": 0.0,
        "cross_source_agreement": 0.0,
        "language_reliability": 0.0,
        "authority": 0.0,
        "penalty": 1.0,
    }

    scoring = service._build_system_scoring(
        LayerOutcome(status="matched", confidence=0.85, explanation="Signed proof matched.", details={"internal_manifest_valid": True}),
        outcome,
        LayerOutcome(
            status="supporting",
            confidence=0.75,
            explanation="Consensus supports the content.",
            details={"vote_count": 4, "majority_vote": "true", "majority_confidence": 0.75, "true_votes": 3, "false_votes": 1, "not_sure_votes": 0},
        ),
        extracted,
    )

    assert scoring["layer_scores"]["nlp"] == 0.0
    assert scoring["final_label"] == "Not Verifiable"
    assert scoring["reason"] == "Could not extract enough article text from the URL, so URL NLP verification could not run."


def test_image_ocr_wrapper_calls_text_pipeline_for_claim_text(db_session, monkeypatch):
    service = VerificationService(db_session)
    captured = {}
    ocr_claim_text = (
        "City council approved a 2.5 billion dollar transit expansion after a Tuesday vote. "
        "Officials said the plan will add three rail stations and expand bus service next year."
    )
    extracted = ExtractedPayload(
        title="claim-image.png",
        author=None,
        publish_time=None,
        canonical_url=None,
        main_text=ocr_claim_text,
        hero_image_url=None,
        normalized_text="city council approved transit expansion",
        content_hash="image-wrapper",
        metadata={
            "ocr_pipeline": {
                    "extracted_text": ocr_claim_text,
                "text_detected": True,
                "ocr_status": "completed",
                "ocr_confidence": 0.91,
                "image_width": 1200,
                "image_height": 800,
            }
        },
    )

    def fake_text_nlp_layer(job, payload, input_text):
        captured["input_text"] = input_text
        captured["metadata"] = payload.metadata
        return LayerOutcome(
            status="supporting",
            confidence=0.72,
            explanation="External evidence supports the image OCR claim.",
            details={
                "text_verification_score": 0.72,
                "text_verification_label": "Likely Verified",
                "extracted_claims": ["City council approved a 2.5 billion dollar transit expansion."],
                "search_queries_used": ["City council approved 2.5 billion transit expansion"],
                "evidence_sources": [
                    {"url": "https://www.reuters.com/example", "domain": "www.reuters.com", "evidence_relation": "supports", "score": 0.8}
                ],
                "text_component_scores": {"S": 0.72, "R": 0.72, "A": 0.72, "D": 0.72, "C": 0.72, "E": 1.0, "P": 1.0},
            },
        )

    monkeypatch.setattr(service, "_text_nlp_layer", fake_text_nlp_layer)
    monkeypatch.setattr(service.vision_reasoning, "describe_image", lambda *args, **kwargs: {"summary": "image"})
    monkeypatch.setattr(service.vision_reasoning, "compare_visual_context", lambda *args, **kwargs: {"score": 0.5})

    outcome = service._image_analysis_layer(extracted, "missing.png", "claim-image.png", job=object())

    assert captured["input_text"].startswith("City council approved")
    assert captured["metadata"]["source_type"] == "image_ocr_text"
    assert outcome.confidence == 0.72
    assert outcome.details["text_verification_score"] == 0.72
    assert outcome.details["image_context_wrapper_debug"]["text_pipeline_called"] is True


def test_image_without_readable_text_does_not_call_text_pipeline(db_session, monkeypatch):
    service = VerificationService(db_session)
    called = False
    extracted = ExtractedPayload(
        title="blank.png",
        author=None,
        publish_time=None,
        canonical_url=None,
        main_text="",
        hero_image_url=None,
        normalized_text="",
        content_hash="image-no-text",
        metadata={"ocr_pipeline": {"extracted_text": "", "text_detected": False, "ocr_status": "ocr_no_text_detected", "ocr_confidence": 0.0}},
    )

    def fake_text_nlp_layer(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("text pipeline should not be called")

    monkeypatch.setattr(service, "_text_nlp_layer", fake_text_nlp_layer)
    monkeypatch.setattr(service.vision_reasoning, "describe_image", lambda *args, **kwargs: {"summary": "image"})
    monkeypatch.setattr(service.vision_reasoning, "compare_visual_context", lambda *args, **kwargs: {"score": 0.0})

    outcome = service._image_analysis_layer(extracted, "missing.png", "blank.png", job=object())

    assert called is False
    assert outcome.confidence == 0.0
    assert outcome.details["image_context_wrapper_debug"]["failure_reason"] == "no_readable_text"
    assert outcome.details["image_context_wrapper_debug"]["text_pipeline_called"] is False


def test_audio_wrapper_fixed_scoring_uses_text_nlp_without_multipliers(db_session):
    service = VerificationService(db_session)
    layer1 = LayerOutcome(status="matched", confidence=0.85, explanation="Audio provenance recorded.", details={"crypto_breakdown": {"hash_integrity": 1.0}})
    layer2 = LayerOutcome(
        status="supporting",
        confidence=0.70,
        explanation="Transcript claim supported.",
        details={
            "text_verification_score": 0.70,
            "text_verification_label": "Likely Verified",
            "audio_analysis_score": 0.2,
            "audio_context_wrapper_debug": {"text_pipeline_called": True, "failure_reason": None},
            "evidence_sources": [{"url": "https://apnews.com/example", "evidence_relation": "supports"}],
            "text_component_scores": {"S": 0.7},
        },
    )
    layer3 = LayerOutcome(status="none", confidence=0.0, explanation="No feedback.", details={})

    scoring = service._build_audio_system_scoring(layer1, layer2, layer3)

    assert scoring["final_score"] == 57.0
    assert scoring["layer_scores"] == {"crypto": 0.85, "nlp": 0.7, "community": 0.0}
    assert scoring["active_layer_weights"] == {"crypto": 0.30, "nlp": 0.45, "community": 0.25}


def test_true_consensus_vote_does_not_reduce_url_primary_score(db_session):
    service = VerificationService(db_session)
    extracted = ExtractedPayload(
        title="Example URL story",
        author="Reporter",
        publish_time=datetime.fromisoformat("2026-04-19T10:30:00"),
        canonical_url="https://example.com/news/story",
        main_text="A URL article body with enough content for normal URL verification.",
        hero_image_url=None,
        normalized_text="a url article body with enough content for normal url verification",
        content_hash="url-community-boost-safety",
        metadata={"fetched_url": "https://example.com/news/story", "source_domain": "example.com"},
    )
    layer1 = LayerOutcome(status="matched", confidence=0.95, explanation="Signed proof matched.", details={"internal_manifest_valid": True})
    layer2 = LayerOutcome(
        status="supporting",
        confidence=0.82,
        explanation="Contextual corroboration found.",
        details={"fact_check_verdict": "content match found", "matches": [{"url": "https://www.reuters.com/world/example"}]},
    )
    no_votes = LayerOutcome(status="none", confidence=0.0, explanation="No votes yet.", details={"vote_count": 0})
    true_vote = LayerOutcome(
        status="supporting",
        confidence=1.0,
        explanation="Consensus supports the content.",
        details={"vote_count": 1, "majority_vote": "true", "majority_confidence": 1.0, "true_votes": 1, "false_votes": 0, "not_sure_votes": 0},
    )

    base = service._build_system_scoring(layer1, layer2, no_votes, extracted)
    boosted = service._build_system_scoring(layer1, layer2, true_vote, extracted)

    assert boosted["final_score"] >= base["final_score"]
    assert boosted["layer_scores"]["crypto"] == base["layer_scores"]["crypto"]
    assert boosted["layer_scores"]["nlp"] == base["layer_scores"]["nlp"]
    assert boosted["community_adjustment"]["mode"] == "positive_consensus_boost"


def test_strong_url_crypto_and_nlp_are_not_downgraded_to_unverified_by_soft_penalty(db_session):
    service = VerificationService(db_session)
    extracted = ExtractedPayload(
        title=None,
        author=None,
        publish_time=None,
        canonical_url="https://example.com/news/story",
        main_text="A URL article body with a clearly extracted factual claim.",
        hero_image_url=None,
        normalized_text="a url article body with a clearly extracted factual claim",
        content_hash="url-primary-layer-floor",
        metadata={"fetched_url": "https://example.com/news/story", "fetch_limited": True},
    )
    layer1 = LayerOutcome(status="matched", confidence=0.85, explanation="Signed proof matched.", details={"internal_manifest_valid": True})
    layer2 = LayerOutcome(
        status="supporting",
        confidence=0.942,
        explanation="Strong contextual corroboration found.",
        details={
            "fact_check_verdict": "content match found",
            "matches": [
                {"url": "https://www.reuters.com/world/example", "domain": "www.reuters.com", "evidence_relation": "supports", "score": 0.94},
                {"url": "https://apnews.com/article/example", "domain": "apnews.com", "evidence_relation": "supports", "score": 0.92},
                {"url": "https://www.bbc.com/news/example", "domain": "www.bbc.com", "evidence_relation": "supports", "score": 0.9},
            ],
            "nlp_component_values": {
                "factual_consistency": 0.94,
                "reference_support": 0.94,
                "cross_source_agreement": 0.94,
                "language_reliability": 0.94,
                "authority": 0.94,
                "penalty": 1.0,
            },
        },
    )
    layer3 = LayerOutcome(status="none", confidence=0.0, explanation="No votes yet.", details={"vote_count": 0})

    scoring = service._build_system_scoring(layer1, layer2, layer3, extracted)

    assert scoring["evidence_multiplier"] == 0.64
    assert scoring["global_penalty"] == 0.6
    assert scoring["final_score"] >= 50
    assert scoring["final_label"] != "Unverified"
    assert scoring["url_multiplier_policy"]["evidence_multiplier_used_as_final_multiplier"] is False
    assert scoring["url_multiplier_policy"]["global_penalty_used_as_final_multiplier"] is False


def test_text_system_scoring_uses_text_formula_components_not_url_user_layer(db_session):
    service = VerificationService(db_session)
    extracted = build_text_payload("A trusted source confirmed the city council approved a new transit plan.")
    layer1 = LayerOutcome(status="none", confidence=0.0, explanation="No provenance.", details={})
    layer2 = LayerOutcome(
        status="supporting",
        confidence=0.75,
        explanation="Text evidence supports the claim.",
        details={
            "claim_results": [
                {
                    "support_score": 0.82,
                    "source_reliability": 0.9,
                    "agreement_score": 0.55,
                    "directness_score": 0.84,
                    "clarity_score": 0.9,
                    "claim_verdict": "supported",
                    "evidence_sources": [
                        {
                            "url": "https://www.reuters.com/world/example",
                            "domain": "www.reuters.com",
                            "source_type": "trusted_news_page",
                            "evidence_relation": "supports",
                            "alignment": "supports",
                        }
                    ],
                }
            ],
            "evidence_multiplier": 0.8,
            "penalty_multiplier": 1.0,
        },
    )
    layer3 = LayerOutcome(status="review_attention", confidence=0.0, explanation="Feedback only.", details={"mode": "feedback"})

    scoring = service._build_text_system_scoring(layer1, layer2, layer3, extracted)

    assert scoring["scoring_method"] == "text_verification_formula"
    assert set(scoring["text_formula_components"]) == {"S", "R", "A", "D", "C", "E", "P"}
    assert "claim_evidence_support" in scoring["layer_scores"]
    assert "source_reliability" in scoring["layer_scores"]
    assert "cross_source_agreement" in scoring["layer_scores"]
    assert "evidence_directness" in scoring["layer_scores"]
    assert "claim_clarity" in scoring["layer_scores"]
    assert "community" not in scoring["layer_scores"]
    assert scoring["display_layer_scores"]["community"] is None


def test_repeated_article_submissions_link_internal_provenance_versions(db_session):
    service = VerificationService(db_session)
    payload = ExtractedPayload(
        title="Repeat provenance story",
        author="Reporter",
        publish_time=datetime.fromisoformat("2026-03-20T10:30:00"),
        canonical_url="https://example.com/news/repeat-provenance-story",
        main_text="A repeated article body with enough content to persist provenance details.",
        hero_image_url=None,
        normalized_text="a repeated article body with enough content to persist provenance details",
        content_hash="repeat-provenance-story",
        metadata={
            "fetched_url": "https://example.com/news/repeat-provenance-story",
            "publisher_name": "Example News",
            "source_domain": "example.com",
        },
    )

    first_job = service._run_pipeline(service._create_job(JobInputType.url, payload.canonical_url), payload)
    second_job = service._run_pipeline(service._create_job(JobInputType.url, payload.canonical_url), payload)

    assert first_job.proof_summary["previous_version_id"] is None
    assert second_job.proof_summary["previous_version_id"] is not None


def test_similarity_matches_endpoint_paginates_ten_at_a_time(client):
    verify_response = client.post("/api/v1/verify/text", json={"text": "This is a long enough body of text for verification."})
    job_id = verify_response.json()["job_id"]

    first_page = client.get(f"/api/v1/results/{job_id}/matches?offset=0&limit=5")
    assert first_page.status_code == 200
    payload = first_page.json()
    assert payload["limit"] == 5
    assert payload["offset"] == 0
    assert "items" in payload


def test_get_similarity_matches_returns_next_ten_page(db_session):
    job = VerificationJob(
        input_type=JobInputType.text,
        input_value="sample",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()

    for index in range(7):
        db_session.add(
            SimilarityMatch(
                job_id=job.id,
                candidate_url=f"https://example.com/{index}",
                candidate_title=f"Match {index}",
                score=1 - (index * 0.01),
                explanation="Stored match",
                metadata_json={"source": "web_search", "domain": "example.com"},
            )
        )
    db_session.commit()

    service = VerificationService(db_session)
    first_page = service.get_similarity_matches(job.id, offset=0, limit=5)
    second_page = service.get_similarity_matches(job.id, offset=5, limit=5)

    assert len(first_page["items"]) == 5
    assert first_page["has_more"] is True
    assert len(second_page["items"]) == 2
    assert second_page["has_more"] is False


def test_build_fact_check_query_uses_url_slug_when_metadata_is_sparse(db_session):
    service = VerificationService(db_session)
    payload = ExtractedPayload(
        title="nytimes.com",
        author=None,
        publish_time=None,
        canonical_url="https://www.nytimes.com/2026/03/18/us/cesar-chavez-sexual-abuse-allegations-ufw.html",
        main_text="Submitted URL from nytimes.com. Full article text could not be fetched server-side.",
        hero_image_url=None,
        normalized_text="submitted url from nytimes com",
        content_hash="slug-query-test",
        metadata={"fetched_url": "https://www.nytimes.com/2026/03/18/us/cesar-chavez-sexual-abuse-allegations-ufw.html"},
    )

    query = service._build_fact_check_query(payload)

    assert "cesar chavez sexual abuse allegations ufw" in query


def test_url_fallback_payload_cannot_get_high_nlp_from_title_or_search(db_session):
    service = VerificationService(db_session)
    service.search.search = lambda query, expected_domain=None, limit=5: [
        WebSearchCandidate(
            title="Fact check: Obama vacation scam was fabricated",
            url="https://www.politifact.com/factchecks/example/obama-vacation-scam-fabricated/",
            snippet="A claim about Obama arranging billions in future vacations is fabricated.",
            domain="www.politifact.com",
            score=0.96,
            explanation="Fact check result",
        )
    ]
    job = VerificationJob(
        input_type=JobInputType.url,
        input_value="https://web.archive.org/web/201701/example",
        status=JobStatus.not_verifiable,
        headline_message="Pending",
        short_reason="Pending",
        content_preview={},
        proof_summary={},
        analysis_report={},
        layer_results={},
    )
    db_session.add(job)
    db_session.flush()
    payload = ExtractedPayload(
        title="Breaking Trump Just Ended Obamas Vacation Scam And Sent Him A Bill You Have To See To Believe",
        author=None,
        publish_time=None,
        canonical_url="https://web.archive.org/web/201701/example",
        main_text="Submitted URL from web.archive.org. Full article text could not be fetched server-side.",
        hero_image_url=None,
        normalized_text="submitted url from web archive org full article text could not be fetched server side",
        content_hash="archive-fallback-no-url-nlp",
        metadata={
            "fetched_url": "https://web.archive.org/web/201701/example",
            "fetch_limited": True,
            "source_type": "url_fallback",
            "source_domain": "web.archive.org",
        },
    )

    outcome = service._layer2_article_url(job, payload, None, [], persist_matches=False)

    assert outcome.status == "none"
    assert outcome.confidence == 0.0
    assert outcome.details["matches"] == []
    assert outcome.details["retrieval_failure_reason"] == "extraction_too_short"
    assert outcome.details["url_context_debug"]["fallback_used"] is True


def test_format_certificate_name_flattens_subject_entries():
    subject = ((("commonName", "edition.cnn.com"),), (("organizationName", "Example Org"),))

    assert _format_certificate_name(subject) == "commonName=edition.cnn.com, organizationName=Example Org"


def test_crypto_score_inputs_treat_verified_platform_manifest_as_platform_issued(db_session):
    service = VerificationService(db_session)
    payload = build_text_payload("Platform-issued provenance scoring check.")
    layer1 = LayerOutcome(
        status="publisher-recognized",
        confidence=0.62,
        explanation="Publisher identified.",
        details={
            "internal_manifest_valid": True,
            "canonical_hash": "abc123",
            "manifest_id": "manifest-1",
            "signed_at": "2026-04-19T00:00:00Z",
        },
    )

    score_inputs = service._crypto_score_inputs(layer1, payload)

    assert score_inputs["signature_validity"] == 1.0
    assert score_inputs["issuer_trust"] == 0.6
    assert score_inputs["timestamp_validity"] == 0.7


def test_decide_returns_likely_verified_when_primary_layers_support_without_votes(db_session):
    service = VerificationService(db_session)
    payload = build_text_payload("Primary layers support this article with strong matching evidence.")
    layer1 = LayerOutcome(
        status="publisher-recognized",
        confidence=0.62,
        explanation="Trusted publisher recognized.",
        details={},
    )
    layer2 = LayerOutcome(
        status="supporting",
        confidence=0.64,
        explanation="Corroborating similarity found.",
        details={"fact_check_verdict": "content match found"},
    )
    layer3 = LayerOutcome(
        status="none",
        confidence=0.0,
        explanation="No community votes yet.",
        details={"vote_count": 0},
    )

    status_value, headline, _, _, _ = service._decide(layer1, layer2, layer3, payload)

    assert status_value == JobStatus.likely_verified
    assert headline == "Likely Verified"


def test_decide_returns_verified_when_all_three_layers_align(db_session):
    service = VerificationService(db_session)
    payload = build_text_payload("All three layers align on this content.")
    layer1 = LayerOutcome(status="matched", confidence=0.99, explanation="Signed proof matched.", details={})
    layer2 = LayerOutcome(
        status="supporting",
        confidence=0.83,
        explanation="Strong corroborating source found.",
        details={"fact_check_verdict": "content match found"},
    )
    layer3 = LayerOutcome(
        status="supporting",
        confidence=0.6,
        explanation="Consensus supports the content.",
        details={"vote_count": 3},
    )

    status_value, headline, _, _, _ = service._decide(layer1, layer2, layer3, payload)

    assert status_value == JobStatus.verified
    assert headline == "Verified"


def test_decide_returns_unverified_when_two_layers_are_negative(db_session):
    service = VerificationService(db_session)
    payload = build_text_payload("Two layers disagree with the content.")
    layer1 = LayerOutcome(status="mismatch", confidence=0.98, explanation="Trusted proof mismatch.", details={})
    layer2 = LayerOutcome(status="none", confidence=0.0, explanation="No NLP support.", details={"fact_check_verdict": "inconclusive"})
    layer3 = LayerOutcome(
        status="disputing",
        confidence=-0.7,
        explanation="Consensus disputes the content.",
        details={"vote_count": 2},
    )

    status_value, headline, _, _, _ = service._decide(layer1, layer2, layer3, payload)

    assert status_value == JobStatus.unverified
    assert headline == "Unverified"
