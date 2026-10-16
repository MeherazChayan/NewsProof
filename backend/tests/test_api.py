from app.services.content_extraction import ExtractedPayload


def test_verify_text_and_history(client):
    response = client.post("/api/v1/verify/text", json={"text": "This is a long enough body of text for verification."})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "Unverified"
    assert len(payload["audit_trail"]) >= 1
    assert "analysis_report" in payload
    assert payload["input_type"] == "text"
    assert payload["proof_summary"]["verification_record_id"]
    assert payload["proof_summary"]["verification_manifest_id"]

    history = client.get("/api/v1/history")
    assert history.status_code == 200
    assert len(history.json()["items"]) == 1


def test_publishers_crud(client):
    create_payload = {
        "name": "Metro Wire",
        "domain": "metro.test",
        "metadata": {"country": "US"},
        "public_keys": [{"key_id": "k1", "algorithm": "Ed25519", "public_key": "abc", "is_active": True}],
    }
    create_response = client.post("/api/v1/publishers", json=create_payload)
    assert create_response.status_code == 200
    publisher_id = create_response.json()["id"]

    list_response = client.get("/api/v1/publishers")
    assert list_response.status_code == 200
    assert len(list_response.json()) == 1

    update_response = client.put(
        f"/api/v1/publishers/{publisher_id}",
        json={**create_payload, "name": "Metro Wire Updated"},
    )
    assert update_response.status_code == 200
    assert update_response.json()["name"] == "Metro Wire Updated"

    delete_response = client.delete(f"/api/v1/publishers/{publisher_id}")
    assert delete_response.status_code == 200


def test_content_records_crud(client):
    publisher_response = client.post(
        "/api/v1/publishers",
        json={
            "name": "Wire Desk",
            "domain": "wiredesk.test",
            "metadata": {"domains": ["www.wiredesk.test"], "names": ["Wire Desk"]},
            "public_keys": [{"key_id": "k1", "algorithm": "Ed25519", "public_key": "abc", "is_active": True}],
        },
    )
    publisher_id = publisher_response.json()["id"]

    create_response = client.post(
        "/api/v1/content-records",
        json={
            "publisher_id": publisher_id,
            "canonical_url": "https://www.wiredesk.test/story",
            "title": "Wire Desk Story",
            "author": "Reporter",
            "publish_time": "2026-03-17T10:30:00",
            "normalized_text": "wire desk story body",
            "content_hash": "hash123",
            "signature": None,
            "key_id": None,
            "source_metadata": {},
        },
    )
    assert create_response.status_code == 200
    record_id = create_response.json()["id"]

    list_response = client.get(f"/api/v1/content-records?publisher_id={publisher_id}")
    assert list_response.status_code == 200
    assert len(list_response.json()) == 1

    update_response = client.put(
        f"/api/v1/content-records/{record_id}",
        json={
            "canonical_url": "https://www.wiredesk.test/story",
            "title": "Wire Desk Story Updated",
            "author": "Reporter",
            "publish_time": "2026-03-17T10:30:00",
            "normalized_text": "wire desk story body",
            "content_hash": "hash123",
            "signature": None,
            "key_id": None,
            "source_metadata": {},
        },
    )
    assert update_response.status_code == 200
    assert update_response.json()["title"] == "Wire Desk Story Updated"

    delete_response = client.delete(f"/api/v1/content-records/{record_id}")
    assert delete_response.status_code == 200


def test_url_verification_happy_path(client, monkeypatch):
    async def fake_fetch(_url: str):
        return ExtractedPayload(
            title="Trusted article",
            author="Reporter",
            publish_time=None,
            canonical_url="https://example.com/story",
            main_text="Trusted article body",
            hero_image_url=None,
            normalized_text="trusted article body",
            content_hash="abc123",
            metadata={},
        )

    monkeypatch.setattr("app.services.verification.fetch_url_content", fake_fetch)
    response = client.post("/api/v1/verify/url", json={"url": "https://example.com/story"})
    assert response.status_code == 200
    assert response.json()["headline_message"] in {"Not Verifiable", "Likely Verified", "Verified", "Unverified"}


def test_consensus_vote_updates_job(client):
    verify_response = client.post("/api/v1/verify/text", json={"text": "This is a long enough body of text for verification."})
    job_id = verify_response.json()["job_id"]

    vote_response = client.post(
        f"/api/v1/consensus/{job_id}/votes",
        json={
            "reporter_name": "Analyst",
            "role": "fact_checker",
            "vote_value": -1,
            "confidence": 0.8,
            "report_reason": "The source looks altered compared with the original publication.",
            "evidence_url": "https://example.com/evidence"
        },
    )
    assert vote_response.status_code == 400


def test_text_feedback_submission_and_summary(client):
    verify_response = client.post("/api/v1/verify/text", json={"text": "This is a long enough body of text for verification."})
    assert verify_response.status_code == 200
    job_id = verify_response.json()["job_id"]

    feedback_response = client.post(
        f"/api/v1/feedback/{job_id}",
        json={
            "feedback_type": "source_suggestion",
            "feedback_text": "Consider the official press release as an additional source.",
            "submitted_by": "Researcher",
            "user_role": "expert",
            "linked_source_url": "https://example.com/press-release",
        },
    )
    assert feedback_response.status_code == 200
    assert feedback_response.json()["feedback_type"] == "source_suggestion"

    summary_response = client.get(f"/api/v1/feedback/{job_id}")
    assert summary_response.status_code == 200
    assert summary_response.json()["totals"]["feedback_count"] == 1
    assert summary_response.json()["feedback_entries"][0]["submitted_by"] == "Researcher"


def test_repeated_text_verification_creates_new_text_version(client):
    first = client.post("/api/v1/verify/text", json={"text": "Repeated verification text should version correctly."})
    assert first.status_code == 200
    assert first.json()["proof_summary"]["current_version"] == 1

    second = client.post("/api/v1/verify/text", json={"text": "Repeated verification text should version correctly."})
    assert second.status_code == 200
    assert second.json()["proof_summary"]["current_version"] == 2
    assert second.json()["proof_summary"]["previous_version_id"] is not None

    summary_response = client.get(f"/api/v1/consensus/{second.json()['job_id']}")
    assert summary_response.status_code == 400


def test_consensus_votes_persist_across_same_link_verifications(client, monkeypatch):
    async def fake_fetch(_url: str):
        return ExtractedPayload(
            title="CNN article",
            author="Hanako Montgomery",
            publish_time=None,
            canonical_url="https://www.cnn.com/2026/03/19/world/example-story",
            main_text="cnn example story body",
            hero_image_url=None,
            normalized_text="cnn example story body",
            content_hash="cnn-same-link-hash",
            metadata={"publisher_name": "CNN", "source_domain": "www.cnn.com"},
        )

    monkeypatch.setattr("app.services.verification.fetch_url_content", fake_fetch)

    first = client.post("/api/v1/verify/url", json={"url": "https://www.cnn.com/2026/03/19/world/example-story"})
    assert first.status_code == 200
    first_job_id = first.json()["job_id"]

    vote_response = client.post(
        f"/api/v1/consensus/{first_job_id}/votes",
        json={"reporter_name": "Reader", "vote_value": 1},
    )
    assert vote_response.status_code == 200

    second = client.post("/api/v1/verify/url", json={"url": "https://www.cnn.com/2026/03/19/world/example-story"})
    assert second.status_code == 200
    second_job_id = second.json()["job_id"]

    summary_response = client.get(f"/api/v1/consensus/{second_job_id}")
    assert summary_response.status_code == 200
    assert summary_response.json()["totals"]["vote_count"] == 1


def test_verify_audio_upload(client):
    response = client.post(
        "/api/v1/verify/audio",
        files={"file": ("sample.wav", b"fake-wave-data", "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "Not Verifiable"
    assert payload["content_preview"]["title"].startswith("Audio upload:")


def test_verify_video_upload(client):
    response = client.post(
        "/api/v1/verify/video",
        files={"file": ("sample.mp4", b"fake-video-data", "video/mp4")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "Not Verifiable"
    assert payload["content_preview"]["title"].startswith("Video upload:")


def test_video_upload_uses_video_size_limit(client):
    response = client.post(
        "/api/v1/verify/video",
        files={"file": ("large.mp4", b"x" * 6_000_000, "video/mp4")},
    )
    assert response.status_code == 200
