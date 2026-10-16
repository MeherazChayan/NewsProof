def test_crypto_article_snapshot_sign_verify_flow(client):
    issuers = client.get("/api/v1/crypto/issuers")
    assert issuers.status_code == 200
    signer_id = issuers.json()[0]["id"]

    snapshot_response = client.post(
        "/api/v1/crypto/article-snapshots",
        json={
            "source_url": "https://example.com/story",
            "publisher_name": "Example News",
            "author_name": "Reporter Name",
            "title": "Example Story",
            "normalized_article_text": "Example article body",
            "media_urls": [],
            "metadata": {"section": "World"},
        },
    )
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()

    manifest_response = client.post(
        "/api/v1/crypto/manifests/sign",
        json={
            "object_type": "article_snapshot",
            "object_id": snapshot["id"],
            "signer_id": signer_id,
        },
    )
    assert manifest_response.status_code == 200
    manifest = manifest_response.json()
    assert manifest["article"]["title"] == "Example Story"
    assert manifest["crypto"]["signature_algorithm"] == "Ed25519"

    verify_response = client.post(
        "/api/v1/crypto/manifests/verify",
        json={"manifest_id": manifest["manifest_id"], "verified_by": "test-suite"},
    )
    assert verify_response.status_code == 200
    verification = verify_response.json()

    history_response = client.get(f"/api/v1/crypto/records/article_snapshot/{snapshot['id']}/history")
    assert history_response.status_code == 200

    assert verification["valid"] is True
    assert verification["status"] == "valid"
    assert verification["confidence"]["score"] >= 90
    assert history_response.json()["versions"][0]["id"] == snapshot["id"]
