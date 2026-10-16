from app.services.scoring import (
    compute_community_score,
    compute_crypto_score,
    compute_evidence_multiplier,
    compute_global_penalty,
    compute_newsproof_score,
    compute_nlp_score,
    calculate_weighted_score,
    generate_newsproof_label,
)


def test_crypto_score_formula_is_applied_exactly():
    result = compute_crypto_score(
        {
            "signature_validity": 1.0,
            "issuer_trust": 1.0,
            "timestamp_validity": 0.7,
            "hash_integrity": 1.0,
            "version_chain": 1.0,
            "metadata_completeness": 0.8,
            "penalty": 1.0,
        }
    )

    expected = (0.35 * 1.0 + 0.20 * 1.0 + 0.15 * 0.7 + 0.15 * 1.0 + 0.10 * 1.0 + 0.05 * 0.8)
    assert result["score"] == expected


def test_nlp_score_formula_is_applied_exactly():
    result = compute_nlp_score(
        {
            "factual_consistency": 0.8,
            "reference_support": 0.6,
            "cross_source_agreement": 0.5,
            "language_reliability": 0.9,
            "authority": 0.7,
            "penalty": 1.0,
        }
    )

    expected = (0.30 * 0.8 + 0.25 * 0.6 + 0.20 * 0.5 + 0.15 * 0.9 + 0.10 * 0.7)
    assert result["score"] == expected


def test_community_score_formula_is_applied_exactly():
    result = compute_community_score(
        {
            "validated_user_agreement": 0.75,
            "expert_assessment": 0.5,
            "reporting_behavior": 0.9,
            "trust_weighted_reputation": 0.8,
            "penalty": 1.0,
        }
    )

    expected = (0.35 * 0.75 + 0.25 * 0.5 + 0.20 * 0.9 + 0.20 * 0.8)
    assert result["score"] == expected


def test_evidence_multiplier_and_global_penalty_follow_rules():
    assert compute_evidence_multiplier(0, 10) == 0.4
    assert compute_evidence_multiplier(10, 10) == 1.0
    assert compute_global_penalty({"confirmed_tampering": True}) == 0.0
    assert compute_global_penalty({"strong_trusted_source_contradiction": True}) == 0.4
    assert compute_global_penalty({"evidence_incomplete_and_suspicious": True}) == 0.6


def test_whole_system_score_supports_forced_not_verifiable():
    result = compute_newsproof_score(
        {
            "crypto": {
                "signature_validity": 0.2,
                "issuer_trust": 0.3,
                "timestamp_validity": 0.0,
                "hash_integrity": 0.5,
                "version_chain": 1.0,
                "metadata_completeness": 0.3,
                "penalty": 1.0,
            },
            "nlp": {
                "factual_consistency": 0.2,
                "reference_support": 0.0,
                "cross_source_agreement": 0.0,
                "language_reliability": 0.7,
                "authority": 0.2,
                "penalty": 1.0,
            },
            "community": {
                "validated_user_agreement": 0.0,
                "expert_assessment": 0.0,
                "reporting_behavior": 0.0,
                "trust_weighted_reputation": 0.0,
                "penalty": 1.0,
            },
            "evidence_present": 1,
            "evidence_total": 10,
            "flags": {"force_not_verifiable": True, "evidence_too_limited": True},
        }
    )

    assert result["final_label"] == "Unverified"
    assert result["reason"] == "Overall support is weak enough that the item remains unverified."
    assert result["explanation"] == "Article has weak provenance and low cross-source support."


def test_whole_system_score_fails_on_tampering():
    result = compute_newsproof_score(
        {
            "crypto": {
                "signature_validity": 0.0,
                "issuer_trust": 1.0,
                "timestamp_validity": 0.7,
                "hash_integrity": 0.0,
                "version_chain": 0.5,
                "metadata_completeness": 0.8,
                "penalty": 0.0,
            },
            "nlp": {
                "factual_consistency": 0.8,
                "reference_support": 0.7,
                "cross_source_agreement": 0.7,
                "language_reliability": 0.9,
                "authority": 0.7,
                "penalty": 1.0,
            },
            "community": {
                "validated_user_agreement": 0.8,
                "expert_assessment": 0.5,
                "reporting_behavior": 0.8,
                "trust_weighted_reputation": 0.7,
                "penalty": 1.0,
            },
            "evidence_present": 8,
            "evidence_total": 10,
            "flags": {"confirmed_tampering": True},
        }
    )

    assert result["final_score"] == 0.0
    assert result["final_label"] == "Unverified"
    assert result["reason"] == "Critical provenance failure or tampering indicators block a verified outcome."
    assert "Tampering" in result["explanation"]


def test_label_generation_bands():
    assert generate_newsproof_label(92, 1.0, {})["final_label"] == "Verified"
    assert generate_newsproof_label(70, 1.0, {})["final_label"] == "Verified"
    assert generate_newsproof_label(65, 1.0, {})["final_label"] == "Likely Verified"
    assert generate_newsproof_label(50, 1.0, {})["final_label"] == "Likely Verified"
    assert generate_newsproof_label(45, 1.0, {})["final_label"] == "Not Verifiable"
    assert generate_newsproof_label(35, 1.0, {})["final_label"] == "Not Verifiable"
    assert generate_newsproof_label(10, 1.0, {})["final_label"] == "Unverified"


def test_label_generation_no_longer_uses_evidence_gate_before_score():
    result = generate_newsproof_label(96, 0.4, {})

    assert result["final_label"] == "Verified"
    assert "Strong overall support" in result["reason"]


def test_label_generation_downgrades_verified_when_contradiction_exists():
    result = generate_newsproof_label(91, 0.8, {"severe_contradiction": True})

    assert result["final_label"] == "Unverified"
    assert "prevent full verification" in result["reason"]


def test_active_layer_weighting_excludes_missing_consensus():
    result = calculate_weighted_score(
        {
            "crypto": {"score": 0.863, "state": "active"},
            "nlp": {"score": 0.610, "state": "active"},
            "community": {"score": 0.10, "state": "missing"},
        },
        evidence_sufficiency=0.76,
        global_penalty=1.0,
    )

    assert round(result["score"], 1) == 54.1
    assert round(result["active_weights"]["crypto"], 2) == 0.40
    assert round(result["active_weights"]["nlp"], 2) == 0.60
    assert "community" not in result["active_weights"]


def test_neutral_consensus_score_can_be_included_when_explicitly_active_mixed():
    result = calculate_weighted_score(
        {
            "crypto": {"score": 0.863, "state": "active"},
            "nlp": {"score": 0.610, "state": "active"},
            "community": {"score": 0.50, "state": "active_mixed"},
        },
        evidence_sufficiency=0.76,
        global_penalty=1.0,
    )

    assert round(result["score"], 1) == 50.0


def test_missing_consensus_status_does_not_use_low_community_score():
    result = compute_newsproof_score(
        {
            "crypto": {"signature_validity": 0.863, "penalty": 1.0},
            "nlp": {"factual_consistency": 0.610, "penalty": 1.0},
            "community": {
                "validated_user_agreement": 0.10,
                "expert_assessment": 0.10,
                "reporting_behavior": 0.10,
                "trust_weighted_reputation": 0.10,
                "penalty": 1.0,
                "status": "missing",
            },
            "evidence_present": 6,
            "evidence_total": 10,
            "flags": {},
        }
    )

    assert result["layer_status"]["community"] == "missing"
    assert "community" not in result["active_layer_weights"]


def test_active_negative_consensus_can_reduce_score():
    result = calculate_weighted_score(
        {
            "crypto": {"score": 0.863, "state": "active"},
            "nlp": {"score": 0.610, "state": "active"},
            "community": {"score": 0.10, "state": "active_negative"},
        },
        evidence_sufficiency=0.76,
        global_penalty=1.0,
    )

    assert round(result["score"], 1) == 42.4
    assert "community" in result["active_weights"]


def test_strong_source_moderate_nlp_floor_when_no_contradiction():
    result = compute_newsproof_score(
        {
            "crypto": {
                "signature_validity": 1.0,
                "issuer_trust": 1.0,
                "timestamp_validity": 0.7,
                "hash_integrity": 1.0,
                "version_chain": 1.0,
                "metadata_completeness": 0.66,
                "penalty": 0.89,
            },
            "nlp": {
                "factual_consistency": 0.61,
                "reference_support": 0.61,
                "cross_source_agreement": 0.61,
                "language_reliability": 0.61,
                "authority": 0.61,
                "penalty": 1.0,
            },
            "community": {"status": "missing"},
            "evidence_present": 6,
            "evidence_total": 10,
            "flags": {},
        }
    )

    assert result["final_score"] >= 50.0
    assert result["final_label"] in {"Likely Verified", "Verified"}
    assert "missing consensus" in result["explanation"].lower() or "community consensus data is unavailable" in result["explanation"].lower()
