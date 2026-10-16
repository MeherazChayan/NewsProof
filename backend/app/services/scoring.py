from __future__ import annotations

from typing import Any


def clamp01(value: float | int | None) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def clamp100(value: float | int | None) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, numeric))


def map_score_to_label(score: float | int | None) -> str:
    numeric = clamp100(score)
    if numeric >= 70:
        return "Verified"
    if numeric >= 50:
        return "Likely Verified"
    if numeric >= 35:
        return "Not Verifiable"
    return "Unverified"


def compute_crypto_score(layer_input: dict[str, Any]) -> dict[str, Any]:
    signature_validity = clamp01(layer_input.get("signature_validity"))
    issuer_trust = clamp01(layer_input.get("issuer_trust"))
    timestamp_validity = clamp01(layer_input.get("timestamp_validity"))
    hash_integrity = clamp01(layer_input.get("hash_integrity"))
    version_chain = clamp01(layer_input.get("version_chain"))
    metadata_completeness = clamp01(layer_input.get("metadata_completeness"))
    penalty = clamp01(layer_input.get("penalty", 1.0))

    score = (
        0.35 * signature_validity
        + 0.20 * issuer_trust
        + 0.15 * timestamp_validity
        + 0.15 * hash_integrity
        + 0.10 * version_chain
        + 0.05 * metadata_completeness
    ) * penalty

    return {
        "score": clamp01(score),
        "breakdown": {
            "signature_validity": signature_validity,
            "issuer_trust": issuer_trust,
            "timestamp_validity": timestamp_validity,
            "hash_integrity": hash_integrity,
            "version_chain": version_chain,
            "metadata_completeness": metadata_completeness,
            "penalty": penalty,
        },
    }


def compute_nlp_score(layer_input: dict[str, Any]) -> dict[str, Any]:
    factual_consistency = clamp01(layer_input.get("factual_consistency"))
    reference_support = clamp01(layer_input.get("reference_support"))
    cross_source_agreement = clamp01(layer_input.get("cross_source_agreement"))
    language_reliability = clamp01(layer_input.get("language_reliability"))
    authority = clamp01(layer_input.get("authority"))
    penalty = clamp01(layer_input.get("penalty", 1.0))

    score = (
        0.30 * factual_consistency
        + 0.25 * reference_support
        + 0.20 * cross_source_agreement
        + 0.15 * language_reliability
        + 0.10 * authority
    ) * penalty

    return {
        "score": clamp01(score),
        "breakdown": {
            "factual_consistency": factual_consistency,
            "reference_support": reference_support,
            "cross_source_agreement": cross_source_agreement,
            "language_reliability": language_reliability,
            "authority": authority,
            "penalty": penalty,
        },
    }


def compute_community_score(layer_input: dict[str, Any]) -> dict[str, Any]:
    validated_user_agreement = clamp01(layer_input.get("validated_user_agreement"))
    expert_assessment = clamp01(layer_input.get("expert_assessment"))
    reporting_behavior = clamp01(layer_input.get("reporting_behavior"))
    trust_weighted_reputation = clamp01(layer_input.get("trust_weighted_reputation"))
    penalty = clamp01(layer_input.get("penalty", 1.0))

    score = (
        0.35 * validated_user_agreement
        + 0.25 * expert_assessment
        + 0.20 * reporting_behavior
        + 0.20 * trust_weighted_reputation
    ) * penalty

    return {
        "score": clamp01(score),
        "breakdown": {
            "validated_user_agreement": validated_user_agreement,
            "expert_assessment": expert_assessment,
            "reporting_behavior": reporting_behavior,
            "trust_weighted_reputation": trust_weighted_reputation,
            "penalty": penalty,
        },
    }


def compute_evidence_multiplier(ep: int | float, et: int | float) -> float:
    try:
        present = float(ep)
        total = float(et)
    except (TypeError, ValueError):
        return 0.4
    if total <= 0:
        return 0.4
    return clamp01(0.4 + 0.6 * (max(0.0, present) / total))


def compute_global_penalty(flags: dict[str, Any] | None) -> float:
    flags = flags or {}
    if flags.get("confirmed_tampering") or flags.get("forged_provenance"):
        return 0.0
    if flags.get("strong_trusted_source_contradiction"):
        return 0.4
    if flags.get("evidence_incomplete_and_suspicious"):
        return 0.6
    return 1.0


BASE_LAYER_WEIGHTS = {"crypto": 0.30, "nlp": 0.45, "community": 0.25}
EXCLUDED_LAYER_STATES = {"missing", "insufficient", "neutral", "excluded"}


def calculate_weighted_score(
    layers: dict[str, dict[str, Any]],
    evidence_sufficiency: float,
    global_penalty: float,
) -> dict[str, Any]:
    active_layers: dict[str, dict[str, Any]] = {}
    layer_status: dict[str, str] = {}
    for name, payload in layers.items():
        state = str(payload.get("state") or "active").lower()
        layer_status[name] = state
        if state in EXCLUDED_LAYER_STATES:
            continue
        active_layers[name] = payload

    if not active_layers:
        return {
            "score": 0.0,
            "base_score": 0.0,
            "active_weights": {},
            "layer_status": layer_status,
        }

    total_weight = sum(BASE_LAYER_WEIGHTS.get(name, 0.0) for name in active_layers)
    if total_weight <= 0:
        return {
            "score": 0.0,
            "base_score": 0.0,
            "active_weights": {},
            "layer_status": layer_status,
        }

    active_weights = {name: BASE_LAYER_WEIGHTS.get(name, 0.0) / total_weight for name in active_layers}
    base_score = sum(active_weights[name] * clamp01(payload.get("score")) for name, payload in active_layers.items())
    score = clamp100(100 * base_score * clamp01(evidence_sufficiency) * clamp01(global_penalty))
    return {
        "score": score,
        "base_score": base_score,
        "active_weights": active_weights,
        "layer_status": layer_status,
    }


def generate_newsproof_label(score: float, evidence_multiplier: float, flags: dict[str, Any] | None = None) -> dict[str, str]:
    flags = flags or {}
    score = clamp100(score)
    evidence_multiplier = clamp01(evidence_multiplier)
    if flags.get("confirmed_tampering") or flags.get("forged_provenance"):
        return {
            "final_label": "Unverified",
            "reason": "Critical provenance failure or tampering indicators block a verified outcome.",
        }
    if flags.get("severe_contradiction") or flags.get("strong_trusted_source_contradiction") or flags.get("contradictory_evidence"):
        return {
            "final_label": "Unverified",
            "reason": "Contradictory evidence or severe factual conflict prevent full verification.",
        }
    label = map_score_to_label(score)
    if label == "Verified":
        return {
            "final_label": label,
            "reason": "Strong overall support with no critical failure detected.",
        }
    if label == "Likely Verified":
        return {
            "final_label": label,
            "reason": "Good overall support, but not strong enough for full verification.",
        }
    if label == "Not Verifiable":
        return {
            "final_label": label,
            "reason": "The available evidence is too limited or mixed for a reliable verification decision.",
        }
    return {
        "final_label": label,
        "reason": "Overall support is weak enough that the item remains unverified.",
    }


def generate_explanation(result: dict[str, Any]) -> str:
    label = result.get("final_label", "")
    final_score = float(result.get("final_score", 0.0))
    crypto = float(result.get("layer_scores", {}).get("crypto", 0.0))
    nlp = float(result.get("layer_scores", {}).get("nlp", 0.0))
    community = float(result.get("layer_scores", {}).get("community", 0.0))
    warnings = result.get("warnings", [])
    if "confirmed tampering or forged provenance" in warnings:
        return "Tampering or forged provenance indicators caused the score to fail."
    layer_status = result.get("layer_status", {}) if isinstance(result.get("layer_status"), dict) else {}
    consensus_missing = layer_status.get("community") in {"missing", "insufficient", "neutral", "excluded"}
    if consensus_missing and crypto >= 0.8 and nlp >= 0.55:
        return "Cryptographic/source signals and NLP evidence provide moderate support, but community consensus data is unavailable. The score was calculated without treating missing consensus as negative evidence."
    if label == "Not Verifiable":
        return "The available evidence is too limited or mixed for a reliable verification decision."
    if crypto >= 0.75 and nlp >= 0.65 and community < 0.4:
        return "Strong provenance and good contextual support, but limited community validation."
    if crypto < 0.4 and nlp < 0.4:
        return "Article has weak provenance and low cross-source support."
    if nlp >= 0.7 and crypto < 0.45:
        return "Contextual evidence is fairly strong, but provenance evidence remains weak."
    if crypto >= 0.7 and nlp < 0.45:
        return "Provenance is strong, but contextual support is limited or inconsistent."
    if final_score >= 70:
        return "Strong provenance, contextual support, and review signals align in favor of the article."
    if final_score >= 50:
        return "The article has meaningful support across the system, but some verification gaps remain."
    if final_score >= 35:
        return "Evidence remains too mixed to classify the item as reliably verified."
    return "The available signals point to weak verification support and possible unreliability."


def compute_newsproof_score(input_data: dict[str, Any]) -> dict[str, Any]:
    crypto = compute_crypto_score(input_data.get("crypto", {}))
    nlp = compute_nlp_score(input_data.get("nlp", {}))
    community = compute_community_score(input_data.get("community", {}))
    community_state = _community_layer_state(input_data.get("community", {}), community)

    evidence_multiplier = compute_evidence_multiplier(
        input_data.get("evidence_present", 0),
        input_data.get("evidence_total", 0),
    )
    global_penalty = compute_global_penalty(input_data.get("flags"))

    weighted = calculate_weighted_score(
        {
            "crypto": {"score": crypto["score"], "state": "active"},
            "nlp": {"score": nlp["score"], "state": "active"},
            "community": {"score": community["score"], "state": community_state},
        },
        evidence_multiplier,
        global_penalty,
    )
    final_score = clamp100(weighted["score"])
    flags = input_data.get("flags") or {}
    final_score = _apply_score_caps(final_score, flags, crypto["score"], nlp["score"], evidence_multiplier, community_state)

    warnings = list(_warning_flags(input_data, crypto, nlp, community, evidence_multiplier, global_penalty))
    label_result = generate_newsproof_label(final_score, evidence_multiplier, input_data.get("flags"))

    result = {
        "final_score": final_score,
        "final_label": label_result["final_label"],
        "reason": label_result["reason"],
        "evidence_multiplier": evidence_multiplier,
        "global_penalty": global_penalty,
        "layer_scores": {
            "crypto": crypto["score"],
            "nlp": nlp["score"],
            "community": community["score"],
        },
        "layer_breakdown": {
            "crypto": {**crypto["breakdown"], "score_state": "active", "weighted_scoring": "Active"},
            "nlp": {**nlp["breakdown"], "score_state": "active", "weighted_scoring": "Active"},
            "community": {
                **community["breakdown"],
                "score_state": community_state,
                "weighted_scoring": _layer_status_message("community", community_state),
            },
        },
        "active_layer_weights": {key: round(float(value), 3) for key, value in weighted.get("active_weights", {}).items()},
        "layer_status": weighted.get("layer_status", {}),
        "warnings": warnings,
    }
    result["explanation"] = generate_explanation(result)
    return result


def _community_layer_state(layer_input: dict[str, Any], community: dict[str, Any]) -> str:
    explicit = str(layer_input.get("status") or layer_input.get("state") or "").strip().lower()
    if explicit:
        aliases = {
            "none": "missing",
            "missing": "missing",
            "insufficient": "insufficient",
            "neutral": "neutral",
            "active_positive": "active_positive",
            "supporting": "active_positive",
            "active_negative": "active_negative",
            "disputing": "active_negative",
            "active_mixed": "active_mixed",
            "mixed": "active_mixed",
        }
        return aliases.get(explicit, explicit)
    vote_count = int(layer_input.get("vote_count") or 0)
    feedback_count = int(layer_input.get("feedback_count") or 0)
    if vote_count <= 0 and feedback_count <= 0:
        return "missing"
    majority_vote = str(layer_input.get("majority_vote") or "").lower()
    if majority_vote == "false":
        return "active_negative"
    if majority_vote == "true":
        return "active_positive"
    return "active_mixed"


def _apply_score_caps(
    score: float,
    flags: dict[str, Any],
    crypto_score: float,
    nlp_score: float,
    evidence_multiplier: float,
    community_state: str,
) -> float:
    capped = score
    if flags.get("no_evidence_found"):
        capped = min(capped, 35.0)
    if flags.get("all_unrelated_evidence"):
        capped = min(capped, 45.0)
    if flags.get("only_partial_or_background_evidence"):
        capped = min(capped, 55.0)
    if flags.get("location_entity_mismatch"):
        capped = min(capped, 45.0)
    if flags.get("contradictory_evidence") or flags.get("strong_trusted_source_contradiction") or flags.get("severe_contradiction"):
        capped = min(capped, 34.0)

    has_negative_flag = any(
        bool(flags.get(name))
        for name in (
            "confirmed_tampering",
            "forged_provenance",
            "strong_trusted_source_contradiction",
            "severe_contradiction",
            "contradictory_evidence",
            "location_entity_mismatch",
            "all_unrelated_evidence",
        )
    )
    if (
        crypto_score >= 0.80
        and nlp_score >= 0.55
        and evidence_multiplier >= 0.70
        and community_state != "active_negative"
        and not has_negative_flag
    ):
        capped = max(capped, 50.0)
    return clamp100(capped)


def _layer_status_message(layer_name: str, state: str) -> str:
    if state == "missing":
        return "Missing: no meaningful layer data was available, so this layer was excluded from weighted scoring."
    if state == "insufficient":
        return "Insufficient: not enough layer data was available, so this layer was treated as neutral and excluded."
    if state == "neutral":
        return "Neutral: this layer did not provide a positive or negative scoring signal."
    if state == "active_negative":
        return "Penalising: this layer contains actual negative evidence and is included in weighted scoring."
    if state == "active_mixed":
        return "Active mixed: this layer has real but inconclusive evidence and is included with its measured score."
    if state == "active_positive":
        return "Active: this layer contains supportive evidence and is included in weighted scoring."
    return "Active: this layer is included in weighted scoring."


def _warning_flags(
    input_data: dict[str, Any],
    crypto: dict[str, Any],
    nlp: dict[str, Any],
    community: dict[str, Any],
    evidence_multiplier: float,
    global_penalty: float,
) -> list[str]:
    flags = input_data.get("flags", {}) or {}
    warnings: list[str] = []
    if flags.get("confirmed_tampering") or flags.get("forged_provenance"):
        warnings.append("confirmed tampering or forged provenance")
    if flags.get("tampering_detected"):
        warnings.append("tampering detected")
    if flags.get("severe_contradiction"):
        warnings.append("severe contradiction detected")
    if flags.get("strong_trusted_source_contradiction"):
        warnings.append("strong contradiction across trusted sources")
    if flags.get("evidence_incomplete_and_suspicious"):
        warnings.append("evidence incomplete and suspicious")
    if flags.get("suspicious_vote_spike"):
        warnings.append("suspicious community vote spike detected")
    if flags.get("clear_vote_manipulation"):
        warnings.append("community signal manipulation detected")
    if evidence_multiplier < 0.55:
        warnings.append("evidence coverage is thin")
    if crypto["score"] < 0.3:
        warnings.append("cryptographic provenance is weak")
    if nlp["score"] < 0.3:
        warnings.append("contextual support is weak")
    if community["score"] == 0 and not input_data.get("community", {}).get("vote_count") and not input_data.get("community", {}).get("feedback_count"):
        warnings.append("community validation is absent")
    if global_penalty == 0.0:
        warnings.append("global penalty forced the score to zero")
    return warnings
