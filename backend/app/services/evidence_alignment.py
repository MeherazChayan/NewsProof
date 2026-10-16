from __future__ import annotations

from app.services.context_match_service import ContextMatchService
from app.services.fact_check_parser import normalize_fact_check_verdict, normalize_fact_check_verdict_label, verdict_direction
import re

from app.utils import normalize_text


def compare_claim_with_evidence(claim: str, evidence_text: str) -> dict:
    matcher = ContextMatchService()
    context = matcher.compare(claim, evidence_text)
    verdict = normalize_fact_check_verdict(evidence_text)
    verdict_label = normalize_fact_check_verdict_label(evidence_text)
    overlap = _overlap(claim, evidence_text)
    entity_overlap = _proper_entity_overlap(claim, evidence_text)
    action_match = _action_object_match(claim, evidence_text)
    aboutness = _detect_aboutness(context, overlap, entity_overlap, action_match, verdict)
    missing_entities = [
        item
        for item in context.get("claim_actors", []) + context.get("claim_locations", []) + context.get("claim_countries", [])
        if item not in context.get("evidence_actors", []) + context.get("evidence_locations", []) + context.get("evidence_countries", [])
    ]
    contradictions = []
    if verdict == "refutes":
        contradictions.append("Fact-check verdict refutes the claim.")
    if context.get("mismatch_reason"):
        contradictions.append(context["mismatch_reason"])

    if verdict == "refutes" and aboutness == "exact_claim":
        alignment = "refutes"
    elif not context.get("location_match_pass"):
        alignment = "irrelevant"
    elif verdict == "supports" and context.get("context_match_pass") and action_match["action_match"]:
        alignment = "supports"
    elif context.get("context_match_pass") and overlap >= 0.82 and action_match["action_match"]:
        alignment = "supports"
    elif overlap >= 0.25 or context.get("entity_match_score", 0) > 0 or entity_overlap["score"] > 0:
        alignment = "mentions_only"
    else:
        alignment = "irrelevant"

    stance = _stance_from_alignment(alignment, verdict, aboutness)
    return {
        "alignment": alignment,
        "aboutness": aboutness,
        "stance": stance,
        "verdict_extracted": verdict_label,
        "verdict_direction": verdict,
        "confidence": round(max(overlap, 0.85 if verdict in {"refutes", "supports"} else 0.0), 3),
        "reason": _reason(alignment, contradictions, aboutness, stance, verdict_label),
        "action_match": action_match["action_match"],
        "matched_actions": action_match["matched_actions"],
        "missing_actions": action_match["missing_actions"],
        "entity_overlap_score": entity_overlap["score"],
        "evidence_quote": _evidence_quote(evidence_text, alignment, action_match),
        "matched_entities": [
            item
            for item in context.get("claim_actors", []) + context.get("claim_locations", []) + context.get("claim_countries", [])
            if item not in missing_entities
        ],
        "missing_entities": missing_entities,
        "contradictions": contradictions,
    }


def classify_alignment(claim: str, evidence_text: str) -> str:
    return compare_claim_with_evidence(claim, evidence_text)["alignment"]


def classify_evidence(claim: str, evidence_text: str, is_fact_check: bool = False) -> dict:
    result = compare_claim_with_evidence(claim, evidence_text)
    if is_fact_check and result["aboutness"] == "exact_claim":
        direction = verdict_direction(result.get("verdict_extracted", "not_found"))
        if direction == "refutes":
            return {**result, "alignment": "refutes", "stance": "refutes", "relation": "refutes", "support_score": 0.0, "refute_score": 1.0}
        if direction == "supports":
            return {**result, "alignment": "supports", "stance": "supports", "relation": "supports", "support_score": 1.0, "refute_score": 0.0}
        if direction == "neutral":
            return {**result, "alignment": "mentions_only", "stance": "neutral", "relation": "not_verifiable", "support_score": 0.0, "refute_score": 0.0}
    return {
        **result,
        "relation": result["alignment"],
        "support_score": 1.0 if result["alignment"] == "supports" else 0.0,
        "refute_score": 1.0 if result["alignment"] == "refutes" else 0.0,
    }


def detect_contradictions(claim: str, evidence_text: str) -> list[str]:
    return compare_claim_with_evidence(claim, evidence_text)["contradictions"]


def _overlap(claim: str, evidence_text: str) -> float:
    claim_terms = {term for term in normalize_text(claim).split() if len(term) > 3}
    evidence_terms = {term for term in normalize_text(evidence_text).split() if len(term) > 3}
    if not claim_terms:
        return 0.0
    return len(claim_terms & evidence_terms) / max(1, len(claim_terms))


def _action_object_match(claim: str, evidence_text: str) -> dict:
    claim_text = normalize_text(claim)
    evidence = normalize_text(evidence_text)
    action_groups = {
        "invest": ("invest", "invested", "investment", "fund", "funded", "finance", "financed"),
        "announce": ("announce", "announced", "says", "said", "confirm", "confirmed"),
        "open": ("open", "opened", "launch", "launched", "start", "started"),
        "hospital": ("hospital", "clinic", "medical center", "healthcare"),
        "homeless": ("homeless", "unhoused", "shelter"),
        "sell": ("sell", "sold", "sale", "transfer", "ownership"),
        "sign_deal": ("sign", "signed", "deal", "agreement", "contract"),
        "run_for_office": ("run", "campaign", "president", "presidential", "election"),
        "renounce": ("renounce", "renounced", "renouncing", "citizenship"),
        "alter": ("altered", "fabricated", "fake", "ai-generated", "digitally"),
        "vegan": ("vegan", "not vegan", "isn't vegan", "is not vegan", "plant-based"),
    }
    claim_actions = [
        key
        for key, terms in action_groups.items()
        if any(re.search(rf"\b{re.escape(term)}\b", claim_text) for term in terms)
    ]
    if not claim_actions:
        return {"action_match": True, "matched_actions": [], "missing_actions": []}
    matched = [
        key
        for key in claim_actions
        if any(re.search(rf"\b{re.escape(term)}\b", evidence) for term in action_groups[key])
    ]
    money_terms = re.findall(r"(?:\$|usd\s*)\s?\d[\d,\.]*(?:\s?(?:million|billion|trillion|k))?|\b\d[\d,\.]*\s?(?:million|billion|trillion|k)\b", claim_text)
    money_match = True
    if money_terms:
        money_match = any(normalize_text(term) in evidence for term in money_terms)
    missing = [key for key in claim_actions if key not in matched]
    return {
        "action_match": bool(matched) and money_match,
        "matched_actions": matched,
        "missing_actions": missing + ([] if money_match else ["money_or_quantity"]),
    }


def _proper_entity_overlap(claim: str, evidence_text: str) -> dict:
    claim_names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", claim or "")
    evidence = normalize_text(evidence_text)
    important_tokens: set[str] = set()
    for name in claim_names:
        for token in normalize_text(name).split():
            if len(token) > 3 and token not in {"republicans", "texas", "senate"}:
                important_tokens.add(token)
    if not important_tokens:
        return {"score": 0.0, "matched": []}
    matched = sorted(token for token in important_tokens if re.search(rf"\b{re.escape(token)}\b", evidence))
    return {"score": len(matched) / max(1, len(important_tokens)), "matched": matched}


def _detect_aboutness(context: dict, overlap: float, entity_overlap: dict, action_match: dict, verdict: str) -> str:
    if not context.get("location_match_pass"):
        return "unrelated"
    entity_score = float(context.get("entity_match_score") or 0.0)
    event_score = float(context.get("event_match_score") or 0.0)
    same_context = bool(context.get("context_match_pass"))
    if same_context and action_match.get("action_match") and (overlap >= 0.55 or verdict in {"refutes", "supports", "neutral"}):
        return "exact_claim"
    if (entity_score > 0 or float(entity_overlap.get("score") or 0.0) > 0) and (event_score > 0 or overlap >= 0.18):
        return "same_entity_related_event"
    if overlap >= 0.25:
        return "same_topic"
    return "unrelated"


def _stance_from_alignment(alignment: str, verdict: str, aboutness: str) -> str:
    if aboutness == "exact_claim" and verdict in {"refutes", "supports", "neutral"}:
        return verdict
    if alignment == "supports":
        return "supports"
    if alignment == "refutes":
        return "refutes"
    if alignment in {"mentions_only", "irrelevant"}:
        return "neutral" if alignment == "mentions_only" else "unknown"
    return "unknown"


def _evidence_quote(evidence_text: str, alignment: str, action_match: dict) -> str:
    if alignment not in {"supports", "refutes"}:
        return ""
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", evidence_text or "") if sentence.strip()]
    matched_actions = set(action_match.get("matched_actions") or [])
    for sentence in sentences:
        normalized = normalize_text(sentence)
        if alignment == "refutes" and re.search(r"\b(false|fake|not true|no evidence|misleading|fabricated|incorrect)\b", normalized):
            return sentence[:300]
        if matched_actions and any(action in normalized for action in matched_actions):
            return sentence[:300]
    return sentences[0][:300] if sentences else ""


def _reason(alignment: str, contradictions: list[str], aboutness: str, stance: str, verdict_label: str) -> str:
    if alignment == "supports":
        return "The source matches the claim context and confirms the same claim direction."
    if alignment == "refutes":
        if aboutness == "exact_claim" and verdict_label not in {"", "not_found"}:
            return f"The source directly discusses the submitted claim and labels it {verdict_label}."
        return contradictions[0] if contradictions else "The source rejects the claim."
    if alignment == "mentions_only":
        if aboutness == "same_entity_related_event":
            return "This source is related to the same person or topic, but it does not directly verify the exact claim."
        return "The source shares keywords or background context but does not verify the exact claim."
    return "The source does not match the claim context closely enough."
