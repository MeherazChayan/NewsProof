from __future__ import annotations


SUPPORTING_TYPES = {"fact_check_page", "trusted_news_page", "official_source_page"}


def score_evidence(evidence_items: list[dict]) -> dict:
    support_score = 0
    refute_score = 0
    scored_items: list[dict] = []
    for item in evidence_items:
        source_type = str(item.get("source_type") or "unknown_or_untrusted_page")
        alignment = str(item.get("alignment") or item.get("evidence_relation") or item.get("relation") or "irrelevant")
        stance = str(item.get("stance") or "")
        verdict = str(item.get("verdict_direction") or item.get("fact_check_verdict") or "")
        if not verdict:
            verdict_label = str(item.get("verdict_extracted") or "")
            verdict = "refutes" if verdict_label in {"false", "fake", "not true", "misleading", "fabricated", "altered", "ai-generated", "digitally altered", "no evidence", "hoax", "scam", "incorrect", "satire"} else "supports" if verdict_label in {"true", "mostly true", "correct", "accurate", "verified"} else "neutral" if verdict_label in {"unproven", "not enough evidence", "not verifiable", "missing context", "partly true", "needs context"} else "not_found"
        score = 0
        if source_type == "fact_check_page" and (verdict == "refutes" or alignment in {"refutes", "contradicts"} or stance == "refutes"):
            score = -5
        elif source_type == "fact_check_page" and (verdict == "supports" or alignment == "supports" or stance == "supports"):
            score = 5
        elif source_type == "official_source_page" and alignment == "supports":
            score = 4
        elif source_type == "official_source_page" and alignment in {"refutes", "contradicts"}:
            score = -4
        elif source_type == "trusted_news_page" and alignment == "supports":
            score = 2
        elif source_type == "trusted_news_page" and (alignment in {"refutes", "contradicts"} or stance == "refutes"):
            score = -2
        if score > 0:
            support_score += score
        elif score < 0:
            refute_score += score
        scored_items.append({**item, "score": score})
    return {
        "support_score": support_score,
        "refute_score": refute_score,
        "items": scored_items,
    }


def decide_final_verdict(scores: dict) -> dict:
    support = int(scores.get("support_score") or 0)
    refute = int(scores.get("refute_score") or 0)
    items = scores.get("items", [])
    trusted_news_support_domains = {
        str(item.get("domain") or item.get("url") or "")
        for item in items
        if item.get("source_type") == "trusted_news_page" and item.get("alignment") == "supports"
    }
    if refute <= -4:
        return {"verdict": "Unverified", "reason": "Trusted evidence refutes the claim."}
    if support >= 5 and refute == 0 and any(item.get("source_type") == "fact_check_page" and item.get("alignment") == "supports" for item in items):
        return {"verdict": "Verified", "reason": "A trusted fact-checker supports the exact claim."}
    if support >= 4 and refute == 0 and any(item.get("source_type") == "official_source_page" and item.get("alignment") == "supports" for item in items):
        return {"verdict": "Verified", "reason": "An official source supports the exact claim."}
    if support >= 4 and refute == 0 and len(trusted_news_support_domains) >= 2:
        return {"verdict": "Verified", "reason": "At least two independent trusted news sources support the exact claim."}
    if support >= 2 and refute == 0:
        return {"verdict": "Likely Verified", "reason": "One trusted source supports the exact claim, with no trusted refutation."}
    return {"verdict": "Not Verifiable", "reason": "No trusted source supports the exact claim."}


def generate_explanation(evidence_items: list[dict], final_verdict: str) -> str:
    if final_verdict == "Unverified":
        return "Trusted evidence refutes the submitted claim."
    if final_verdict in {"Verified", "Likely Verified"}:
        return "Trusted evidence supports the submitted claim rather than merely mentioning related keywords."
    return "The retrieved pages did not provide enough trusted, exact claim support."
