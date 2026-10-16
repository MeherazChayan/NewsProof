from __future__ import annotations

import re

from app.utils import normalize_text


class ReasoningService:
    """Claim-vs-evidence reasoning over retrieved passages."""

    def verify_claim_with_evidence(self, claim: str, passages: list[dict]) -> dict:
        if not claim.strip():
            return self._empty_result(claim, "insufficient_evidence", "No checkable claim was available.")
        if not passages:
            return self._empty_result(claim, "insufficient_evidence", "No relevant evidence passages were retrieved.")

        best = passages[0]
        support_score = max(self._support_score(claim, passage) for passage in passages[:5])
        reliability = self._source_reliability(passages[:5])
        agreement = self._agreement_score(passages[:5])
        directness = self._directness_score(claim, passages[:5])
        contradiction = self._contradiction_score(claim, passages[:5])
        confidence = max(0.0, min(1.0, 0.45 * support_score + 0.20 * reliability + 0.20 * agreement + 0.15 * directness))

        if contradiction >= 0.6 and support_score < 0.45:
            verdict = "contradicted"
            rationale = "The strongest evidence passages conflict with key claim details."
        elif support_score >= 0.65:
            verdict = "supported"
            rationale = "The strongest evidence passages directly support the claim."
        else:
            verdict = "insufficient_evidence"
            rationale = "The retrieved passages are relevant, but not strong enough for a firm conclusion."

        return {
            "claim_text": claim,
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "rationale": rationale,
            "support_score": round(support_score, 3),
            "source_reliability": round(reliability, 3),
            "agreement_score": round(agreement, 3),
            "directness_score": round(directness, 3),
            "contradiction_score": round(contradiction, 3),
            "top_evidence_passages": passages[:3],
            "evidence_source_urls": [item.get("url") for item in passages[:3] if item.get("url")],
        }

    def aggregate_claim_results(self, results: list[dict]) -> dict:
        if not results:
            return {
                "verdict": "insufficient_evidence",
                "confidence": 0.0,
                "supported_count": 0,
                "contradicted_count": 0,
                "insufficient_count": 0,
            }
        supported = sum(1 for item in results if item.get("verdict") == "supported")
        contradicted = sum(1 for item in results if item.get("verdict") == "contradicted")
        insufficient = sum(1 for item in results if item.get("verdict") == "insufficient_evidence")
        average_confidence = sum(float(item.get("confidence") or 0.0) for item in results) / len(results)

        if contradicted and contradicted >= supported:
            verdict = "contradicted"
        elif supported:
            verdict = "supported"
        else:
            verdict = "insufficient_evidence"

        return {
            "verdict": verdict,
            "confidence": round(average_confidence, 3),
            "supported_count": supported,
            "contradicted_count": contradicted,
            "insufficient_count": insufficient,
        }

    def _empty_result(self, claim: str, verdict: str, rationale: str) -> dict:
        return {
            "claim_text": claim,
            "verdict": verdict,
            "confidence": 0.0,
            "rationale": rationale,
            "support_score": 0.0,
            "source_reliability": 0.0,
            "agreement_score": 0.0,
            "directness_score": 0.0,
            "contradiction_score": 0.0,
            "top_evidence_passages": [],
            "evidence_source_urls": [],
        }

    def _support_score(self, claim: str, passage: dict) -> float:
        text = " ".join(filter(None, [str(passage.get("title") or ""), str(passage.get("passage") or ""), str(passage.get("snippet") or "")]))
        return self._overlap(claim, text)

    def _directness_score(self, claim: str, passages: list[dict]) -> float:
        return max((self._overlap(claim, str(item.get("title") or "")) + self._overlap(claim, str(item.get("passage") or ""))) / 2 for item in passages)

    def _agreement_score(self, passages: list[dict]) -> float:
        domains = {str(item.get("domain") or "") for item in passages if item.get("domain")}
        if len(domains) >= 4:
            return 1.0
        if len(domains) == 3:
            return 0.85
        if len(domains) == 2:
            return 0.7
        if len(domains) == 1:
            return 0.45
        return 0.0

    def _source_reliability(self, passages: list[dict]) -> float:
        known = ("reuters", "apnews", "bbc", "nytimes", "washingtonpost", "theguardian", "aljazeera", "cnn")
        scores = []
        for item in passages:
            domain = str(item.get("domain") or "")
            if any(name in domain for name in known):
                scores.append(0.85)
            elif domain:
                scores.append(0.65)
        return sum(scores) / len(scores) if scores else 0.0

    def _contradiction_score(self, claim: str, passages: list[dict]) -> float:
        claim_numbers = set(re.findall(r"\b\d[\d,\.]*\b", claim))
        if not claim_numbers:
            return 0.0
        contradiction = 0.0
        for item in passages:
            passage_text = " ".join(filter(None, [str(item.get("title") or ""), str(item.get("passage") or "")]))
            passage_numbers = set(re.findall(r"\b\d[\d,\.]*\b", passage_text))
            if passage_numbers and claim_numbers and not (claim_numbers & passage_numbers):
                contradiction = max(contradiction, 0.7)
        return contradiction

    def _overlap(self, left: str, right: str) -> float:
        left_terms = set(normalize_text(left).split())
        right_terms = set(normalize_text(right).split())
        if not left_terms or not right_terms:
            return 0.0
        return len(left_terms & right_terms) / max(1, len(left_terms))
