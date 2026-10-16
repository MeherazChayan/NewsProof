from __future__ import annotations

import re

from app.services.trusted_sources import get_sources_for_country


def build_country_aware_queries(claim_context: dict, limit: int = 8) -> list[str]:
    claim = " ".join(str(claim_context.get("main_claim") or "").split())
    claim_terms = _search_terms(claim)
    sources = get_sources_for_country(claim_context.get("country"))
    queries: list[str] = []
    for source in sources:
        domain = str(source.get("domain") or "").strip()
        if domain and claim_terms:
            queries.append(f"site:{domain} {claim_terms}")
        if len(queries) >= max(1, limit - 4):
            break
    quoted = f"\"{claim_terms}\"" if claim_terms else ""
    queries.extend(
        query
        for query in (
            f"{quoted} false",
            f"{quoted} fake",
            f"{quoted} fact check",
            f"{claim_terms} fact check",
        )
        if query.strip()
    )
    return _dedupe(queries)[:limit]


def search_trusted_sources_only(claim_context: dict) -> list[dict]:
    return []


def rank_evidence_pages(results: list[dict]) -> list[dict]:
    return sorted(results, key=lambda item: float(item.get("score") or item.get("rerank_score") or 0.0), reverse=True)


def _search_terms(text: str) -> str:
    tokens = re.findall(r"\$?\b[\w.-]+\b", text or "")
    useful = [token for token in tokens if len(token) > 2 or token.startswith("$")]
    return " ".join(useful[:14])


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = " ".join(item.split())
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            deduped.append(cleaned)
    return deduped
