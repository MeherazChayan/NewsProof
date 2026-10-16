from __future__ import annotations

import re

from app.services.context_match_service import ContextMatchService


COUNTRY_NAMES = {
    "bangladesh": "Bangladesh",
    "india": "India",
    "united states": "United States",
    "china": "China",
    "south africa": "South Africa",
    "united kingdom": "United Kingdom",
    "australia": "Australia",
    "canada": "Canada",
    "pakistan": "Pakistan",
    "sri lanka": "Sri Lanka",
    "malaysia": "Malaysia",
}


def extract_claim_context(text: str) -> dict:
    matcher = ContextMatchService()
    terms = matcher.extract_context_terms(text or "")
    country = detect_country(text)
    entities = extract_entities(text)
    return {
        "main_claim": " ".join((text or "").split()).strip(),
        "country": country,
        "location": terms.get("locations", [None])[0] if terms.get("locations") else None,
        "entities": entities,
        "person_names": re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text or ""),
        "organisation_names": [item for item in entities if any(word in item.lower() for word in ("government", "company", "conglomerate", "agency", "party"))],
        "date_or_time": re.findall(r"\b(?:20\d{2}|19\d{2})\b|today|yesterday|tomorrow", text or "", flags=re.IGNORECASE),
        "event_type": terms.get("events", []),
        "topic": " / ".join(terms.get("events", [])[:3]) or "factual claim",
        "language": "bn" if re.search(r"[\u0980-\u09ff]", text or "") else "en",
        "claim_polarity": "negative_or_refuting" if re.search(r"\b(no|not|never|false|fake|did not|has not)\b", text or "", re.I) else "affirming",
    }


def detect_country(text: str) -> str | None:
    terms = ContextMatchService().extract_context_terms(text or "")
    countries = terms.get("countries", [])
    if countries:
        return COUNTRY_NAMES.get(countries[0], countries[0].title())
    for location in terms.get("locations", []):
        country = ContextMatchService().country_for_location(location)
        if country:
            return COUNTRY_NAMES.get(country, country.title())
    return None


def extract_entities(text: str) -> list[str]:
    matcher = ContextMatchService()
    terms = matcher.extract_context_terms(text or "")
    entities = [*terms.get("actors", []), *terms.get("locations", []), *terms.get("countries", [])]
    entities.extend(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text or ""))
    deduped: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        key = entity.lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(entity)
    return deduped
