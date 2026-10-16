from __future__ import annotations

import re

from app.services.trusted_sources import is_fact_check_domain
from app.utils import normalize_text


REFUTING = (
    "false", "fake", "not true", "misleading", "fabricated", "altered", "ai-generated",
    "digitally altered", "no evidence", "hoax", "scam", "incorrect", "satire", "pants on fire"
)
SUPPORTING = ("mostly true", "true", "correct", "accurate", "verified")
NEUTRAL = ("unproven", "not enough evidence", "not verifiable", "missing context", "partly true", "needs context")


def extract_fact_check_verdict(page_text: str) -> str:
    label = extract_fact_check_verdict_label(page_text)
    return verdict_direction(label)


def extract_fact_check_verdict_label(page_text: str) -> str:
    text = normalize_text(page_text or "")
    if not text:
        return "not_found"
    verdict_windows = re.findall(r"(?:rating|verdict|claim|conclusion|our ruling)\s*[:\-]?\s*.{0,160}", text)
    haystacks = verdict_windows or [text[:1200]]
    for haystack in haystacks:
        verdict = normalize_fact_check_verdict_label(haystack)
        if verdict != "not_found":
            return verdict
    return "not_found"


def normalize_fact_check_verdict(raw_verdict: str) -> str:
    return verdict_direction(normalize_fact_check_verdict_label(raw_verdict))


def normalize_fact_check_verdict_label(raw_verdict: str) -> str:
    text = normalize_text(raw_verdict or "")
    if not text:
        return "not_found"
    for term in REFUTING:
        if re.search(rf"\b{re.escape(term)}\b", text):
            return term
    for term in NEUTRAL:
        if re.search(rf"\b{re.escape(term)}\b", text):
            return term
    for term in SUPPORTING:
        if re.search(rf"\b{re.escape(term)}\b", text):
            return term
    return "not_found"


def verdict_direction(verdict_label: str) -> str:
    label = normalize_text(verdict_label or "")
    if label in REFUTING:
        return "refutes"
    if label in SUPPORTING:
        return "supports"
    if label in NEUTRAL:
        return "neutral"
    return "not_found"


def is_fact_check_page(domain_or_url: str) -> bool:
    return is_fact_check_domain(domain_or_url)
