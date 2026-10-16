from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "trusted_sources.json"


def _host_and_path(value: str) -> tuple[str, str]:
    raw = (value or "").strip().lower()
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = parsed.netloc or parsed.path.split("/")[0]
    path = parsed.path if parsed.netloc else "/" + "/".join(parsed.path.split("/")[1:])
    if host.startswith("www."):
        host = host[4:]
    return host, path.rstrip("/")


@lru_cache(maxsize=1)
def load_trusted_sources() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _entry(domain: str, source_type: str, priority: int = 2, country: str | None = None) -> dict:
    return {"domain": domain, "type": source_type, "priority": priority, "country": country}


def get_sources_for_country(country: str | None) -> list[dict]:
    registry = load_trusted_sources()
    sources: list[dict] = []
    country_key = next((key for key in registry.get("countries", {}) if key.lower() == (country or "").lower()), None)
    if country_key:
        country_data = registry["countries"][country_key]
        sources.extend(_entry(domain, "news", 2, country_key) for domain in country_data.get("news", []))
        sources.extend(_entry(domain, "fact_check", 1, country_key) for domain in country_data.get("fact_check", []))
    elif country is None:
        for key, country_data in registry.get("countries", {}).items():
            sources.extend(_entry(domain, "news", 2, key) for domain in country_data.get("news", []))
            sources.extend(_entry(domain, "fact_check", 1, key) for domain in country_data.get("fact_check", []))
    sources.extend(registry.get("global_news_agencies", []))
    sources.extend(registry.get("global_fact_checkers", []))

    deduped: list[dict] = []
    seen: set[str] = set()
    for source in sources:
        key = str(source.get("domain", "")).lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(source)
    return deduped


def is_trusted_domain(domain_or_url: str) -> bool:
    return get_source_type(domain_or_url) in {"fact_check_page", "trusted_news_page", "official_source_page"}


def is_fact_check_domain(domain_or_url: str) -> bool:
    return get_source_type(domain_or_url) == "fact_check_page"


def get_source_type(domain_or_url: str) -> str:
    host, path = _host_and_path(domain_or_url)
    if not host:
        return "unknown_or_untrusted_page"
    if host.endswith(".gov") or ".gov." in host or host.endswith("gov.bd"):
        return "official_source_page"
    if any(host == social or host.endswith(f".{social}") for social in ("facebook.com", "x.com", "twitter.com", "youtube.com", "tiktok.com", "instagram.com")):
        return "social_media_page"

    for source in get_sources_for_country(None):
        source_host, source_path = _host_and_path(str(source.get("domain") or ""))
        if not source_host:
            continue
        host_matches = host == source_host or host.endswith(f".{source_host}")
        path_matches = not source_path or source_path == "/" or path.startswith(source_path)
        if host_matches and path_matches:
            if str(source.get("type")) == "fact_check":
                return "fact_check_page"
            return "trusted_news_page"
    return "unknown_or_untrusted_page"
