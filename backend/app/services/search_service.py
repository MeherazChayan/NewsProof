from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

from app.config import get_settings
from app.services.search_providers import BraveSearchProvider, SearchProvider


@dataclass
class WebSearchCandidate:
    title: str
    url: str
    snippet: str
    domain: str | None
    score: float
    explanation: str


class SearchService:
    MAX_QUERY_VARIANTS = 1

    def __init__(self) -> None:
        self.settings = get_settings()
        self.providers: list[SearchProvider] = [
            BraveSearchProvider(self.settings),
        ]

    def search(self, query: str, expected_domain: str | None = None, limit: int = 5) -> list[WebSearchCandidate]:
        return self.search_with_debug(query, expected_domain=expected_domain, limit=limit)["results"]

    def search_with_debug(self, query: str, expected_domain: str | None = None, limit: int = 5) -> dict:
        if not query.strip():
            return {
                "results": [],
                "failure_reason": "empty_query",
                "provider_attempts": [],
                "providers_used": [],
                "provider_status": {"brave": "not_called"},
                "stopped_by_budget": False,
            }

        brave_provider = self.providers[0]
        if not brave_provider.is_configured():
            return {
                "results": [],
                "failure_reason": "provider_not_configured",
                "provider_attempts": [
                    {
                        "provider": brave_provider.provider_name,
                        "query": query,
                        "result_count": 0,
                        "failure_reason": "provider_not_configured",
                        "returned_urls": [],
                        "returned_titles": [],
                    }
                ],
                "providers_used": [],
                "provider_status": {"brave": "missing"},
                "stopped_by_budget": False,
            }

        combined: list[WebSearchCandidate] = []
        provider_attempts: list[dict] = []
        seen_urls: set[str] = set()

        for query_variant in self._query_variants(query, expected_domain)[: self.MAX_QUERY_VARIANTS]:
            variant_results: list[WebSearchCandidate] = []
            response = brave_provider.search(query_variant, expected_domain=expected_domain, limit=limit)
            results = [
                WebSearchCandidate(
                    title=str(item.get("title") or ""),
                    url=str(item.get("url") or ""),
                    snippet=str(item.get("snippet") or ""),
                    domain=item.get("domain"),
                    score=float(item.get("score") or 0.0),
                    explanation=str(item.get("explanation") or ""),
                )
                for item in response.results
                if str(item.get("url") or "").strip()
            ]
            provider_attempts.append(
                {
                    "provider": response.provider,
                    "query": query_variant,
                    "result_count": len(results),
                    "failure_reason": response.failure_reason,
                    "returned_urls": [item.url for item in results[:limit]],
                    "returned_titles": [item.title for item in results[:limit]],
                }
            )
            if results:
                variant_results = results
            else:
                if response.failure_reason in {"provider_not_configured", "provider_auth_failed"}:
                    failure_reason = response.failure_reason
                    return {
                        "results": [],
                        "failure_reason": failure_reason,
                        "provider_attempts": provider_attempts,
                        "providers_used": [],
                        "provider_status": {"brave": failure_reason},
                        "stopped_by_budget": False,
                    }
            for candidate in variant_results:
                if candidate.url in seen_urls:
                    continue
                seen_urls.add(candidate.url)
                combined.append(candidate)
                if len(combined) >= limit:
                    break
            if combined:
                break

        failure_reason = None
        if not combined:
            failures = [str(item.get("failure_reason") or "") for item in provider_attempts if item.get("failure_reason")]
            failure_reason = " | ".join(item for item in failures if item) or "no_results_returned"

        results = self._diversify_results(combined, expected_domain=expected_domain, limit=limit)
        providers_used = list(dict.fromkeys(str(item.get("provider")) for item in provider_attempts if int(item.get("result_count") or 0) > 0))
        return {
            "results": results,
            "failure_reason": failure_reason,
            "provider_attempts": provider_attempts,
            "providers_used": providers_used,
            "provider_status": {"brave": "configured" if providers_used else ("no_results" if failure_reason == "no_results_returned" else failure_reason or "configured")},
            "stopped_by_budget": False,
        }

    def _query_variants(self, query: str, expected_domain: str | None) -> list[str]:
        variants: list[str] = []
        trimmed = " ".join(query.split())
        if trimmed:
            variants.append(trimmed)
            if self._is_exact_phrase_query(trimmed):
                unquoted = trimmed.strip('"').strip()
                if unquoted:
                    variants.append(unquoted)
            elif expected_domain and not trimmed.startswith("site:"):
                variants.append(f"site:{expected_domain} {trimmed}")

        deduped: list[str] = []
        seen = set()
        for variant in variants:
            key = variant.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(variant)
        return deduped

    def _is_exact_phrase_query(self, query: str) -> bool:
        cleaned = (query or "").strip()
        return len(cleaned) >= 2 and cleaned.startswith('"') and cleaned.endswith('"')

    def _diversify_results(
        self,
        candidates: list[WebSearchCandidate],
        expected_domain: str | None,
        limit: int,
    ) -> list[WebSearchCandidate]:
        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        if not ranked:
            return []

        selected: list[WebSearchCandidate] = []
        used_urls: set[str] = set()
        used_domains: dict[str, int] = {}

        for prefer_external in (True, False):
            for candidate in ranked:
                if candidate.url in used_urls:
                    continue
                domain = candidate.domain or ""
                same_domain = bool(expected_domain and domain and (domain == expected_domain or domain.endswith(f".{expected_domain}")))
                if prefer_external and same_domain:
                    continue
                if not prefer_external and not same_domain:
                    continue
                if domain and used_domains.get(domain, 0) >= 2:
                    continue
                selected.append(candidate)
                used_urls.add(candidate.url)
                if domain:
                    used_domains[domain] = used_domains.get(domain, 0) + 1
                if len(selected) >= limit:
                    return selected

        for candidate in ranked:
            if candidate.url in used_urls:
                continue
            domain = candidate.domain or ""
            if domain and used_domains.get(domain, 0) >= 3:
                continue
            selected.append(candidate)
            used_urls.add(candidate.url)
            if domain:
                used_domains[domain] = used_domains.get(domain, 0) + 1
            if len(selected) >= limit:
                break
        return selected

    def _resolve_result_url(self, href: str) -> str:
        parsed = urlparse(href)
        if "duckduckgo.com" not in parsed.netloc:
            return href
        query = parse_qs(parsed.query)
        if "uddg" in query and query["uddg"]:
            return unquote(query["uddg"][0])
        return href
