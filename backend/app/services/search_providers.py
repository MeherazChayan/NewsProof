from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import quote_plus

import httpx

from app.config import Settings
from app.utils import normalize_text


@dataclass
class ProviderSearchResult:
    provider: str
    results: list[dict]
    failure_reason: str | None


class SearchProvider(ABC):
    provider_name: str

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def is_configured(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, expected_domain: str | None = None, limit: int = 5) -> ProviderSearchResult:
        raise NotImplementedError

    def _request_timeout(self) -> httpx.Timeout:
        timeout_seconds = max(2.0, min(float(self.settings.request_timeout_seconds), 8.0))
        return httpx.Timeout(connect=min(3.0, timeout_seconds), read=timeout_seconds, write=timeout_seconds, pool=timeout_seconds)

    def _score_result(self, query: str, title: str, snippet: str, rank: int, expected_domain: str | None, domain: str | None) -> float:
        query_tokens = set(normalize_text(query).split())
        if not query_tokens:
            return max(0.1, 1.0 - (rank * 0.08))
        title_tokens = set(normalize_text(title).split())
        snippet_tokens = set(normalize_text(snippet).split())
        overlap = len(query_tokens & (title_tokens | snippet_tokens)) / max(1, len(query_tokens))
        rank_score = max(0.1, 1.0 - (rank * 0.08))
        domain_bonus = 0.08 if expected_domain and domain and (domain == expected_domain or domain.endswith(f".{expected_domain}")) else 0.0
        return round(min(1.0, (overlap * 0.7) + (rank_score * 0.3) + domain_bonus), 3)


class GoogleSearchProvider(SearchProvider):
    provider_name = "google_programmable_search"
    endpoint = "https://www.googleapis.com/customsearch/v1"

    def is_configured(self) -> bool:
        return bool(self.settings.google_programmable_search_api_key and self.settings.google_programmable_search_engine_id)

    def search(self, query: str, expected_domain: str | None = None, limit: int = 5) -> ProviderSearchResult:
        if not self.is_configured():
            return ProviderSearchResult(provider=self.provider_name, results=[], failure_reason="provider_not_configured")

        params = {
            "key": self.settings.google_programmable_search_api_key,
            "cx": self.settings.google_programmable_search_engine_id,
            "q": query,
            "num": max(1, min(limit, 10)),
            "safe": "off",
        }
        if expected_domain:
            params["siteSearch"] = expected_domain
            params["siteSearchFilter"] = "i"

        try:
            with httpx.Client(timeout=self._request_timeout(), follow_redirects=True) as client:
                response = client.get(self.endpoint, params=params)
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            return ProviderSearchResult(provider=self.provider_name, results=[], failure_reason=f"request_failed:{exc.__class__.__name__}")

        items = payload.get("items") or []
        if not items:
            return ProviderSearchResult(provider=self.provider_name, results=[], failure_reason="no_results")

        results: list[dict] = []
        for rank, item in enumerate(items[:limit]):
            url = str(item.get("link") or "").strip()
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            domain = self._extract_domain(url)
            if not url:
                continue
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "domain": domain,
                    "score": self._score_result(query, title, snippet, rank, expected_domain, domain),
                    "explanation": "Found through Google Programmable Search and ranked by result position plus query overlap.",
                }
            )
        if not results:
            return ProviderSearchResult(provider=self.provider_name, results=[], failure_reason="no_results")
        return ProviderSearchResult(provider=self.provider_name, results=results, failure_reason=None)

    def _extract_domain(self, url: str) -> str | None:
        try:
            return httpx.URL(url).host.lower() if httpx.URL(url).host else None
        except Exception:
            return None


class BraveSearchProvider(SearchProvider):
    provider_name = "brave"
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def is_configured(self) -> bool:
        return bool((self.settings.brave_search_api_key or "").strip())

    def search(self, query: str, expected_domain: str | None = None, limit: int = 5) -> ProviderSearchResult:
        if not self.is_configured():
            return ProviderSearchResult(provider=self.provider_name, results=[], failure_reason="provider_not_configured")

        q = query if not expected_domain else f"site:{expected_domain} {query}"
        url = f"{self.endpoint}?q={quote_plus(q)}&count={max(1, min(limit, 10))}&text_decorations=false"
        headers = {"X-Subscription-Token": str((self.settings.brave_search_api_key or "").strip()), "Accept": "application/json"}

        try:
            with httpx.Client(timeout=self._request_timeout(), headers=headers, follow_redirects=True) as client:
                response = client.get(url)
                if response.status_code in {401, 403}:
                    return ProviderSearchResult(provider=self.provider_name, results=[], failure_reason="provider_auth_failed")
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            return ProviderSearchResult(provider=self.provider_name, results=[], failure_reason=f"request_failed:{exc.__class__.__name__}")

        items = ((payload.get("web") or {}).get("results")) or []
        if not items:
            return ProviderSearchResult(provider=self.provider_name, results=[], failure_reason="no_results")

        results: list[dict] = []
        for rank, item in enumerate(items[:limit]):
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("description") or item.get("snippet") or "").strip()
            domain = self._extract_domain(url)
            if not url:
                continue
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "domain": domain,
                    "provider": self.provider_name,
                    "score": self._score_result(query, title, snippet, rank, expected_domain, domain),
                    "explanation": "Found through Brave Search API and ranked by result position plus query overlap.",
                }
            )
        if not results:
            return ProviderSearchResult(provider=self.provider_name, results=[], failure_reason="no_results")
        return ProviderSearchResult(provider=self.provider_name, results=results, failure_reason=None)

    def _extract_domain(self, url: str) -> str | None:
        try:
            return httpx.URL(url).host.lower() if httpx.URL(url).host else None
        except Exception:
            return None
