from __future__ import annotations

from collections import Counter
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.services.cohere_service import CohereService
from app.services.embeddings_service import EmbeddingsService
from app.services.evidence_store_service import EvidenceStoreService
from app.services.search_service import SearchService
from app.utils import normalize_text


class RetrievalService:
    """Hybrid retrieval: lexical + semantic + rerank + passage selection."""

    MAX_SEARCH_RESULTS = 3
    MAX_FULL_PAGE_FETCHES = 1
    MAX_CHUNKS_PER_PAGE = 2

    def __init__(
        self,
        search_service: SearchService,
        embeddings_service: EmbeddingsService,
        cohere_service: CohereService,
        evidence_store: EvidenceStoreService,
    ) -> None:
        self.search = search_service
        self.embeddings = embeddings_service
        self.cohere = cohere_service
        self.store = evidence_store

    def chunk_document(self, text: str, chunk_size: int = 90, overlap: int = 20) -> list[str]:
        tokens = [token for token in text.split() if token]
        if not tokens:
            return []
        chunks: list[str] = []
        step = max(1, chunk_size - overlap)
        for index in range(0, len(tokens), step):
            chunk = " ".join(tokens[index : index + chunk_size]).strip()
            if len(chunk.split()) >= 25:
                chunks.append(chunk)
            if len(chunks) >= 12:
                break
        return chunks

    def retrieve_lexical(self, query: str, limit: int = 8) -> list[dict]:
        passages = self._collect_passages(query, limit=max(3, min(limit, 5)))
        return sorted(passages, key=lambda item: item["lexical_score"], reverse=True)[:limit]

    def retrieve_semantic(self, query: str, limit: int = 8) -> list[dict]:
        passages = self._collect_passages(query, limit=max(3, min(limit, 5)))
        return sorted(passages, key=lambda item: item["semantic_score"], reverse=True)[:limit]

    def retrieve_hybrid(self, query: str, limit: int = 8) -> list[dict]:
        cache_key = f"hybrid::{normalize_text(query)}::{limit}"
        cached = self.store.get(cache_key)
        if cached is not None:
            return cached

        working_limit = max(3, min(limit, 5))
        merged = self._collect_passages(query, limit=working_limit)
        reranked = self._rerank(query, merged, limit=limit)
        return self.store.put(cache_key, reranked[:limit])

    def fetch_page_text(self, url: str | None) -> str:
        return self._fetch_page_text(url)

    def fetch_page_document(self, url: str | None) -> dict:
        if not url:
            return {
                "url": None,
                "fetch_status": "fail",
                "content_type": None,
                "title": None,
                "extracted_text": "",
                "extracted_text_length": 0,
                "extract_status": "missing_url",
            }
        cache_key = f"page_document::{url}"
        cached = self.store.get(cache_key)
        if cached is not None:
            return cached[0] if cached else {
                "url": url,
                "fetch_status": "fail",
                "content_type": None,
                "title": None,
                "extracted_text": "",
                "extracted_text_length": 0,
                "extract_status": "cached_failure",
            }

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            )
        }
        try:
            with httpx.Client(timeout=2.4, headers=headers, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
        except Exception:
            document = {
                "url": url,
                "fetch_status": "fail",
                "content_type": None,
                "title": None,
                "extracted_text": "",
                "extracted_text_length": 0,
                "extract_status": "fetch_failed",
            }
            self.store.put(cache_key, [document])
            return document

        content_type = str(response.headers.get("content-type") or "").lower()
        if "html" not in content_type:
            document = {
                "url": url,
                "fetch_status": "success",
                "content_type": content_type,
                "title": None,
                "extracted_text": "",
                "extracted_text_length": 0,
                "extract_status": "non_html_response",
            }
            self.store.put(cache_key, [document])
            return document

        soup = self._parse_html(response.text)
        if soup is None:
            document = {
                "url": url,
                "fetch_status": "success",
                "content_type": content_type,
                "title": None,
                "extracted_text": "",
                "extracted_text_length": 0,
                "extract_status": "parse_failed",
            }
            self.store.put(cache_key, [document])
            return document

        title = soup.title.get_text(" ", strip=True) if soup.title else None
        segments = [
            " ".join(node.get_text(" ", strip=True) for node in soup.select("article p")),
            " ".join(node.get_text(" ", strip=True) for node in soup.select("main p")),
            " ".join(node.get_text(" ", strip=True) for node in soup.select("[itemprop='articleBody'] p")),
            " ".join(node.get_text(" ", strip=True) for node in soup.select("p")),
        ]
        extracted_text = ""
        for segment in segments:
            normalized = " ".join((segment or "").split())
            if len(normalized.split()) >= 40:
                extracted_text = normalized
                break
        if not extracted_text:
            description = ""
            for meta in soup.find_all("meta"):
                key = str(meta.get("property") or meta.get("name") or "").lower()
                if key in {"og:description", "description", "twitter:description"}:
                    description = str(meta.get("content") or "").strip()
                    if description:
                        break
            extracted_text = description

        document = {
            "url": url,
            "fetch_status": "success",
            "content_type": content_type,
            "title": title,
            "extracted_text": extracted_text,
            "extracted_text_length": len(extracted_text),
            "extract_status": "extracted" if extracted_text else "extraction_empty",
        }
        self.store.put(cache_key, [document])
        return document

    def _collect_passages(self, query: str, limit: int) -> list[dict]:
        results = self.search.search(query, expected_domain=None, limit=min(limit, self.MAX_SEARCH_RESULTS))
        query_embedding = self.embeddings.embed_query(query)
        query_terms = Counter(normalize_text(query).split())
        passages: list[dict] = []

        for index, result in enumerate(results):
            page_text = self._fetch_page_text(result.url) if index < self.MAX_FULL_PAGE_FETCHES else ""
            chunks = self.chunk_document(page_text) if page_text else []
            if not chunks:
                chunks = [f"{result.title or ''}. {result.snippet or ''}".strip()]

            for chunk in chunks[: self.MAX_CHUNKS_PER_PAGE]:
                lexical_score = self._lexical_score(query_terms, chunk)
                semantic_score = self.embeddings.similarity(query_embedding, self.embeddings.embed_query(chunk))
                passages.append(
                    {
                        "title": result.title,
                        "url": result.url,
                        "domain": result.domain or urlparse(result.url or "").netloc.lower(),
                        "passage": chunk,
                        "snippet": result.snippet,
                        "retrieval_score": round(max(lexical_score, semantic_score), 3),
                        "lexical_score": round(lexical_score, 3),
                        "semantic_score": round(semantic_score, 3),
                        "source": "full_page_passage" if page_text else "search_snippet_fallback",
                    }
                )

        return sorted(passages, key=lambda item: item["retrieval_score"], reverse=True)[: max(limit, 4)]

    def _rerank(self, query: str, passages: list[dict], limit: int) -> list[dict]:
        if not passages:
            return []
        documents = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "text": " ".join(filter(None, [str(item.get("title") or ""), str(item.get("passage") or "")])),
                "passage": item.get("passage"),
                "domain": item.get("domain"),
                "source": item.get("source"),
                "snippet": item.get("snippet"),
                "lexical_score": item.get("lexical_score"),
                "semantic_score": item.get("semantic_score"),
            }
            for item in passages[:12]
        ]
        ranked = self.cohere.rank_candidates(query, documents)
        output: list[dict] = []
        for item in ranked[:limit]:
            output.append(
                {
                    "title": item.title,
                    "url": item.url,
                    "domain": item.metadata.get("domain"),
                    "passage": item.metadata.get("passage"),
                    "snippet": item.metadata.get("snippet"),
                    "score": round(float(item.score), 3),
                    "lexical_score": item.metadata.get("lexical_score"),
                    "semantic_score": item.metadata.get("semantic_score"),
                    "source": item.metadata.get("source"),
                    "explanation": item.explanation,
                }
            )
        return output

    def _fetch_page_text(self, url: str | None) -> str:
        document = self.fetch_page_document(url)
        return str(document.get("extracted_text") or "")

    def _parse_html(self, html: str) -> BeautifulSoup | None:
        if not html:
            return None
        for parser in ("lxml", "html.parser"):
            try:
                return BeautifulSoup(html, parser)
            except Exception:
                continue
        return None

    def _lexical_score(self, query_terms: Counter[str], passage: str) -> float:
        passage_terms = Counter(normalize_text(passage).split())
        if not query_terms or not passage_terms:
            return 0.0
        overlap = sum(min(query_terms[term], passage_terms.get(term, 0)) for term in query_terms)
        return overlap / max(1, sum(query_terms.values()))
