from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings

try:
    import cohere
except Exception:
    cohere = None


@dataclass
class SimilarityCandidate:
    title: str | None
    url: str | None
    score: float
    explanation: str
    metadata: dict


class CohereService:
    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.cohere_api_key
        self.client = cohere.ClientV2(self.api_key) if settings.enable_external_rerank and self.api_key and cohere else None

    def rank_candidates(self, query: str, documents: list[dict]) -> list[SimilarityCandidate]:
        if not documents:
            return []
        if not self.client:
            return self._fallback_rank(query, documents)

        rerank_response = self.client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=[doc["text"] for doc in documents],
            top_n=min(5, len(documents)),
        )
        ranked: list[SimilarityCandidate] = []
        for item in rerank_response.results:
            source = documents[item.index]
            ranked.append(
                SimilarityCandidate(
                    title=source.get("title"),
                    url=source.get("url"),
                    score=float(item.relevance_score),
                    explanation="Ranked with Cohere Rerank.",
                    metadata=source,
                )
            )
        return ranked

    def _fallback_rank(self, query: str, documents: list[dict]) -> list[SimilarityCandidate]:
        query_tokens = set(query.split())
        ranked: list[SimilarityCandidate] = []
        for doc in documents[:5]:
            doc_tokens = set(doc["text"].split())
            shared_tokens = len(query_tokens & doc_tokens)
            overlap_score = shared_tokens / max(1, len(query_tokens))
            title_tokens = set(str(doc.get("title") or "").lower().split())
            title_score = len(query_tokens & title_tokens) / max(1, len(title_tokens)) if title_tokens else 0.0
            score = min(1.0, (overlap_score * 0.8) + (title_score * 0.2))
            ranked.append(
                SimilarityCandidate(
                    title=doc.get("title"),
                    url=doc.get("url"),
                    score=score,
                    explanation=f"Fallback lexical ranking found {shared_tokens} overlapping tokens because Cohere is unavailable.",
                    metadata=doc,
                )
            )
        return sorted(ranked, key=lambda item: item.score, reverse=True)
