from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.services.cohere_service import CohereService

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


class TextRerankerService:
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, cohere_service: CohereService) -> None:
        self.cohere = cohere_service
        self.settings = get_settings()
        self._cross_encoder: Any | None = None
        self._load_error: str | None = None

    def rerank(self, query: str, documents: list[dict], limit: int = 5) -> tuple[list[dict], str | None]:
        if not documents:
            return [], "no_documents_to_rerank"

        if self.cohere.client:
            ranked = self.cohere.rank_candidates(
                query,
                [
                    {
                        "title": document.get("title"),
                        "url": document.get("url"),
                        "text": str(document.get("text") or ""),
                        "passage": document.get("passage"),
                        "domain": document.get("domain"),
                        "source": document.get("source"),
                        "snippet": document.get("snippet"),
                        "bm25_score": document.get("bm25_score"),
                        "semantic_score": document.get("semantic_score"),
                        "hybrid_score": document.get("hybrid_score"),
                    }
                    for document in documents
                ],
            )
            output = []
            for item in ranked[:limit]:
                output.append(
                    {
                        **item.metadata,
                        "title": item.title,
                        "url": item.url,
                        "rerank_score": round(float(item.score), 3),
                        "rerank_explanation": item.explanation,
                    }
                )
            return output, None

        model = self._get_model() if self.settings.enable_local_ml_rerank else None
        if model is None:
            fallback = []
            for document in sorted(
                documents,
                key=lambda item: (
                    float(item.get("hybrid_score") or 0.0),
                    float(item.get("source_recovery_score") or 0.0),
                    float(item.get("bm25_score") or 0.0),
                    float(item.get("semantic_score") or 0.0),
                ),
                reverse=True,
            )[:limit]:
                fallback.append(
                    {
                        **document,
                        "rerank_score": round(
                            max(
                                float(document.get("hybrid_score") or 0.0),
                                float(document.get("source_recovery_score") or 0.0),
                            ),
                            3,
                        ),
                        "rerank_explanation": "Reranker unavailable, so hybrid retrieval scores were used directly.",
                    }
                )
            return fallback, self._load_error or "reranker_disabled"

        try:
            scores = model.predict([(query, str(document.get("text") or "")) for document in documents])
        except Exception as exc:
            self._load_error = f"rerank_failed:{exc.__class__.__name__}"
            return [], self._load_error

        ranked = []
        raw_pairs = list(zip(documents, [float(score) for score in scores]))
        max_score = max((score for _, score in raw_pairs), default=0.0)
        min_score = min((score for _, score in raw_pairs), default=0.0)
        denominator = (max_score - min_score) or 1.0
        for document, raw_score in sorted(raw_pairs, key=lambda item: item[1], reverse=True)[:limit]:
            normalized = (raw_score - min_score) / denominator
            ranked.append(
                {
                    **document,
                    "rerank_score": round(max(0.0, min(1.0, normalized)), 3),
                    "rerank_explanation": "Reranked with local CrossEncoder relevance model.",
                }
            )
        return ranked, None

    def _get_model(self):
        if self._cross_encoder is not None:
            return self._cross_encoder
        if self._load_error is not None:
            return None
        if CrossEncoder is None:
            self._load_error = "cross_encoder_unavailable"
            return None
        try:
            self._cross_encoder = CrossEncoder(self.MODEL_NAME)
            return self._cross_encoder
        except Exception as exc:
            self._load_error = f"reranker_load_failed:{exc.__class__.__name__}"
            return None
