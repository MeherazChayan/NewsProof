from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.utils import normalize_text

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class SemanticRetrievalService:
    """Semantic retrieval backed by a real sentence embedding model."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Any | None = None
        self._load_error: str | None = None

    def rank(self, queries: list[str], documents: list[dict], limit: int = 5) -> tuple[list[dict], str | None]:
        if not self.settings.enable_local_ml_rerank:
            return [], "semantic_model_disabled"
        model = self._get_model()
        if model is None:
            return [], self._load_error or "semantic_model_unavailable"

        prepared = [document for document in documents if str(document.get("text") or "").strip()]
        if not prepared:
            return [], "no_candidate_documents"

        clean_queries = [" ".join(query.split()) for query in queries if normalize_text(query)]
        if not clean_queries:
            return [], "no_semantic_queries"

        try:
            query_vectors = model.encode(clean_queries, normalize_embeddings=True)
            document_vectors = model.encode([str(document.get("text") or "") for document in prepared], normalize_embeddings=True)
        except Exception as exc:
            self._load_error = f"semantic_encode_failed:{exc.__class__.__name__}"
            return [], self._load_error

        combined_scores = []
        for index, document in enumerate(prepared):
            score = max(float(query_vector @ document_vectors[index]) for query_vector in query_vectors)
            combined_scores.append((document, score))

        ranked: list[dict] = []
        for document, score in sorted(combined_scores, key=lambda item: item[1], reverse=True)[:limit]:
            ranked.append(
                {
                    **document,
                    "semantic_score": round(max(0.0, min(1.0, score)), 3),
                }
            )
        return ranked, None

    def _get_model(self):
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            return None
        if SentenceTransformer is None:
            self._load_error = "sentence_transformers_unavailable"
            return None
        try:
            self._model = SentenceTransformer(self.MODEL_NAME)
            return self._model
        except Exception as exc:
            self._load_error = f"semantic_model_load_failed:{exc.__class__.__name__}"
            return None
