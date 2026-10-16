from __future__ import annotations

from collections import Counter
from math import sqrt

from app.utils import normalize_text


class EmbeddingsService:
    """Lightweight local embedding fallback for passage-level retrieval.

    This is intentionally simple and deterministic. It gives the retrieval
    layer a semantic-ish signal without making snippet overlap the verdict
    engine.
    """

    def embed_texts(self, texts: list[str]) -> list[dict[str, float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> dict[str, float]:
        return self._embed(query)

    def similarity(self, left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        shared = set(left) & set(right)
        numerator = sum(left[token] * right[token] for token in shared)
        left_norm = sqrt(sum(value * value for value in left.values()))
        right_norm = sqrt(sum(value * value for value in right.values()))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _embed(self, text: str) -> dict[str, float]:
        tokens = [token for token in normalize_text(text).split() if token]
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = float(sum(counts.values()))
        return {token: value / total for token, value in counts.items()}
