from __future__ import annotations

import math
from collections import Counter

from app.utils import normalize_text


class BM25RetrievalService:
    """Small BM25 implementation for fetched candidate documents."""

    K1 = 1.5
    B = 0.75

    def rank(self, queries: list[str], documents: list[dict], limit: int = 5) -> list[dict]:
        prepared = []
        corpus_tokens: list[list[str]] = []
        document_frequencies: Counter[str] = Counter()
        lengths: list[int] = []

        for document in documents:
            tokens = [token for token in normalize_text(str(document.get("text") or "")).split() if token]
            if not tokens:
                continue
            prepared.append(document)
            corpus_tokens.append(tokens)
            lengths.append(len(tokens))
            for token in set(tokens):
                document_frequencies[token] += 1

        if not prepared:
            return []

        avg_doc_len = sum(lengths) / max(1, len(lengths))
        query_tokens = []
        for query in queries:
            tokens = [token for token in normalize_text(query).split() if token]
            if tokens:
                query_tokens.append(tokens)
        if not query_tokens:
            return []

        scored = []
        total_docs = len(corpus_tokens)
        for index, tokens in enumerate(corpus_tokens):
            score = 0.0
            term_counts = Counter(tokens)
            doc_len = len(tokens)
            for query in query_tokens:
                query_score = 0.0
                for term in query:
                    if term not in term_counts:
                        continue
                    df = document_frequencies.get(term, 0)
                    idf = math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))
                    tf = term_counts[term]
                    numerator = tf * (self.K1 + 1)
                    denominator = tf + self.K1 * (1 - self.B + self.B * (doc_len / max(1.0, avg_doc_len)))
                    query_score += idf * (numerator / max(1e-9, denominator))
                score = max(score, query_score)
            scored.append((prepared[index], score))

        max_score = max((score for _, score in scored), default=0.0)
        ranked = []
        for document, raw_score in sorted(scored, key=lambda item: item[1], reverse=True)[:limit]:
            ranked.append(
                {
                    **document,
                    "bm25_score": round((raw_score / max_score) if max_score > 0 else 0.0, 3),
                    "bm25_raw_score": round(raw_score, 4),
                }
            )
        return ranked
