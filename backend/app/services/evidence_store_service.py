from __future__ import annotations


class EvidenceStoreService:
    """Small in-process cache for retrieval and passage chunking results."""

    def __init__(self) -> None:
        self._store: dict[str, list[dict]] = {}

    def get(self, key: str) -> list[dict] | None:
        return self._store.get(key)

    def put(self, key: str, passages: list[dict]) -> list[dict]:
        self._store[key] = passages
        return passages
