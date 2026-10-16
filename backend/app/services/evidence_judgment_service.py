from __future__ import annotations

import math
from typing import Any

from app.config import get_settings

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


class EvidenceJudgmentService:
    MODEL_NAME = "cross-encoder/nli-distilroberta-base"
    LABELS = ["contradiction", "entailment", "neutral"]

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Any | None = None
        self._load_error: str | None = None

    def judge(self, claim: str, evidence_items: list[dict]) -> tuple[list[dict], str | None]:
        if not evidence_items:
            return [], "no_evidence_items"

        model = self._get_model() if self.settings.enable_local_ml_rerank else None
        if model is None:
            judgments = []
            for item in evidence_items:
                source_score = max(
                    float(item.get("source_recovery_score") or 0.0),
                    float(item.get("rerank_score") or 0.0),
                    float(item.get("bm25_score") or 0.0),
                )
                contradiction_signal = float(item.get("contradiction_score") or 0.0)
                if source_score >= 0.78:
                    label = "supports"
                    confidence = source_score
                elif source_score >= 0.52:
                    label = "partially_supports"
                    confidence = source_score
                elif contradiction_signal >= 0.55:
                    label = "contradicts"
                    confidence = contradiction_signal
                else:
                    label = "unrelated"
                    confidence = max(0.0, 1.0 - source_score)
                judgments.append(
                    {
                        **item,
                        "judgment_label": label,
                        "judgment_confidence": round(float(confidence), 3),
                        "entailment_score": round(float(source_score), 3),
                        "contradiction_score": round(float(contradiction_signal), 3),
                        "neutral_score": round(max(0.0, 1.0 - float(confidence)), 3),
                    }
                )
            return judgments, self._load_error or "judgment_model_unavailable"

        pairs = [(claim, str(item.get("text") or item.get("passage") or item.get("snippet") or "")) for item in evidence_items]
        try:
            logits = model.predict(pairs)
        except Exception as exc:
            self._load_error = f"judgment_failed:{exc.__class__.__name__}"
            return [], self._load_error

        judgments = []
        for item, row in zip(evidence_items, logits):
            contradiction, entailment, neutral = self._softmax(row)
            label = "unrelated"
            confidence = neutral
            if entailment >= 0.72:
                label = "supports"
                confidence = entailment
            elif entailment >= 0.5:
                label = "partially_supports"
                confidence = entailment
            elif contradiction >= 0.55:
                label = "contradicts"
                confidence = contradiction
            judgments.append(
                {
                    **item,
                    "judgment_label": label,
                    "judgment_confidence": round(float(confidence), 3),
                    "entailment_score": round(float(entailment), 3),
                    "contradiction_score": round(float(contradiction), 3),
                    "neutral_score": round(float(neutral), 3),
                }
            )
        return judgments, None

    def _get_model(self):
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            return None
        if CrossEncoder is None:
            self._load_error = "cross_encoder_unavailable"
            return None
        try:
            self._model = CrossEncoder(self.MODEL_NAME)
            return self._model
        except Exception as exc:
            self._load_error = f"judgment_model_load_failed:{exc.__class__.__name__}"
            return None

    def _softmax(self, row) -> tuple[float, float, float]:
        values = [float(item) for item in row]
        max_value = max(values)
        exps = [math.exp(item - max_value) for item in values]
        total = sum(exps) or 1.0
        normalized = [item / total for item in exps]
        if len(normalized) >= 3:
            return normalized[0], normalized[1], normalized[2]
        padded = normalized + [0.0] * (3 - len(normalized))
        return padded[0], padded[1], padded[2]
