from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.services.context_match_service import ContextMatchService
from app.utils import normalize_text

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
except Exception:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    torch = None


LABELS = ["supports", "contradicts", "partially_related", "background_only", "unrelated", "insufficient"]


@dataclass
class ClaimEvidenceVerification:
    relation_label: str
    confidence: float
    explanation: str
    model_used: str
    context_match: dict
    raw_scores: dict[str, float]


class ClaimEvidenceVerifier:
    """Trainable claim-evidence verifier with rule-based fallback.

    If `CLAIM_VERIFIER_MODEL_PATH` points to a fine-tuned local model, this
    loads it lazily and classifies the pair. Otherwise it uses the context
    matcher as a deterministic fallback. In both modes, hard context mismatches
    override semantic/model support.
    """

    def __init__(self, context_matcher: ContextMatchService | None = None) -> None:
        self.settings = get_settings()
        self.context_matcher = context_matcher or ContextMatchService()
        self.model_path = str(getattr(self.settings, "claim_verifier_model_path", "") or "").strip()
        self.enabled = bool(getattr(self.settings, "enable_claim_verifier_model", False))
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._load_error: str | None = None

    def verify(self, claim_text: str, evidence: dict, relevance_score: float = 0.0, fallback_label: str = "unrelated") -> ClaimEvidenceVerification:
        context_match = self.context_matcher.compare(claim_text, evidence)
        model_label, confidence, raw_scores, model_used = self._predict_with_model(claim_text, evidence)
        if model_label is None:
            model_label = self._fallback_label(context_match, relevance_score, fallback_label)
            confidence = self._fallback_confidence(context_match, relevance_score)
            raw_scores = {label: (confidence if label == model_label else 0.0) for label in LABELS}
            model_used = self._load_error or "rule_based_context_fallback"

        relation = self.context_matcher.relation_from_context(model_label, context_match, relevance_score)
        if model_label == "supports" and relation != "supports":
            confidence = min(confidence, 0.55)
        if relation == "unrelated" and not bool(context_match.get("location_match_pass")):
            confidence = max(confidence, 0.85)
        explanation = self.context_matcher.relation_explanation(relation, context_match)
        return ClaimEvidenceVerification(
            relation_label=relation,
            confidence=round(max(0.0, min(1.0, float(confidence))), 3),
            explanation=explanation,
            model_used=model_used,
            context_match=context_match,
            raw_scores={key: round(float(value), 3) for key, value in raw_scores.items()},
        )

    def _predict_with_model(self, claim_text: str, evidence: dict) -> tuple[str | None, float, dict[str, float], str]:
        if not self.enabled or not self.model_path:
            return None, 0.0, {}, "claim_verifier_model_disabled"
        if AutoModelForSequenceClassification is None or AutoTokenizer is None or torch is None:
            self._load_error = "transformers_or_torch_unavailable"
            return None, 0.0, {}, self._load_error
        if not Path(self.model_path).exists():
            self._load_error = f"claim_verifier_model_missing:{self.model_path}"
            return None, 0.0, {}, self._load_error
        try:
            model, tokenizer = self._load_model()
            encoded = tokenizer(
                self._format_pair(claim_text, evidence),
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(**encoded).logits[0]
                probs = torch.softmax(logits, dim=-1).tolist()
            id2label = getattr(model.config, "id2label", {}) or {}
            scores = {
                str(id2label.get(index, LABELS[index] if index < len(LABELS) else index)).lower(): float(score)
                for index, score in enumerate(probs)
            }
            normalized_scores = {self._normalize_label(label): score for label, score in scores.items()}
            label = max(normalized_scores, key=normalized_scores.get)
            return label, float(normalized_scores[label]), normalized_scores, self.model_path
        except Exception as exc:
            self._load_error = f"claim_verifier_inference_failed:{exc.__class__.__name__}"
            return None, 0.0, {}, self._load_error

    def _load_model(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self._model.eval()
        return self._model, self._tokenizer

    def _fallback_label(self, context_match: dict, relevance_score: float, fallback_label: str) -> str:
        if not bool(context_match.get("location_match_pass")):
            return "unrelated"
        if fallback_label == "contradicts":
            return "contradicts"
        if fallback_label == "supports" and bool(context_match.get("context_match_pass")) and relevance_score >= 0.8:
            return "supports"
        if relevance_score >= 0.6:
            return "partially_related"
        if relevance_score >= 0.45:
            return "background_only"
        return "insufficient"

    def _fallback_confidence(self, context_match: dict, relevance_score: float) -> float:
        if not bool(context_match.get("location_match_pass")):
            return 0.9
        context_average = (
            float(context_match.get("entity_match_score") or 0.0)
            + float(context_match.get("location_match_score") or 0.0)
            + float(context_match.get("event_match_score") or 0.0)
        ) / 3
        return max(0.2, min(0.95, 0.55 * relevance_score + 0.45 * context_average))

    def _format_pair(self, claim_text: str, evidence: dict) -> str:
        evidence_text = " ".join(
            str(evidence.get(key) or "")
            for key in ("title", "snippet", "passage", "text", "evidence_body")
        )
        return f"CLAIM: {claim_text}\nEVIDENCE: {evidence_text}"

    def _normalize_label(self, label: str) -> str:
        normalized = normalize_text(label).replace(" ", "_").replace("-", "_")
        aliases = {
            "entailment": "supports",
            "support": "supports",
            "supports": "supports",
            "refutes": "contradicts",
            "refuted": "contradicts",
            "contradiction": "contradicts",
            "contradicts": "contradicts",
            "neutral": "insufficient",
            "not_enough_info": "insufficient",
            "nei": "insufficient",
            "partial": "partially_related",
            "partially_supports": "partially_related",
            "partially_related": "partially_related",
            "background": "background_only",
            "background_only": "background_only",
            "unrelated": "unrelated",
            "insufficient": "insufficient",
        }
        return aliases.get(normalized, normalized if normalized in LABELS else "insufficient")
