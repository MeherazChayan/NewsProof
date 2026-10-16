from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.utils import normalize_text


class VisionReasoningService:
    """Lightweight visual-context reasoning wrapper.

    This is intentionally separate from OCR so image/video context does not
    depend on text extraction alone.
    """

    def describe_image(self, path: str, context: str | None = None, ocr_text: str | None = None) -> dict:
        width = height = 0
        mode = ""
        try:
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
        except Exception:
            pass

        clues = []
        if ocr_text:
            clues.append(f"Visible text suggests: {ocr_text[:180]}")
        if width and height:
            clues.append(f"Image resolution is {width}x{height}")
        if mode:
            clues.append(f"Color mode is {mode}")
        if context:
            clues.append(f"Claimed context references: {context[:180]}")

        summary = ". ".join(clues) if clues else f"Uploaded image {Path(path).name} was available for visual context review."
        objects = self._extract_context_terms(" ".join(filter(None, [context or "", ocr_text or "", Path(path).stem.replace('-', ' ').replace('_', ' ')])))
        return {
            "scene_description": summary,
            "visible_entities": objects[:8],
            "location_clues": [term for term in objects if term.istitle()][:5],
            "event_summary": context[:200] if context else "",
        }

    def describe_video_frames(self, frame_paths: list[str], context: str | None = None, ocr_texts: list[str] | None = None) -> dict:
        ocr_texts = ocr_texts or []
        descriptions = [self.describe_image(path, context=context, ocr_text=ocr_texts[index] if index < len(ocr_texts) else "") for index, path in enumerate(frame_paths)]
        entities: list[str] = []
        for description in descriptions:
            entities.extend(description.get("visible_entities", []))
        deduped_entities = []
        seen = set()
        for entity in entities:
            key = entity.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped_entities.append(entity)
        summary = " ".join(item.get("scene_description", "") for item in descriptions[:3]).strip()
        return {
            "scene_description": summary,
            "visible_entities": deduped_entities[:12],
            "event_summary": context[:200] if context else "",
            "frame_count": len(frame_paths),
        }

    def compare_visual_context(self, description: dict, claim: str) -> dict:
        summary = " ".join(
            filter(
                None,
                [
                    str(description.get("scene_description") or ""),
                    " ".join(description.get("visible_entities", [])),
                    str(description.get("event_summary") or ""),
                ],
            )
        )
        score = self._overlap(claim, summary) if claim else 0.6
        if not claim:
            verdict = "neutral"
        elif score >= 0.6:
            verdict = "supports_context"
        elif score <= 0.2:
            verdict = "contradicts_context"
        else:
            verdict = "mixed_context"
        return {
            "verdict": verdict,
            "score": round(score, 3),
            "summary": summary[:400],
        }

    def _extract_context_terms(self, text: str) -> list[str]:
        return [token for token in text.split() if len(token) > 3][:20]

    def _overlap(self, left: str, right: str) -> float:
        left_terms = set(normalize_text(left).split())
        right_terms = set(normalize_text(right).split())
        if not left_terms or not right_terms:
            return 0.0
        return len(left_terms & right_terms) / max(1, len(left_terms))
