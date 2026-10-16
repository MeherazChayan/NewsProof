from __future__ import annotations

from dataclasses import dataclass

from app.services.content_extraction import ExtractedPayload


AI_TOOL_KEYWORDS = {
    "midjourney",
    "stable diffusion",
    "dall-e",
    "firefly",
    "runway",
    "sora",
    "elevenlabs",
    "descript",
    "respeecher",
    "synthesia",
}

EDITING_TOOL_KEYWORDS = {
    "photoshop",
    "gimp",
    "lightroom",
    "canva",
    "capcut",
    "after effects",
    "premiere",
    "final cut",
    "davinci",
    "filmora",
}


@dataclass
class MediaAuthenticityAssessment:
    status: str
    risk_level: str
    score: float
    explanation: str
    signals: list[str]
    limitations: list[str]
    details: dict

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "risk_level": self.risk_level,
            "score": self.score,
            "explanation": self.explanation,
            "signals": self.signals,
            "limitations": self.limitations,
            "details": self.details,
        }


class MediaAuthenticityService:
    def analyze(self, extracted: ExtractedPayload) -> dict:
        media_type = str(extracted.metadata.get("media_type") or "")
        if media_type not in {"image", "audio", "video"}:
            return MediaAuthenticityAssessment(
                status="not_applicable",
                risk_level="none",
                score=0.0,
                explanation="No uploaded media forensic assessment was needed for this submission type.",
                signals=[],
                limitations=["Media authenticity checks only run on uploaded image, audio, and video files."],
                details={},
            ).as_dict()

        forensics = extracted.metadata.get("forensics", {})
        if media_type == "image":
            return self._analyze_image(forensics, bool(extracted.metadata.get("ocr_text_present"))).as_dict()
        if media_type == "audio":
            return self._analyze_audio(forensics).as_dict()
        return self._analyze_video(forensics, extracted.metadata.get("derived_from") == "video_keyframe_ocr").as_dict()

    def _analyze_image(self, forensics: dict, has_ocr_text: bool) -> MediaAuthenticityAssessment:
        fields = forensics.get("fields", {}) if isinstance(forensics, dict) else {}
        signals: list[str] = []
        limitations = [
            "This is a forensic manipulation assessment, not proof that a claim is true or false.",
            "Missing EXIF metadata alone does not prove an image is fake.",
        ]

        software = str(fields.get("software") or "").lower()
        make = str(fields.get("device_make") or "").strip()
        model = str(fields.get("device_model") or "").strip()
        captured_at = str(fields.get("captured_at") or "").strip()

        if _contains_keyword(software, AI_TOOL_KEYWORDS):
            signals.append(f"Image metadata references an AI-generation or synthetic media tool: {fields.get('software')}.")
            return MediaAuthenticityAssessment(
                status="high_deepfake_risk",
                risk_level="high",
                score=0.9,
                explanation="The image metadata points to a synthetic-media tool, so the upload carries a high deepfake or AI-generated content risk.",
                signals=signals,
                limitations=limitations,
                details={"forensics": forensics, "has_ocr_text": has_ocr_text},
            )

        if _contains_keyword(software, EDITING_TOOL_KEYWORDS):
            signals.append(f"Image metadata references editing software: {fields.get('software')}.")
            if not (make or model or captured_at):
                signals.append("No camera make, model, or original capture timestamp was preserved alongside the editing tag.")
                return MediaAuthenticityAssessment(
                    status="suspicious_manipulation",
                    risk_level="medium",
                    score=0.72,
                    explanation="The image includes editing-software traces but does not preserve strong original capture metadata, so it looks edited or transformed.",
                    signals=signals,
                    limitations=limitations,
                    details={"forensics": forensics, "has_ocr_text": has_ocr_text},
                )

        if make or model or captured_at:
            signals.append("Original-looking camera or capture metadata is present.")
            return MediaAuthenticityAssessment(
                status="likely_authentic",
                risk_level="low",
                score=0.68,
                explanation="The image retains camera or capture metadata and no obvious synthetic-media software marker was found.",
                signals=signals,
                limitations=limitations,
                details={"forensics": forensics, "has_ocr_text": has_ocr_text},
            )

        return MediaAuthenticityAssessment(
            status="inconclusive",
            risk_level="moderate",
            score=0.35,
            explanation="The image was checked for forensic clues, but the available metadata was too limited to determine whether it is authentic or manipulated.",
            signals=signals or ["No strong camera or software markers were available in the parsed image metadata."],
            limitations=limitations,
            details={"forensics": forensics, "has_ocr_text": has_ocr_text},
        )

    def _analyze_audio(self, forensics: dict) -> MediaAuthenticityAssessment:
        fields = forensics.get("fields", {}) if isinstance(forensics, dict) else {}
        streams = forensics.get("streams", []) if isinstance(forensics, dict) else []
        signals: list[str] = []
        limitations = [
            "This check does not prove speaker identity.",
            "Container metadata can be stripped or changed without proving a file is fake.",
        ]

        combined = " ".join(str(value) for value in fields.values()).lower()
        if _contains_keyword(combined, AI_TOOL_KEYWORDS):
            signals.append("Audio metadata references a synthetic voice or AI media tool.")
            return MediaAuthenticityAssessment(
                status="high_deepfake_risk",
                risk_level="high",
                score=0.88,
                explanation="The audio container metadata references tools commonly used for synthetic speech or AI-generated media.",
                signals=signals,
                limitations=limitations,
                details={"forensics": forensics},
            )

        if _contains_keyword(combined, EDITING_TOOL_KEYWORDS):
            signals.append("Audio metadata references editing software.")
            return MediaAuthenticityAssessment(
                status="suspicious_manipulation",
                risk_level="medium",
                score=0.7,
                explanation="The audio file appears to have been exported or edited in post-production, so manipulation risk is elevated.",
                signals=signals,
                limitations=limitations,
                details={"forensics": forensics},
            )

        if streams:
            signals.append("Audio streams and container structure were parsed successfully.")
            return MediaAuthenticityAssessment(
                status="likely_authentic",
                risk_level="low",
                score=0.61,
                explanation="No synthetic-audio software markers were found in the parsed container metadata.",
                signals=signals,
                limitations=limitations,
                details={"forensics": forensics},
            )

        return MediaAuthenticityAssessment(
            status="inconclusive",
            risk_level="moderate",
            score=0.32,
            explanation="The audio file was checked, but the available metadata was too limited to assess deepfake risk confidently.",
            signals=["No strong container or tool markers were available."],
            limitations=limitations,
            details={"forensics": forensics},
        )

    def _analyze_video(self, forensics: dict, has_keyframe_ocr: bool) -> MediaAuthenticityAssessment:
        fields = forensics.get("fields", {}) if isinstance(forensics, dict) else {}
        streams = forensics.get("streams", []) if isinstance(forensics, dict) else []
        signals: list[str] = []
        limitations = [
            "This release uses forensic metadata and simple container heuristics, not frame-level face reenactment models.",
            "A suspicious result is not proof that the underlying claim is false.",
        ]

        combined = " ".join(str(value) for value in fields.values()).lower()
        if _contains_keyword(combined, AI_TOOL_KEYWORDS):
            signals.append("Video metadata references an AI-generation or synthetic media tool.")
            return MediaAuthenticityAssessment(
                status="high_deepfake_risk",
                risk_level="high",
                score=0.9,
                explanation="The video metadata includes synthetic-media tooling markers, which creates a high deepfake risk signal.",
                signals=signals,
                limitations=limitations,
                details={"forensics": forensics, "keyframe_ocr_used": has_keyframe_ocr},
            )

        if _contains_keyword(combined, EDITING_TOOL_KEYWORDS):
            signals.append("Video metadata references editing or compositing software.")
            return MediaAuthenticityAssessment(
                status="suspicious_manipulation",
                risk_level="medium",
                score=0.74,
                explanation="The video appears to have passed through editing or compositing software, so manipulation risk is elevated.",
                signals=signals,
                limitations=limitations,
                details={"forensics": forensics, "keyframe_ocr_used": has_keyframe_ocr},
            )

        has_video_stream = any(str(stream.get("codec_type")) == "video" for stream in streams if isinstance(stream, dict))
        if has_video_stream:
            signals.append("Video stream metadata was parsed successfully.")
            if has_keyframe_ocr:
                signals.append("A readable keyframe was extracted from the upload.")
            return MediaAuthenticityAssessment(
                status="likely_authentic",
                risk_level="low",
                score=0.63,
                explanation="No synthetic-media tooling markers were found in the parsed video metadata.",
                signals=signals,
                limitations=limitations,
                details={"forensics": forensics, "keyframe_ocr_used": has_keyframe_ocr},
            )

        return MediaAuthenticityAssessment(
            status="inconclusive",
            risk_level="moderate",
            score=0.34,
            explanation="The video file was checked, but the available metadata was too limited for a confident authenticity assessment.",
            signals=["No strong stream or tooling markers were available."],
            limitations=limitations,
            details={"forensics": forensics, "keyframe_ocr_used": has_keyframe_ocr},
        )


def _contains_keyword(value: str, keywords: set[str]) -> bool:
    lowered = str(value or "").lower()
    return any(keyword in lowered for keyword in keywords)
