from __future__ import annotations

import logging
import math
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from app.config import get_settings
from app.services.media import probe_media
logger = logging.getLogger(__name__)


class TranscriptionService:
    """Local faster-whisper transcription with ffmpeg preprocessing."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None

    def transcribe_audio(self, path: str) -> dict[str, Any]:
        return self._transcribe_local(path, media_kind="audio")

    def transcribe_video_audio(self, path: str) -> dict[str, Any]:
        return self._transcribe_local(path, media_kind="video")

    def _transcribe_local(self, path: str, media_kind: str) -> dict[str, Any]:
        probe = probe_media(path)
        format_info = probe.get("format", {}) if isinstance(probe, dict) else {}
        file_format = str(format_info.get("format_name") or Path(path).suffix.lstrip(".") or "unknown")
        duration = self._safe_float(format_info.get("duration"))

        prepared_path, preprocessing = self._prepare_audio_for_transcription(path)
        if not prepared_path:
            return {
                "transcript_text": "",
                "cleaned_transcript": "",
                "transcript_status": "transcription_failed",
                "transcript_confidence": 0.0,
                "transcript_summary": f"Local transcription failed because audio preprocessing could not be completed for this {media_kind} file.",
                "timestamps": [],
                "detected_language": None,
                "language_probability": None,
                "chunk_count": 0,
                "engine": "faster-whisper",
                "model": self._setting("faster_whisper_model_size", "small"),
                "file_format": file_format,
                "duration_seconds": duration,
                "preprocessing_status": preprocessing.get("status"),
                "preprocessing_details": preprocessing,
                "failure_reason": preprocessing.get("failure_reason"),
            }

        try:
            model = self._get_model()
        except Exception as exc:
            logger.warning("faster-whisper model initialization failed: %s", exc)
            self._cleanup_temp_file(prepared_path, path)
            return {
                "transcript_text": "",
                "cleaned_transcript": "",
                "transcript_status": "transcription_failed",
                "transcript_confidence": 0.0,
                "transcript_summary": f"Local transcription could not start for this {media_kind} file.",
                "timestamps": [],
                "detected_language": None,
                "language_probability": None,
                "chunk_count": 0,
                "engine": "faster-whisper",
                "model": self._setting("faster_whisper_model_size", "small"),
                "file_format": file_format,
                "duration_seconds": duration,
                "preprocessing_status": preprocessing.get("status"),
                "preprocessing_details": preprocessing,
                "failure_reason": str(exc),
            }

        try:
            segments, info = model.transcribe(
                prepared_path,
                beam_size=int(self._setting("faster_whisper_beam_size", 5)),
                vad_filter=True,
                condition_on_previous_text=False,
            )
            collected_segments = list(segments)
        except Exception as exc:
            logger.warning("faster-whisper transcription failed for %s: %s", path, exc)
            self._cleanup_temp_file(prepared_path, path)
            return {
                "transcript_text": "",
                "cleaned_transcript": "",
                "transcript_status": "transcription_failed",
                "transcript_confidence": 0.0,
                "transcript_summary": f"Local transcription failed for this {media_kind} file.",
                "timestamps": [],
                "detected_language": None,
                "language_probability": None,
                "chunk_count": 0,
                "engine": "faster-whisper",
                "model": self._setting("faster_whisper_model_size", "small"),
                "file_format": file_format,
                "duration_seconds": duration,
                "preprocessing_status": preprocessing.get("status"),
                "preprocessing_details": preprocessing,
                "failure_reason": str(exc),
            }
        finally:
            self._cleanup_temp_file(prepared_path, path)

        raw_transcript = " ".join(str(segment.text or "").strip() for segment in collected_segments if str(segment.text or "").strip()).strip()
        cleaned_transcript = " ".join(raw_transcript.split())
        timestamps = [
            {
                "start": round(float(segment.start or 0.0), 3),
                "end": round(float(segment.end or 0.0), 3),
                "text": str(segment.text or "").strip(),
            }
            for segment in collected_segments
            if str(segment.text or "").strip()
        ]
        segment_confidences = [
            max(
                0.0,
                min(
                    1.0,
                    1.0 - float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
                ),
            )
            for segment in collected_segments
        ]
        transcript_confidence = round(sum(segment_confidences) / len(segment_confidences), 3) if segment_confidences else 0.0
        detected_language = getattr(info, "language", None)
        language_probability = self._safe_float(getattr(info, "language_probability", None))

        if cleaned_transcript:
            status = "completed"
            summary = f"Local faster-whisper transcription produced {len(cleaned_transcript.split())} words for this {media_kind} file."
        elif collected_segments:
            status = "speech_not_detected"
            summary = f"No meaningful speech could be recovered from this {media_kind} file."
        else:
            status = "transcript_empty"
            summary = f"No transcript text was produced for this {media_kind} file."

        return {
            "transcript_text": raw_transcript,
            "cleaned_transcript": cleaned_transcript,
            "transcript_status": status,
            "transcript_confidence": transcript_confidence,
            "transcript_summary": summary,
            "timestamps": timestamps,
            "detected_language": detected_language,
            "language_probability": language_probability,
            "chunk_count": 1 if prepared_path else 0,
            "engine": "faster-whisper",
            "model": self._setting("faster_whisper_model_size", "small"),
            "file_format": file_format,
            "duration_seconds": duration,
            "preprocessing_status": preprocessing.get("status"),
            "preprocessing_details": preprocessing,
            "failure_reason": None,
        }

    def _prepare_audio_for_transcription(self, path: str) -> tuple[str | None, dict[str, Any]]:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            return None, {
                "status": "ffmpeg_missing",
                "target_format": "mono_16khz_wav",
                "failure_reason": "ffmpeg is not available in the current environment.",
            }

        temp_file = NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()
        output_path = temp_file.name
        command = [
            ffmpeg_path,
            "-y",
            "-i",
            path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            "loudnorm",
            output_path,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            return output_path, {
                "status": "completed",
                "target_format": "mono_16khz_wav",
                "normalization": "loudnorm",
                "chunk_count": 1,
            }
        except (OSError, subprocess.SubprocessError) as exc:
            self._cleanup_temp_file(output_path, path)
            return None, {
                "status": "failed",
                "target_format": "mono_16khz_wav",
                "failure_reason": str(exc),
            }

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("faster-whisper is not installed in the backend environment.") from exc

        self._model = WhisperModel(
            self._setting("faster_whisper_model_size", "small"),
            device=self._setting("faster_whisper_device", "cpu"),
            compute_type=self._setting("faster_whisper_compute_type", "int8"),
            cpu_threads=int(self._setting("faster_whisper_cpu_threads", 4)),
        )
        return self._model

    def _cleanup_temp_file(self, prepared_path: str | None, original_path: str) -> None:
        if not prepared_path or prepared_path == original_path:
            return
        try:
            Path(prepared_path).unlink(missing_ok=True)
        except Exception:
            logger.debug("Could not remove temporary transcription file %s", prepared_path)

    def _safe_float(self, value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return round(numeric, 3)

    def _setting(self, name: str, default: Any) -> Any:
        return getattr(self.settings, name, default)
