from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from io import BytesIO
from pathlib import Path

from PIL import ExifTags, Image

from app.services.content_extraction import ExtractedPayload, build_text_payload
from app.services.ocr import extract_text_from_image
from app.utils import normalize_text, sha256_text


def probe_media(path: str) -> dict:
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        return {}
    try:
        completed = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {}

    try:
        return json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return {}


def build_audio_payload(path: str, original_filename: str, probe: dict | None = None) -> ExtractedPayload:
    probe = probe or {}
    summary = _build_media_summary("audio", original_filename, probe)
    normalized = normalize_text(summary)
    return ExtractedPayload(
        title=f"Audio upload: {original_filename}",
        author=None,
        publish_time=None,
        canonical_url=None,
        main_text=summary,
        hero_image_url=None,
        normalized_text=normalized,
        content_hash=sha256_text(normalized),
        metadata={
            "media_type": "audio",
            "probe": probe,
            "transcript_status": "pending_transcription",
            "c2pa_credentials": _extract_media_c2pa_credentials(probe),
            "forensics": extract_media_forensics(path, "audio", probe=probe),
        },
    )


def build_image_payload(path: str, original_filename: str, ocr_text: str | None = None) -> ExtractedPayload:
    forensics = extract_media_forensics(path, "image")
    if ocr_text:
        payload = build_text_payload(ocr_text)
        payload.title = f"Image upload: {original_filename}"
        payload.metadata = {
            **payload.metadata,
            "media_type": "image",
            "forensics": forensics,
            "ocr_text_present": True,
            "c2pa_credentials": [],
        }
        return payload

    summary = _build_image_summary(original_filename, forensics)
    normalized = normalize_text(summary)
    return ExtractedPayload(
        title=f"Image upload: {original_filename}",
        author=None,
        publish_time=None,
        canonical_url=None,
        main_text=summary,
        hero_image_url=None,
        normalized_text=normalized,
        content_hash=sha256_text(normalized),
        metadata={
            "media_type": "image",
            "forensics": forensics,
            "ocr_text_present": False,
            "c2pa_credentials": [],
        },
    )


def compute_image_hashes(path: str) -> dict[str, str | None]:
    data = Path(path).read_bytes()
    sha256_hash = hashlib.sha256(data).hexdigest()
    perceptual_hash: str | None = None
    try:
        with Image.open(BytesIO(data)) as image:
            grayscale = image.convert("L").resize((8, 8))
            pixels = list(grayscale.getdata())
            average = sum(pixels) / len(pixels)
            bits = "".join("1" if pixel >= average else "0" for pixel in pixels)
            perceptual_hash = f"{int(bits, 2):016x}"
    except Exception:
        perceptual_hash = None
    return {
        "sha256": sha256_hash,
        "perceptual_hash": perceptual_hash,
    }


def compute_binary_hashes(path: str) -> dict[str, str | None]:
    data = Path(path).read_bytes()
    sha256_hash = hashlib.sha256(data).hexdigest()
    sample = data[:65536] + data[-65536:] if len(data) > 65536 else data
    fingerprint = hashlib.sha256(sample).hexdigest()[:32] if sample else None
    return {
        "sha256": sha256_hash,
        "fingerprint": fingerprint,
    }


def build_video_payload(path: str, original_filename: str, upload_dir: str, probe: dict | None = None) -> ExtractedPayload:
    probe = probe or {}
    frame_path = extract_video_keyframe(path, upload_dir)
    if frame_path:
        frame_text = extract_text_from_image(frame_path)
        if frame_text:
            payload = build_text_payload(frame_text)
            payload.title = f"Video upload: {original_filename}"
            payload.metadata = {
                **payload.metadata,
                "media_type": "video",
                "probe": probe,
                "keyframe_path": frame_path,
                "derived_from": "video_keyframe_ocr",
                "c2pa_credentials": _extract_media_c2pa_credentials(probe),
                "forensics": extract_media_forensics(path, "video", probe=probe),
            }
            return payload

    summary = _build_media_summary("video", original_filename, probe)
    normalized = normalize_text(summary)
    return ExtractedPayload(
        title=f"Video upload: {original_filename}",
        author=None,
        publish_time=None,
        canonical_url=None,
        main_text=summary,
        hero_image_url=None,
        normalized_text=normalized,
        content_hash=sha256_text(normalized),
        metadata={
            "media_type": "video",
            "probe": probe,
            "keyframe_path": frame_path,
            "c2pa_credentials": _extract_media_c2pa_credentials(probe),
            "forensics": extract_media_forensics(path, "video", probe=probe),
        },
    )


def extract_video_keyframe(path: str, upload_dir: str) -> str | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return None

    output_path = Path(upload_dir) / f"{Path(path).stem}-keyframe.jpg"
    try:
        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-i",
                path,
                "-vf",
                "select=eq(n\\,0)",
                "-vframes",
                "1",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    return str(output_path) if output_path.exists() else None


def extract_audio_track(path: str, upload_dir: str) -> str | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return None

    output_path = Path(upload_dir) / f"{Path(path).stem}-audio.wav"
    try:
        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-i",
                path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    return str(output_path) if output_path.exists() else None


def sample_video_frames(path: str, upload_dir: str, count: int = 4) -> list[str]:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path or count <= 0:
        return []

    probe = probe_media(path)
    duration = 0.0
    try:
        duration = float((probe.get("format", {}) or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    timestamps = [0.0]
    if duration > 0.0:
        timestamps = sorted({round(duration * fraction, 2) for fraction in (0.0, 0.25, 0.5, 0.75, 0.95)})[:count]

    frame_paths: list[str] = []
    for index, timestamp in enumerate(timestamps[:count]):
        output_path = Path(upload_dir) / f"{Path(path).stem}-frame-{index}.jpg"
        try:
            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-ss",
                    str(timestamp),
                    "-i",
                    path,
                    "-vframes",
                    "1",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if output_path.exists():
            frame_paths.append(str(output_path))
    return frame_paths


def _build_media_summary(kind: str, original_filename: str, probe: dict) -> str:
    format_data = probe.get("format", {}) if isinstance(probe, dict) else {}
    streams = probe.get("streams", []) if isinstance(probe, dict) else []
    duration = format_data.get("duration")
    format_name = format_data.get("format_name")
    stream_types = ", ".join(sorted({stream.get("codec_type", "unknown") for stream in streams if isinstance(stream, dict)}))

    parts = [
        f"Uploaded {kind} file named {original_filename}.",
        f"Detected container or format: {format_name or 'unknown'}.",
    ]
    if duration:
        parts.append(f"Approximate duration: {duration} seconds.")
    if stream_types:
        parts.append(f"Detected streams: {stream_types}.")
    parts.append(f"No direct {'speech transcript' if kind == 'audio' else 'full frame text'} was available, so metadata-only preprocessing was used.")
    return " ".join(parts)


def _build_image_summary(original_filename: str, forensics: dict) -> str:
    parts = [f"Uploaded image file named {original_filename}."]
    if forensics.get("format"):
        parts.append(f"Detected format: {forensics['format']}.")
    if forensics.get("width") and forensics.get("height"):
        parts.append(f"Dimensions: {forensics['width']} by {forensics['height']} pixels.")
    fields = forensics.get("fields", {}) if isinstance(forensics, dict) else {}
    if isinstance(fields, dict):
        if fields.get("device_model"):
            parts.append(f"Recorded device model: {fields['device_model']}.")
        if fields.get("software"):
            parts.append(f"Recorded software tag: {fields['software']}.")
        if fields.get("captured_at"):
            parts.append(f"Recorded capture timestamp: {fields['captured_at']}.")
    parts.append("No direct OCR text was available, so metadata-only preprocessing was used.")
    return " ".join(parts)


def _extract_media_c2pa_credentials(probe: dict) -> list[dict]:
    credentials: list[dict] = []
    format_tags = probe.get("format", {}).get("tags", {}) if isinstance(probe, dict) else {}
    for key, value in format_tags.items():
        normalized_key = str(key).lower()
        if "c2pa" not in normalized_key and "content" not in normalized_key and "manifest" not in normalized_key:
            continue
        credentials.append(
            {
                "source": "ffprobe",
                "type": str(key),
                "value": str(value),
            }
        )
    return credentials


def extract_media_forensics(path: str, media_kind: str, probe: dict | None = None) -> dict:
    if media_kind == "image":
        return _extract_image_forensics(path)
    return _extract_probe_forensics(media_kind, probe or {})


def extract_image_forensics_from_bytes(data: bytes, source: str = "remote_sample") -> dict:
    if not data:
        return {
            "media_kind": "image",
            "source": source,
            "metadata_status": "absent",
            "notes": ["No image bytes were available for forensic parsing."],
        }
    try:
        with Image.open(BytesIO(data)) as image:
            return _image_forensics_from_image(image, source)
    except Exception:
        return {
            "media_kind": "image",
            "source": source,
            "metadata_status": "limited",
            "notes": ["Image bytes were available, but the sample was not sufficient for full forensic parsing."],
        }


def _extract_image_forensics(path: str) -> dict:
    try:
        with Image.open(path) as image:
            return _image_forensics_from_image(image, "uploaded_file")
    except Exception:
        return {
            "media_kind": "image",
            "source": "uploaded_file",
            "metadata_status": "absent",
            "notes": ["The uploaded image could not be parsed for forensic metadata."],
        }


def _image_forensics_from_image(image: Image.Image, source: str) -> dict:
    exif_data = {}
    try:
        raw_exif = image.getexif()
    except Exception:
        raw_exif = None

    if raw_exif:
        for key, value in raw_exif.items():
            tag_name = ExifTags.TAGS.get(key, str(key))
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore")
            exif_data[tag_name] = value

    selected_fields = {
        "device_make": exif_data.get("Make"),
        "device_model": exif_data.get("Model"),
        "captured_at": exif_data.get("DateTimeOriginal") or exif_data.get("DateTime"),
        "software": exif_data.get("Software"),
        "artist": exif_data.get("Artist"),
        "copyright": exif_data.get("Copyright"),
        "gps_info_present": bool(exif_data.get("GPSInfo")),
    }
    selected_fields = {key: value for key, value in selected_fields.items() if value not in (None, "", False)}

    return {
        "media_kind": "image",
        "source": source,
        "format": image.format,
        "mode": image.mode,
        "width": image.width,
        "height": image.height,
        "metadata_status": "present" if exif_data else "absent",
        "exif_present": bool(exif_data),
        "fields": selected_fields,
        "notes": [
            "Embedded EXIF metadata was present in the image." if exif_data else "No embedded EXIF metadata was present in the parsed image data."
        ],
    }


def _extract_probe_forensics(media_kind: str, probe: dict) -> dict:
    format_data = probe.get("format", {}) if isinstance(probe, dict) else {}
    streams = probe.get("streams", []) if isinstance(probe, dict) else []
    tags = format_data.get("tags", {}) if isinstance(format_data, dict) else {}

    interesting_tags = {
        "title": tags.get("title"),
        "artist": tags.get("artist"),
        "encoder": tags.get("encoder"),
        "software": tags.get("software"),
        "creation_time": tags.get("creation_time"),
        "comment": tags.get("comment"),
    }
    interesting_tags = {key: value for key, value in interesting_tags.items() if value not in (None, "")}

    stream_summaries = []
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        entry = {
            "codec_type": stream.get("codec_type"),
            "codec_name": stream.get("codec_name"),
            "duration": stream.get("duration"),
            "bit_rate": stream.get("bit_rate"),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "sample_rate": stream.get("sample_rate"),
            "channels": stream.get("channels"),
            "avg_frame_rate": stream.get("avg_frame_rate"),
        }
        stream_summaries.append({key: value for key, value in entry.items() if value not in (None, "", "0/0")})

    return {
        "media_kind": media_kind,
        "source": "ffprobe",
        "format_name": format_data.get("format_name"),
        "format_long_name": format_data.get("format_long_name"),
        "duration_seconds": format_data.get("duration"),
        "size_bytes": format_data.get("size"),
        "bit_rate": format_data.get("bit_rate"),
        "metadata_status": "present" if interesting_tags else "limited",
        "fields": interesting_tags,
        "streams": stream_summaries,
        "notes": [
            "Container-level metadata tags were present." if interesting_tags else "No useful container-level authoring tags were present in the media container."
        ],
    }
