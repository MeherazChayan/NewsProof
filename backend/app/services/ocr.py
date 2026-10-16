from io import BytesIO
from pathlib import Path
from statistics import mean

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError
import pytesseract


def extract_text_from_image(path: str) -> str:
    return str(extract_text_details_from_image(path).get("extracted_text") or "").strip()


def extract_text_details_from_image(path: str) -> dict:
    try:
        with Image.open(Path(path)) as image:
            return _extract_text_details_from_image_object(image, source=str(path))
    except FileNotFoundError:
        return _ocr_failure("image_not_downloaded", "The image file could not be found for OCR.")
    except UnidentifiedImageError:
        return _ocr_failure("invalid_image_format", "The image file could not be decoded as a supported image.")
    except Exception:
        return _ocr_failure("ocr_error", "OCR could not be completed for the supplied image.")


def extract_text_details_from_image_bytes(data: bytes, source: str = "remote_image") -> dict:
    if not data:
        return _ocr_failure("image_not_downloaded", "No image bytes were available for OCR.")
    try:
        with Image.open(BytesIO(data)) as image:
            return _extract_text_details_from_image_object(image, source=source)
    except UnidentifiedImageError:
        return _ocr_failure("invalid_image_format", "Downloaded bytes did not decode into a valid image.")
    except Exception:
        return _ocr_failure("ocr_error", "OCR could not be completed for the downloaded image.")


def _extract_text_details_from_image_object(image: Image.Image, source: str) -> dict:
    decoded = image.copy()
    width, height = decoded.size
    image_format = decoded.format or image.format or "unknown"
    failure_reasons: list[str] = []

    if min(width, height) < 48:
        failure_reasons.append("image_too_small_for_ocr")
        return {
            "image_fetch_status": "downloaded" if source.startswith("http") else "not_applicable",
            "image_decode_status": "decoded",
            "ocr_status": "image_too_small_for_ocr",
            "ocr_text": "",
            "extracted_text": "",
            "ocr_confidence": 0.0,
            "text_detected": False,
            "image_width": width,
            "image_height": height,
            "image_format": image_format,
            "preprocess_status": "skipped",
            "corroboration_status": "not_checked",
            "failure_reasons": failure_reasons,
            "ocr_summary": "The image decoded correctly, but it was too small for reliable OCR.",
        }

    processed = _preprocess_for_ocr(decoded)
    try:
        data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
    except Exception:
        return _ocr_failure("ocr_error", "OCR could not be completed for the decoded image.", width=width, height=height, image_format=image_format)

    words: list[str] = []
    confidences: list[float] = []
    text_items = data.get("text", []) if isinstance(data, dict) else []
    conf_items = data.get("conf", []) if isinstance(data, dict) else []
    for raw_text, raw_conf in zip(text_items, conf_items):
        text = str(raw_text or "").strip()
        if not text:
            continue
        try:
            conf = float(raw_conf)
        except (TypeError, ValueError):
            conf = -1.0
        if conf >= 0:
            confidences.append(conf)
        words.append(text)

    extracted_text = " ".join(words).strip()
    normalized_confidence = round((mean(confidences) / 100.0), 3) if confidences else 0.0
    if not extracted_text:
        failure_reasons.append("ocr_no_text_detected")

    return {
        "image_fetch_status": "downloaded" if source.startswith("http") else "not_applicable",
        "image_decode_status": "decoded",
        "ocr_status": "completed" if extracted_text else "ocr_no_text_detected",
        "ocr_text": extracted_text,
        "extracted_text": extracted_text,
        "ocr_confidence": normalized_confidence,
        "text_detected": bool(extracted_text),
        "image_width": width,
        "image_height": height,
        "image_format": image_format,
        "preprocess_status": "completed",
        "corroboration_status": "not_checked",
        "failure_reasons": failure_reasons,
        "ocr_summary": (
            "Text was extracted successfully after image preprocessing."
            if extracted_text
            else "The image was decoded and preprocessed, but OCR did not detect readable text."
        ),
    }


def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    working = image.convert("RGB")
    grayscale = ImageOps.grayscale(working)
    bbox = ImageOps.invert(grayscale).getbbox()
    if bbox:
        grayscale = grayscale.crop(bbox)
    width, height = grayscale.size
    min_dimension = min(width, height)
    if min_dimension < 900 and min_dimension > 0:
        scale = max(1.0, 900 / float(min_dimension))
        grayscale = grayscale.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    grayscale = ImageOps.autocontrast(grayscale)
    grayscale = ImageEnhance.Contrast(grayscale).enhance(1.35)
    grayscale = grayscale.filter(ImageFilter.MedianFilter(size=3))
    return grayscale.point(lambda pixel: 255 if pixel > 170 else 0)


def _ocr_failure(reason: str, summary: str, width: int | None = None, height: int | None = None, image_format: str | None = None) -> dict:
    return {
        "image_fetch_status": "not_downloaded" if reason == "image_not_downloaded" else "unknown",
        "image_decode_status": "invalid" if reason == "invalid_image_format" else "failed",
        "ocr_status": reason,
        "ocr_text": "",
        "extracted_text": "",
        "ocr_confidence": 0.0,
        "text_detected": False,
        "image_width": width,
        "image_height": height,
        "image_format": image_format,
        "preprocess_status": "failed",
        "corroboration_status": "not_checked",
        "failure_reasons": [reason],
        "ocr_summary": summary,
    }
