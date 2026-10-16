from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from app.config import get_settings
from app.services.content_extraction import ExtractedPayload
from app.services.ocr import extract_text_details_from_image_bytes
from app.utils import canonicalize_url, extract_domain, normalize_text, sha256_text


logger = logging.getLogger(__name__)


@dataclass
class NormalizedSourceMetadata:
    source_name: str
    publisher_or_channel: str | None
    publisher_id: str | None
    author_or_uploader: str | None
    publish_time: str | None
    canonical_url: str | None
    platform_type: str
    source_type: str
    metadata_completeness: float
    provenance_available: bool
    transcript_status: str | None = None


class SourceAdapterService:
    """Normalizes source metadata before verification layers run.

    NewsProof should not treat YouTube, generic webpages, social posts, and
    uploaded media as the same source type. This adapter layer produces a
    normalized source view that downstream scoring and rendering can trust.
    """

    YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
    YOUTUBE_BLOCKED_PAGE_PATTERNS = (
        "our systems have detected unusual traffic",
        "not a robot",
        "captcha",
        "violation of the terms of service",
        "verify you are human",
    )

    def __init__(self) -> None:
        self.settings = get_settings()

    async def adapt_url_payload(self, url: str, payload: ExtractedPayload) -> ExtractedPayload:
        platform = self._platform_type(url, payload.metadata)
        if platform == "youtube":
            return await self._adapt_youtube_payload(url, payload)
        if platform in {"facebook", "instagram", "pinterest", "linkedin", "x"}:
            payload = await self._repair_social_payload(url, payload)
            return self._apply_normalized_metadata(payload, self._build_social_metadata(url, payload))
        return self._apply_normalized_metadata(payload, self._build_webpage_metadata(url, payload))

    def adapt_uploaded_media_payload(self, payload: ExtractedPayload, source_type: str) -> ExtractedPayload:
        metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
        normalized = NormalizedSourceMetadata(
            source_name="uploaded_media",
            publisher_or_channel=None,
            publisher_id=None,
            author_or_uploader=None,
            publish_time=None,
            canonical_url=None,
            platform_type="uploaded_media",
            source_type=source_type,
            metadata_completeness=self._completeness_score(
                [metadata.get("media_type"), metadata.get("forensics"), payload.title]
            ),
            provenance_available=bool(metadata.get("c2pa_credentials") or metadata.get("forensics")),
            transcript_status=metadata.get("transcript_status") if source_type in {"audio", "video"} else None,
        )
        return self._apply_normalized_metadata(payload, normalized)

    async def _adapt_youtube_payload(self, url: str, payload: ExtractedPayload) -> ExtractedPayload:
        video_id = self._youtube_video_id(url)
        metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
        blocked_page_reason = self._youtube_blocked_page_reason(
            "\n".join(
                part
                for part in [
                    str(payload.title or ""),
                    str(payload.main_text or ""),
                    str(metadata.get("description") or ""),
                ]
                if str(part or "").strip()
            )
        )
        blocked_page_detected = bool(blocked_page_reason)
        oembed_data = await self._fetch_youtube_oembed_metadata(url)
        api_data = await self._fetch_youtube_api_metadata(video_id) if video_id else {}
        snippet = api_data.get("snippet", {}) if isinstance(api_data, dict) else {}
        channel = api_data.get("channel", {}) if isinstance(api_data, dict) else {}
        transcript_text = self._youtube_transcript_text(metadata)
        normalized_url = self._normalized_youtube_url(video_id, url)
        youtube_signals = metadata.get("youtube_signals", {}) if isinstance(metadata.get("youtube_signals"), dict) else {}
        payload_title = None if blocked_page_detected else self._clean_youtube_text(payload.title)
        payload_description = None if blocked_page_detected else self._clean_youtube_text(metadata.get("description"))
        title = (
            self._clean_youtube_text(snippet.get("title"))
            or self._clean_youtube_text(oembed_data.get("title"))
            or payload_title
        )
        description = (
            self._clean_youtube_text(snippet.get("description"))
            or payload_description
        )
        publisher = (
            self._clean_youtube_channel_name(snippet.get("channelTitle"))
            or self._clean_youtube_channel_name(channel.get("title"))
            or self._clean_youtube_channel_name(oembed_data.get("author_name"))
            or self._clean_youtube_channel_name(metadata.get("channel_title"))
            or self._clean_youtube_channel_name(metadata.get("publisher_name"))
            or self._clean_youtube_channel_name(payload.author)
        )
        channel_id = str(snippet.get("channelId") or metadata.get("channel_id") or "").strip() or None
        publish_time = self._safe_iso(snippet.get("publishedAt") or payload.publish_time)
        transcript_status = str(
            metadata.get("transcript_status")
            or ("available" if transcript_text else "unavailable")
        )
        channel_verification_status = self._youtube_channel_verification_status(youtube_signals)
        metadata_fetch_status = self._youtube_metadata_fetch_status(api_data, oembed_data, blocked_page_detected)

        if title:
            payload.title = title
        elif blocked_page_detected:
            payload.title = "Video metadata could not be retrieved"
        verification_parts = [part for part in [title, description, publisher, transcript_text] if part]
        payload.main_text = "\n".join(verification_parts[:3] + ([transcript_text] if transcript_text else []))[:4000] if verification_parts else ""
        payload.publish_time = payload.publish_time or self._parse_dt(publish_time)
        payload.canonical_url = normalized_url
        payload.normalized_text = normalize_text(" ".join(part for part in [title, description, publisher, transcript_text] if part))
        payload.content_hash = sha256_text(payload.normalized_text or normalized_url)

        metadata.update(
            {
                "detected_youtube_url": url,
                "normalized_url": normalized_url,
                "metadata_fetch_status": metadata_fetch_status,
                "blocked_page_detected": blocked_page_detected,
                "blocked_page_reason": blocked_page_reason,
                "title": title,
                "description": description,
                "publisher_name": publisher,
                "channel_title": publisher,
                "channel_id": channel_id,
                "channel_verification_status": channel_verification_status,
                "publish_time": publish_time,
                "thumbnails": snippet.get("thumbnails") or oembed_data.get("thumbnail_url") or metadata.get("thumbnails") or {},
                "thumbnail_url": oembed_data.get("thumbnail_url") or self._thumbnail_url_from_snippet(snippet.get("thumbnails")) or metadata.get("thumbnail_url"),
                "transcript_status": transcript_status,
                "transcript_text": transcript_text,
                "transcript_length": len(transcript_text.split()) if transcript_text else 0,
                "youtube_api_enriched": bool(api_data),
                "youtube_oembed_enriched": bool(oembed_data),
                "youtube_video_id": video_id,
            }
        )
        normalized = NormalizedSourceMetadata(
            source_name="youtube",
            publisher_or_channel=publisher,
            publisher_id=channel_id,
            author_or_uploader=publisher,
            publish_time=publish_time,
            canonical_url=normalized_url,
            platform_type="youtube",
            source_type="social_post",
            metadata_completeness=self._completeness_score(
                [title, publisher, channel_id, publish_time, description, metadata.get("thumbnails")]
            ),
            provenance_available=bool(metadata.get("youtube_signals") or metadata.get("c2pa_credentials")),
            transcript_status=transcript_status,
        )
        payload.metadata = metadata
        return self._apply_normalized_metadata(payload, normalized)

    async def _fetch_youtube_oembed_metadata(self, url: str) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=6.0, follow_redirects=True) as client:
                response = await client.get(
                    "https://www.youtube.com/oembed",
                    params={"url": url, "format": "json"},
                )
                response.raise_for_status()
                data = response.json() or {}
                return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.debug("YouTube oEmbed metadata enrichment failed for %s: %s", url, exc)
            return {}

    def _build_webpage_metadata(self, url: str, payload: ExtractedPayload) -> NormalizedSourceMetadata:
        metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
        return NormalizedSourceMetadata(
            source_name="webpage",
            publisher_or_channel=str(metadata.get("publisher_name") or "").strip() or None,
            publisher_id=None,
            author_or_uploader=payload.author,
            publish_time=self._safe_iso(payload.publish_time or metadata.get("publish_time")),
            canonical_url=payload.canonical_url or canonicalize_url(url),
            platform_type="webpage",
            source_type="article",
            metadata_completeness=self._completeness_score(
                [payload.title, payload.author, metadata.get("publisher_name"), payload.publish_time, payload.canonical_url]
            ),
            provenance_available=bool(metadata.get("c2pa_credentials") or metadata.get("remote_media_provenance")),
        )

    def _build_social_metadata(self, url: str, payload: ExtractedPayload) -> NormalizedSourceMetadata:
        metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
        platform = self._platform_type(url, metadata)
        account = str(metadata.get("publisher_name") or metadata.get("account_name") or "").strip() or None
        return NormalizedSourceMetadata(
            source_name=platform,
            publisher_or_channel=account,
            publisher_id=str(metadata.get("account_id") or "").strip() or None,
            author_or_uploader=payload.author or account,
            publish_time=self._safe_iso(payload.publish_time or metadata.get("publish_time")),
            canonical_url=payload.canonical_url or canonicalize_url(url),
            platform_type=platform,
            source_type="social_post",
            metadata_completeness=self._completeness_score(
                [payload.title, account, payload.publish_time, payload.canonical_url]
            ),
            provenance_available=bool(metadata.get("c2pa_credentials") or metadata.get("remote_media_provenance")),
        )

    async def _repair_social_payload(self, url: str, payload: ExtractedPayload) -> ExtractedPayload:
        metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
        platform_preview = await self._fetch_page_preview(url)
        shared_target_url = self._social_shared_target_url(url)
        target_preview = await self._fetch_target_preview(shared_target_url) if shared_target_url else {}
        if platform_preview:
            metadata["platform_preview"] = platform_preview
        if shared_target_url:
            metadata["shared_target_url"] = shared_target_url
        if target_preview:
            metadata["shared_target_preview"] = target_preview
        description = str(metadata.get("description") or "").strip()
        if self._looks_like_social_error_text(description):
            description = ""
        if not description:
            description = str(platform_preview.get("description") or "").strip()
        if not description:
            description = str(target_preview.get("description") or "").strip()
        social_caption = self._social_caption_text(metadata)
        hero = (
            payload.hero_image_url
            or str(metadata.get("hero_image_url") or "").strip()
            or str(platform_preview.get("image_url") or "").strip()
            or str(target_preview.get("image_url") or "").strip()
            or None
        )
        media_assets = metadata.get("media_assets", [])
        if not isinstance(media_assets, list):
            media_assets = []
        if hero and not any(isinstance(asset, dict) and asset.get("url") == hero for asset in media_assets):
            media_assets.insert(
                0,
                {
                    "type": "image",
                    "url": hero,
                    "source": "social_fallback",
                    "caption": description or social_caption or None,
                    "platform": self._platform_type(url, metadata),
                },
            )
        social_ocr_results = await self._collect_social_image_ocr_results(hero, media_assets)
        social_ocr_texts = self._dedupe_text_list(
            [str(item.get("ocr_text") or item.get("extracted_text") or "").strip() for item in social_ocr_results]
        )
        social_ocr = social_ocr_texts[0] if social_ocr_texts else ""
        if social_ocr:
            metadata["social_image_ocr"] = social_ocr
        metadata["social_image_ocr_texts"] = social_ocr_texts
        metadata["social_image_ocr_results"] = social_ocr_results
        metadata["social_image_ocr_result"] = social_ocr_results[0] if social_ocr_results else {
            "image_fetch_status": "not_downloaded",
            "image_decode_status": "not_attempted",
            "ocr_status": "image_not_downloaded",
            "ocr_text": "",
            "ocr_confidence": 0.0,
            "text_detected": False,
            "failure_reasons": ["image_not_downloaded"],
            "ocr_summary": "No remote post image was available for OCR.",
        }
        usable_summary = description or social_caption or social_ocr
        title_looked_broken = self._looks_like_social_error_text(payload.title)
        text_looked_broken = self._looks_like_social_error_text(payload.main_text)
        if title_looked_broken:
            if usable_summary:
                payload.title = usable_summary[:120]
            elif platform_preview.get("title"):
                payload.title = str(platform_preview.get("title"))[:120]
            elif target_preview.get("title"):
                payload.title = str(target_preview.get("title"))[:120]
        if text_looked_broken:
            if usable_summary:
                payload.main_text = usable_summary
                payload.normalized_text = normalize_text(usable_summary)
                payload.content_hash = sha256_text(payload.normalized_text)
        if not payload.main_text and usable_summary:
            payload.main_text = usable_summary
            payload.normalized_text = normalize_text(usable_summary)
            payload.content_hash = sha256_text(payload.normalized_text)

        if hero and not payload.hero_image_url:
            payload.hero_image_url = hero

        metadata["media_assets"] = media_assets
        if usable_summary:
            metadata["summary"] = usable_summary
        if description:
            metadata["description"] = description
        if hero:
            metadata["hero_image_url"] = hero
        if not usable_summary:
            metadata["caption_unavailable_reason"] = "The platform did not expose a usable post caption or description to the server."
        if title_looked_broken or text_looked_broken:
            metadata["fetch_limited"] = True
            metadata["social_payload_repaired"] = True
        payload.metadata = metadata
        return payload

    async def _collect_social_image_ocr_results(self, hero_image_url: str | None, media_assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        image_urls = self._social_image_urls(hero_image_url, media_assets)
        if not image_urls:
            return [
                {
                    "image_fetch_status": "not_downloaded",
                    "image_decode_status": "not_attempted",
                    "ocr_status": "image_not_downloaded",
                    "ocr_text": "",
                    "ocr_confidence": 0.0,
                    "text_detected": False,
                    "failure_reasons": ["image_not_downloaded"],
                    "ocr_summary": "No remote post image was available for OCR.",
                }
            ]
        results: list[dict[str, Any]] = []
        for image_url in image_urls[:4]:
            results.append(await self._fetch_social_image_ocr(image_url))
        return results

    def _social_image_urls(self, hero_image_url: str | None, media_assets: list[dict[str, Any]]) -> list[str]:
        urls: list[str] = []
        if hero_image_url:
            urls.append(str(hero_image_url).strip())
        for asset in media_assets[:6]:
            if not isinstance(asset, dict):
                continue
            if str(asset.get("type") or "").lower() != "image":
                continue
            url = str(asset.get("url") or "").strip()
            if url:
                urls.append(url)
        return self._dedupe_text_list(urls)

    def _dedupe_text_list(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = normalize_text(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(value.strip())
        return deduped

    def _social_shared_target_url(self, url: str) -> str | None:
        parsed = urlparse(url)
        if "facebook.com" not in (parsed.netloc or "").lower():
            return None
        query = parse_qs(parsed.query)
        for key in ("u", "url", "href", "story_url", "link"):
            values = query.get(key) or []
            for value in values:
                decoded = unquote(str(value)).strip()
                if decoded.startswith("http://") or decoded.startswith("https://"):
                    return decoded
        return None

    async def _fetch_target_preview(self, target_url: str | None) -> dict[str, str]:
        if not target_url:
            return {}
        try:
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
                response = await client.get(target_url)
                response.raise_for_status()
            return self._extract_structured_preview(response.text, target_url)
        except Exception as exc:
            logger.debug("Shared target preview fetch failed for %s: %s", target_url, exc)
            return {}

    async def _fetch_page_preview(self, target_url: str | None) -> dict[str, str]:
        if not target_url:
            return {}
        try:
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
                response = await client.get(target_url)
                response.raise_for_status()
            return self._extract_structured_preview(response.text, target_url)
        except Exception as exc:
            logger.debug("Platform preview fetch failed for %s: %s", target_url, exc)
            return {}

    async def _fetch_social_image_ocr(self, image_url: str | None) -> dict[str, Any]:
        if not image_url:
            return {
                "image_fetch_status": "not_downloaded",
                "image_decode_status": "not_attempted",
                "ocr_status": "image_not_downloaded",
                "ocr_text": "",
                "ocr_confidence": 0.0,
                "text_detected": False,
                "failure_reasons": ["image_not_downloaded"],
                "ocr_summary": "No remote post image was available for OCR.",
            }
        try:
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
                response = await client.get(image_url)
                response.raise_for_status()
                content_type = str(response.headers.get("content-type") or "").lower()
                if not content_type.startswith("image/"):
                    return {
                        "image_fetch_status": "downloaded",
                        "image_decode_status": "invalid",
                        "ocr_status": "invalid_image_format",
                        "ocr_text": "",
                        "ocr_confidence": 0.0,
                        "text_detected": False,
                        "failure_reasons": ["invalid_image_format"],
                        "ocr_summary": "The remote media URL did not return an image file for OCR.",
                        "image_url": image_url,
                    }
            result = extract_text_details_from_image_bytes(response.content, source=image_url)
            result["image_url"] = image_url
            return result
        except Exception as exc:
            logger.debug("Social image OCR fetch failed for %s: %s", image_url, exc)
            return {
                "image_fetch_status": "not_downloaded",
                "image_decode_status": "failed",
                "ocr_status": "image_not_downloaded",
                "ocr_text": "",
                "ocr_confidence": 0.0,
                "text_detected": False,
                "failure_reasons": ["image_not_downloaded"],
                "ocr_summary": "The social post image could not be downloaded for OCR.",
                "image_url": image_url,
            }

    def _social_caption_text(self, metadata: dict[str, Any]) -> str:
        assets = metadata.get("media_assets", [])
        if not isinstance(assets, list):
            return ""
        texts: list[str] = []
        for asset in assets[:4]:
            if not isinstance(asset, dict):
                continue
            for key in ("caption", "alt", "description", "label", "title"):
                value = str(asset.get(key) or "").strip()
                if value:
                    texts.append(value)
        deduped: list[str] = []
        seen: set[str] = set()
        for item in texts:
            key = normalize_text(item)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return " ".join(deduped)

    def _looks_like_social_error_text(self, value: str | None) -> bool:
        normalized = normalize_text(value or "")
        if not normalized:
            return True
        markers = (
            "sorry something went wrong",
            "sorry, something went wrong",
            "go back",
            "we're working on getting this fixed",
            "this content isn't available",
            "this page isn't available",
            "log into facebook",
        )
        return any(marker in normalized for marker in markers)

    def _meta_content(self, soup: BeautifulSoup, names: list[str]) -> str | None:
        wanted = {name.lower() for name in names}
        for tag in soup.find_all("meta"):
            content = tag.get("content")
            if not content:
                continue
            for attribute_name in ("name", "property", "itemprop", "http-equiv"):
                value = tag.get(attribute_name)
                if value and str(value).strip().lower() in wanted:
                    return str(content)
        return None

    def _extract_structured_preview(self, html: str, source_url: str) -> dict[str, str]:
        soup = BeautifulSoup(html, "lxml")
        title = (
            self._meta_content(soup, ["og:title", "twitter:title"])
            or self._json_ld_field(soup, ("headline", "name", "title"))
            or (soup.title.get_text(" ", strip=True) if soup.title else "")
        )
        description = (
            self._meta_content(soup, ["og:description", "twitter:description", "description"])
            or self._json_ld_field(soup, ("description", "caption", "articleBody"))
            or self._visible_text_summary(soup)
        )
        image_url = (
            self._meta_content(soup, ["og:image", "og:image:url", "twitter:image"])
            or self._json_ld_field(soup, ("image", "thumbnailUrl"))
            or ""
        )
        canonical = self._canonical_href(soup) or canonicalize_url(source_url) or ""
        return {
            "title": str(title or "").strip(),
            "description": str(description or "").strip(),
            "image_url": str(image_url or "").strip(),
            "canonical_url": str(canonical or "").strip(),
        }

    def _json_ld_field(self, soup: BeautifulSoup, names: tuple[str, ...]) -> str | None:
        wanted = {name.lower() for name in names}
        for node in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = (node.string or node.get_text() or "").strip()
            if not raw:
                continue
            try:
                import json

                payload = json.loads(raw)
            except Exception:
                continue
            value = self._find_json_ld_value(payload, wanted)
            if value:
                return value
        return None

    def _find_json_ld_value(self, payload: Any, wanted: set[str]) -> str | None:
        if isinstance(payload, dict):
            for key, value in payload.items():
                if str(key).lower() in wanted:
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and item.strip():
                                return item
                            if isinstance(item, dict) and item.get("url"):
                                return str(item.get("url"))
                    if isinstance(value, dict):
                        if value.get("url"):
                            return str(value.get("url"))
                        if value.get("@id"):
                            return str(value.get("@id"))
                    if isinstance(value, str) and value.strip():
                        return value
                nested = self._find_json_ld_value(value, wanted)
                if nested:
                    return nested
        elif isinstance(payload, list):
            for item in payload:
                nested = self._find_json_ld_value(item, wanted)
                if nested:
                    return nested
        return None

    def _canonical_href(self, soup: BeautifulSoup) -> str | None:
        link = soup.find("link", attrs={"rel": lambda value: value and "canonical" in str(value).lower()})
        href = link.get("href") if link else None
        return str(href).strip() if href else None

    def _visible_text_summary(self, soup: BeautifulSoup) -> str | None:
        body = soup.body
        if not body:
            return None
        text = body.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if not text:
            return None
        return text[:600]

    async def _fetch_youtube_api_metadata(self, video_id: str | None) -> dict[str, Any]:
        if not video_id or not self.settings.youtube_data_api_key:
            return {}
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                video_resp = await client.get(
                    f"{self.YOUTUBE_API_BASE}/videos",
                    params={
                        "id": video_id,
                        "part": "snippet,contentDetails,status",
                        "key": self.settings.youtube_data_api_key,
                    },
                )
                video_resp.raise_for_status()
                video_items = (video_resp.json() or {}).get("items", [])
                if not video_items:
                    return {}
                video_item = video_items[0]
                snippet = video_item.get("snippet", {}) if isinstance(video_item, dict) else {}
                channel_id = snippet.get("channelId")
                channel_data: dict[str, Any] = {}
                if channel_id:
                    channel_resp = await client.get(
                        f"{self.YOUTUBE_API_BASE}/channels",
                        params={
                            "id": channel_id,
                            "part": "snippet,brandingSettings,statistics",
                            "key": self.settings.youtube_data_api_key,
                        },
                    )
                    channel_resp.raise_for_status()
                    channel_items = (channel_resp.json() or {}).get("items", [])
                    channel_data = channel_items[0] if channel_items else {}
                return {
                    "snippet": snippet,
                    "contentDetails": video_item.get("contentDetails", {}),
                    "status": video_item.get("status", {}),
                    "channel": channel_data.get("snippet", {}) if isinstance(channel_data, dict) else {},
                }
        except Exception as exc:
            logger.warning("YouTube metadata enrichment failed for %s: %s", video_id, exc)
            return {}

    def _youtube_transcript_text(self, metadata: dict[str, Any]) -> str:
        for key in ("transcript_text", "captions_text", "caption_text", "transcript", "captions"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _youtube_blocked_page_reason(self, text: str) -> str | None:
        normalized = normalize_text(text)
        if not normalized:
            return None
        for pattern in self.YOUTUBE_BLOCKED_PAGE_PATTERNS:
            if pattern in normalized:
                return pattern
        return None

    def _clean_youtube_text(self, value: Any) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        if self._youtube_blocked_page_reason(text):
            return None
        return text

    def _clean_youtube_channel_name(self, value: Any) -> str | None:
        text = self._clean_youtube_text(value)
        if not text:
            return None
        if normalize_text(text) == "youtube":
            return None
        return text

    def _youtube_metadata_fetch_status(self, api_data: dict[str, Any], oembed_data: dict[str, Any], blocked_page_detected: bool) -> str:
        if api_data:
            return "api_metadata_loaded"
        if oembed_data:
            return "oembed_metadata_loaded"
        if blocked_page_detected:
            return "blocked_page_detected_metadata_unavailable"
        return "metadata_unavailable"

    def _thumbnail_url_from_snippet(self, thumbnails: Any) -> str | None:
        if not isinstance(thumbnails, dict):
            return None
        for key in ("maxres", "standard", "high", "medium", "default"):
            item = thumbnails.get(key)
            if isinstance(item, dict) and item.get("url"):
                return str(item["url"]).strip()
        return None

    def _apply_normalized_metadata(self, payload: ExtractedPayload, normalized: NormalizedSourceMetadata) -> ExtractedPayload:
        metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
        metadata["normalized_source"] = asdict(normalized)
        metadata["platform_type"] = normalized.platform_type
        metadata["source_type"] = normalized.source_type
        metadata["metadata_completeness"] = normalized.metadata_completeness
        metadata["provenance_available"] = normalized.provenance_available
        if normalized.publisher_or_channel and not metadata.get("publisher_name"):
            metadata["publisher_name"] = normalized.publisher_or_channel
        if normalized.publish_time and not metadata.get("publish_time"):
            metadata["publish_time"] = normalized.publish_time
        if normalized.canonical_url and not payload.canonical_url:
            payload.canonical_url = normalized.canonical_url
        payload.metadata = metadata
        return payload

    def _platform_type(self, url: str | None, metadata: dict[str, Any] | None = None) -> str:
        metadata = metadata or {}
        domain = extract_domain(url or metadata.get("fetched_url") or metadata.get("canonical_url")) or ""
        if "youtube.com" in domain or "youtu.be" in domain:
            return "youtube"
        if "facebook.com" in domain:
            return "facebook"
        if "instagram.com" in domain:
            return "instagram"
        if "pinterest." in domain:
            return "pinterest"
        if "linkedin.com" in domain:
            return "linkedin"
        if "x.com" in domain or "twitter.com" in domain:
            return "x"
        return "webpage"

    def _youtube_video_id(self, url: str) -> str | None:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if "youtu.be" in host:
            return parsed.path.strip("/").split("/")[0] or None
        if "youtube.com" not in host:
            return None
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith("/embed/"):
            return parsed.path.split("/embed/", 1)[1].split("/", 1)[0] or None
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/shorts/", 1)[1].split("/", 1)[0] or None
        return None

    def _normalized_youtube_url(self, video_id: str | None, original_url: str) -> str:
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
        return canonicalize_url(original_url)

    def _youtube_channel_verification_status(self, youtube_signals: dict[str, Any]) -> str:
        status = str(youtube_signals.get("channel_verification_status") or "").strip().lower()
        if status in {"verified", "not_confirmed", "not_available"}:
            return status
        if youtube_signals.get("channel_verified") is True:
            return "verified"
        return "not_confirmed"

    def _safe_iso(self, value: Any) -> str | None:
        if isinstance(value, datetime):
            return value.isoformat()
        if not value:
            return None
        try:
            return date_parser.parse(str(value)).isoformat()
        except Exception:
            return str(value)

    def _parse_dt(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return date_parser.parse(value)
        except Exception:
            return None

    def _completeness_score(self, fields: list[Any]) -> float:
        if not fields:
            return 0.0
        present = sum(1 for field in fields if field not in (None, "", [], {}))
        return round(present / len(fields), 3)
