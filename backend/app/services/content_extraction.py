from dataclasses import dataclass
from datetime import datetime
import json
import re
import socket
import ssl
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from readability import Document

from app.config import get_settings
from app.services.soft_binding import SoftBindingRecoveryService
from app.utils import canonicalize_url, extract_domain, normalize_text, sha256_text


@dataclass
class ExtractedPayload:
    title: str | None
    author: str | None
    publish_time: datetime | None
    canonical_url: str | None
    main_text: str
    hero_image_url: str | None
    normalized_text: str
    content_hash: str
    metadata: dict


@dataclass
class UrlFetchError(Exception):
    kind: str
    message: str
    status_code: int | None = None
    extracted_title: str | None = None

    def __str__(self) -> str:
        return self.message


async def fetch_url_content(url: str, fast_mode: bool = False) -> ExtractedPayload:
    settings = get_settings()
    timeout = httpx.Timeout(6.0 if fast_mode else settings.request_timeout_seconds)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
            response = await client.get(url)
    except httpx.TimeoutException as exc:
        raise UrlFetchError(kind="timeout", message=f"Timed out while fetching {url}") from exc
    except httpx.HTTPError as exc:
        raise UrlFetchError(kind="network", message=f"Could not fetch {url}: {exc}") from exc

    content_type = str(response.headers.get("content-type") or "")
    response_kind = _response_kind_from_content_type(content_type)
    if response_kind in {"image", "video", "audio"}:
        raise UrlFetchError(
            kind="non_html_content",
            message=f"Source URL returned non-HTML content ({content_type}) for {url}",
            status_code=response.status_code,
        )
    html = response.text
    payload = _build_payload_from_html(url, html)
    payload.metadata["url_fetch_debug"] = {
        "submitted_url": url,
        "normalized_url": canonicalize_url(url),
        "final_redirected_url": str(response.url),
        "http_status_code": response.status_code,
        "content_type": content_type,
        "response_size_bytes": len(response.content or b""),
        "raw_html_length": len(html or ""),
        "readable_html_present": bool(_parse_html(html or "")),
        "raw_preview": (html or "")[:500],
        "exception_message": None,
    }
    payload.metadata["fetch_status_code"] = response.status_code
    payload.metadata["final_redirected_url"] = str(response.url)
    payload.metadata["content_type"] = content_type
    payload.metadata["transport_evidence"] = _fetch_transport_evidence(str(response.url))
    remote_media_provenance = await _inspect_remote_media_assets(
        client_headers=headers,
        media_assets=payload.metadata.get("media_assets", []),
        existing_credentials=payload.metadata.get("c2pa_credentials", []),
        page_context=payload.metadata,
        timeout=timeout,
        max_assets=1 if fast_mode else 4,
    )
    payload.metadata["remote_media_provenance"] = remote_media_provenance
    payload.metadata["fast_mode"] = fast_mode
    payload.metadata["c2pa_credentials"] = _merge_content_credentials(
        payload.metadata.get("c2pa_credentials", []),
        _credentials_from_remote_media(remote_media_provenance),
    )
    if response.status_code >= 400:
        if _payload_has_meaningful_content(payload):
            payload.metadata["fetch_limited"] = True
            payload.metadata["http_status"] = response.status_code
            return payload
        raise UrlFetchError(
            kind="blocked_by_site" if response.status_code in {401, 403} else "fetch_failed",
            message=f"Source site responded with HTTP {response.status_code} for {url}",
            status_code=response.status_code,
            extracted_title=payload.title,
        )
    return payload


def _fetch_transport_evidence(url: str) -> dict:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        return {"https_enabled": parsed.scheme.lower() == "https"}

    host = parsed.hostname
    port = parsed.port or 443
    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=5.0) as sock:
            with context.wrap_socket(sock, server_hostname=host) as tls_sock:
                cert = tls_sock.getpeercert()
        return {
            "https_enabled": True,
            "certificate_subject": _format_certificate_name(cert.get("subject")),
            "certificate_issuer": _format_certificate_name(cert.get("issuer")),
            "certificate_valid_from": cert.get("notBefore"),
            "certificate_valid_to": cert.get("notAfter"),
            "certificate_hostname_match": True,
        }
    except Exception:
        return {
            "https_enabled": True,
            "certificate_hostname_match": None,
        }


async def _inspect_remote_media_assets(
    client_headers: dict[str, str],
    media_assets: list[dict],
    existing_credentials: list[dict],
    page_context: dict,
    timeout: httpx.Timeout,
    max_assets: int,
) -> list[dict]:
    if not isinstance(media_assets, list) or not media_assets:
        return []

    findings: list[dict] = []
    seen_urls: set[str] = set()
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=client_headers) as client:
        for asset in media_assets[:max_assets]:
            if not isinstance(asset, dict):
                continue
            url = str(asset.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            finding = await _inspect_remote_media_asset(client, asset, existing_credentials, page_context)
            if finding:
                findings.append(finding)
    return findings


async def _inspect_remote_media_asset(
    client: httpx.AsyncClient,
    asset: dict,
    existing_credentials: list[dict],
    page_context: dict,
) -> dict | None:
    url = str(asset.get("url") or "").strip()
    if not url:
        return None

    candidate_urls = _candidate_media_urls(asset, url)
    primary_url = candidate_urls[0]
    findings: dict[str, object] = {
        "url": primary_url,
        "type": asset.get("type"),
        "source": asset.get("source"),
        "signals": [],
        "checked": True,
        "checked_candidates": candidate_urls,
        "platform": _platform_from_url(primary_url),
    }

    best_result: dict[str, object] | None = None
    for candidate_url in candidate_urls:
        candidate_result = await _inspect_single_media_candidate(client, candidate_url, asset, existing_credentials, page_context)
        if not best_result:
            best_result = candidate_result
        if str(candidate_result.get("verification_state") or "") in {
            "credentials_verified",
            "remote_manifest_verified",
            "sidecar_manifest_verified",
        }:
            best_result = candidate_result
            break
        if candidate_result.get("direct_media_response") and not best_result.get("direct_media_response"):
            best_result = candidate_result
        elif candidate_result.get("evaluation_state") == "checked_no_credentials" and best_result.get("evaluation_state") != "checked_no_credentials":
            best_result = candidate_result

    if not best_result:
        findings["checked"] = False
        findings["evaluation_state"] = "not_evaluated"
        findings["description"] = "The media asset could not be evaluated for content credentials."
        return findings

    findings.update(best_result)
    findings["checked_candidates"] = candidate_urls
    findings["description"] = _describe_remote_provenance(findings)
    return findings


async def _inspect_single_media_candidate(
    client: httpx.AsyncClient,
    url: str,
    asset: dict,
    existing_credentials: list[dict],
    page_context: dict,
) -> dict:
    platform = _platform_from_url(url)
    linkedin_page_signal = _linkedin_page_signal_from_context(page_context) if platform == "linkedin" else {}
    youtube_page_signal = _youtube_signal_from_context(page_context) if platform == "youtube" else {}
    findings: dict[str, object] = {
        "url": url,
        "type": asset.get("type"),
        "source": asset.get("source"),
        "signals": [],
        "checked": True,
        "platform": platform,
        "evaluation_state": "not_evaluated",
        "direct_media_response": False,
        "verification_state": "checked_no_retrievable_credentials",
        "page_level_platform_signal_found": bool(linkedin_page_signal.get("found")),
        "page_level_platform_signal_markers": linkedin_page_signal.get("markers", []),
        "youtube_page_provenance_signal_found": bool(youtube_page_signal.get("page_level_provenance_signal")),
        "youtube_original_content_claim": bool(youtube_page_signal.get("original_content_claim")),
        "youtube_license_label": youtube_page_signal.get("license_label"),
        "sidecar_tried_urls": [],
    }

    headers = {}
    head_error: str | None = None
    try:
        head_response = await client.head(url)
        headers = dict(head_response.headers)
        findings["http_status"] = head_response.status_code
        findings["final_fetched_url"] = str(head_response.url)
    except httpx.HTTPError as exc:
        head_error = str(exc)

    header_signals = _header_c2pa_indicators(headers)
    if header_signals:
        findings["signals"] = header_signals.copy()
    if headers.get("content-type"):
        findings["content_type"] = headers.get("content-type")
    if headers.get("content-length"):
        findings["content_length"] = headers.get("content-length")
    if headers.get("etag"):
        findings["etag"] = headers.get("etag")
    if headers.get("last-modified"):
        findings["last_modified"] = headers.get("last-modified")
    if headers.get("accept-ranges"):
        findings["accept_ranges"] = headers.get("accept-ranges")

    header_content_type = str(headers.get("content-type") or "").lower()
    if header_content_type:
        findings["response_kind"] = _response_kind_from_content_type(header_content_type)
        findings["direct_media_response"] = findings["response_kind"] in {"image", "video", "audio"}

    manifest_url = _header_manifest_url(headers, url)
    if manifest_url:
        findings["manifest_url"] = manifest_url
        findings["signals"] = [*list(findings["signals"]), "manifest_link_header"]
        manifest_summary = await _fetch_manifest_summary(client, manifest_url)
        if manifest_summary:
            findings["manifest_summary"] = manifest_summary
        manifest_ok, manifest_details = await _verify_remote_manifest_reference(client, manifest_url)
        findings["remote_manifest_ref_found"] = True
        findings["remote_manifest_verified"] = manifest_ok
        findings["remote_manifest_details"] = manifest_details
        if manifest_ok:
            findings["evidence_found"] = True
            findings["verification_state"] = "remote_manifest_verified"
            findings["evaluation_state"] = "credentials_detected"
            findings["forensics"] = _build_remote_media_forensics(asset, findings, None)
            findings["note"] = _provenance_note(findings)
            return findings
        if manifest_details.get("status") == "verification_failed":
            findings["evidence_found"] = False
            findings["verification_state"] = "verification_failed"
            findings["evaluation_state"] = "checked_failed"
            findings["forensics"] = _build_remote_media_forensics(asset, findings, None)
            findings["note"] = _provenance_note(findings)
            return findings

    byte_signals: list[str] = []
    sample: bytes | None = None
    try:
        sample = await _fetch_media_provenance_sample(client, url, findings)
        byte_signals = _byte_c2pa_indicators(sample)
    except httpx.HTTPError:
        byte_signals = []

    if byte_signals:
        findings["signals"] = sorted(set([*list(findings["signals"]), *byte_signals]))

    linked_credentials = [
        credential
        for credential in existing_credentials
        if isinstance(credential, dict) and str(credential.get("url") or "").strip() == manifest_url
    ]
    if linked_credentials:
        findings["linked_credentials"] = linked_credentials

    response_kind = str(findings.get("response_kind") or "")
    if response_kind and response_kind not in {"image", "video", "audio"}:
        if platform == "youtube" and (
            findings["youtube_page_provenance_signal_found"]
            or findings["youtube_original_content_claim"]
            or _is_youtube_player_or_page_url(url)
        ):
            findings["evidence_found"] = True
            findings["verification_state"] = "platform_signal_detected_manifest_not_retrievable"
            findings["evaluation_state"] = "platform_signal_detected"
        else:
            findings["evidence_found"] = False
            findings["verification_state"] = "could_not_evaluate_asset"
            findings["evaluation_state"] = "not_evaluated"
        findings["forensics"] = _build_remote_media_forensics(asset, findings, sample)
    elif byte_signals:
        findings["evidence_found"] = True
        findings["verification_state"] = "credentials_verified"
        findings["evaluation_state"] = "credentials_detected"
        findings["forensics"] = _build_remote_media_forensics(asset, findings, sample)
    else:
        sidecar_urls = _candidate_sidecar_manifest_urls(url)
        findings["sidecar_tried_urls"] = sidecar_urls
        sidecar_url, sidecar_details = await _verify_sidecar_manifest(client, sidecar_urls)
        findings["sidecar_manifest_verified"] = bool(sidecar_url)
        findings["sidecar_manifest_url"] = sidecar_url
        findings["sidecar_manifest_details"] = sidecar_details
        if sidecar_url:
            findings["evidence_found"] = True
            findings["verification_state"] = "sidecar_manifest_verified"
            findings["evaluation_state"] = "credentials_detected"
        elif platform == "linkedin" and findings["page_level_platform_signal_found"]:
            findings["evidence_found"] = False
            findings["verification_state"] = "platform_signal_detected_manifest_not_retrievable"
            findings["evaluation_state"] = "platform_restricted"
        elif platform == "youtube" and (
            findings["youtube_page_provenance_signal_found"]
            or findings["youtube_original_content_claim"]
            or _is_youtube_player_or_page_url(url)
        ):
            findings["evidence_found"] = True
            findings["verification_state"] = "platform_signal_detected_manifest_not_retrievable"
            findings["evaluation_state"] = "platform_signal_detected"
        elif findings.get("direct_media_response"):
            findings["evidence_found"] = False
            findings["verification_state"] = "checked_no_retrievable_credentials"
            findings["evaluation_state"] = "checked_no_credentials"
        else:
            findings["evidence_found"] = False
            findings["verification_state"] = "could_not_evaluate_asset"
            findings["evaluation_state"] = "not_evaluated"

        soft_binding = SoftBindingRecoveryService().recoverDecoupledManifest(sample, url, str(findings.get("content_type") or None))
        findings["soft_binding_status"] = soft_binding.status
        findings["soft_binding_generated"] = bool(soft_binding.binding)
        findings["soft_binding_repository_configured"] = soft_binding.repository_configured
        findings["soft_binding_reason"] = soft_binding.reason
        findings["forensics"] = _build_remote_media_forensics(asset, findings, sample)

    if head_error and not headers:
        findings["head_error"] = head_error
    findings["note"] = _provenance_note(findings)
    return findings


def build_url_fallback_payload(url: str, title: str | None = None) -> ExtractedPayload:
    parsed = urlparse(url)
    host = parsed.netloc or "submitted source"
    headline = title or _title_from_path(parsed.path) or host
    main_text = f"Submitted URL from {host}. Full article text could not be fetched server-side."
    normalized = normalize_text(f"{headline} {url}")
    return ExtractedPayload(
        title=headline,
        author=None,
        publish_time=None,
        canonical_url=canonicalize_url(url),
        main_text=main_text,
        hero_image_url=None,
        normalized_text=normalized,
        content_hash=sha256_text(normalized),
        metadata={
            "fetched_url": url,
            "fetch_limited": True,
            "source_type": "url_fallback",
            "url_fetch_debug": {
                "submitted_url": url,
                "normalized_url": canonicalize_url(url),
                "final_redirected_url": None,
                "http_status_code": None,
                "content_type": None,
                "response_size_bytes": 0,
                "raw_html_length": 0,
                "readable_html_present": False,
                "raw_preview": "",
                "exception_message": "fallback_payload_created_without_fetchable_article_body",
            },
        },
    )


def _build_payload_from_html(url: str, html: str) -> ExtractedPayload:
    try:
        doc = Document(html)
        summary_html = doc.summary()
    except Exception:
        doc = None
        summary_html = ""
    soup = _parse_html(summary_html)
    full_soup = _parse_html(html)
    if soup is None or full_soup is None:
        return build_url_fallback_payload(url)
    _remove_wayback_noise(soup)
    _remove_wayback_noise(full_soup)

    json_ld = _extract_article_json_ld(full_soup)
    fallback_title = _clean_title_text(full_soup.title.string) if full_soup.title and full_soup.title.string else None
    title = _select_title(full_soup, json_ld, doc, fallback_title)
    author = (
        _meta_content(
            full_soup,
            ["author", "article:author", "parsely-author", "byl", "sailthru.author", "dc.creator", "twitter:creator"],
        )
        or _json_ld_author(json_ld)
        or _visible_author(full_soup)
    )
    publisher_name = (
        _meta_content(
            full_soup,
            ["og:site_name", "application-name", "publisher", "parsely-site", "al:android:app_name"],
        )
        or _json_ld_publisher(json_ld)
        or _visible_publisher(full_soup)
    )
    publish_time_raw = _meta_content(
        full_soup,
        [
            "article:published_time",
            "pubdate",
            "date",
            "parsely-pub-date",
            "publication_date",
            "dc.date",
            "displaydate",
            "sailthru.date",
        ],
    ) or json_ld.get("datePublished") or _visible_publish_time(full_soup)
    publish_time = _parse_date(publish_time_raw)
    canonical = _meta_content(full_soup, ["og:url"]) or _canonical_href(full_soup) or url
    media_assets = _extract_media_assets(full_soup, canonical or url)
    hero_image = _meta_content(full_soup, ["og:image", "twitter:image"]) or _first_media_url(media_assets, "image")
    c2pa_credentials = _extract_c2pa_credentials(full_soup, canonical or url)
    related_links = _extract_related_links(full_soup, canonical or url)
    linkedin_page_signal = _detect_linkedin_page_signal(full_soup, canonical or url)
    youtube_signals = _extract_youtube_signals(full_soup, canonical or url, html)
    if youtube_signals.get("page_level_provenance_signal"):
        c2pa_credentials = _merge_content_credentials(
            c2pa_credentials,
            [
                {
                    "source": "youtube_page",
                    "type": "platform_signal",
                    "url": canonicalize_url(canonical or url),
                    "description": "YouTube indicates content credentials or provenance information in the page experience.",
                }
            ],
        )
    readable_text = soup.get_text(" ", strip=True)
    article_text = _extract_visible_article_text(full_soup)
    description = _meta_content(full_soup, ["description", "og:description", "twitter:description"])
    main_text, extraction_method, extraction_candidates = _extract_layered_article_text(
        full_soup=full_soup,
        readable_text=readable_text,
        article_text=article_text,
        json_ld=json_ld,
        title=title,
        author=author,
        publisher_name=publisher_name,
        description=description,
    )
    normalized_source = main_text
    if not normalized_source.strip():
        normalized_source = " ".join(part for part in [title, author, publisher_name] if part)
    normalized = normalize_text(normalized_source)
    keywords = _extract_keywords(full_soup, json_ld)
    section = _meta_content(full_soup, ["article:section", "section", "parsely-section"]) or _json_ld_section(json_ld)
    original_archived_url = _wayback_original_url(url)

    return ExtractedPayload(
        title=title,
        author=author,
        publish_time=publish_time,
        canonical_url=canonicalize_url(canonical),
        main_text=main_text,
        hero_image_url=hero_image,
        normalized_text=normalized,
        content_hash=sha256_text(normalized),
        metadata={
            "fetched_url": url,
            "document_title": title,
            "source_domain": extract_domain(canonical) or extract_domain(url),
            "original_archived_url": original_archived_url,
            "publisher_name": publisher_name,
            "author": author,
            "publish_time": publish_time.isoformat() if publish_time else None,
            "keywords": keywords,
            "description": description,
            "section": section,
            "hero_image_url": hero_image,
            "media_assets": media_assets,
            "c2pa_credentials": c2pa_credentials,
            "related_links": related_links,
            "linkedin_page_signal": linkedin_page_signal,
            "youtube_signals": youtube_signals,
            "extraction_method": extraction_method,
            "cleaned_article_length": len(main_text or ""),
            "cleaned_article_preview": (main_text or "")[:500],
            "extraction_candidates": extraction_candidates,
            "extraction_quality": _summarize_extraction_quality(
                title=title,
                author=author,
                publisher_name=publisher_name,
                publish_time=publish_time,
                main_text=main_text,
            ),
        },
    )


def build_text_payload(text: str) -> ExtractedPayload:
    normalized = normalize_text(text)
    preview_title = text[:80].strip()
    return ExtractedPayload(
        title=preview_title or "Submitted text",
        author=None,
        publish_time=None,
        canonical_url=None,
        main_text=text,
        hero_image_url=None,
        normalized_text=normalized,
        content_hash=sha256_text(normalized),
        metadata={"source_type": "text_submission"},
    )


def _meta_content(soup: BeautifulSoup, names: list[str]) -> str | None:
    wanted = {name.lower() for name in names}
    for tag in soup.find_all("meta"):
        content = tag.get("content")
        if not content:
            continue
        for attribute_name in ("name", "property", "itemprop", "http-equiv"):
            value = tag.get(attribute_name)
            if value and str(value).strip().lower() in wanted:
                return content
    return None


def _canonical_href(soup: BeautifulSoup) -> str | None:
    link = soup.find("link", attrs={"rel": "canonical"})
    if link and link.get("href"):
        return link["href"]
    return None


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return date_parser.parse(value)
    except Exception:
        return None


def _extract_title_from_html(html: str) -> str | None:
    soup = _parse_html(html or "")
    if soup is None:
        return None
    json_ld = _extract_article_json_ld(soup)
    fallback_title = _clean_title_text(soup.title.string) if soup.title and soup.title.string else None
    return _select_title(soup, json_ld, None, fallback_title)


def _extract_readable_text(html: str) -> str:
    if not html:
        return ""
    doc = Document(html)
    summary_html = doc.summary()
    soup = _parse_html(summary_html)
    if soup is None:
        return ""
    return soup.get_text(" ", strip=True)


def _parse_html(html: str) -> BeautifulSoup | None:
    if not html:
        return None
    for parser in ("lxml", "html.parser"):
        try:
            return BeautifulSoup(html, parser)
        except Exception:
            continue
    return None


def _title_from_path(path: str) -> str | None:
    slug = path.strip("/").split("/")[-1]
    if not slug:
        return None
    return slug.replace("-", " ").replace("_", " ").strip().title() or None


def _select_title(soup: BeautifulSoup, json_ld: dict, doc: Document | None, fallback_title: str | None) -> str | None:
    meta_title = _meta_content(soup, ["og:title", "twitter:title", "parsely-title", "hdl", "headline"])
    visible_title = _visible_headline(soup)
    json_ld_title = _clean_title_text(json_ld.get("headline")) if isinstance(json_ld.get("headline"), str) else None
    readability_title = _clean_title_text(doc.short_title()) if doc else None
    for candidate in [meta_title, json_ld_title, visible_title, readability_title, fallback_title]:
        cleaned = _clean_title_text(candidate)
        if cleaned:
            return cleaned
    return None


def _extract_article_json_ld(soup: BeautifulSoup) -> dict:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue

        for candidate in _flatten_json_ld(parsed):
            if not isinstance(candidate, dict):
                continue
            candidate_type = candidate.get("@type")
            if isinstance(candidate_type, list):
                types = {item.lower() for item in candidate_type if isinstance(item, str)}
            else:
                types = {str(candidate_type).lower()} if candidate_type else set()

            if {"newsarticle", "article", "reportage"}.intersection(types):
                return candidate
    return {}


def _flatten_json_ld(value):
    if isinstance(value, list):
        for item in value:
            yield from _flatten_json_ld(item)
    elif isinstance(value, dict):
        if "@graph" in value and isinstance(value["@graph"], list):
            for item in value["@graph"]:
                yield from _flatten_json_ld(item)
        yield value


def _json_ld_author(payload: dict) -> str | None:
    author = payload.get("author")
    if isinstance(author, dict):
        return author.get("name")
    if isinstance(author, list):
        names = [item.get("name") for item in author if isinstance(item, dict) and item.get("name")]
        return ", ".join(names) if names else None
    if isinstance(author, str):
        return author
    return None


def _json_ld_publisher(payload: dict) -> str | None:
    publisher = payload.get("publisher")
    if isinstance(publisher, dict):
        return publisher.get("name")
    if isinstance(publisher, list):
        names = [item.get("name") for item in publisher if isinstance(item, dict) and item.get("name")]
        return ", ".join(names) if names else None
    if isinstance(publisher, str):
        return publisher
    return None


def _json_ld_section(payload: dict) -> str | None:
    for key in ("articleSection", "section"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _visible_headline(soup: BeautifulSoup) -> str | None:
    for selector in ["h1", '[data-testid="headline"]', '[itemprop="headline"]']:
        node = soup.select_one(selector)
        text = _clean_node_text(node)
        if text:
            return text
    return None


def _visible_author(soup: BeautifulSoup) -> str | None:
    selectors = [
        '[itemprop="author"]',
        '[rel="author"]',
        '[data-testid*="author"]',
        '[class*="author"]',
        '[class*="byline"]',
    ]
    for selector in selectors:
        for node in soup.select(selector):
            text = _clean_node_text(node)
            if text and len(text) <= 120:
                return text.removeprefix("By ").strip()
    return None


def _visible_publisher(soup: BeautifulSoup) -> str | None:
    selectors = [
        '[itemprop="publisher"]',
        '[data-testid*="source"]',
        '[class*="publisher"]',
        '[class*="source"]',
    ]
    for selector in selectors:
        for node in soup.select(selector):
            text = _clean_node_text(node)
            if text and len(text) <= 120:
                return text
    return None


def _visible_publish_time(soup: BeautifulSoup) -> str | None:
    time_node = soup.find("time")
    if time_node:
        return time_node.get("datetime") or _clean_node_text(time_node)

    selectors = ['[data-testid*="timestamp"]', '[class*="timestamp"]', '[class*="date"]', '[class*="time"]']
    for selector in selectors:
        for node in soup.select(selector):
            text = node.get("datetime") or _clean_node_text(node)
            if text and len(text) <= 80:
                return text
    return None


def _extract_keywords(soup: BeautifulSoup, json_ld: dict) -> list[str]:
    values: list[str] = []
    for meta_name in ["keywords", "news_keywords", "article:tag"]:
        meta_value = _meta_content(soup, [meta_name])
        if meta_value:
            values.extend([item.strip() for item in meta_value.split(",") if item.strip()])

    json_ld_keywords = json_ld.get("keywords")
    if isinstance(json_ld_keywords, str):
        values.extend([item.strip() for item in json_ld_keywords.split(",") if item.strip()])
    elif isinstance(json_ld_keywords, list):
        values.extend([str(item).strip() for item in json_ld_keywords if str(item).strip()])

    deduped: list[str] = []
    seen = set()
    for value in values:
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(value)
    return deduped


def _extract_visible_article_text(soup: BeautifulSoup) -> str:
    containers = [
        '[itemprop="articleBody"]',
        "article",
        "main",
        '[data-testid="article-body"]',
        '[class*="article-body"]',
        '[class*="story-body"]',
        '[class*="post-content"]',
        '[class*="entry-content"]',
        '[class*="article" i]',
        '[class*="story" i]',
        '[class*="content" i]',
        '[class*="post" i]',
        '[class*="entry" i]',
        '[class*="news" i]',
        '[id*="article" i]',
        '[id*="story" i]',
        '[id*="content" i]',
        '[id*="post" i]',
        '[id*="entry" i]',
        '[id*="news" i]',
    ]
    best = ""
    for selector in containers:
        node = soup.select_one(selector)
        text = _clean_article_node_text(node)
        if text and len(text) > len(best):
            best = text
    if best and len(best) > 120:
        return best
    return _extract_paragraph_fallback_text(soup)


def _extract_layered_article_text(
    full_soup: BeautifulSoup,
    readable_text: str,
    article_text: str,
    json_ld: dict,
    title: str | None,
    author: str | None,
    publisher_name: str | None,
    description: str | None,
) -> tuple[str, str, list[dict]]:
    candidates: list[tuple[str, str]] = []
    if isinstance(json_ld.get("articleBody"), str):
        candidates.append(("json_ld_article_body", json_ld.get("articleBody", "")))
    if article_text:
        candidates.append(("beautifulsoup_article_containers", article_text))
    if readable_text:
        candidates.append(("readability_lxml", readable_text))
    paragraph_text = _extract_paragraph_fallback_text(full_soup)
    if paragraph_text:
        candidates.append(("beautifulsoup_paragraph_fallback", paragraph_text))
    if description:
        candidates.append(("meta_description_limited", description))

    seen: set[str] = set()
    scored: list[tuple[int, str, str]] = []
    debug: list[dict] = []
    for method, raw_text in candidates:
        cleaned = _normalize_article_text(raw_text)
        if not cleaned:
            continue
        key = normalize_text(cleaned[:1000])
        if key in seen:
            continue
        seen.add(key)
        blocked = _looks_like_block_page(cleaned)
        score = _article_text_quality_score(cleaned, title)
        debug.append(
            {
                "method": method,
                "length": len(cleaned),
                "word_count": len(cleaned.split()),
                "looks_blocked": blocked,
                "quality_score": score,
                "preview": cleaned[:240],
            }
        )
        if not blocked:
            scored.append((score, method, cleaned))

    if scored:
        scored.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
        _, method, text = scored[0]
        return text, method, debug

    metadata_summary = " ".join(part for part in [title, description, author, publisher_name] if part).strip()
    if metadata_summary:
        cleaned = _normalize_article_text(metadata_summary)
        debug.append(
            {
                "method": "metadata_title_description_fallback",
                "length": len(cleaned),
                "word_count": len(cleaned.split()),
                "looks_blocked": False,
                "quality_score": _article_text_quality_score(cleaned, title),
                "preview": cleaned[:240],
            }
        )
        return cleaned, "metadata_title_description_fallback", debug
    return "", "extraction_empty", debug


def _extract_paragraph_fallback_text(soup: BeautifulSoup) -> str:
    clone = _parse_html(str(soup))
    if clone is None:
        return ""
    _remove_noise_nodes(clone)
    selectors = [
        "article p",
        "main p",
        '[itemprop="articleBody"] p',
        '[class*="article" i] p',
        '[class*="story" i] p',
        '[class*="content" i] p',
        '[class*="post" i] p',
        '[class*="entry" i] p',
        '[class*="news" i] p',
        '[id*="article" i] p',
        '[id*="story" i] p',
        '[id*="content" i] p',
        "p",
    ]
    best = ""
    for selector in selectors:
        paragraphs = []
        for node in clone.select(selector):
            text = _clean_node_text(node)
            if not text:
                continue
            normalized = _normalize_article_text(text)
            if len(normalized) < 35:
                continue
            if _looks_like_boilerplate_text(normalized):
                continue
            paragraphs.append(normalized)
        combined = _normalize_article_text(" ".join(dict.fromkeys(paragraphs)))
        if len(combined) > len(best):
            best = combined
        if len(best) >= 800:
            break
    return best


def _extract_media_assets(soup: BeautifulSoup, base_url: str) -> list[dict]:
    assets: list[dict] = []
    seen: set[tuple[str, str]] = set()
    social_video_post = _looks_like_social_video_post(base_url)

    meta_assets = [
        (
            "video",
            _meta_content(soup, ["og:video", "og:video:url", "twitter:player:stream", "twitter:player"]),
            {"source": "meta", "embed": True},
        ),
    ]
    if not social_video_post:
        meta_assets.insert(0, ("image", _meta_content(soup, ["og:image", "og:image:url", "twitter:image"]), {"source": "meta"}))

    for media_type, raw_url, metadata in meta_assets:
        _append_media_asset(
            assets,
            seen,
            media_type=media_type,
            raw_url=raw_url,
            base_url=base_url,
            metadata=metadata,
        )

    containers = soup.select("article, main, figure, [itemprop='articleBody']")
    if not containers:
        containers = [soup]

    for container in containers:
        for image_node in container.select("img"):
            figure = image_node.find_parent("figure")
            caption = _clean_node_text(figure.find("figcaption")) if figure else None
            _append_media_asset(
                assets,
                seen,
                media_type="image",
                raw_url=image_node.get("src") or image_node.get("data-src") or image_node.get("srcset"),
                base_url=base_url,
                metadata={
                    "source": "article",
                    "alt": image_node.get("alt"),
                    "caption": caption,
                    "width": image_node.get("width"),
                    "height": image_node.get("height"),
                    "title": image_node.get("title"),
                },
            )

        for video_node in container.select("video"):
            figure = video_node.find_parent("figure")
            caption = _clean_node_text(figure.find("figcaption")) if figure else None
            source_node = video_node.find("source")
            _append_media_asset(
                assets,
                seen,
                media_type="video",
                raw_url=video_node.get("src") or (source_node.get("src") if source_node else None),
                base_url=base_url,
                metadata={
                    "source": "article",
                    "caption": caption,
                    "poster": _resolve_media_url(video_node.get("poster"), base_url),
                    "width": video_node.get("width"),
                    "height": video_node.get("height"),
                    "mime_type": source_node.get("type") if source_node else video_node.get("type"),
                    "title": video_node.get("title"),
                },
            )

        for iframe_node in container.select("iframe"):
            embed_url = _resolve_media_url(iframe_node.get("src"), base_url)
            if not embed_url or not _looks_like_video_embed(embed_url):
                continue
            figure = iframe_node.find_parent("figure")
            caption = _clean_node_text(figure.find("figcaption")) if figure else None
            _append_media_asset(
                assets,
                seen,
                media_type="video",
                raw_url=embed_url,
                base_url=base_url,
                metadata={
                    "source": "article_embed",
                    "caption": caption,
                    "width": iframe_node.get("width"),
                    "height": iframe_node.get("height"),
                    "title": iframe_node.get("title"),
                    "embed": True,
                    "platform": _platform_from_url(embed_url),
                },
            )

    if social_video_post and not any(asset.get("type") == "video" for asset in assets):
        _append_media_asset(
            assets,
            seen,
            media_type="video",
            raw_url=base_url,
            base_url=base_url,
            metadata={
                "source": "social_post",
                "embed": True,
                "platform": _platform_from_url(base_url),
                "title": _meta_content(soup, ["og:title", "twitter:title"]),
            },
        )

    if social_video_post:
        poster_url = _meta_content(soup, ["og:image", "og:image:url", "twitter:image"])
        if poster_url:
            _append_media_asset(
                assets,
                seen,
                media_type="image",
                raw_url=poster_url,
                base_url=base_url,
                metadata={
                    "source": "video_poster",
                    "platform": _platform_from_url(base_url),
                },
            )
    return assets


def _extract_c2pa_credentials(soup: BeautifulSoup, base_url: str) -> list[dict]:
    credentials: list[dict] = []
    seen: set[str] = set()

    meta_names = [
        "c2pa:manifest",
        "c2pa.manifest",
        "contentcredentials",
        "content-credentials",
    ]
    for name in meta_names:
        raw_value = _meta_content(soup, [name])
        resolved = _resolve_media_url(raw_value, base_url)
        if resolved and not _looks_like_generic_web_manifest(resolved) and resolved not in seen:
            seen.add(resolved)
            credentials.append(
                {
                    "source": "meta",
                    "type": "manifest",
                    "url": resolved,
                    "description": "Content credentials manifest advertised in page metadata.",
                }
            )

    for link in soup.find_all("link"):
        rel_values = link.get("rel") or []
        rel_normalized = " ".join(item.lower() for item in rel_values if isinstance(item, str))
        href = link.get("href")
        type_value = str(link.get("type") or "").lower()
        if not href:
            continue
        resolved = _resolve_media_url(href, base_url)
        if _looks_like_c2pa_manifest_reference(resolved, rel_normalized, type_value):
            if resolved and resolved not in seen:
                seen.add(resolved)
                credentials.append(
                    {
                        "source": "link",
                        "type": type_value or rel_normalized or "manifest",
                        "url": resolved,
                        "description": "Content credentials manifest linked from the article page.",
                    }
                )
    return credentials


def _merge_content_credentials(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for collection in (existing, incoming):
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            key = (str(item.get("url") or item.get("value") or "").strip(), str(item.get("type") or "").strip())
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _credentials_from_remote_media(findings: list[dict]) -> list[dict]:
    credentials: list[dict] = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        manifest_url = str(finding.get("manifest_url") or "").strip()
        verification_state = str(finding.get("verification_state") or "").strip()
        if manifest_url and verification_state in {"remote_manifest_verified", "sidecar_manifest_verified"}:
            credentials.append(
                {
                    "source": "remote_media",
                    "type": verification_state or "remote_manifest",
                    "url": manifest_url,
                    "description": str(finding.get("description") or "Remote media advertised content credentials."),
                }
            )
        elif finding.get("signals") and verification_state == "credentials_verified":
            credentials.append(
                {
                    "source": "remote_media",
                    "type": "embedded_signals",
                    "url": str(finding.get("url") or "").strip() or None,
                    "description": str(finding.get("description") or "Remote media exposed content-credential signals."),
                }
            )
    return credentials


async def _fetch_manifest_summary(client: httpx.AsyncClient, manifest_url: str) -> str | None:
    try:
        response = await client.get(manifest_url, headers={"Range": "bytes=0-8191"})
        response.raise_for_status()
    except httpx.HTTPError:
        return None

    content_type = str(response.headers.get("content-type") or "").lower()
    if "json" in content_type:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            for key in ("title", "label", "description", "summary", "manifest", "claim_generator"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    text = response.text[:200].strip()
    if text:
        return " ".join(text.split())[:160]
    return None


def _describe_remote_provenance(finding: dict[str, object]) -> str:
    signals = [str(signal) for signal in finding.get("signals", [])] if isinstance(finding.get("signals"), list) else []
    manifest_url = str(finding.get("manifest_url") or "").strip()
    manifest_summary = str(finding.get("manifest_summary") or "").strip()
    media_type = str(finding.get("type") or "media")
    verification_state = str(finding.get("verification_state") or "")
    if verification_state == "credentials_verified":
        return "Signed credentials were fetched and verified from the served asset."
    if verification_state in {"remote_manifest_verified", "sidecar_manifest_verified"}:
        return "A remote or sidecar provenance record was recovered and verified."
    if verification_state == "platform_signal_detected_manifest_not_retrievable":
        return "LinkedIn appears to signal Content Credentials, but the served media did not expose a retrievable raw manifest."
    if verification_state == "checked_no_retrievable_credentials":
        return "The asset was checked successfully, but no retrievable credentials were found."
    if verification_state == "could_not_evaluate_asset":
        return "The fetched asset could not be evaluated because it did not return a direct media file."
    if verification_state == "verification_failed":
        return "A provenance record was found, but verification did not succeed."
    if manifest_url and manifest_summary:
        return f"{media_type.title()} asset exposed a content credentials manifest. {manifest_summary}"
    if manifest_url:
        return f"{media_type.title()} asset exposed a content credentials manifest link."
    if signals:
        return f"{media_type.title()} asset exposed content-credential signals: {', '.join(signals)}."
    if str(finding.get("evaluation_state") or "") == "platform_restricted":
        return (
            f"{media_type.title()} asset could not be fully evaluated for content credentials because the platform "
            "returned a transformed or non-direct media response."
        )
    return f"{media_type.title()} asset was checked for content credentials, but no cryptographic provenance evidence was detected."


async def _verify_remote_manifest_reference(client: httpx.AsyncClient, manifest_url: str) -> tuple[bool, dict]:
    try:
        response = await client.get(manifest_url, headers={"Range": "bytes=0-32767"})
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return False, {"status": "fetch_failed", "reason": str(exc)}

    content_type = str(response.headers.get("content-type") or "").lower()
    body = response.content or b""
    indicators = _byte_c2pa_indicators(body)
    if indicators or "json" in content_type or "c2pa" in content_type:
        return True, {"status": "verified", "content_type": content_type, "signals": indicators}
    return False, {"status": "verification_failed", "reason": "Manifest content did not expose recognizable provenance markers."}


async def _verify_sidecar_manifest(client: httpx.AsyncClient, candidate_urls: list[str]) -> tuple[str | None, dict]:
    for sidecar_url in candidate_urls:
        ok, details = await _verify_remote_manifest_reference(client, sidecar_url)
        if ok:
            return sidecar_url, details
    return None, {"status": "not_found"}


async def _fetch_media_provenance_sample(
    client: httpx.AsyncClient,
    url: str,
    findings: dict[str, object],
) -> bytes:
    samples: list[bytes] = []
    ranged = await client.get(url, headers={"Range": "bytes=0-262143"})
    ranged.raise_for_status()
    first_chunk = ranged.content[:262144]
    if first_chunk:
        samples.append(first_chunk)
    response_headers = dict(ranged.headers)
    if response_headers.get("content-type") and not findings.get("content_type"):
        findings["content_type"] = response_headers.get("content-type")
    if response_headers.get("content-length") and not findings.get("content_length"):
        findings["content_length"] = response_headers.get("content-length")

    total_length = _content_length_from_headers(response_headers)
    if total_length and total_length > 262144:
        tail_start = max(total_length - 262144, 0)
        tail_response = await client.get(url, headers={"Range": f"bytes={tail_start}-{total_length - 1}"})
        tail_response.raise_for_status()
        tail_chunk = tail_response.content[-262144:]
        if tail_chunk:
            samples.append(tail_chunk)
    return b"".join(samples)


def _content_length_from_headers(headers: dict[str, str]) -> int | None:
    content_range = str(headers.get("content-range") or "").strip()
    if "/" in content_range:
        total = content_range.rsplit("/", 1)[-1].strip()
        if total.isdigit():
            return int(total)
    content_length = str(headers.get("content-length") or "").strip()
    if content_length.isdigit():
        return int(content_length)
    return None


def _candidate_sidecar_manifest_urls(url: str) -> list[str]:
    parsed = urlparse(url)
    base_path = parsed.path
    if not base_path:
        return []

    candidates: list[str] = []
    stem, dot, ext = base_path.rpartition(".")
    possible_paths = []
    if stem and dot:
        possible_paths.extend([f"{stem}.c2pa", f"{stem}.manifest.c2pa", f"{stem}.{ext}.c2pa"])
    possible_paths.append(f"{base_path}.c2pa")
    possible_paths.append(f"{base_path}.manifest")

    for candidate_path in possible_paths:
        rebuilt = parsed._replace(path=candidate_path, query="").geturl()
        if rebuilt not in candidates:
            candidates.append(rebuilt)
    return candidates[:6]


def _response_kind_from_content_type(content_type: str) -> str:
    lowered = str(content_type).lower()
    if lowered.startswith("image/"):
        return "image"
    if lowered.startswith("video/"):
        return "video"
    if lowered.startswith("audio/"):
        return "audio"
    if "json" in lowered:
        return "json"
    if "html" in lowered or "xhtml" in lowered:
        return "html"
    return "unknown"


def _header_c2pa_indicators(headers: dict[str, str]) -> list[str]:
    indicators: list[str] = []
    for key, value in headers.items():
        normalized_key = str(key).lower()
        normalized_value = str(value).lower()
        if "c2pa" in normalized_key or "content-credentials" in normalized_key or "contentcredentials" in normalized_key:
            indicators.append(f"header:{normalized_key}")
        elif "manifest" in normalized_key and ("c2pa" in normalized_value or "content" in normalized_value):
            indicators.append(f"header:{normalized_key}")
        elif normalized_key == "link" and ("c2pa" in normalized_value or "content-credentials" in normalized_value):
            indicators.append("header:link")
    return sorted(set(indicators))


def _header_manifest_url(headers: dict[str, str], base_url: str) -> str | None:
    link_header = headers.get("link") or headers.get("Link")
    if not link_header:
        return None
    for part in str(link_header).split(","):
        segment = part.strip()
        if "<" not in segment or ">" not in segment:
            continue
        target = segment[segment.find("<") + 1 : segment.find(">")].strip()
        meta = segment[segment.find(">") + 1 :].lower()
        resolved = urljoin(base_url, target)
        if _looks_like_c2pa_manifest_reference(resolved, meta, ""):
            return resolved
    return None


def _byte_c2pa_indicators(sample: bytes) -> list[str]:
    if not sample:
        return []
    lowered = sample.lower()
    indicators: list[str] = []
    if b"c2pa" in lowered:
        indicators.append("bytes:c2pa")
    if b"jumb" in lowered:
        indicators.append("bytes:jumbf")
    if b"content credentials" in lowered or b"contentcredentials" in lowered:
        indicators.append("bytes:content_credentials")
    if b"manifest" in lowered and (b"c2pa" in lowered or b"jumb" in lowered):
        indicators.append("bytes:manifest")
    return sorted(set(indicators))


def _extract_related_links(soup: BeautifulSoup, base_url: str) -> list[dict]:
    related: list[dict] = []
    seen: set[str] = set()
    base_domain = extract_domain(base_url)

    for link in soup.find_all("a"):
        href = _resolve_media_url(link.get("href"), base_url)
        if not href or href in seen:
            continue
        domain = extract_domain(href)
        if not domain or not base_domain or domain != base_domain:
            continue
        path = urlparse(href).path.strip("/")
        if not _looks_article_like_url(path):
            continue
        title_candidates = [
            _clean_node_text(link),
            link.get("title"),
            link.get("aria-label"),
        ]
        title = max((candidate.strip() for candidate in title_candidates if isinstance(candidate, str) and candidate.strip()), key=len, default=None)
        if not title or len(title.strip()) < 8:
            continue
        seen.add(href)
        related.append({"title": title.strip(), "url": href, "domain": domain, "source": "related_link"})
        if len(related) >= 8:
            break

    return related


def _looks_like_c2pa_manifest_reference(url: str | None, relation: str, type_value: str) -> bool:
    if not url:
        return False
    lowered_url = str(url).lower()
    lowered_relation = str(relation).lower()
    lowered_type = str(type_value).lower()

    if _looks_like_generic_web_manifest(lowered_url):
        return False

    explicit_relation = any(
        token in lowered_relation
        for token in ("c2pa", "content-credentials", "contentcredentials", "credential")
    )
    explicit_type = any(
        token in lowered_type
        for token in ("application/c2pa", "application/x-c2pa", "content-credentials", "contentcredentials")
    )
    explicit_url = any(
        token in lowered_url
        for token in (".c2pa", "/c2pa/", "content-credentials", "contentcredentials")
    )

    return explicit_relation or explicit_type or explicit_url


def _looks_like_generic_web_manifest(url: str) -> bool:
    lowered = str(url).lower()
    return lowered.endswith(".webmanifest") or "manifest.webmanifest" in lowered


def _looks_article_like_url(path: str) -> bool:
    if not path:
        return False
    segments = [segment for segment in path.split("/") if segment]
    if len(segments) >= 2:
        return True
    last_segment = segments[-1] if segments else ""
    if "-" in last_segment:
        return True
    return any(segment.isdigit() and len(segment) == 4 for segment in segments)


def _looks_like_video_embed(url: str) -> bool:
    lowered = str(url).lower()
    return any(
        token in lowered
        for token in (
            "youtube.com/embed/",
            "youtube.com/watch",
            "youtu.be/",
            "player.vimeo.com/video/",
            "vimeo.com/",
            "twitter.com/i/videos",
            "twitter.com/i/cards",
            "x.com/i/videos",
            "facebook.com/plugins/video.php",
            "facebook.com/watch/",
            "fb.watch/",
            "jwplatform.com/players/",
            "brightcove",
            "dailymotion.com/embed",
        )
    )


def _looks_like_social_video_post(url: str) -> bool:
    lowered = str(url).lower()
    return any(
        token in lowered
        for token in (
            "facebook.com/reel/",
            "facebook.com/watch/",
            "facebook.com/share/r/",
            "facebook.com/video.php",
            "fb.watch/",
            "instagram.com/reel/",
            "instagram.com/reels/",
        )
    )


def _is_youtube_player_or_page_url(url: str) -> bool:
    lowered = str(url).lower()
    return any(
        token in lowered
        for token in (
            "youtube.com/embed/",
            "youtube.com/watch",
            "youtu.be/",
            "youtube-nocookie.com/embed/",
        )
    )


def _append_media_asset(
    assets: list[dict],
    seen: set[tuple[str, str]],
    media_type: str,
    raw_url: str | None,
    base_url: str,
    metadata: dict,
) -> None:
    asset_url = _resolve_media_url(raw_url, base_url)
    if not asset_url:
        return
    key = (media_type, asset_url)
    if key in seen:
        return
    seen.add(key)
    cleaned_metadata = {name: value for name, value in metadata.items() if value not in (None, "", [])}
    cleaned_metadata["type"] = media_type
    cleaned_metadata["url"] = asset_url
    assets.append(cleaned_metadata)


def _resolve_media_url(raw_url: str | None, base_url: str) -> str | None:
    if not raw_url:
        return None
    primary_candidate = str(raw_url).split(",")[0].strip().split(" ")[0].strip()
    if not primary_candidate:
        return None
    return urljoin(base_url, primary_candidate)


def _candidate_media_urls(asset: dict, default_url: str) -> list[str]:
    candidates: list[str] = []
    for raw in [asset.get("url"), asset.get("poster"), default_url]:
        if not raw:
            continue
        candidate = str(raw).strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
        extracted_original = _extract_original_media_url(candidate)
        if extracted_original and extracted_original not in candidates:
            candidates.insert(0, extracted_original)
    return candidates[:4] or [default_url]


def _extract_original_media_url(url: str) -> str | None:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    preferred_keys = [
        "url",
        "u",
        "media_url",
        "mediaurl",
        "imgurl",
        "image_url",
        "video_url",
        "src",
    ]
    for key in preferred_keys:
        values = query.get(key) or []
        for value in values:
            decoded = unquote(str(value)).strip()
            if decoded.startswith("http://") or decoded.startswith("https://"):
                return decoded
    return None


def _platform_from_url(url: str) -> str | None:
    domain = (extract_domain(url) or "").lower()
    if not domain:
        return None
    if "youtube.com" in domain or "youtu.be" in domain or "googlevideo.com" in domain:
        return "youtube"
    if "facebook.com" in domain or "fbcdn.net" in domain or "fbsbx.com" in domain:
        return "facebook"
    if "pinterest.com" in domain or "pinimg.com" in domain:
        return "pinterest"
    if "instagram.com" in domain or "cdninstagram.com" in domain:
        return "instagram"
    if "linkedin.com" in domain or "licdn.com" in domain:
        return "linkedin"
    return domain


def _build_remote_media_forensics(asset: dict, finding: dict[str, object], sample: bytes | None) -> dict:
    from app.services.media import extract_image_forensics_from_bytes

    media_type = str(asset.get("type") or finding.get("type") or "media")
    content_type = str(finding.get("content_type") or asset.get("mime_type") or "")
    base = {
        "media_kind": media_type,
        "source": "remote_asset",
        "content_type": content_type or None,
        "content_length": finding.get("content_length"),
        "direct_media_response": bool(finding.get("direct_media_response")),
        "metadata_status": "limited",
        "fields": {},
        "notes": [],
    }

    if not finding.get("direct_media_response"):
        base["metadata_status"] = "not_evaluable"
        base["notes"] = ["The fetched endpoint did not return a direct media binary, so forensic parsing could not continue."]
        return base

    if media_type == "image":
        image_forensics = extract_image_forensics_from_bytes(sample or b"")
        image_forensics["source"] = "remote_asset"
        image_forensics["content_type"] = content_type or None
        image_forensics["content_length"] = finding.get("content_length")
        return image_forensics

    fields = {
        "format_hint": content_type.split(";", 1)[0] if content_type else None,
        "width": asset.get("width"),
        "height": asset.get("height"),
        "duration_hint": asset.get("duration"),
        "poster": asset.get("poster"),
        "mime_type": asset.get("mime_type"),
    }
    fields = {key: value for key, value in fields.items() if value not in (None, "")}
    base["fields"] = fields
    if media_type == "video":
        base["notes"] = [
            "Remote video forensics are limited unless the platform exposes a direct video binary or rich container metadata."
        ]
    elif media_type == "audio":
        base["notes"] = [
            "Remote audio forensics are limited unless the platform exposes a direct audio binary or rich container metadata."
        ]
    else:
        base["notes"] = ["Only limited remote media metadata was available for forensic inspection."]
    return base


def _provenance_note(finding: dict[str, object]) -> str | None:
    response_kind = str(finding.get("response_kind") or "")
    content_type = str(finding.get("content_type") or "")
    platform = str(finding.get("platform") or "")
    verification_state = str(finding.get("verification_state") or "")
    if response_kind == "json":
        return f"The endpoint returned JSON ({content_type}) instead of a direct media file."
    if response_kind == "html":
        return f"The endpoint returned HTML ({content_type}) instead of a direct media file."
    if response_kind == "unknown" and content_type:
        return f"The endpoint returned {content_type}, which does not look like a direct image, video, or audio file."
    if verification_state == "platform_signal_detected_manifest_not_retrievable":
        return "LinkedIn appears to signal Content Credentials, but the served media did not expose a retrievable raw manifest."
    if verification_state == "checked_no_retrievable_credentials":
        return "The asset was checked successfully, but no retrievable credentials were found."
    if verification_state == "could_not_evaluate_asset":
        return "The fetched LinkedIn asset did not return a direct media binary."
    if verification_state == "verification_failed":
        return "A provenance record was found, but verification did not succeed."
    if str(finding.get("evaluation_state") or "") == "platform_restricted" and platform in {"facebook", "pinterest", "instagram"}:
        return f"{platform.title()} appears to be serving a transformed or gated asset rather than the original media file."
    return None


def _detect_linkedin_page_signal(soup: BeautifulSoup, page_url: str) -> dict:
    if _platform_from_url(page_url) != "linkedin":
        return {"found": False, "markers": []}

    markers: list[str] = []
    page_text = " ".join(soup.get_text(" ", strip=True).lower().split())
    for marker in ("content credentials", "about this image", "credential"):
        if marker in page_text:
            markers.append(marker)

    for value in (_meta_content(soup, ["description", "og:description"]), _meta_content(soup, ["og:title", "twitter:title"])):
        lowered = str(value or "").lower()
        if "content credentials" in lowered and "meta:content credentials" not in markers:
            markers.append("meta:content credentials")

    return {"found": bool(markers), "markers": markers}


def _extract_youtube_signals(soup: BeautifulSoup, page_url: str, html: str) -> dict:
    if _platform_from_url(page_url) != "youtube":
        return {}

    page_text = " ".join(soup.get_text(" ", strip=True).split()).lower()
    html_lower = html.lower()
    page_level_provenance_signal = any(
        marker in html_lower or marker in page_text
        for marker in (
            "content credentials",
            "contentcredentials",
            "provenance",
            "about this video",
        )
    )

    verified_markers = (
        "badge_style_type_verified",
        "badge_style_type_verified_artist",
        "official artist channel",
    )
    channel_verified = any(marker in html_lower for marker in verified_markers)

    license_label = None
    if "creative commons attribution license" in html_lower or "creative commons attribution license" in page_text:
        license_label = "Creative Commons Attribution license"
    elif "standard youtube license" in html_lower or "standard youtube license" in page_text:
        license_label = "Standard YouTube license"

    creative_commons = bool(license_label and "creative commons" in license_label.lower())
    original_content_claim = bool(channel_verified and license_label and not creative_commons)

    signals: list[str] = []
    if channel_verified:
        signals.append("verified_channel")
    if license_label:
        signals.append("license_detected")
    if page_level_provenance_signal:
        signals.append("page_level_provenance_signal")
    if original_content_claim:
        signals.append("original_content_claim")

    return {
        "page_level_provenance_signal": page_level_provenance_signal,
        "channel_verified": channel_verified,
        "channel_verification_status": "verified" if channel_verified else "not_confirmed",
        "license_label": license_label,
        "creative_commons": creative_commons,
        "original_content_claim": original_content_claim,
        "signals": signals,
    }


def _linkedin_page_signal_from_context(page_context: dict) -> dict:
    if not isinstance(page_context, dict):
        return {"found": False, "markers": []}
    signal = page_context.get("linkedin_page_signal", {})
    return signal if isinstance(signal, dict) else {"found": False, "markers": []}


def _youtube_signal_from_context(page_context: dict) -> dict:
    if not isinstance(page_context, dict):
        return {}
    signal = page_context.get("youtube_signals", {})
    return signal if isinstance(signal, dict) else {}


def _first_media_url(media_assets: list[dict], media_type: str) -> str | None:
    for asset in media_assets:
        if asset.get("type") == media_type and asset.get("url"):
            return str(asset["url"])
    return None


def _select_main_text(
    readable_text: str,
    article_text: str,
    title: str | None,
    author: str | None,
    publisher_name: str | None,
) -> str:
    readable = readable_text.strip()
    article = article_text.strip()
    candidates = [candidate for candidate in [article, readable] if candidate]
    for candidate in candidates:
        if not _looks_like_block_page(candidate):
            return candidate

    metadata_summary = " ".join(part for part in [title, author, publisher_name] if part).strip()
    if candidates:
        return max(candidates, key=len)
    return metadata_summary


def _normalize_article_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def _article_text_quality_score(text: str, title: str | None) -> int:
    normalized = normalize_text(text)
    words = normalized.split()
    score = min(len(text), 4000)
    if len(words) >= 80:
        score += 1200
    elif len(words) >= 35:
        score += 500
    if title:
        title_terms = {token for token in normalize_text(title).split() if len(token) > 3}
        overlap = len(title_terms & {token for token in words if len(token) > 3})
        score += overlap * 80
    boilerplate_hits = sum(
        1
        for marker in (
            "subscribe",
            "sign up",
            "cookie",
            "privacy policy",
            "advertisement",
            "share this",
            "related articles",
        )
        if marker in normalized
    )
    score -= boilerplate_hits * 350
    return score


def _remove_noise_nodes(soup: BeautifulSoup) -> None:
    for selector in [
        "script",
        "style",
        "noscript",
        "svg",
        "form",
        "nav",
        "header",
        "footer",
        "aside",
        "iframe",
        "button",
        "#wm-ipp",
        "#wm-ipp-base",
        ".wm-ipp",
        '[role="navigation"]',
        '[role="complementary"]',
        '[aria-label*="share" i]',
        '[class*="share" i]',
        '[class*="social" i]',
        '[class*="comment" i]',
        '[class*="related" i]',
        '[class*="recommend" i]',
        '[class*="newsletter" i]',
        '[class*="subscribe" i]',
        '[class*="cookie" i]',
        '[class*="advert" i]',
        '[class*="ad-" i]',
        '[id*="share" i]',
        '[id*="comment" i]',
        '[id*="related" i]',
        '[id*="newsletter" i]',
        '[id*="subscribe" i]',
        '[id*="cookie" i]',
        '[id*="advert" i]',
    ]:
        for node in soup.select(selector):
            node.decompose()


def _remove_wayback_noise(soup: BeautifulSoup) -> None:
    _remove_noise_nodes(soup)
    for selector in ["#wm-ipp", "#wm-ipp-base", "script[src*='archive.org']", "link[href*='archive.org']"]:
        for node in soup.select(selector):
            node.decompose()


def _wayback_original_url(url: str) -> str | None:
    parsed = urlparse(url)
    if "web.archive.org" not in parsed.netloc.lower():
        return None
    path = parsed.path.strip("/")
    if not path.startswith("web/"):
        return None
    parts = path.split("/", 2)
    if len(parts) < 3:
        return None
    original = unquote(parts[2])
    return original if original.startswith(("http://", "https://")) else None


def _looks_like_boilerplate_text(text: str) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 25:
        return True
    boilerplate = (
        "cookie policy",
        "privacy policy",
        "terms of use",
        "all rights reserved",
        "subscribe to",
        "sign up for",
        "share this",
        "follow us",
        "advertisement",
        "related article",
        "enable javascript",
    )
    return any(marker in normalized for marker in boilerplate)


def _looks_like_block_page(text: str) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 120:
        return True
    block_markers = [
        "please enable js",
        "disable any ad blocker",
        "verify you are human",
        "access denied",
        "captcha",
        "javascript is disabled",
    ]
    return any(marker in normalized for marker in block_markers)


def _payload_has_meaningful_content(payload: ExtractedPayload) -> bool:
    if payload.title or payload.author or payload.publish_time or payload.metadata.get("publisher_name"):
        return True
    if payload.metadata.get("media_assets") or payload.metadata.get("c2pa_credentials"):
        return True
    return bool(payload.main_text and not _looks_like_block_page(payload.main_text))


def _clean_title_text(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = " ".join(str(value).split()).strip()
    if not cleaned:
        return None
    separators = [" | ", " - ", " :: "]
    for separator in separators:
        if separator in cleaned:
            parts = [part.strip() for part in cleaned.split(separator) if part.strip()]
            if parts:
                cleaned = max(parts, key=len)
                break
    return cleaned or None


def _summarize_extraction_quality(
    title: str | None,
    author: str | None,
    publisher_name: str | None,
    publish_time: datetime | None,
    main_text: str,
) -> str:
    populated_fields = sum(bool(value) for value in [title, author, publisher_name, publish_time])
    if populated_fields >= 4 and main_text and not _looks_like_block_page(main_text):
        return "high"
    if populated_fields >= 2:
        return "medium"
    return "low"


def _clean_node_text(node) -> str | None:
    if not node:
        return None
    text = node.get_text(" ", strip=True)
    return text or None


def _clean_article_node_text(node) -> str | None:
    if not node:
        return None
    node = _parse_html(str(node))
    if node is None:
        return None
    _remove_noise_nodes(node)

    paragraphs: list[str] = []
    for child in node.select("p, h1, h2, time"):
        text = " ".join(child.get_text(" ", strip=True).split())
        if not text or _looks_like_article_noise_line(text):
            continue
        paragraphs.append(text)

    if not paragraphs:
        text = _normalize_article_text(node.get_text(" ", strip=True))
        return text or None
    return " ".join(paragraphs) or None


def _looks_like_article_noise_line(text: str) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 3:
        return True
    noise_patterns = [
        r"^(advertisement|sponsored|share|follow us|subscribe|sign up|read more|related articles?)$",
        r"^(facebook|twitter|x|linkedin|whatsapp|telegram|copy link)$",
        r"^(comments?|all rights reserved|copyright)\b",
        r"\b(enable notifications|newsletter|download our app)\b",
    ]
    return any(re.search(pattern, normalized) for pattern in noise_patterns)


def _format_certificate_name(entries) -> str | None:
    if not entries:
        return None
    parts: list[str] = []
    for rdn in entries:
        for key, value in rdn:
            normalized_key = str(key)
            normalized_value = str(value).strip()
            if normalized_value:
                parts.append(f"{normalized_key}={normalized_value}")
    return ", ".join(parts) if parts else None
