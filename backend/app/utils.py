import hashlib
import re
from urllib.parse import parse_qs, urlparse, urlunparse


def normalize_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip().lower()
    return compact


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    if "youtube.com" in netloc and path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        query = f"v={video_id}" if video_id else ""
        return urlunparse((scheme, netloc, path, "", query, ""))
    return urlunparse((scheme, netloc, path, "", "", ""))


def extract_domain(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    hostname = parsed.netloc.lower().split(":")[0]
    return hostname or None


def domain_matches(hostname: str | None, candidate_domain: str | None) -> bool:
    if not hostname or not candidate_domain:
        return False
    host = hostname.lower()
    domain = candidate_domain.lower()
    return host == domain or host.endswith(f".{domain}")


def make_audit_hash(previous_hash: str | None, event_type: str, payload: dict) -> str:
    body = f"{previous_hash or ''}:{event_type}:{payload}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()
