from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.config import get_settings


@dataclass
class TimestampVerificationResult:
    valid: bool
    provider: str
    reason: str | None = None


class BaseTimestampProvider:
    provider_name = "base"

    def issue(self, digest_hex: str) -> dict:
        raise NotImplementedError

    def verify(self, token: dict | None, digest_hex: str) -> TimestampVerificationResult:
        raise NotImplementedError


class InternalTimestampProvider(BaseTimestampProvider):
    provider_name = "internal"

    def issue(self, digest_hex: str) -> dict:
        return {
            "mode": "internal",
            "provider": self.provider_name,
            "hashed_message": digest_hex,
            "issued_at": datetime.now(timezone.utc).isoformat(),
        }

    def verify(self, token: dict | None, digest_hex: str) -> TimestampVerificationResult:
        if not isinstance(token, dict):
            return TimestampVerificationResult(valid=False, provider=self.provider_name, reason="missing token")
        if token.get("hashed_message") != digest_hex:
            return TimestampVerificationResult(valid=False, provider=self.provider_name, reason="digest mismatch")
        if not token.get("issued_at"):
            return TimestampVerificationResult(valid=False, provider=self.provider_name, reason="missing issued_at")
        return TimestampVerificationResult(valid=True, provider=self.provider_name)


class RFC3161TimestampProvider(BaseTimestampProvider):
    provider_name = "rfc3161"

    def __init__(self, tsa_url: str):
        self.tsa_url = tsa_url

    def issue(self, digest_hex: str) -> dict:
        return {
            "mode": "rfc3161",
            "provider": self.provider_name,
            "tsa_url": self.tsa_url,
            "hashed_message": digest_hex,
            "issued_at": datetime.now(timezone.utc).isoformat(),
            "token": None,
            "note": "RFC 3161 integration point configured; internal fallback token issued in MVP.",
        }

    def verify(self, token: dict | None, digest_hex: str) -> TimestampVerificationResult:
        if not isinstance(token, dict):
            return TimestampVerificationResult(valid=False, provider=self.provider_name, reason="missing token")
        if token.get("hashed_message") != digest_hex:
            return TimestampVerificationResult(valid=False, provider=self.provider_name, reason="digest mismatch")
        return TimestampVerificationResult(valid=True, provider=self.provider_name)


def get_timestamp_provider() -> BaseTimestampProvider:
    settings = get_settings()
    if settings.timestamp_mode.lower() == "rfc3161" and settings.tsa_url:
        return RFC3161TimestampProvider(settings.tsa_url)
    return InternalTimestampProvider()
