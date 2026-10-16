from __future__ import annotations

import logging
from functools import lru_cache

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from app.config import get_settings
from app.security import encode_private_key, encode_public_key, load_private_key, sign_ed25519_message


logger = logging.getLogger(__name__)


@lru_cache
def get_managed_private_key_b64() -> str:
    settings = get_settings()
    if settings.platform_signing_private_key_b64:
        return settings.platform_signing_private_key_b64
    logger.warning("PLATFORM_SIGNING_PRIVATE_KEY_B64 is not configured; generating an ephemeral development signing key")
    return encode_private_key(Ed25519PrivateKey.generate())


def get_managed_public_key_b64() -> str:
    private_key = load_private_key(get_managed_private_key_b64())
    return encode_public_key(private_key.public_key())


def sign_digest(digest_hex: str) -> str:
    return sign_ed25519_message(get_managed_private_key_b64(), digest_hex.encode("utf-8"))
