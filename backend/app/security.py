import base64

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def encode_public_key(public_key: Ed25519PublicKey) -> str:
    return base64.b64encode(public_key.public_bytes_raw()).decode("utf-8")


def encode_private_key(private_key: Ed25519PrivateKey) -> str:
    return base64.b64encode(private_key.private_bytes_raw()).decode("utf-8")


def load_private_key(private_key_b64: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(base64.b64decode(private_key_b64))


def sign_ed25519_message(private_key_b64: str, message: bytes) -> str:
    private_key = load_private_key(private_key_b64)
    signature = private_key.sign(message)
    return base64.b64encode(signature).decode("utf-8")


def verify_ed25519_signature(public_key_b64: str, signature_b64: str, message: bytes) -> bool:
    public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64))
    signature = base64.b64decode(signature_b64)
    try:
        public_key.verify(signature, message)
        return True
    except Exception:
        return False
