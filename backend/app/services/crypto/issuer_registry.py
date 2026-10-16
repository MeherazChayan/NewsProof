from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import IssuerCredential, IssuerType
from app.services.crypto.signer import get_managed_public_key_b64


def ensure_default_issuers(db: Session) -> None:
    settings = get_settings()
    issuers = {
        settings.platform_issuer_name: {
            "issuer_type": IssuerType.platform,
            "public_key": get_managed_public_key_b64(),
            "status": "active",
            "metadata_json": {"managed_signing_key": True},
        },
        "NewsProof Demo Publisher": {
            "issuer_type": IssuerType.publisher,
            "public_key": get_managed_public_key_b64(),
            "status": "active",
            "metadata_json": {"demo": True},
        },
        "NewsProof Fact Checker": {
            "issuer_type": IssuerType.fact_checker,
            "public_key": get_managed_public_key_b64(),
            "status": "active",
            "metadata_json": {"demo": True},
        },
        "NewsProof Moderator": {
            "issuer_type": IssuerType.moderator,
            "public_key": get_managed_public_key_b64(),
            "status": "active",
            "metadata_json": {"demo": True},
        },
    }

    changed = False
    for issuer_name, payload in issuers.items():
        issuer = db.scalar(select(IssuerCredential).where(IssuerCredential.issuer_name == issuer_name))
        if issuer:
            if issuer.public_key != payload["public_key"] or issuer.status != payload["status"]:
                issuer.public_key = payload["public_key"]
                issuer.status = payload["status"]
                issuer.metadata_json = payload["metadata_json"]
                issuer.issued_at = issuer.issued_at or datetime.utcnow()
                changed = True
            continue
        db.add(
            IssuerCredential(
                issuer_name=issuer_name,
                issuer_type=payload["issuer_type"],
                public_key=payload["public_key"],
                status=payload["status"],
                issued_at=datetime.utcnow(),
                metadata_json=payload["metadata_json"],
            )
        )
        changed = True
    if changed:
        db.commit()
