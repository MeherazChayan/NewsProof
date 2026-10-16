from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError

from app.models import TrustedPublisher


DEFAULT_TRUSTED_PUBLISHERS = [
    {
        "name": "The New York Times",
        "domain": "nytimes.com",
        "metadata": {
            "domains": ["www.nytimes.com"],
            "names": ["The New York Times", "New York Times", "NYTimes"],
        },
    },
    {
        "name": "Reuters",
        "domain": "reuters.com",
        "metadata": {
            "domains": ["www.reuters.com"],
            "names": ["Reuters"],
        },
    },
    {
        "name": "BBC News",
        "domain": "bbc.com",
        "metadata": {
            "domains": ["www.bbc.com", "bbc.co.uk", "www.bbc.co.uk"],
            "names": ["BBC News", "BBC"],
        },
    },
    {
        "name": "Associated Press",
        "domain": "apnews.com",
        "metadata": {
            "domains": ["www.apnews.com", "apnews.com"],
            "names": ["Associated Press", "AP News", "The Associated Press"],
        },
    },
    {
        "name": "The Washington Post",
        "domain": "washingtonpost.com",
        "metadata": {
            "domains": ["www.washingtonpost.com"],
            "names": ["The Washington Post", "Washington Post"],
        },
    },
    {
        "name": "The Guardian",
        "domain": "theguardian.com",
        "metadata": {
            "domains": ["www.theguardian.com"],
            "names": ["The Guardian", "Guardian"],
        },
    },
    {
        "name": "CNN",
        "domain": "cnn.com",
        "metadata": {
            "domains": ["www.cnn.com", "edition.cnn.com"],
            "names": ["CNN", "CNN News"],
        },
    },
    {
        "name": "CNBC",
        "domain": "cnbc.com",
        "metadata": {
            "domains": ["www.cnbc.com"],
            "names": ["CNBC", "@CNBC"],
        },
    },
]


def ensure_default_publishers(db: Session) -> None:
    try:
        bind = db.get_bind()
        if not inspect(bind).has_table("trusted_publishers"):
            return

        db.execute(delete(TrustedPublisher).where((TrustedPublisher.name == "") | (TrustedPublisher.domain == "")))
    except OperationalError:
        db.rollback()
        return

    existing = {
        publisher.domain.lower(): publisher
        for publisher in db.scalars(select(TrustedPublisher))
        if publisher.domain
    }

    for seed in DEFAULT_TRUSTED_PUBLISHERS:
        domain = seed["domain"].lower()
        publisher = existing.get(domain)
        if publisher:
            metadata = publisher.metadata_json if isinstance(publisher.metadata_json, dict) else {}
            merged_domains = _merge_aliases(metadata.get("domains", []), seed["metadata"].get("domains", []))
            merged_names = _merge_aliases(metadata.get("names", []), seed["metadata"].get("names", []))
            publisher.metadata_json = {
                **metadata,
                "domains": merged_domains,
                "names": merged_names,
            }
            if not publisher.name:
                publisher.name = seed["name"]
            continue

        db.add(
            TrustedPublisher(
                name=seed["name"],
                domain=seed["domain"],
                metadata_json=seed["metadata"],
            )
        )

    db.commit()


def _merge_aliases(current: list[str], incoming: list[str]) -> list[str]:
    values: list[str] = []
    seen = set()
    for item in [*(current or []), *(incoming or [])]:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(normalized)
    return values
