from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AuditLogEntry, VerificationJob
from app.utils import make_audit_hash


def append_audit_entry(db: Session, job: VerificationJob, event_type: str, event_payload: dict) -> AuditLogEntry:
    previous_entry = db.scalar(
        select(AuditLogEntry)
        .where(AuditLogEntry.job_id == job.id)
        .order_by(AuditLogEntry.id.desc())
        .limit(1)
    )
    previous_hash = previous_entry.entry_hash if previous_entry else None
    entry_hash = make_audit_hash(previous_hash, event_type, event_payload)
    entry = AuditLogEntry(
        job_id=job.id,
        event_type=event_type,
        event_payload=event_payload,
        previous_hash=previous_hash,
        entry_hash=entry_hash,
    )
    db.add(entry)
    db.flush()
    return entry
