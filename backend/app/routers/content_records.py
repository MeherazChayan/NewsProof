from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import ContentRecord, TrustedPublisher
from app.schemas import ContentRecordCreate, ContentRecordResponse, ContentRecordUpdate


router = APIRouter(prefix="/content-records", tags=["content-records"])


@router.get("", response_model=list[ContentRecordResponse])
def list_content_records(publisher_id: int | None = None, db: Session = Depends(get_db)):
    query = select(ContentRecord).order_by(ContentRecord.created_at.desc())
    if publisher_id is not None:
        query = query.where(ContentRecord.publisher_id == publisher_id)
    return [_to_response(record) for record in db.scalars(query)]


@router.post("", response_model=ContentRecordResponse)
def create_content_record(payload: ContentRecordCreate, db: Session = Depends(get_db)):
    publisher = db.get(TrustedPublisher, payload.publisher_id)
    if not publisher:
        raise HTTPException(status_code=404, detail="Publisher not found")

    record = ContentRecord(
        publisher_id=payload.publisher_id,
        canonical_url=payload.canonical_url,
        title=payload.title,
        content_hash=payload.content_hash,
        signature=payload.signature,
        key_id=payload.key_id,
        source_metadata=_build_source_metadata(payload),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return _to_response(record)


@router.put("/{record_id}", response_model=ContentRecordResponse)
def update_content_record(record_id: int, payload: ContentRecordUpdate, db: Session = Depends(get_db)):
    record = db.get(ContentRecord, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Content record not found")

    record.canonical_url = payload.canonical_url
    record.title = payload.title
    record.content_hash = payload.content_hash
    record.signature = payload.signature
    record.key_id = payload.key_id
    record.source_metadata = _build_source_metadata(payload)
    db.commit()
    db.refresh(record)
    return _to_response(record)


@router.delete("/{record_id}")
def delete_content_record(record_id: int, db: Session = Depends(get_db)):
    record = db.get(ContentRecord, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Content record not found")
    db.delete(record)
    db.commit()
    return {"ok": True}


def _build_source_metadata(payload: ContentRecordCreate | ContentRecordUpdate) -> dict:
    metadata = dict(payload.source_metadata or {})
    metadata["normalized_text"] = payload.normalized_text
    metadata["author"] = payload.author
    metadata["publish_time"] = payload.publish_time.isoformat() if payload.publish_time else None
    return metadata


def _to_response(record: ContentRecord) -> ContentRecordResponse:
    return ContentRecordResponse(
        id=record.id,
        publisher_id=record.publisher_id,
        canonical_url=record.canonical_url,
        title=record.title,
        author=record.source_metadata.get("author"),
        publish_time=record.source_metadata.get("publish_time"),
        normalized_text=record.source_metadata.get("normalized_text", ""),
        content_hash=record.content_hash,
        signature=record.signature,
        key_id=record.key_id,
        source_metadata=record.source_metadata,
        created_at=record.created_at,
    )
