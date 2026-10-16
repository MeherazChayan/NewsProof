from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import PublicKey, TrustedPublisher
from app.schemas import PublisherCreate, PublisherResponse, PublisherUpdate


router = APIRouter(prefix="/publishers", tags=["publishers"])


@router.get("", response_model=list[PublisherResponse])
def list_publishers(db: Session = Depends(get_db)):
    return list(db.scalars(select(TrustedPublisher).order_by(TrustedPublisher.name.asc())))


@router.post("", response_model=PublisherResponse)
def create_publisher(payload: PublisherCreate, db: Session = Depends(get_db)):
    publisher = TrustedPublisher(name=payload.name, domain=payload.domain, metadata_json=payload.metadata)
    for key in payload.public_keys:
        publisher.public_keys.append(
            PublicKey(
                key_id=key.key_id,
                algorithm=key.algorithm,
                public_key=key.public_key,
                is_active=key.is_active,
            )
        )
    db.add(publisher)
    db.commit()
    db.refresh(publisher)
    return publisher


@router.put("/{publisher_id}", response_model=PublisherResponse)
def update_publisher(publisher_id: int, payload: PublisherUpdate, db: Session = Depends(get_db)):
    publisher = db.get(TrustedPublisher, publisher_id)
    if not publisher:
        raise HTTPException(status_code=404, detail="Publisher not found")
    publisher.name = payload.name
    publisher.domain = payload.domain
    publisher.metadata_json = payload.metadata
    db.execute(delete(PublicKey).where(PublicKey.publisher_id == publisher.id))
    db.flush()
    for key in payload.public_keys:
        publisher.public_keys.append(
            PublicKey(
                key_id=key.key_id,
                algorithm=key.algorithm,
                public_key=key.public_key,
                is_active=key.is_active,
            )
        )
    db.commit()
    db.refresh(publisher)
    return publisher


@router.delete("/{publisher_id}")
def delete_publisher(publisher_id: int, db: Session = Depends(get_db)):
    publisher = db.get(TrustedPublisher, publisher_id)
    if not publisher:
        raise HTTPException(status_code=404, detail="Publisher not found")
    db.delete(publisher)
    db.commit()
    return {"ok": True}
