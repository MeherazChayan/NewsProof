from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import AuditEntryResponse
from app.services.verification import VerificationService


router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/{job_id}", response_model=list[AuditEntryResponse])
def get_audit(job_id: UUID, db: Session = Depends(get_db)):
    return VerificationService(db).get_audit(job_id)

