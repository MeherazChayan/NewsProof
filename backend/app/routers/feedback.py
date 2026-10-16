from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import FeedbackEntryRequest, FeedbackEntryResponse, FeedbackSummaryResponse
from app.services.verification import VerificationService


router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.get("/{job_id}", response_model=FeedbackSummaryResponse)
def get_feedback(job_id: UUID, db: Session = Depends(get_db)):
    return VerificationService(db).get_feedback(job_id)


@router.post("/{job_id}", response_model=FeedbackEntryResponse)
def submit_feedback(job_id: UUID, payload: FeedbackEntryRequest, db: Session = Depends(get_db)):
    return VerificationService(db).submit_feedback(job_id, payload)
