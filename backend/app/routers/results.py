from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api_helpers import history_to_response, job_to_response
from app.database import get_db
from app.schemas import HistoryResponse, SimilarityMatchesPageResponse, VerificationResultResponse
from app.services.verification import VerificationService


router = APIRouter(tags=["results"])


@router.get("/results/{job_id}", response_model=VerificationResultResponse)
def get_result(job_id: UUID, db: Session = Depends(get_db)):
    job = VerificationService(db).get_result(job_id)
    return job_to_response(job)


@router.get("/history", response_model=HistoryResponse)
def get_history(db: Session = Depends(get_db)):
    jobs = VerificationService(db).get_history()
    return history_to_response(jobs)


@router.get("/results/{job_id}/matches", response_model=SimilarityMatchesPageResponse)
def get_similarity_matches(
    job_id: UUID,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=10, ge=1, le=10),
    db: Session = Depends(get_db),
):
    return VerificationService(db).get_similarity_matches(job_id, offset=offset, limit=limit)
