from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import ConsensusSummaryResponse, ConsensusVoteRequest, ConsensusVoteResponse
from app.services.verification import VerificationService


router = APIRouter(prefix="/consensus", tags=["consensus"])


@router.get("/{job_id}", response_model=ConsensusSummaryResponse)
def get_consensus(job_id: UUID, db: Session = Depends(get_db)):
    return VerificationService(db).get_consensus(job_id)


@router.post("/{job_id}/votes", response_model=ConsensusVoteResponse)
def create_consensus_vote(job_id: UUID, payload: ConsensusVoteRequest, db: Session = Depends(get_db)):
    return VerificationService(db).add_consensus_vote(job_id, payload)
