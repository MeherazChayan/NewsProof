from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy.orm import Session

from app.api_helpers import job_to_response
from app.database import get_db
from app.redis_client import rate_limiter
from app.schemas import TextVerificationRequest, UrlVerificationRequest, VerificationResultResponse
from app.services.verification import VerificationService


router = APIRouter(prefix="/verify", tags=["verify"])


@router.post("/url", response_model=VerificationResultResponse)
async def verify_url(payload: UrlVerificationRequest, request: Request, fast: bool = Query(False), db: Session = Depends(get_db)):
    _enforce_rate_limit(request)
    job = await VerificationService(db).verify_url(str(payload.url), fast_mode=fast)
    return job_to_response(job)


@router.post("/text", response_model=VerificationResultResponse)
def verify_text(payload: TextVerificationRequest, request: Request, db: Session = Depends(get_db)):
    _enforce_rate_limit(request)
    job = VerificationService(db).verify_text(payload.text)
    return job_to_response(job)


@router.post("/image", response_model=VerificationResultResponse)
async def verify_image(
    request: Request,
    file: UploadFile = File(...),
    caption_or_claim: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    _enforce_rate_limit(request)
    job = await VerificationService(db).verify_image(file, caption_or_claim=caption_or_claim)
    return job_to_response(job)


@router.post("/audio", response_model=VerificationResultResponse)
async def verify_audio(
    request: Request,
    file: UploadFile = File(...),
    caption_or_claim: str | None = Form(default=None),
    claimed_speaker: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    _enforce_rate_limit(request)
    job = await VerificationService(db).verify_audio(file, caption_or_claim=caption_or_claim, claimed_speaker=claimed_speaker)
    return job_to_response(job)


@router.post("/video", response_model=VerificationResultResponse)
async def verify_video(
    request: Request,
    file: UploadFile = File(...),
    caption_or_claim: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    _enforce_rate_limit(request)
    job = await VerificationService(db).verify_video(file, caption_or_claim=caption_or_claim)
    return job_to_response(job)


def _enforce_rate_limit(request: Request) -> None:
    client_host = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_host):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
