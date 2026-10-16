from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import CryptoObjectType
from app.schemas import (
    ArticleSnapshotCreate,
    ArticleSnapshotResponse,
    EvidenceBundleCreate,
    EvidenceBundleResponse,
    IssuerCredentialCreate,
    IssuerCredentialResponse,
    ManifestSignRequest,
    ManifestVerificationResponse,
    ManifestVerifyRequest,
    ProvenanceHistoryResponse,
    ProvenanceManifestResponse,
    ReviewEventCreate,
    ReviewEventResponse,
    VersionCompareResponse,
)
from app.services.crypto import CryptoProvenanceService


router = APIRouter(prefix="/crypto", tags=["crypto"])


@router.get("/issuers", response_model=list[IssuerCredentialResponse])
def list_issuers(db: Session = Depends(get_db)):
    service = CryptoProvenanceService(db)
    service.ensure_default_issuers()
    return service.list_issuers()


@router.get("/issuers/{issuer_id}", response_model=IssuerCredentialResponse)
def get_issuer(issuer_id: UUID, db: Session = Depends(get_db)):
    try:
        return CryptoProvenanceService(db).get_issuer(issuer_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/issuers", response_model=IssuerCredentialResponse)
def create_issuer(payload: IssuerCredentialCreate, db: Session = Depends(get_db)):
    service = CryptoProvenanceService(db)
    return service.create_issuer(
        issuer_type=payload.issuer_type,
        issuer_name=payload.issuer_name,
        public_key=payload.public_key,
        issued_at=payload.issued_at,
        expires_at=payload.expires_at,
    )


@router.post("/issuers/{issuer_id}/revoke", response_model=IssuerCredentialResponse)
def revoke_issuer(issuer_id: UUID, db: Session = Depends(get_db)):
    try:
        return CryptoProvenanceService(db).revoke_issuer(issuer_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/article-snapshots", response_model=ArticleSnapshotResponse)
def create_article_snapshot(payload: ArticleSnapshotCreate, db: Session = Depends(get_db)):
    return CryptoProvenanceService(db).create_article_snapshot(payload.model_dump())


@router.post("/evidence-bundles", response_model=EvidenceBundleResponse)
def create_evidence_bundle(payload: EvidenceBundleCreate, db: Session = Depends(get_db)):
    return CryptoProvenanceService(db).create_evidence_bundle(payload.model_dump())


@router.post("/review-events", response_model=ReviewEventResponse)
def create_review_event(payload: ReviewEventCreate, db: Session = Depends(get_db)):
    return CryptoProvenanceService(db).create_review_event(payload.model_dump())


@router.post("/manifests/sign", response_model=ProvenanceManifestResponse)
def sign_manifest(payload: ManifestSignRequest, db: Session = Depends(get_db)):
    try:
        service = CryptoProvenanceService(db)
        manifest = service.sign_object(payload.object_type, payload.object_id, payload.signer_id)
        return ProvenanceManifestResponse.model_validate(service.serialize_manifest(manifest))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/manifests/verify", response_model=ManifestVerificationResponse)
def verify_manifest(payload: ManifestVerifyRequest, db: Session = Depends(get_db)):
    try:
        result = CryptoProvenanceService(db).verify_manifest(payload.manifest_id, verified_by=payload.verified_by)
        return ManifestVerificationResponse.model_validate(result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/manifests/{manifest_id}", response_model=ProvenanceManifestResponse)
def get_manifest(manifest_id: UUID, db: Session = Depends(get_db)):
    try:
        service = CryptoProvenanceService(db)
        manifest = service.get_manifest(manifest_id)
        return ProvenanceManifestResponse.model_validate(service.serialize_manifest(manifest))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/records/{object_type}/{record_id}/history", response_model=ProvenanceHistoryResponse)
def get_record_history(object_type: CryptoObjectType, record_id: UUID, db: Session = Depends(get_db)):
    try:
        result = CryptoProvenanceService(db).record_history(object_type, record_id)
        return ProvenanceHistoryResponse.model_validate(result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/records/{object_type}/{record_id}/compare", response_model=VersionCompareResponse)
def compare_record_versions(
    object_type: CryptoObjectType,
    record_id: UUID,
    left_version_id: UUID,
    right_version_id: UUID,
    db: Session = Depends(get_db),
):
    try:
        result = CryptoProvenanceService(db).compare_versions(object_type, left_version_id, right_version_id)
        return VersionCompareResponse.model_validate(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
