from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import OperationalError

from app.database import SessionLocal
from app.config import get_settings
from app.logging import configure_logging
from app.routers import audit, consensus, content_records, crypto, feedback, publishers, results, verify
from app.services.bootstrap import ensure_default_publishers
from app.services.crypto.issuer_registry import ensure_default_issuers


configure_logging()
settings = get_settings()
app = FastAPI(title="NewsProof API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.backend_cors_origins,
    allow_origin_regex=settings.effective_backend_cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(verify.router, prefix="/api/v1")
app.include_router(results.router, prefix="/api/v1")
app.include_router(audit.router, prefix="/api/v1")
app.include_router(consensus.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")
app.include_router(publishers.router, prefix="/api/v1")
app.include_router(content_records.router, prefix="/api/v1")
app.include_router(crypto.router, prefix="/api/v1")


@app.on_event("startup")
def bootstrap_registry() -> None:
    db = SessionLocal()
    try:
        try:
            ensure_default_publishers(db)
            ensure_default_issuers(db)
        except OperationalError:
            pass
    finally:
        db.close()


@app.get("/health")
def healthcheck():
    return {"status": "ok"}
