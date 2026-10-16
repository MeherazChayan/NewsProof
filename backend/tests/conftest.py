import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ["DATABASE_URL"] = "sqlite://"
os.environ["UPLOAD_DIR"] = "/tmp/test_uploads"

from app.database import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.routers import verify as verify_router  # noqa: E402

engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


@pytest.fixture(autouse=True)
def setup_database():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    original_allow = verify_router.rate_limiter.allow
    verify_router.rate_limiter.allow = lambda _client_host: True
    with TestClient(app) as test_client:
        yield test_client
    verify_router.rate_limiter.allow = original_allow
    app.dependency_overrides.clear()
