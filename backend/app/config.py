from functools import lru_cache
from typing import List

from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    database_url: str = "postgresql+psycopg://postgres:postgres@postgres:5432/newsproof"
    redis_url: str = "redis://redis:6379/0"
    cohere_api_key: str | None = None
    google_programmable_search_api_key: str | None = None
    google_programmable_search_engine_id: str | None = None
    brave_search_api_key: str | None = None
    youtube_data_api_key: str | None = None
    faster_whisper_model_size: str = "small"
    faster_whisper_device: str = "cpu"
    faster_whisper_compute_type: str = "int8"
    faster_whisper_cpu_threads: int = 4
    faster_whisper_beam_size: int = 5
    enable_external_rerank: bool = False
    enable_local_ml_rerank: bool = False
    enable_claim_verifier_model: bool = False
    claim_verifier_model_path: str | None = None
    backend_cors_origins_raw: str = "http://localhost:3000"
    backend_cors_origin_regex: str | None = r"^https://[a-zA-Z0-9-]+\.(ngrok-free\.app|ngrok\.io)$"
    upload_dir: str = "/app/uploads"
    image_upload_max_bytes: int = 5_000_000
    audio_upload_max_bytes: int = 25_000_000
    video_upload_max_bytes: int = 100_000_000
    request_timeout_seconds: int = 15
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    platform_signing_private_key_b64: str | None = None
    platform_issuer_name: str = "NewsProof Platform"
    timestamp_mode: str = "internal"
    tsa_url: str | None = None

    @field_validator(
        "cohere_api_key",
        "google_programmable_search_api_key",
        "google_programmable_search_engine_id",
        "brave_search_api_key",
        "youtube_data_api_key",
        "platform_signing_private_key_b64",
        "tsa_url",
        mode="before",
    )
    @classmethod
    def _strip_optional_secret_values(cls, value):
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @computed_field
    @property
    def backend_cors_origins(self) -> List[str]:
        origins = [item.strip() for item in self.backend_cors_origins_raw.split(",") if item.strip()]
        if self.app_env.lower() == "production":
            return [origin for origin in origins if origin != "*"]
        return origins

    @computed_field
    @property
    def effective_backend_cors_origin_regex(self) -> str | None:
        if self.app_env.lower() == "production":
            return None
        return self.backend_cors_origin_regex


@lru_cache
def get_settings() -> Settings:
    return Settings()
