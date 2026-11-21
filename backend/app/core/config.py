"""Application settings, read from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR.parent / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AutoCare AI"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    database_url: str = f"sqlite:///{(BACKEND_DIR / 'autocare.db').as_posix()}"
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["http://localhost:5173"])

    artifacts_dir: Path = BACKEND_DIR / "artifacts"
    knowledge_dir: Path = BACKEND_DIR / "knowledge"
    vectorstore_dir: Path = BACKEND_DIR / "vectorstore"
    collection_name: str = "manuals"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_provider: Literal["groq", "gemini", "none"] = "groq"
    llm_api_key: str = ""
    llm_model: str = ""  # empty -> provider default
    llm_timeout_seconds: float = 30.0

    retrieval_k: int = 5
    retrieval_min_score: float = 0.2

    guardrail_enabled: bool = True
    guardrail_model: str = "cross-encoder/nli-deberta-v3-xsmall"
    guardrail_entailment_threshold: float = 0.5
    guardrail_min_supported_share: float = 0.5

    diagnose_rate_limit_per_minute: int = 10
    batch_max_rows: int = 1000

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip().startswith("["):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @field_validator("artifacts_dir", "knowledge_dir", "vectorstore_dir", mode="after")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        return v if v.is_absolute() else BACKEND_DIR / v


@lru_cache
def get_settings() -> Settings:
    return Settings()
