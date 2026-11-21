"""Builds the services once at startup (FastAPI lifespan) and holds them."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.runtime import on_vercel, run_migrations, writable_vectorstore
from app.db import models  # noqa: F401 - register tables
from app.db.base import Base, make_engine, make_session_factory
from app.services.diagnostic_agent import DiagnosticAgent
from app.services.fault_predictor import FaultPredictor, ModelNotReadyError
from app.services.guardrail import Guardrail, build_guardrail
from app.services.llm import LLMClient, build_llm
from app.services.manual_retriever import ManualRetriever, Retriever

log = logging.getLogger(__name__)


class _EmptyRetriever:
    """Used when the vector store could not be opened."""

    def __init__(self, error: str):
        self.error = error

    def search(self, query: str, k: int = 5) -> list[Any]:
        return []

    def status(self) -> dict[str, Any]:
        return {"ready": False, "chunks": 0, "error": self.error}


@dataclass
class Container:
    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    predictor: FaultPredictor | None
    retriever: Retriever
    llm: LLMClient
    guardrail: Guardrail
    agent: DiagnosticAgent
    startup_errors: dict[str, str] = field(default_factory=dict)


def build_container(settings: Settings) -> Container:
    errors: dict[str, str] = {}
    if on_vercel():
        try:
            run_migrations(settings.database_url)
        except Exception as exc:  # noqa: BLE001 - report via /api/health instead of crashing
            errors["database"] = str(exc)
            log.exception("Migrations failed")
    engine = make_engine(settings.database_url)
    if on_vercel():
        # Safety net for a fresh /tmp SQLite file: create any table the migration missed (idempotent).
        Base.metadata.create_all(engine)
    predictor: FaultPredictor | None
    try:
        predictor = FaultPredictor(settings.artifacts_dir)
        log.info("Loaded model %s", predictor.version)
    except (ModelNotReadyError, FileNotFoundError) as exc:
        predictor, errors["model"] = None, str(exc)
        log.warning("Model not loaded: %s", exc)
    retriever: Retriever
    try:
        retriever = ManualRetriever(
            writable_vectorstore(settings.vectorstore_dir),
            settings.collection_name,
            settings.embedding_model,
            settings.retrieval_min_score,
        )
    except Exception as exc:  # noqa: BLE001 - keep the API up without RAG
        retriever, errors["vector_store"] = _EmptyRetriever(str(exc)), str(exc)
        log.warning("Vector store unavailable: %s", exc)
    llm = build_llm(
        settings.llm_provider, settings.llm_api_key, settings.llm_model, settings.llm_timeout_seconds
    )
    guardrail = build_guardrail(
        settings.guardrail_enabled, settings.guardrail_model, settings.guardrail_entailment_threshold
    )
    agent = DiagnosticAgent(
        retriever, llm, guardrail, settings.retrieval_k, settings.guardrail_min_supported_share
    )
    return Container(
        settings, engine, make_session_factory(engine), predictor, retriever, llm, guardrail, agent, errors
    )
