from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import BACKEND_DIR, Settings
from app.core.container import Container
from app.db.base import Base, make_engine, make_session_factory
from app.main import create_app
from app.services.diagnostic_agent import DiagnosticAgent
from app.services.fault_predictor import FaultPredictor
from app.services.guardrail import LexicalGuardrail
from app.services.llm import ChatTurn, LLMClient
from app.services.manual_retriever import Chunk

ARTIFACTS = BACKEND_DIR / "artifacts"
HAS_ARTIFACTS = (ARTIFACTS / "model_meta.json").exists()
needs_artifacts = pytest.mark.skipif(not HAS_ARTIFACTS, reason="run `python -m ml.train` first")

OIL_CHUNK = Chunk(
    "c1",
    "Owner's Manual",
    42,
    "If the oil pressure warning light comes on, stop the engine and check the oil level. "
    "Low oil pressure can mean the oil level is low or the oil pump is worn.",
    0.81,
)


class FakeRetriever:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = [OIL_CHUNK] if chunks is None else chunks
        self.queries: list[str] = []

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        self.queries.append(query)
        return self.chunks[:k]

    def status(self) -> dict[str, Any]:
        return {"ready": bool(self.chunks), "chunks": len(self.chunks)}


class FakeLLM(LLMClient):
    """Stands in for Groq/Gemini so CI never calls a paid API."""

    name = "fake"

    def __init__(self, reply: str | None = None):
        self.reply = reply or (
            "Low oil pressure can mean the oil level is low or the oil pump is worn [Owner's Manual, page 42]. "
            "Stop the engine and check the oil level [Owner's Manual, page 42]. "
            "Have a qualified mechanic inspect it."
        )
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, history: list[ChatTurn] | None = None) -> str:
        self.calls.append((system, user))
        return self.reply


SAMPLE = {
    "engine_rpm": 700.0,
    "lub_oil_pressure": 2.5,
    "fuel_pressure": 11.0,
    "coolant_pressure": 3.1,
    "lub_oil_temp": 84.0,
    "coolant_temp": 81.0,
}


@pytest.fixture(scope="session")
def predictor() -> FaultPredictor:
    if not HAS_ARTIFACTS:
        pytest.skip("run `python -m ml.train` first")
    return FaultPredictor(ARTIFACTS)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(predictor: FaultPredictor, fake_llm: FakeLLM) -> Iterator[TestClient]:
    settings = Settings(
        database_url="sqlite://",
        environment="test",
        diagnose_rate_limit_per_minute=3,
        batch_max_rows=50,
    )

    def factory(s: Settings) -> Container:
        engine = make_engine(s.database_url)
        Base.metadata.create_all(engine)
        retriever = FakeRetriever()
        guardrail = LexicalGuardrail()
        agent = DiagnosticAgent(retriever, fake_llm, guardrail)
        return Container(
            s, engine, make_session_factory(engine), predictor, retriever, fake_llm, guardrail, agent
        )

    with TestClient(create_app(settings, factory)) as c:
        yield c
