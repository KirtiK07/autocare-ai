"""Pydantic request/response models (the API contract)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RiskBand = Literal["low", "medium", "high"]


class SensorReading(BaseModel):
    """One snapshot of engine sensors. Bounds reject physically impossible values."""

    model_config = ConfigDict(extra="forbid")

    engine_rpm: float = Field(..., ge=0, le=8000, description="Engine speed (rpm)")
    lub_oil_pressure: float = Field(..., ge=0, le=30, description="Lubricant oil pressure (bar)")
    fuel_pressure: float = Field(..., ge=0, le=50, description="Fuel pressure (bar)")
    coolant_pressure: float = Field(..., ge=0, le=30, description="Coolant pressure (bar)")
    lub_oil_temp: float = Field(..., ge=-40, le=200, description="Lubricant oil temperature (°C)")
    coolant_temp: float = Field(..., ge=-40, le=250, description="Coolant temperature (°C)")


class FeatureContribution(BaseModel):
    feature: str
    label: str
    value: float
    deviation: float = Field(description="Signed distance from the healthy mean, in std units")
    direction: Literal["high", "low"]


class Prediction(BaseModel):
    probability: float
    label: Literal["faulty", "healthy"]
    risk_band: RiskBand
    top_features: list[FeatureContribution]
    model_version: str


class PredictionRecord(Prediction):
    id: int
    created_at: datetime
    inputs: SensorReading


class ChatMessageOut(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    citations: list[Citation] = []
    supported: bool | None = None
    created_at: datetime


class PredictionDetail(PredictionRecord):
    messages: list[ChatMessageOut] = []


class PredictionPage(BaseModel):
    items: list[PredictionRecord]
    total: int
    page: int
    page_size: int


class BatchRowResult(BaseModel):
    row: int
    inputs: dict[str, Any]
    prediction: Prediction | None = None
    error: str | None = None


class BatchResponse(BaseModel):
    rows: int
    succeeded: int
    failed: int
    results: list[BatchRowResult]


class Citation(BaseModel):
    source: str
    page: int | None = None


class Snippet(BaseModel):
    id: str
    source: str
    page: int | None = None
    text: str
    score: float


class SentenceCheck(BaseModel):
    text: str
    supported: bool
    score: float | None = None


class DiagnoseRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    prediction_id: int | None = None


class DiagnoseResponse(BaseModel):
    answer: str
    citations: list[Citation]
    snippets: list[Snippet]
    sentences: list[SentenceCheck]
    supported: bool
    covered: bool
    mode: str = Field(description="LLM provider used, or 'extractive' when no key is configured")
    message_id: int | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
    vector_store: dict[str, Any]
    llm_provider: str
    guardrail: str
    database: bool


class ErrorResponse(BaseModel):
    error: str
    detail: Any = None


ChatMessageOut.model_rebuild()
PredictionDetail.model_rebuild()
