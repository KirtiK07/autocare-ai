from __future__ import annotations

from datetime import UTC, datetime

from app.db.models import ChatMessageRow, PredictionRow
from app.schemas import (
    ChatMessageOut,
    Citation,
    FeatureContribution,
    Prediction,
    PredictionDetail,
    PredictionRecord,
    SensorReading,
)


def utc(dt: datetime) -> datetime:
    """SQLite drops tzinfo; values are always stored as UTC, so re-attach it."""
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def to_prediction(row: PredictionRow) -> Prediction:
    return Prediction(
        probability=row.probability,
        label=row.label,  # type: ignore[arg-type]
        risk_band=row.risk_band,  # type: ignore[arg-type]
        top_features=[FeatureContribution(**f) for f in row.top_features],
        model_version=row.model_version,
    )


def to_record(row: PredictionRow) -> PredictionRecord:
    return PredictionRecord(
        id=row.id,
        created_at=utc(row.created_at),
        inputs=SensorReading(**row.inputs),
        **to_prediction(row).model_dump(),
    )


def to_message(m: ChatMessageRow) -> ChatMessageOut:
    return ChatMessageOut(
        id=m.id,
        role=m.role,  # type: ignore[arg-type]
        content=m.content,
        citations=[Citation(**c) for c in m.citations or []],
        supported=m.supported,
        created_at=utc(m.created_at),
    )


def to_detail(row: PredictionRow) -> PredictionDetail:
    return PredictionDetail(**to_record(row).model_dump(), messages=[to_message(m) for m in row.messages])
