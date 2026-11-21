"""Data-access classes: the only code that writes SQL (via the ORM)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import ChatMessageRow, PredictionRow
from app.schemas import Prediction, SensorReading


class PredictionRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, reading: SensorReading, prediction: Prediction) -> PredictionRow:
        row = PredictionRow(
            inputs=reading.model_dump(),
            probability=prediction.probability,
            label=prediction.label,
            risk_band=prediction.risk_band,
            top_features=[f.model_dump() for f in prediction.top_features],
            model_version=prediction.model_version,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def get(self, prediction_id: int) -> PredictionRow | None:
        return self.session.get(PredictionRow, prediction_id)

    def list(
        self, page: int, page_size: int, risk_band: str | None = None
    ) -> tuple[list[PredictionRow], int]:
        query = select(PredictionRow)
        count = select(func.count()).select_from(PredictionRow)
        if risk_band:
            query = query.where(PredictionRow.risk_band == risk_band)
            count = count.where(PredictionRow.risk_band == risk_band)
        query = query.order_by(PredictionRow.created_at.desc(), PredictionRow.id.desc())
        rows = self.session.scalars(query.offset((page - 1) * page_size).limit(page_size)).all()
        return list(rows), int(self.session.scalar(count) or 0)


class ChatRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(
        self,
        prediction_id: int | None,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        supported: bool | None = None,
    ) -> ChatMessageRow:
        row = ChatMessageRow(
            prediction_id=prediction_id,
            role=role,
            content=content,
            citations=citations or [],
            supported=supported,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def recent(self, prediction_id: int, limit: int = 6) -> list[ChatMessageRow]:
        query = (
            select(ChatMessageRow)
            .where(ChatMessageRow.prediction_id == prediction_id)
            .order_by(ChatMessageRow.id.desc())
            .limit(limit)
        )
        return list(reversed(self.session.scalars(query).all()))
