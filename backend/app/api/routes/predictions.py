from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_session
from app.api.routes._convert import to_detail, to_record
from app.core.errors import NotFoundError
from app.db.repositories import PredictionRepository
from app.schemas import PredictionDetail, PredictionPage

router = APIRouter(tags=["history"])


@router.get("/predictions", response_model=PredictionPage)
def list_predictions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    risk_band: Literal["low", "medium", "high"] | None = None,
    session: Session = Depends(get_session),
) -> PredictionPage:
    """Paginated prediction history, newest first."""
    rows, total = PredictionRepository(session).list(page, page_size, risk_band)
    return PredictionPage(items=[to_record(r) for r in rows], total=total, page=page, page_size=page_size)


@router.get("/predictions/{prediction_id}", response_model=PredictionDetail)
def get_prediction(prediction_id: int, session: Session = Depends(get_session)) -> PredictionDetail:
    """One prediction with its inputs, drivers and chat history."""
    row = PredictionRepository(session).get(prediction_id)
    if row is None:
        raise NotFoundError(f"Prediction {prediction_id} not found")
    return to_detail(row)
