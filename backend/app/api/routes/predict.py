from __future__ import annotations

import csv
import io
from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.api.deps import get_container, get_predictor, get_session
from app.api.routes._convert import to_record
from app.core.container import Container
from app.core.errors import ServiceError
from app.db.repositories import PredictionRepository
from app.schemas import BatchResponse, BatchRowResult, PredictionRecord, SensorReading
from app.services.fault_predictor import FaultPredictor
from ml.features import RAW_COLUMN_ALIASES, SENSOR_COLUMNS

router = APIRouter(tags=["predict"])
MAX_UPLOAD_BYTES = 2_000_000


@router.post("/predict", response_model=PredictionRecord, status_code=201)
def predict(
    reading: SensorReading,
    predictor: FaultPredictor = Depends(get_predictor),
    session: Session = Depends(get_session),
) -> PredictionRecord:
    """Score one reading and store it in the history."""
    prediction = predictor.predict(reading)
    row = PredictionRepository(session).add(reading, prediction)
    return to_record(row)


def _normalise_header(name: str) -> str:
    name = name.strip()
    return RAW_COLUMN_ALIASES.get(name, name.lower().replace(" ", "_"))


def _row_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{err['loc'][0]}: {err['msg']}" for err in exc.errors())
    return "non-numeric or missing value"


@router.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(
    file: UploadFile = File(..., description="CSV with the six sensor columns"),
    predictor: FaultPredictor = Depends(get_predictor),
    container: Container = Depends(get_container),
) -> BatchResponse:
    """Score up to BATCH_MAX_ROWS rows from a CSV upload (not stored in history)."""
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ServiceError("CSV is larger than 2 MB")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ServiceError("CSV must be UTF-8 encoded") from exc
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ServiceError("CSV has no header row")
    header_map = {h: _normalise_header(h) for h in reader.fieldnames}
    missing = [c for c in SENSOR_COLUMNS if c not in header_map.values()]
    if missing:
        raise ServiceError(f"CSV is missing columns: {', '.join(missing)}")

    limit = container.settings.batch_max_rows
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(reader):
        if i >= limit:
            raise ServiceError(f"CSV has more than {limit} rows")
        rows.append(
            {header_map[k]: v for k, v in r.items() if k in header_map and header_map[k] in SENSOR_COLUMNS}
        )

    results: list[BatchRowResult] = []
    valid: list[tuple[int, SensorReading]] = []
    for i, r in enumerate(rows, start=1):
        try:
            valid.append((i, SensorReading(**{k: float(v) for k, v in r.items()})))
        except (ValidationError, ValueError, TypeError) as exc:
            results.append(BatchRowResult(row=i, inputs=r, error=_row_error(exc)))
    predictions = predictor.predict_many([reading for _, reading in valid])
    for (i, reading), pred in zip(valid, predictions, strict=True):
        results.append(BatchRowResult(row=i, inputs=reading.model_dump(), prediction=pred))
    results.sort(key=lambda x: x.row)
    ok = sum(r.prediction is not None for r in results)
    return BatchResponse(rows=len(results), succeeded=ok, failed=len(results) - ok, results=results)
