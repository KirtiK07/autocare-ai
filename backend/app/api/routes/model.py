from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends

from app.api.deps import get_container, get_predictor
from app.core.container import Container
from app.core.errors import ModelUnavailableError
from app.services.fault_predictor import FaultPredictor
from ml.features import SENSOR_LABELS

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/metrics")
def metrics(container: Container = Depends(get_container)) -> dict[str, Any]:
    """Model comparison table, ROC curve, confusion matrices and feature importance."""
    path = container.settings.artifacts_dir / "metrics.json"
    if not path.exists():
        raise ModelUnavailableError("metrics.json not found; run `python -m ml.train`")
    return dict(json.loads(path.read_text(encoding="utf-8")))


@router.get("/info")
def info(predictor: FaultPredictor = Depends(get_predictor)) -> dict[str, Any]:
    """Sample readings for the UI, sensor ranges and labels, threshold and version."""
    return {
        "model_version": predictor.version,
        "served_model": predictor.meta.get("served_model_name"),
        "threshold": predictor.threshold,
        "risk_bands": predictor.meta.get("risk_bands"),
        "samples": predictor.samples(),
        "sensor_ranges": predictor.sensor_ranges(),
        "sensor_labels": SENSOR_LABELS,
        "dataset": predictor.meta.get("dataset"),
    }
