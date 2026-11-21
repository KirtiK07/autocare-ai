"""FastAPI dependencies: the service container and a DB session per request."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from app.core.container import Container
from app.core.errors import ModelUnavailableError
from app.db.base import session_scope
from app.services.fault_predictor import FaultPredictor


def get_container(request: Request) -> Container:
    from app.main import ensure_container

    return ensure_container(request.app)


def get_session(request: Request) -> Iterator[Session]:
    yield from session_scope(get_container(request).session_factory)


def get_predictor(request: Request) -> FaultPredictor:
    predictor = get_container(request).predictor
    if predictor is None:
        raise ModelUnavailableError("Model artifacts not found. Run `python -m ml.train` and restart.")
    return predictor
