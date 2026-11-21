from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text

from app.api.deps import get_container
from app.core.container import Container
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(container: Container = Depends(get_container)) -> HealthResponse:
    """Liveness plus model version and vector-store status."""
    try:
        with container.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False
    predictor = container.predictor
    ok = predictor is not None and db_ok
    return HealthResponse(
        status="ok" if ok else "degraded",
        model_loaded=predictor is not None,
        model_version=predictor.version if predictor else None,
        vector_store=container.retriever.status(),
        llm_provider=container.llm.name,
        guardrail=container.guardrail.name,
        database=db_ok,
    )
