"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import diagnose, health, model, predict, predictions
from app.core.config import Settings, get_settings
from app.core.container import Container, build_container
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging
from app.core.ratelimit import RateLimiter


def create_app(
    settings: Settings | None = None,
    container_factory: Callable[[Settings], Container] = build_container,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Models and the vector store load once here, not per request.
        app.state.container = container_factory(settings)
        yield
        app.state.container.engine.dispose()

    app = FastAPI(
        title="AutoCare AI API",
        version="1.0.0",
        description="Engine fault prediction and a manual-grounded diagnostic assistant.",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.diagnose_limiter = RateLimiter(settings.diagnose_rate_limit_per_minute)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    register_error_handlers(app)
    for module in (health, predict, predictions, diagnose, model):
        app.include_router(module.router, prefix="/api")
    return app


app = create_app()
