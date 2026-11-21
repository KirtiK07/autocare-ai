"""FastAPI application factory."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
        # Models and the vector store load once per process, not per request.
        ensure_container(app)
        yield
        container = getattr(app.state, "container", None)
        if container is not None:
            container.engine.dispose()

    app = FastAPI(
        title="AutoCare AI API",
        version="1.0.0",
        description="Engine fault prediction and a manual-grounded diagnostic assistant.",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.container_factory = container_factory
    app.state.container_lock = threading.Lock()
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
    if settings.static_dir and (settings.static_dir / "index.html").exists():
        mount_spa(app, settings.static_dir)
    return app


def ensure_container(app: FastAPI) -> Container:
    """Build the service container once per process.

    Called from the lifespan hook and, as a fallback, lazily from the first request, so
    serverless runtimes that skip ASGI lifespan events (e.g. Vercel Functions) still work.
    """
    container = getattr(app.state, "container", None)
    if container is None:
        with app.state.container_lock:
            container = getattr(app.state, "container", None)
            if container is None:
                container = app.state.container_factory(app.state.settings)
                app.state.container = container
    return container  # type: ignore[no-any-return]


def mount_spa(app: FastAPI, static_dir: Path) -> None:
    """Serve the built React app from the same container (single-container hosting).

    Hashed assets are served as files; every other non-API path returns index.html so
    client-side routes (/history, /model, ...) work on reload.
    """
    app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")
    index = static_dir / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404)
        candidate = (static_dir / path).resolve()
        if path and candidate.is_file() and static_dir.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index)


app = create_app()
