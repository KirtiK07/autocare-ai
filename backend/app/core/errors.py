"""Service errors and one place that maps them to clean JSON responses."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.services.llm import LLMError, LLMTimeoutError

log = logging.getLogger(__name__)


class ServiceError(Exception):
    status_code = 400
    error = "bad_request"

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class NotFoundError(ServiceError):
    status_code = 404
    error = "not_found"


class ModelUnavailableError(ServiceError):
    status_code = 503
    error = "model_unavailable"


class RateLimitedError(ServiceError):
    status_code = 429
    error = "rate_limited"


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def _service(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse({"error": exc.error, "detail": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [
            {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
            for e in exc.errors()
        ]
        return JSONResponse({"error": "validation_error", "detail": fields}, status_code=422)

    @app.exception_handler(LLMTimeoutError)
    async def _llm_timeout(_: Request, exc: LLMTimeoutError) -> JSONResponse:
        return JSONResponse(
            {"error": "llm_timeout", "detail": "The assistant took too long. Please retry in a few seconds."},
            status_code=503,
            headers={"Retry-After": "5"},
        )

    @app.exception_handler(LLMError)
    async def _llm(_: Request, exc: LLMError) -> JSONResponse:
        log.error("LLM error: %s", exc)
        return JSONResponse(
            {"error": "llm_unavailable", "detail": "The assistant is unavailable right now. Please retry."},
            status_code=503,
            headers={"Retry-After": "10"},
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error")
        return JSONResponse({"error": "internal_error", "detail": "Something went wrong."}, status_code=500)
