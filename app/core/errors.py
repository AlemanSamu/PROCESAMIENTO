import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    status_code = 400
    error_code = "app_error"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class BadRequestError(AppError):
    status_code = 400
    error_code = "bad_request"


class AuthenticationError(AppError):
    status_code = 401
    error_code = "authentication_error"


class ProjectNotFoundError(AppError):
    status_code = 404
    error_code = "project_not_found"


class InvalidProjectStateError(AppError):
    status_code = 409
    error_code = "invalid_project_state"


class StorageError(AppError):
    status_code = 500
    error_code = "storage_error"


class ProcessingError(AppError):
    status_code = 500
    error_code = "processing_error"

    def __init__(
        self,
        message: str,
        *,
        reason_code: str | None = None,
        current_stage: str | None = None,
        metadata: dict[str, Any] | None = None,
        allow_fallback: bool = True,
    ) -> None:
        self.reason_code = reason_code
        self.current_stage = current_stage
        self.metadata = dict(metadata or {})
        self.allow_fallback = allow_fallback
        super().__init__(message)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.error_code, "message": exc.message}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": str(exc)}},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_server_error",
                    "message": "Error interno no controlado.",
                }
            },
        )