from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas.api_error import ApiError, ErrorResponse

LOG = logging.getLogger("ocrlty.errors")


def _request_id(request: Request) -> str | None:
    rid = getattr(request.state, "request_id", None)
    if isinstance(rid, str) and rid:
        return rid
    rid = request.headers.get("x-request-id") or request.headers.get("x-request_id")
    return rid or None


def _error_payload(
    *,
    code: str,
    message: str,
    request_id: str | None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return ErrorResponse(
        error=ApiError(
            code=code,
            message=message,
            request_id=request_id,
            details=details,
        )
    ).model_dump()


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException):
        rid = _request_id(request)
        details: dict[str, Any] | None = None
        code = "http_error"
        message = str(exc.detail) if exc.detail is not None else "Request failed."

        if isinstance(exc.detail, dict):
            details = dict(exc.detail)
            code = str(details.pop("code", code))
            message = str(details.pop("message", message))
            if not details:
                details = None

        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(code=code, message=message, request_id=rid, details=details),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _request_validation_handler(request: Request, exc: RequestValidationError):
        rid = _request_id(request)
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                code="validation_error",
                message="Request validation failed.",
                request_id=rid,
                details={"errors": exc.errors()},
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        rid = _request_id(request)
        LOG.exception("unhandled_exception", extra={"request_id": rid})
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                code="internal_error",
                message="Internal server error.",
                request_id=rid,
            ),
        )
