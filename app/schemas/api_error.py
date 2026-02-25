from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ApiError(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ApiError


_DEFAULT_STATUS_DESCRIPTIONS: dict[int, str] = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    422: "Validation Error",
    500: "Internal Server Error",
}


def common_error_responses(*codes: int) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for code in codes:
        out[int(code)] = {
            "model": ErrorResponse,
            "description": _DEFAULT_STATUS_DESCRIPTIONS.get(int(code), "Error"),
        }
    return out
