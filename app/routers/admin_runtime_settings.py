from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.auth import require_scopes
from app.auth_store import ApiPrincipal
from app.schemas.api_error import common_error_responses
from app.services import runtime_config
from app.services.runtime_settings_store import RuntimeSettingsStore

router = APIRouter(prefix="/v1/admin/runtime_settings", tags=["admin_runtime_settings"])
_ERR = common_error_responses(400, 401, 403, 404, 422, 500)


def _require_admin(principal: ApiPrincipal) -> None:
    if (principal.role or "").strip().lower() != "admin":
        raise HTTPException(status_code=403, detail="Admin role required.")


def _store(request: Request) -> RuntimeSettingsStore:
    s = getattr(request.app.state, "runtime_settings_store", None)
    if s is None:
        raise HTTPException(status_code=500, detail="Runtime settings store is not initialized.")
    return s


class RuntimeSettingView(BaseModel):
    key: str
    value: Optional[Any] = None
    effective_value: Optional[Any] = None
    source: str
    updated_at: Optional[str] = None
    is_secret: bool
    is_set: bool


class RuntimeSettingListResponse(BaseModel):
    items: List[RuntimeSettingView] = Field(default_factory=list)


class RuntimeSettingUpdateRequest(BaseModel):
    value: str


@router.get(
    "",
    response_model=RuntimeSettingListResponse,
    responses=_ERR,
    dependencies=[Depends(require_scopes([]))],
)
async def list_runtime_settings(
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes([])),
) -> RuntimeSettingListResponse:
    _store(request)
    _require_admin(principal)
    return RuntimeSettingListResponse(items=[RuntimeSettingView(**item) for item in runtime_config.list_effective_settings()])


@router.put(
    "/{key}",
    response_model=RuntimeSettingView,
    responses=_ERR,
    dependencies=[Depends(require_scopes([]))],
)
async def update_runtime_setting(
    key: str,
    body: RuntimeSettingUpdateRequest,
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes([])),
) -> RuntimeSettingView:
    _store(request)
    _require_admin(principal)
    try:
        runtime_config.set_value(key=key, value=body.value, updated_by_key_id=principal.key_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Runtime setting not found.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    item = next((it for it in runtime_config.list_effective_settings() if it["key"] == key), None)
    if item is None:
        raise HTTPException(status_code=500, detail="Runtime setting update failed.")
    return RuntimeSettingView(**item)
