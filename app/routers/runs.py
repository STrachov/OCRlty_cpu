from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.auth import require_scopes
from app.auth_store import ApiPrincipal
from app.schemas.api_error import common_error_responses
from app.services.artifacts import find_batch_artifact_path, list_batch_artifacts, read_artifact_json

router = APIRouter(prefix="/v1/runs", tags=["runs"])
_ERR = common_error_responses(400, 401, 403, 404, 422, 500)


class RunSummary(BaseModel):
    run_id: str
    created_at: Optional[str] = None
    task_id: Optional[str] = None
    item_count: Optional[int] = None
    ok_count: Optional[int] = None
    error_count: Optional[int] = None
    artifact_rel: Optional[str] = None


class RunsListResponse(BaseModel):
    items: List[RunSummary] = Field(default_factory=list)
    limit: int
    next_cursor: Optional[str] = None


def _encode_cursor(created_at: str, run_id: str) -> str:
    return f"{created_at}|{run_id}"


def _decode_cursor(cursor: str) -> tuple[str, str]:
    raw = (cursor or "").strip()
    if not raw or "|" not in raw:
        raise HTTPException(status_code=400, detail="Invalid cursor format.")
    created_at, run_id = raw.split("|", 1)
    created_at = created_at.strip()
    run_id = run_id.strip()
    if not created_at or not run_id:
        raise HTTPException(status_code=400, detail="Invalid cursor format.")
    return created_at, run_id


def _ensure_owner_or_admin(principal: ApiPrincipal, owner_key_id: Optional[str]) -> None:
    if principal.role == "admin":
        return
    if not owner_key_id or owner_key_id != principal.key_id:
        raise HTTPException(status_code=403, detail="You may only access your own runs.")


@router.get(
    "",
    response_model=RunsListResponse,
    responses=_ERR,
    dependencies=[Depends(require_scopes(["extract:run"]))],
)
async def list_runs(
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: Optional[str] = Query(default=None),
) -> RunsListResponse:
    cursor_created_at: Optional[str] = None
    cursor_run_id: Optional[str] = None
    if cursor:
        cursor_created_at, cursor_run_id = _decode_cursor(cursor)

    owner_filter = None if principal.role == "admin" else principal.key_id
    rows, next_cur = list_batch_artifacts(
        limit=limit,
        owner_key_id=owner_filter,
        cursor_created_at=cursor_created_at,
        cursor_run_id=cursor_run_id,
    )

    items: List[RunSummary] = []
    for r in rows:
        run_id = str(r.get("artifact_id") or "")
        full_ref = str(r.get("full_ref") or "")
        rel_ref = str(r.get("rel_ref") or "")
        if not run_id or not full_ref:
            continue
        try:
            obj = read_artifact_json(full_ref)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        items.append(
            RunSummary(
                run_id=run_id,
                created_at=obj.get("created_at"),
                task_id=obj.get("task_id"),
                item_count=obj.get("item_count"),
                ok_count=obj.get("ok_count"),
                error_count=obj.get("error_count"),
                artifact_rel=rel_ref or None,
            )
        )

    next_cursor = _encode_cursor(next_cur[0], next_cur[1]) if next_cur else None
    return RunsListResponse(items=items, limit=limit, next_cursor=next_cursor)


@router.get(
    "/{run_id}",
    response_model=Dict[str, Any],
    responses=_ERR,
    dependencies=[Depends(require_scopes(["extract:run"]))],
)
async def get_run(
    run_id: str,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
) -> Dict[str, Any]:
    ref = find_batch_artifact_path(run_id, date=None, max_days=3650)
    if ref is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    obj = read_artifact_json(ref)
    if not isinstance(obj, dict):
        raise HTTPException(status_code=500, detail="Invalid run JSON.")

    auth = obj.get("auth") if isinstance(obj.get("auth"), dict) else {}
    owner_key_id = auth.get("key_id") if isinstance(auth, dict) else None
    _ensure_owner_or_admin(principal, str(owner_key_id) if owner_key_id else None)
    return obj
