from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.auth import require_scopes
from app.auth_store import ApiPrincipal
from app.schemas.api_error import common_error_responses
from app.handlers import _validate_request_id_for_fs
from app.services.artifacts import (
    artifact_ref_from_rel,
    find_artifact_path,
    find_batch_artifact_path,
    list_batch_artifacts,
    read_artifact_json,
)
from app.settings import settings

router = APIRouter(prefix="/v1/runs", tags=["runs"])
_ERR = common_error_responses(400, 401, 403, 404, 422, 500)

class EvalSummary(BaseModel):
    items:Optional[int] = None
    gt_found:Optional[int] = None
    gt_missing:Optional[int] = None
    pred_found:Optional[int] = None
    pred_missing:Optional[int] = None
    str_mode:Optional[str] = None
    decimal_sep:Optional[str] = None
    mismatched:Optional[int] = None

class RunSummary(BaseModel):
    run_id: str
    created_at: Optional[str] = None
    task_id: Optional[str] = None
    item_count: Optional[int] = None
    ok_count: Optional[int] = None
    error_count: Optional[int] = None
    artifact_rel: Optional[str] = None
    eval_summary: Optional[EvalSummary] = None


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


def _resolve_eval_owner_key_id(eval_obj: Dict[str, Any]) -> Optional[str]:
    auth = eval_obj.get("auth") if isinstance(eval_obj.get("auth"), dict) else {}
    owner_key_id = auth.get("key_id") if isinstance(auth, dict) else None
    if owner_key_id:
        return str(owner_key_id)

    batch_artifact_rel = eval_obj.get("batch_artifact_rel")
    if not isinstance(batch_artifact_rel, str) or not batch_artifact_rel:
        return None

    batch_obj = read_artifact_json(artifact_ref_from_rel(batch_artifact_rel))
    batch_auth = batch_obj.get("auth") if isinstance(batch_obj.get("auth"), dict) else {}
    batch_owner_key_id = batch_auth.get("key_id") if isinstance(batch_auth, dict) else None
    return str(batch_owner_key_id) if batch_owner_key_id else None


@router.get(
    "",
    response_model=RunsListResponse,
    responses=_ERR
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
        eval_obj = obj.get("eval")
        summary = eval_obj.get("summary") if isinstance(eval_obj, dict) else None
        by_request_id = eval_obj.get("by_request_id") if isinstance(eval_obj, dict) else None
        mismatched = None
        if isinstance(by_request_id, dict):
            mismatched = sum(
                1
                for row in by_request_id.values()
                if isinstance(row, dict) and int(row.get("mismatches_count", 0) or 0) > 0
            )
        items.append(
            RunSummary(
                run_id=run_id,
                created_at=obj.get("created_at"),
                task_id=obj.get("task_id"),
                item_count=obj.get("item_count"),
                ok_count=obj.get("ok_count"),
                error_count=obj.get("error_count"),
                artifact_rel=rel_ref or None,
                eval_summary=EvalSummary(
                    **summary,
                    mismatched=mismatched,
                ) if isinstance(summary, dict) else None
            )
        )
    next_cursor = _encode_cursor(next_cur[0], next_cur[1]) if next_cur else None
    return RunsListResponse(items=items, limit=limit, next_cursor=next_cursor)


@router.get(
    "/eval",
    response_model=Dict[str, Any],
    responses=_ERR,
)
async def get_eval_artifact(
    artifact_rel: str = Query(..., description="Relative eval artifact path, e.g. evals/YYYY-MM-DD/<eval_id>.json"),
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
) -> Dict[str, Any]:
    rel_norm = str(artifact_rel or "").replace("\\", "/").lstrip("/")
    if not rel_norm.startswith("evals/"):
        raise HTTPException(status_code=400, detail="Invalid artifact_rel.")

    obj = read_artifact_json(artifact_ref_from_rel(rel_norm))
    if not isinstance(obj, dict):
        raise HTTPException(status_code=500, detail="Invalid eval JSON.")

    owner_key_id = _resolve_eval_owner_key_id(obj)
    _ensure_owner_or_admin(principal, owner_key_id)
    return obj


@router.get(
    "/{run_id}",
    response_model=Dict[str, Any],
    responses=_ERR
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

@router.get(
    "/item/{request_id}",
    response_model=Dict[str, Any],
    responses=_ERR,
)
def get_extract_artifact(
    request_id: str,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
) -> Dict[str, Any]:
    """
    Drill-down по одному item (extract artifact).

    - 1 чтение JSON из storage
    - владелец: artifact.auth.key_id == principal.key_id (если AUTH_ENABLED)
    - raw: только при scope debug:read_raw, иначе raw=null
    """
    _validate_request_id_for_fs(request_id)

    ref = find_artifact_path(request_id)
    if not ref:
        raise HTTPException(status_code=404, detail="Not found")

    obj = read_artifact_json(ref)

    # Owner check (hide existence)
    if settings.AUTH_ENABLED:
        owner_key_id = None
        auth = obj.get("auth")
        if isinstance(auth, dict):
            owner_key_id = auth.get("key_id")

        is_admin = (principal.role or "").strip().lower() == "admin"
        if (not is_admin) and (owner_key_id != principal.key_id):
            raise HTTPException(status_code=404, detail="Not found")

    # Raw gating (no debug by default)
    if "debug:read_raw" not in getattr(principal, "scopes", set()):
        if "raw" in obj:
            obj["raw"] = None

    return obj
