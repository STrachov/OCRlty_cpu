from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.auth import require_scopes
from app.auth_store import ApiPrincipal
from app.schemas.api_error import common_error_responses
from app.handlers import _validate_request_id_for_fs
from app.services.artifacts import (
    artifact_ref_from_rel,
    delete_indexed_artifact,
    delete_run_artifacts,
    find_artifact_path,
    find_batch_artifact_path,
    list_batch_artifacts,
    read_artifact_json,
)
from app.services.jobs_store import JobsStore
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
    fields: Optional[List[Dict[str, Any]]] = None

class RunSummary(BaseModel):
    run_id: str
    created_at: Optional[str] = None
    task_id: Optional[str] = None
    model_id: Optional[str] = None
    prompt_sha256: Optional[str] = None
    schema_sha256: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    code_version: Optional[str] = None
    item_count: Optional[int] = None
    ok_count: Optional[int] = None
    error_count: Optional[int] = None
    total_time: Optional[int] = None
    artifact_rel: Optional[str] = None
    eval_summary: Optional[EvalSummary] = None


class RunsListResponse(BaseModel):
    items: List[RunSummary] = Field(default_factory=list)
    limit: int
    next_cursor: Optional[str] = None


class RunCatalogItem(BaseModel):
    run_id: str
    created_at: Optional[str] = None


class RunCatalogResponse(BaseModel):
    items: List[RunCatalogItem] = Field(default_factory=list)
    limit: int
    next_cursor: Optional[str] = None


class DeleteRunResponse(BaseModel):
    ok: bool
    run_id: str
    deleted_extracts: int = 0
    deleted_evals: int = 0
    deleted_job_results: int = 0
    deleted_job_errors: int = 0
    deleted_jobs: int = 0


def _build_eval_summary(eval_obj: Any) -> Optional[EvalSummary]:
    if not isinstance(eval_obj, dict):
        return None

    raw_summary = eval_obj.get("summary")
    summary = raw_summary if isinstance(raw_summary, dict) else {}
    raw_fields = eval_obj.get("fields")
    fields = raw_fields if isinstance(raw_fields, list) else None
    by_request_id = eval_obj.get("by_request_id")

    mismatched: Optional[int] = None

    if isinstance(by_request_id, dict):
        mismatched = sum(
            1
            for row in by_request_id.values()
            if isinstance(row, dict) and int(row.get("mismatches_count", 0) or 0) > 0
        )

    if not summary and mismatched is None and fields is None:
        return None

    return EvalSummary(
        **summary,
        mismatched=mismatched,
        fields=fields,
    )


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


def _jobs_store(request: Request) -> JobsStore:
    store = getattr(request.app.state, "jobs_store", None)
    if store is None:
        raise HTTPException(status_code=500, detail="Jobs store is not initialized.")
    return store


@router.get(
    "/catalog",
    response_model=RunCatalogResponse,
    responses=_ERR,
)
async def list_runs_catalog(
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: Optional[str] = Query(default=None),
) -> RunCatalogResponse:
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

    items = [
        RunCatalogItem(
            run_id=str(r.get("artifact_id") or ""),
            created_at=str(r.get("created_at") or "") or None,
        )
        for r in rows
        if str(r.get("artifact_id") or "")
    ]
    next_cursor = _encode_cursor(next_cur[0], next_cur[1]) if next_cur else None
    return RunCatalogResponse(items=items, limit=limit, next_cursor=next_cursor)


@router.get(
    "",
    response_model=RunsListResponse,
    responses=_ERR
)
async def list_runs(
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: Optional[str] = Query(default=None),
    task_id: Optional[str] = Query(default=None),
) -> RunsListResponse:
    cursor_created_at: Optional[str] = None
    cursor_run_id: Optional[str] = None
    if cursor:
        cursor_created_at, cursor_run_id = _decode_cursor(cursor)

    owner_filter = None if principal.role == "admin" else principal.key_id
    task_id_filter = str(task_id or "").strip() or None

    items: List[RunSummary] = []
    next_cursor: Optional[str] = None

    if task_id_filter is None:
        rows, next_cur = list_batch_artifacts(
            limit=limit,
            owner_key_id=owner_filter,
            cursor_created_at=cursor_created_at,
            cursor_run_id=cursor_run_id,
        )

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
            timings_ms = obj.get("timings_ms")
            total_time = timings_ms.get("total") if isinstance(timings_ms, dict) else None
            eval_summary = _build_eval_summary(obj.get("eval"))
            items.append(
                RunSummary(
                    run_id=run_id,
                    created_at=obj.get("created_at"),
                    task_id=obj.get("task_id"),
                    model_id=obj.get("model_id"),
                    prompt_sha256=obj.get("prompt_sha256"),
                    schema_sha256=obj.get("schema_sha256"),
                    temperature=obj.get("temperature"),
                    max_tokens=obj.get("max_tokens"),
                    code_version=obj.get("code_version"),
                    item_count=obj.get("item_count"),
                    ok_count=obj.get("ok_count"),
                    error_count=obj.get("error_count"),
                    artifact_rel=rel_ref or None,
                    total_time=total_time,
                    eval_summary=eval_summary,
                )
            )
        next_cursor = _encode_cursor(next_cur[0], next_cur[1]) if next_cur else None
    else:
        scan_created_at = cursor_created_at
        scan_run_id = cursor_run_id
        scan_limit = min(max(limit * 3, 50), 200)
        matched_positions: List[tuple[str, str]] = []

        while len(items) < limit + 1:
            rows, next_cur = list_batch_artifacts(
                limit=scan_limit,
                owner_key_id=owner_filter,
                cursor_created_at=scan_created_at,
                cursor_run_id=scan_run_id,
            )
            if not rows:
                break

            for r in rows:
                run_id = str(r.get("artifact_id") or "")
                full_ref = str(r.get("full_ref") or "")
                rel_ref = str(r.get("rel_ref") or "")
                row_created_at = str(r.get("created_at") or "")
                if not run_id or not full_ref or not row_created_at:
                    continue
                try:
                    obj = read_artifact_json(full_ref)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                if str(obj.get("task_id") or "").strip() != task_id_filter:
                    continue

                timings_ms = obj.get("timings_ms")
                total_time = timings_ms.get("total") if isinstance(timings_ms, dict) else None
                eval_summary = _build_eval_summary(obj.get("eval"))
                items.append(
                    RunSummary(
                        run_id=run_id,
                        created_at=obj.get("created_at"),
                        task_id=obj.get("task_id"),
                        model_id=obj.get("model_id"),
                        prompt_sha256=obj.get("prompt_sha256"),
                        schema_sha256=obj.get("schema_sha256"),
                        temperature=obj.get("temperature"),
                        max_tokens=obj.get("max_tokens"),
                        code_version=obj.get("code_version"),
                        item_count=obj.get("item_count"),
                        ok_count=obj.get("ok_count"),
                        error_count=obj.get("error_count"),
                        artifact_rel=rel_ref or None,
                        total_time=total_time,
                        eval_summary=eval_summary,
                    )
                )
                matched_positions.append((row_created_at, run_id))
                if len(items) >= limit + 1:
                    break

            if len(items) >= limit + 1 or next_cur is None:
                break
            scan_created_at, scan_run_id = next_cur

        if len(items) > limit:
            page_positions = matched_positions[:limit]
            items = items[:limit]
            if page_positions:
                next_cursor = _encode_cursor(page_positions[-1][0], page_positions[-1][1])

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


@router.delete(
    "/{run_id}",
    response_model=DeleteRunResponse,
    responses=_ERR,
)
async def delete_run(
    run_id: str,
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes(["runs:delete"])),
) -> DeleteRunResponse:
    if (principal.role or "").strip().lower() != "admin":
        raise HTTPException(status_code=403, detail="Only admin may delete runs.")

    batch_ref = find_batch_artifact_path(run_id, date=None, max_days=3650)
    if batch_ref is None:
        raise HTTPException(status_code=404, detail="Run not found.")

    batch_obj = read_artifact_json(batch_ref)
    if not isinstance(batch_obj, dict):
        raise HTTPException(status_code=500, detail="Invalid run JSON.")

    auth = batch_obj.get("auth") if isinstance(batch_obj.get("auth"), dict) else {}
    owner_key_id = auth.get("key_id") if isinstance(auth, dict) else None
    _ensure_owner_or_admin(principal, str(owner_key_id) if owner_key_id else None)

    jobs_store = _jobs_store(request)
    related_jobs = jobs_store.list_related_jobs(run_id)
    active_jobs = [job.job_id for job in related_jobs if job.status in {"queued", "running"}]
    if active_jobs:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete run while related jobs are active: {', '.join(active_jobs[:10])}",
        )

    deleted_job_results = 0
    deleted_job_errors = 0
    deleted_jobs = 0
    for job in related_jobs:
        if delete_indexed_artifact(kind="job_result", artifact_id=job.job_id, ref=job.result_ref):
            deleted_job_results += 1
        if delete_indexed_artifact(kind="job_error", artifact_id=job.job_id, ref=job.error_ref):
            deleted_job_errors += 1
        if jobs_store.delete_job(job.job_id):
            deleted_jobs += 1

    summary = delete_run_artifacts(run_id, max_days=3650)
    return DeleteRunResponse(
        ok=True,
        run_id=run_id,
        deleted_extracts=int(summary.get("deleted_extracts") or 0),
        deleted_evals=int(summary.get("deleted_evals") or 0),
        deleted_job_results=deleted_job_results,
        deleted_job_errors=deleted_job_errors,
        deleted_jobs=deleted_jobs,
    )

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
