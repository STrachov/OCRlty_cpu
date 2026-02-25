# app/routers/jobs.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from app.auth import require_scopes
from app.auth_store import ApiPrincipal
from app.schemas.api_error import common_error_responses
from app.services.jobs_store import JobsStore, JobRecord
from app.services.jobs_runner import BaseJobRunner
from app.services.artifacts import read_artifact_json

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])
_ERR = common_error_responses(400, 401, 403, 404, 422, 500)


def _store(request: Request) -> JobsStore:
    s = getattr(request.app.state, "jobs_store", None)
    if s is None:
        raise HTTPException(status_code=500, detail="Jobs store is not initialized.")
    return s


def _runner(request: Request) -> BaseJobRunner:
    r = getattr(request.app.state, "jobs_runner", None)
    if r is None:
        raise HTTPException(status_code=500, detail="Jobs runner is not initialized.")
    return r


def _job_to_dict(j: JobRecord) -> Dict[str, Any]:
    # NOTE: "result" is intentionally *thin* (summary only). Full result JSON lives in artifacts.
    return {
        "job_id": j.job_id,
        "kind": j.kind,
        "status": j.status,
        "created_at": j.created_at,
        "updated_at": j.updated_at,
        "started_at": j.started_at,
        "finished_at": j.finished_at,
        "owner_key_id": j.owner_key_id,
        "owner_role": j.owner_role,
        "cancel_requested": bool(j.cancel_requested),

        "request": j.request,
        "progress": j.progress,

        "result_ref": j.result_ref,
        "result_meta": j.result_meta,
        "result_bytes": j.result_bytes,
        "result_sha256": j.result_sha256,

        # Backward-compatible alias:
        "result": j.result_meta,

        "error_ref": j.error_ref,
        "error": j.error,
    }


def _enforce_owner_or_admin(principal: ApiPrincipal, job: JobRecord) -> None:
    if principal.role == "admin":
        return
    if not job.owner_key_id or job.owner_key_id != principal.key_id:
        raise HTTPException(status_code=403, detail="You may only access your own jobs.")


# ---- Request models (JSON) ----

class BatchExtractDirJobRequest(BaseModel):
    images_dir: str = Field(default="/workspace/src/data/cord_subset/images")
    glob: str = Field(default="**/*")
    exts: List[str] = Field(default_factory=lambda: ["png", "jpg", "jpeg", "webp"])
    limit: Optional[int] = Field(default=None, ge=1)

    task_id: str = Field(default="receipt_fields_v1")
    concurrency: int = Field(default=4, ge=1, le=32)
    run_id: Optional[str] = Field(default=None)

    gt_path: Optional[str] = Field(default=None)
    gt_image_key: str = Field(default="file")


class BatchRerunJobRequest(BaseModel):
    source_run_id: str
    source_batch_date: Optional[str] = None
    max_days: int = Field(default=90, ge=1, le=365)

    task_id: str = Field(default="receipt_fields_v1")
    concurrency: int = Field(default=4, ge=1, le=32)
    run_id: Optional[str] = None

    gt_path: Optional[str] = None
    gt_image_key: str = Field(default="file")


class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    poll_url: str


class JobView(BaseModel):
    job_id: str
    kind: str
    status: str
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    owner_key_id: Optional[str] = None
    owner_role: Optional[str] = None
    cancel_requested: bool

    request: Dict[str, Any] = Field(default_factory=dict)
    progress: Optional[Dict[str, Any]] = None

    result_ref: Optional[str] = None
    result_meta: Optional[Dict[str, Any]] = None
    result_bytes: Optional[int] = None
    result_sha256: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    error_ref: Optional[str] = None
    error: Optional[Dict[str, Any]] = None


class JobListResponse(BaseModel):
    items: List[JobView]


class JobCancelResponse(BaseModel):
    ok: bool


# ---- Endpoints ----

@router.post(
    "/batch_extract_dir",
    response_model=JobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=_ERR,
    dependencies=[Depends(require_scopes(["extract:run"]))],
)
async def enqueue_batch_extract_dir(
    req: BatchExtractDirJobRequest,
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
) -> JobCreateResponse:
    store = _store(request)
    runner = _runner(request)

    job = store.create_job(
        kind="batch_extract_dir",
        request=req.model_dump(),
        principal=principal,
    )
    await runner.enqueue(job.job_id)
    return JobCreateResponse(job_id=job.job_id, status=job.status, poll_url=f"/v1/jobs/{job.job_id}")


@router.post(
    "/batch_extract_rerun",
    response_model=JobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=_ERR,
    dependencies=[Depends(require_scopes(["extract:run"]))],
)
async def enqueue_batch_extract_rerun(
    req: BatchRerunJobRequest,
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
) -> JobCreateResponse:
    store = _store(request)
    runner = _runner(request)

    job = store.create_job(
        kind="batch_extract_rerun",
        request=req.model_dump(),
        principal=principal,
    )
    await runner.enqueue(job.job_id)
    return JobCreateResponse(job_id=job.job_id, status=job.status, poll_url=f"/v1/jobs/{job.job_id}")


@router.get(
    "",
    response_model=JobListResponse,
    responses=_ERR,
    dependencies=[Depends(require_scopes(["extract:run"]))],
)
async def list_jobs(
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
    status_: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    store = _store(request)

    owner_key_id = None if principal.role == "admin" else principal.key_id
    jobs = store.list_jobs(owner_key_id=owner_key_id, status=status_, limit=limit, offset=offset)
    return JobListResponse(items=[JobView(**_job_to_dict(j)) for j in jobs])


@router.get(
    "/{job_id}",
    response_model=JobView,
    responses=_ERR,
    dependencies=[Depends(require_scopes(["extract:run"]))],
)
async def get_job(
    job_id: str,
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
):
    store = _store(request)
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    _enforce_owner_or_admin(principal, job)
    return JobView(**_job_to_dict(job))


@router.get(
    "/{job_id}/result",
    response_model=Dict[str, Any],
    responses=_ERR,
    dependencies=[Depends(require_scopes(["extract:run"]))],
)
async def get_job_result(
    job_id: str,
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
):
    store = _store(request)
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    _enforce_owner_or_admin(principal, job)
    if not job.result_ref:
        raise HTTPException(status_code=404, detail="Job result is not available.")
    return read_artifact_json(job.result_ref)


@router.get(
    "/{job_id}/error",
    response_model=Dict[str, Any],
    responses=_ERR,
    dependencies=[Depends(require_scopes(["extract:run"]))],
)
async def get_job_error_details(
    job_id: str,
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
):
    store = _store(request)
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    _enforce_owner_or_admin(principal, job)
    if not job.error_ref:
        raise HTTPException(status_code=404, detail="Job error details are not available.")
    return read_artifact_json(job.error_ref)


@router.post(
    "/{job_id}/cancel",
    response_model=JobCancelResponse,
    responses=_ERR,
    dependencies=[Depends(require_scopes(["extract:run"]))],
)
async def cancel_job(
    job_id: str,
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
):
    store = _store(request)
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    _enforce_owner_or_admin(principal, job)

    # Best-effort cancel (queued -> canceled immediately; running -> mark cancel_requested)
    store.request_cancel(job_id)
    job2 = store.get_job(job_id)
    if job2 and job2.status == "queued":
        store.mark_canceled(job_id)

    return JobCancelResponse(ok=True)
