# app/routers/extract.py
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, Request, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import ApiPrincipal, require_scopes
from app.handlers import get_vllm_client
from app.vllm_client import VLLMClient
from app.services import extract_service
from app.services.jobs_store import JobsStore
from app.services.jobs_runner import BaseJobRunner


router = APIRouter(prefix="/v1", tags=["extract"])


# ---- helpers for async jobs ----

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


class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    poll_url: str = Field(description="Use this URL to poll job status/result.")


# ---- sync endpoints (as before) ----

@router.post("/extract", response_model=extract_service.ExtractResponse)
async def extract(
    req: extract_service.ExtractRequest,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
    vllm_client: VLLMClient = Depends(get_vllm_client),
) -> extract_service.ExtractResponse:
    return await extract_service.extract(req, principal=principal, vllm_client=vllm_client)


@router.post("/batch_extract", response_model=extract_service.BatchExtractResponse)
async def batch_extract(
    req: extract_service.BatchExtractRequest,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
    vllm_client: VLLMClient = Depends(get_vllm_client),
) -> extract_service.BatchExtractResponse:
    return await extract_service.batch_extract(req, principal=principal, vllm_client=vllm_client)


@router.post("/batch_extract_upload", response_model=extract_service.BatchExtractResponse)
async def batch_extract_upload(
    task_id: str = Form(...),
    persist_inputs: bool = Form(False),
    concurrency: int = Form(4),
    run_id: Optional[str] = Form(None),
    gt_path: Optional[str] = Form(None),
    gt_image_key: str = Form("image"),
    gt_record_key: Optional[str] = Form(None),
    include_worst: bool = Form(False),
    limit: Optional[int] = Form(None),
    files: List[UploadFile] = File(...),
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
    vllm_client: VLLMClient = Depends(get_vllm_client),
) -> extract_service.BatchExtractResponse:
    return await extract_service.batch_extract_upload(
        task_id=task_id,
        persist_inputs=persist_inputs,
        concurrency=concurrency,
        run_id=run_id,
        files=files,
        gt_path=gt_path,
        gt_image_key=gt_image_key,
        gt_record_key=gt_record_key,
        include_worst=include_worst,
        limit=limit,
        principal=principal,
        vllm_client=vllm_client,
    )


@router.post("/batch_extract_rerun", response_model=extract_service.BatchExtractResponse)
async def batch_extract_rerun(
    req: extract_service.BatchRerunRequest,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
    vllm_client: VLLMClient = Depends(get_vllm_client),
) -> extract_service.BatchExtractResponse:
    return await extract_service.batch_extract_rerun(req, principal=principal, vllm_client=vllm_client)


# ---- async endpoints via local job store ----
# Result polling: GET /v1/jobs/{job_id} (make sure jobs router is included in main.py)

@router.post(
    "/extract_async",
    response_model=JobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def extract_async(
    req: extract_service.ExtractRequest,
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
) -> JobCreateResponse:
    store = _store(request)
    runner = _runner(request)

    job = store.create_job(kind="extract", request=req.model_dump(), principal=principal)
    await runner.enqueue(job.job_id)
    return JobCreateResponse(job_id=job.job_id, status=job.status, poll_url=f"/v1/jobs/{job.job_id}")


@router.post(
    "/batch_extract_async",
    response_model=JobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def batch_extract_async(
    req: extract_service.BatchExtractRequest,
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
) -> JobCreateResponse:
    store = _store(request)
    runner = _runner(request)

    # Keep legacy kind name used by jobs router
    job = store.create_job(kind="batch_extract_dir", request=req.model_dump(), principal=principal)
    await runner.enqueue(job.job_id)
    return JobCreateResponse(job_id=job.job_id, status=job.status, poll_url=f"/v1/jobs/{job.job_id}")


@router.post(
    "/batch_extract_rerun_async",
    response_model=JobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def batch_extract_rerun_async(
    req: extract_service.BatchRerunRequest,
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
) -> JobCreateResponse:
    store = _store(request)
    runner = _runner(request)

    job = store.create_job(kind="batch_extract_rerun", request=req.model_dump(), principal=principal)
    await runner.enqueue(job.job_id)
    return JobCreateResponse(job_id=job.job_id, status=job.status, poll_url=f"/v1/jobs/{job.job_id}")
