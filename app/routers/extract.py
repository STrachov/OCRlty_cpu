# app/routers/extract.py
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.auth import ApiPrincipal, require_scopes
from app.handlers import get_vllm_client
from app.vllm_client import VLLMClient
from app.services import extract_service


router = APIRouter(prefix="/v1", tags=["extract"])


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
