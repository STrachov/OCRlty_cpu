# app/services/jobs_executor.py
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from app.auth_store import ApiPrincipal
from app.handlers import setup_logging
from app.settings import settings
from app.services import extract_service
from app.vllm_client import VLLMClient

LOG = logging.getLogger("ocrlty.jobs")

_http: Optional[httpx.AsyncClient] = None
_vllm: Optional[VLLMClient] = None


def _get_vllm_client() -> VLLMClient:
    """Create a vLLM client for background jobs.

    Note: FastAPI request-state client isn't available here; we keep our own singleton.
    """
    global _http, _vllm
    if _vllm is not None:
        return _vllm

    timeout = httpx.Timeout(
        connect=float(getattr(settings, "VLLM_CONNECT_TIMEOUT_S", 10.0) or 10.0),
        read=float(getattr(settings, "VLLM_READ_TIMEOUT_S", 180.0) or 180.0),
        write=float(getattr(settings, "VLLM_WRITE_TIMEOUT_S", 30.0) or 30.0),
        pool=float(getattr(settings, "VLLM_POOL_TIMEOUT_S", 5.0) or 5.0),
    )
    _http = httpx.AsyncClient(timeout=timeout, headers={"Accept": "application/json", "User-Agent": "ocrlty-jobs/1.0"})
    _vllm = VLLMClient(_http, base_url=str(settings.VLLM_BASE_URL), api_key=str(getattr(settings, "VLLM_API_KEY", "") or ""))
    return _vllm


def _mk_principal(owner: Dict[str, Any]) -> ApiPrincipal:
    return ApiPrincipal(
        api_key_id=int(owner.get("api_key_id") or 0),
        key_id=str(owner.get("key_id") or "anonymous"),
        role=str(owner.get("role") or "anonymous"),
        scopes=set(owner.get("scopes") or []),
    )


async def execute_job(*, kind: str, request: Dict[str, Any], owner: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a job in-process (LocalJobRunner) or in celery worker.

    The runner is responsible for marking job status in JobsStore.
    This function must return a JSON-serializable dict.

    Supported kinds (current + legacy):
      - extract
      - batch_extract_dir   (legacy name; maps to extract_service.batch_extract)
      - batch_extract       (alias)
      - batch_extract_rerun (legacy name; maps to extract_service.batch_extract_rerun)
    """
    setup_logging()
    vllm = _get_vllm_client()
    principal = _mk_principal(owner)

    if kind == "extract":
        req = extract_service.ExtractRequest(**request)
        resp = await extract_service.extract(req, principal=principal, vllm_client=vllm)
        return resp.model_dump()

    if kind in {"batch_extract_dir", "batch_extract"}:
        req = extract_service.BatchExtractRequest(**request)
        resp = await extract_service.batch_extract(req, principal=principal, vllm_client=vllm)
        return resp.model_dump()

    if kind == "batch_extract_rerun":
        # Accept both shapes:
        # - new: BatchRerunRequest fields (run_id, batch_date, new_run_id, ...)
        # - old: jobs router fields (source_run_id, source_batch_date, run_id as new id)
        if "source_run_id" in request:
            mapped = {
                "task_id": request.get("task_id") or request.get("task_id", "receipt_fields_v1"),
                "run_id": request.get("source_run_id"),
                "batch_date": request.get("source_batch_date"),
                "max_days": request.get("max_days", 30),
                "concurrency": request.get("concurrency", 4),
                "new_run_id": request.get("run_id"),
                "gt_path": request.get("gt_path"),
                "gt_image_key": request.get("gt_image_key", "image"),
                "gt_record_key": request.get("gt_record_key"),
                "include_worst": bool(request.get("include_worst", False)),
                "limit": request.get("limit"),
            }
            req = extract_service.BatchRerunRequest(**mapped)
        else:
            req = extract_service.BatchRerunRequest(**request)

        resp = await extract_service.batch_extract_rerun(req, principal=principal, vllm_client=vllm)
        return resp.model_dump()

    raise ValueError(f"Unknown job kind: {kind!r}")
