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
    global _http, _vllm
    if _vllm is not None:
        return _vllm

    timeout = httpx.Timeout(
        connect=float(getattr(settings, "VLLM_CONNECT_TIMEOUT", 5.0) or 5.0),
        read=float(getattr(settings, "VLLM_READ_TIMEOUT", 60.0) or 60.0),
        write=float(getattr(settings, "VLLM_WRITE_TIMEOUT", 60.0) or 60.0),
        pool=float(getattr(settings, "VLLM_POOL_TIMEOUT", 5.0) or 5.0),
    )
    _http = httpx.AsyncClient(timeout=timeout)
    _vllm = VLLMClient(
        _http,
        base_url=str(settings.VLLM_BASE_URL),
        api_key=str(getattr(settings, "VLLM_API_KEY", "") or ""),
    )
    return _vllm


def _principal_from_owner(owner: Optional[Dict[str, Any]]) -> ApiPrincipal:
    if not owner:
        # best-effort fallback
        return ApiPrincipal(api_key_id=-1, key_id="system", role="system", scopes={"extract:run"}, is_active=True)

    scopes = owner.get("scopes") or []
    if isinstance(scopes, str):
        scopes = [scopes]
    return ApiPrincipal(
        api_key_id=int(owner.get("api_key_id") or -1),
        key_id=str(owner.get("key_id") or "unknown"),
        role=str(owner.get("role") or "user"),
        scopes=set(scopes),
        is_active=True,
    )


def _map_batch_extract_dir_legacy(req: Dict[str, Any]) -> Dict[str, Any]:
    # jobs router uses BatchExtractDirJobRequest {images_dir, glob, exts, limit, task_id, concurrency, run_id,...}
    return {
        "task_id": req.get("task_id"),
        "images_dir": req.get("images_dir"),
        "glob": req.get("glob", "**/*"),
        "concurrency": req.get("concurrency", 4),
        "run_id": req.get("run_id"),
        "gt_path": req.get("gt_path"),
        "gt_image_key": req.get("gt_image_key", "image"),
        "gt_record_key": req.get("gt_record_key"),
        "include_worst": bool(req.get("include_worst", False)),
        "limit": req.get("limit"),
    }


def _map_batch_rerun_legacy(req: Dict[str, Any]) -> Dict[str, Any]:
    # jobs router uses {source_run_id, source_batch_date, max_days, task_id, concurrency, run_id,...}
    return {
        "task_id": req.get("task_id"),
        "run_id": req.get("source_run_id") or req.get("run_id"),
        "batch_date": req.get("source_batch_date") or req.get("batch_date"),
        "max_days": req.get("max_days", 90),
        "concurrency": req.get("concurrency", 4),
        "new_run_id": req.get("run_id") or req.get("new_run_id"),
        "gt_path": req.get("gt_path"),
        "gt_image_key": req.get("gt_image_key", "image"),
        "gt_record_key": req.get("gt_record_key"),
        "include_worst": bool(req.get("include_worst", False)),
        "limit": req.get("limit"),
    }


async def execute_job(kind: str, request: Dict[str, Any], owner: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute a job in-process.

    Kinds supported:
      - extract
      - batch_extract
      - batch_extract_inputs   (refs-only upload)
      - batch_extract_rerun
      - batch_extract_dir      (legacy compatibility with /v1/jobs router)
    """
    setup_logging()
    vllm = _get_vllm_client()
    principal = _principal_from_owner(owner)

    if kind == "extract":
        req = extract_service.ExtractRequest(**(request or {}))
        resp = await extract_service.extract(req, principal=principal, vllm_client=vllm)
        return resp.model_dump()

    if kind == "batch_extract":
        req = extract_service.BatchExtractRequest(**(request or {}))
        resp = await extract_service.batch_extract(req, principal=principal, vllm_client=vllm)
        return resp.model_dump()

    if kind == "batch_extract_inputs":
        req = extract_service.BatchExtractInputsRequest(**(request or {}))
        resp = await extract_service.batch_extract_inputs(req, principal=principal, vllm_client=vllm)
        return resp.model_dump()

    if kind == "batch_extract_dir":
        mapped = _map_batch_extract_dir_legacy(request or {})
        req = extract_service.BatchExtractRequest(**mapped)
        resp = await extract_service.batch_extract(req, principal=principal, vllm_client=vllm)
        return resp.model_dump()

    if kind == "batch_extract_rerun":
        # support both new and legacy shape
        mapped = _map_batch_rerun_legacy(request or {}) if ("source_run_id" in (request or {})) else (request or {})
        req = extract_service.BatchRerunRequest(**mapped)
        resp = await extract_service.batch_extract_rerun(req, principal=principal, vllm_client=vllm)
        return resp.model_dump()

    raise ValueError(f"Unknown job kind: {kind!r}")
