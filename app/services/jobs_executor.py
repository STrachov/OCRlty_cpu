# app/services/jobs_executor.py
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Set

import httpx

from app.auth_store import ApiPrincipal
from app.settings import settings
from app.vllm_client import VLLMClient
from app.handlers import setup_logging
from app.services import extract_service

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
    _vllm = VLLMClient(_http, base_url=str(settings.VLLM_BASE_URL), api_key=str(getattr(settings, "VLLM_API_KEY", "") or ""))
    return _vllm


def _system_principal(extra_scopes: Optional[Set[str]] = None) -> ApiPrincipal:
    scopes: Set[str] = {"extract:run"}
    if extra_scopes:
        scopes |= set(extra_scopes)
    # In internal jobs we act as system (no API key id).
    return ApiPrincipal(api_key_id=-1, key_id="system", role="system", scopes=scopes, is_active=True)


async def execute_job(kind: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a job in-process.

    Supported kinds:
      - extract
      - batch_extract
      - batch_extract_rerun
    """
    setup_logging()
    vllm = _get_vllm_client()

    # Allow jobs to run eval only if DEBUG_MODE is enabled (extract_service enforces it).
    principal = _system_principal(extra_scopes={"debug:run", "debug:raw"})

    if kind == "extract":
        req = extract_service.ExtractRequest(**params)
        resp = await extract_service.extract(req, principal=principal, vllm_client=vllm)
        return resp.model_dump()

    if kind == "batch_extract":
        req = extract_service.BatchExtractRequest(**params)
        resp = await extract_service.batch_extract(req, principal=principal, vllm_client=vllm)
        return resp.model_dump()

    if kind == "batch_extract_rerun":
        req = extract_service.BatchRerunRequest(**params)
        resp = await extract_service.batch_extract_rerun(req, principal=principal, vllm_client=vllm)
        return resp.model_dump()

    raise ValueError(f"Unknown job kind: {kind!r}")
