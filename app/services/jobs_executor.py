# app/services/jobs_executor.py
from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException

from app import handlers
from app.auth_store import ApiPrincipal


def _mk_principal(owner: Dict[str, Any]) -> ApiPrincipal:
    return ApiPrincipal(
        api_key_id=int(owner.get("api_key_id") or 0),
        key_id=str(owner.get("key_id") or "anonymous"),
        role=str(owner.get("role") or "anonymous"),
        scopes=set(owner.get("scopes") or []),
    )


async def execute_job(*, kind: str, request: Dict[str, Any], owner: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns JSON-serializable dict to store into result_json.
    Raises exceptions => runner will mark job failed.
    """
    principal = _mk_principal(owner)

    if kind == "batch_extract_dir":
        # JSON body -> call handler (Form-params function) directly by kwargs
        resp = await handlers.batch_extract(
            task_id=request.get("task_id", "receipt_fields_v1"),
            images_dir=request.get("images_dir", handlers.DEFAULT_IMAGES_DIR),
            glob=request.get("glob", "**/*"),
            exts=request.get("exts", ["png", "jpg", "jpeg", "webp"]),
            limit=request.get("limit"),
            concurrency=int(request.get("concurrency", 4)),
            run_id=request.get("run_id"),
            gt_path=request.get("gt_path"),
            gt_image_key=request.get("gt_image_key", "file"),
            principal=principal,
        )
        return resp.model_dump()

    if kind == "batch_extract_rerun":
        resp = await handlers.batch_extract_rerun(
            source_run_id=str(request.get("source_run_id") or ""),
            source_batch_date=request.get("source_batch_date"),
            max_days=int(request.get("max_days", 90)),
            task_id=request.get("task_id", "receipt_fields_v1"),
            concurrency=int(request.get("concurrency", 4)),
            run_id=request.get("run_id"),
            gt_path=request.get("gt_path"),
            gt_image_key=request.get("gt_image_key", "file"),
            principal=principal,
        )
        return resp.model_dump()

    raise HTTPException(status_code=400, detail=f"Unknown job kind: {kind!r}")
