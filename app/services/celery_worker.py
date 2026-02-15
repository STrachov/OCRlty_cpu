# app/services/celery_worker.py
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import traceback
from typing import Any, Dict

from celery import Celery

from app.settings import settings
from app.services.artifacts import save_job_error_artifact, save_job_result_artifact
from app.services.jobs_executor import execute_job
from app.services.jobs_store import JobsStore

log = logging.getLogger("ocrlty.celery")

celery_app = Celery(
    "ocrlty",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND or settings.CELERY_BROKER_URL,
)

_store = JobsStore(settings.JOBS_DB_PATH)


def _thin_result_meta(kind: str, result: Dict[str, Any]) -> Dict[str, Any]:
    k = (kind or "").lower()
    if k == "extract":
        return {
            "type": "extract",
            "task_id": result.get("task_id"),
            "request_id": result.get("request_id"),
            "artifact_path": result.get("artifact_path"),
            "schema_valid": result.get("schema_valid"),
            "model_id": result.get("model_id"),
            "prompt_sha256": result.get("prompt_sha256"),
            "schema_sha256": result.get("schema_sha256"),
        }
    if k.startswith("batch_extract"):
        meta = {
            "type": "batch",
            "task_id": result.get("task_id"),
            "run_id": result.get("run_id"),
            "artifact_path": result.get("artifact_path"),
            "concurrency": result.get("concurrency"),
            "item_count": result.get("item_count"),
            "ok_count": result.get("ok_count"),
            "error_count": result.get("error_count"),
            "images_dir": result.get("images_dir"),
            "glob": result.get("glob"),
        }
        if isinstance(result.get("eval"), dict):
            meta["eval_artifact_path"] = result["eval"].get("eval_artifact_path") or result["eval"].get("artifact_path")
        elif result.get("eval_artifact_path"):
            meta["eval_artifact_path"] = result.get("eval_artifact_path")
        return meta
    return {"type": kind, "keys": list(result.keys())[:20]}


@celery_app.task(name="ocrlty.execute_job")
def execute_job_task(job_id: str) -> Dict[str, Any]:
    """Execute one job by id in celery worker process."""
    _store.ensure_init()

    job = _store.get_job(job_id)
    if job is None:
        return {"ok": False, "error": "Job not found"}

    try:
        # queued-cancel: instant
        if job.cancel_requested and job.status == "queued":
            _store.mark_canceled(job_id)
            return {"ok": True, "status": "canceled"}

        _store.mark_running(job_id)

        job = _store.get_job(job_id)
        if job is None:
            return {"ok": False, "error": "Job disappeared"}

        owner = {
            "api_key_id": 0,
            "key_id": job.owner_key_id,
            "role": job.owner_role,
            "scopes": job.owner_scopes,
        }

        _store.set_progress(job_id, {"stage": "started"})

        # execute_job is async; run it in event loop.
        result = asyncio.run(execute_job(kind=job.kind, request=job.request, owner=owner))
        if not isinstance(result, dict):
            result = {"value": result}

        job2 = _store.get_job(job_id)
        if job2 and job2.cancel_requested:
            _store.mark_canceled(job_id)
            return {"ok": True, "status": "canceled"}

        result_ref = save_job_result_artifact(job_id, result, owner_key_id=job.owner_key_id)
        meta = _thin_result_meta(job.kind, result)
        body = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        sha = hashlib.sha256(body).hexdigest()

        _store.mark_succeeded(job_id, result_ref=str(result_ref), result_meta=meta, result_bytes=len(body), result_sha256=sha)
        return {"ok": True, "status": "succeeded"}

    except Exception as e:
        log.exception("celery job failed job_id=%s", job_id)
        tb = traceback.format_exc(limit=200)
        err = {"type": type(e).__name__, "message": str(e)}
        error_payload = {"job_id": job_id, "error": err, "traceback": tb}
        try:
            error_ref = save_job_error_artifact(job_id, error_payload, owner_key_id=job.owner_key_id)
        except Exception:
            error_ref = None
        _store.mark_failed(job_id, err, error_ref=str(error_ref) if error_ref else None)
        return {"ok": False, "status": "failed", "error": err}
