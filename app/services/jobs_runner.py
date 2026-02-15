# app/services/jobs_runner.py
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import traceback
from typing import Any, Dict, Optional

from app.services.artifacts import save_job_error_artifact, save_job_result_artifact
from app.services.jobs_executor import execute_job
from app.services.jobs_store import JobsStore

log = logging.getLogger("ocrlty.jobs")


class BaseJobRunner:
    async def enqueue(self, job_id: str) -> None:
        raise NotImplementedError


def _thin_result_meta(kind: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Build a *small* summary for JobsStore; full result lives in artifacts."""
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
        # Covers: batch_extract, batch_extract_inputs, batch_extract_dir, batch_extract_rerun
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
        # Optional eval pointer(s) if present
        if isinstance(result.get("eval"), dict):
            meta["eval_artifact_path"] = result["eval"].get("eval_artifact_path") or result["eval"].get("artifact_path")
        elif result.get("eval_artifact_path"):
            meta["eval_artifact_path"] = result.get("eval_artifact_path")
        return meta

    # Default
    return {"type": kind, "keys": list(result.keys())[:20]}


class LocalJobRunner(BaseJobRunner):
    """
    In-process background runner:
    - persists jobs in SQLite
    - executes them in same FastAPI process with concurrency cap
    """

    def __init__(self, store: JobsStore, max_concurrency: int) -> None:
        self._store = store
        self._q: asyncio.Queue[str] = asyncio.Queue()
        self._sem = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._main_task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._main_task is None:
            self._main_task = asyncio.create_task(self._loop())

    async def shutdown(self) -> None:
        self._stop.set()
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except Exception:
                pass

    async def enqueue(self, job_id: str) -> None:
        await self._q.put(job_id)

    async def _loop(self) -> None:
        while not self._stop.is_set():
            job_id = await self._q.get()
            await self._sem.acquire()
            asyncio.create_task(self._run_one(job_id))

    async def _run_one(self, job_id: str) -> None:
        try:
            job = self._store.get_job(job_id)
            if job is None:
                return

            # queued-cancel: instant
            if job.cancel_requested and job.status == "queued":
                self._store.mark_canceled(job_id)
                return

            self._store.mark_running(job_id)

            job = self._store.get_job(job_id)
            if job is None:
                return

            owner = {
                "api_key_id": 0,  # not used by handlers, but keep shape
                "key_id": job.owner_key_id,
                "role": job.owner_role,
                "scopes": job.owner_scopes,
            }

            # Minimal progress marker
            self._store.set_progress(job_id, {"stage": "started"})

            result = await execute_job(kind=job.kind, request=job.request, owner=owner)
            if not isinstance(result, dict):
                result = {"value": result}

            # If cancel was requested during run: mark canceled (best-effort)
            job2 = self._store.get_job(job_id)
            if job2 and job2.cancel_requested:
                self._store.mark_canceled(job_id)
                return

            # Persist full ("thick") JSON to artifacts, store only pointer + meta in DB.
            result_ref = save_job_result_artifact(job_id, result, owner_key_id=job.owner_key_id)
            meta = _thin_result_meta(job.kind, result)

            # Deterministic sha/bytes (same formatting used for artifacts).
            body = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            sha = hashlib.sha256(body).hexdigest()
            self._store.mark_succeeded(job_id, result_ref=str(result_ref), result_meta=meta, result_bytes=len(body), result_sha256=sha)

        except Exception as e:
            log.exception("job_failed job_id=%s", job_id)
            tb = traceback.format_exc(limit=200)
            err = {"type": type(e).__name__, "message": str(e)}
            error_payload = {"job_id": job_id, "error": err, "traceback": tb}

            # Store full error details in artifacts, keep only small summary in DB.
            try:
                job = self._store.get_job(job_id)
                owner_key_id = job.owner_key_id if job else None
            except Exception:
                owner_key_id = None

            try:
                error_ref = save_job_error_artifact(job_id, error_payload, owner_key_id=owner_key_id)
            except Exception:
                error_ref = None

            self._store.mark_failed(job_id, err, error_ref=str(error_ref) if error_ref else None)

        finally:
            try:
                self._sem.release()
            except Exception:
                pass


class CeleryJobRunner(BaseJobRunner):
    """
    Placeholder runner: enqueue to Celery queue.
    Requires celery+redis installed and a worker running.
    """

    def __init__(self) -> None:
        pass

    async def enqueue(self, job_id: str) -> None:
        try:
            from app.services.celery_worker import celery_app  # lazy import
        except Exception as e:
            raise RuntimeError(
                "Celery backend selected but celery_worker import failed. "
                "Install celery/redis deps and add worker module."
            ) from e
        celery_app.send_task("ocrlty.execute_job", args=[job_id])
