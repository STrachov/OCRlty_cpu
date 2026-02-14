# app/services/celery_worker.py
from __future__ import annotations

import asyncio
import logging

from app.settings import settings
from app.services.jobs_store import JobsStore
from app.services.jobs_executor import execute_job

# Celery import guarded
from celery import Celery

log = logging.getLogger("ocrlty.celery")

celery_app = Celery(
    "ocrlty",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND or settings.CELERY_BROKER_URL,
)

_store = JobsStore(settings.JOBS_DB_PATH)


@celery_app.task(name="ocrlty.execute_job")
def execute_job_task(job_id: str) -> None:
    asyncio.run(_run(job_id))


async def _run(job_id: str) -> None:
    _store.ensure_init()
    job = _store.get_job(job_id)
    if job is None:
        return

    if job.cancel_requested and job.status == "queued":
        _store.mark_canceled(job_id)
        return

    _store.mark_running(job_id)

    job = _store.get_job(job_id)
    if job is None:
        return

    owner = {
        "api_key_id": 0,
        "key_id": job.owner_key_id,
        "role": job.owner_role,
        "scopes": job.owner_scopes,
    }

    try:
        _store.set_progress(job_id, {"stage": "started"})
        result = await execute_job(kind=job.kind, request=job.request, owner=owner)

        job2 = _store.get_job(job_id)
        if job2 and job2.cancel_requested:
            _store.mark_canceled(job_id)
            return

        _store.mark_succeeded(job_id, result)
    except Exception as e:
        log.exception("celery job failed job_id=%s", job_id)
        _store.mark_failed(job_id, {"type": type(e).__name__, "message": str(e)})
