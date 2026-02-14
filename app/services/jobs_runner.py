# app/services/jobs_runner.py
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.services.jobs_store import JobsStore
from app.services.jobs_executor import execute_job
from app.settings import settings

log = logging.getLogger("ocrlty.jobs")


class BaseJobRunner:
    async def enqueue(self, job_id: str) -> None:
        raise NotImplementedError


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

            # Optional: minimal progress marker
            self._store.set_progress(job_id, {"stage": "started"})

            result = await execute_job(kind=job.kind, request=job.request, owner=owner)

            # If cancel was requested during run: mark canceled (best-effort)
            job2 = self._store.get_job(job_id)
            if job2 and job2.cancel_requested:
                self._store.mark_canceled(job_id)
                return

            self._store.mark_succeeded(job_id, result)

        except Exception as e:
            log.exception("job_failed job_id=%s", job_id)
            self._store.mark_failed(job_id, {"type": type(e).__name__, "message": str(e)})
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
