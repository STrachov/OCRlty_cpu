from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.auth import init_auth_db
from app.settings import settings
from app.routers.core import router as core_router
from app.routers.extract import router as extract_router
from app.routers.debug import router as debug_router
from app.handlers import make_request_id, REQUEST_ID_CTX, setup_logging, load_tasks
from app.services.artifacts import artifact_index_init

import httpx
from app.vllm_client import VLLMClient

from app.routers.jobs import router as jobs_router
from app.services.jobs_store import JobsStore
from app.services.jobs_runner import LocalJobRunner, CeleryJobRunner

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- logging (best-effort) ---
    try:
        setup_logging()
    except Exception:
        logging.getLogger("ocrlty").exception("setup_logging_failed")

    http: httpx.AsyncClient | None = None
    runner = None

    try:
        # --- init DB/index/tasks first (so background jobs won't start before tasks exist) ---
        try:
            init_auth_db()
        except Exception:
            logging.getLogger("ocrlty").exception("auth_db_init_failed")

        try:
            artifact_index_init()
        except Exception:
            logging.getLogger("ocrlty").exception("artifact_index_init_failed")

        try:
            load_tasks()
        except Exception:
            logging.getLogger("ocrlty").exception("tasks_load_failed")

        # --- vLLM HTTP client (keep-alive) ---
        limits = httpx.Limits(
            max_connections=settings.VLLM_HTTP_MAX_CONNECTIONS,
            max_keepalive_connections=settings.VLLM_HTTP_MAX_KEEPALIVE,
            keepalive_expiry=settings.VLLM_HTTP_KEEPALIVE_EXPIRY_S,
        )
        timeout = httpx.Timeout(
            connect=settings.VLLM_CONNECT_TIMEOUT_S,
            read=settings.VLLM_READ_TIMEOUT_S,
            write=settings.VLLM_WRITE_TIMEOUT_S,
            pool=settings.VLLM_POOL_TIMEOUT_S,
        )

        http = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            headers={
                "Accept": "application/json",
                "User-Agent": "ocrlty/1.0",
            },
        )
        app.state.vllm_http = http
        app.state.vllm_client = VLLMClient(
            http,
            base_url=settings.VLLM_BASE_URL,
            api_key=settings.VLLM_API_KEY,
        )

        # --- Jobs store/runner (best-effort) ---
        try:
            js = JobsStore(settings.JOBS_DB_PATH)
            js.ensure_init()
            app.state.jobs_store = js

            backend = (settings.JOBS_BACKEND or "local").strip().lower()
            if backend == "celery":
                app.state.jobs_runner = CeleryJobRunner()
            else:
                stale = js.mark_stale_running_as_failed()
                if stale:
                    logging.getLogger("ocrlty").warning("jobs_marked_stale=%s", stale)

                r = LocalJobRunner(store=js, max_concurrency=settings.JOBS_MAX_CONCURRENCY)
                r.start()
                app.state.jobs_runner = r

                # Requeue queued jobs from previous run (best-effort)
                try:
                    queued = js.list_queued_jobs(limit=500)
                    for j in queued:
                        await r.enqueue(j.job_id)
                    if queued:
                        logging.getLogger("ocrlty").info("jobs_requeued=%s", len(queued))
                except Exception:
                    logging.getLogger("ocrlty").exception("jobs_requeue_failed")
        except Exception:
            logging.getLogger("ocrlty").exception("jobs_init_failed")

        yield

    finally:
        # Stop jobs runner first (so no tasks try to use http during shutdown)
        runner = getattr(app.state, "jobs_runner", None)
        try:
            if runner and hasattr(runner, "shutdown"):
                await runner.shutdown()
        except Exception:
            logging.getLogger("ocrlty").exception("jobs_shutdown_failed")

        # Close vLLM HTTP client
        try:
            if http is not None:
                await http.aclose()
        except Exception:
            logging.getLogger("ocrlty").exception("vllm_http_close_failed")

app = FastAPI(title="OCRlty", version="0.1", lifespan=lifespan)


@app.middleware("http")
async def request_id_ctx_mw(request: Request, call_next):
    hdr = request.headers.get("x-request-id") or request.headers.get("x-request_id")
    rid = make_request_id(hdr)
    token = REQUEST_ID_CTX.set(rid)
    try:
        resp = await call_next(request)
        resp.headers["x-request-id"] = rid
        return resp
    finally:
        try:
            REQUEST_ID_CTX.reset(token)
        except Exception:
            pass


# Routers
app.include_router(core_router)
app.include_router(extract_router)
app.include_router(debug_router)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "auth_enabled": bool(settings.AUTH_ENABLED),
        "debug_mode": bool(settings.DEBUG_MODE),
        "inference_backend": settings.INFERENCE_BACKEND,
        "model": settings.VLLM_MODEL,
    }
