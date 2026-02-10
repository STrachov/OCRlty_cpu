from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.auth import init_auth_db
from app.settings import settings
from app.routers.core import router as core_router
from app.routers.extract import router as extract_router
from app.routers.debug import router as debug_router
from app.handlers import make_request_id, REQUEST_ID_CTX, setup_logging
from app.services.artifacts import artifact_index_init


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Logging
    try:
        setup_logging()
    except Exception:
        logging.basicConfig(level=logging.INFO)

    # Auth DB
    try:
        init_auth_db()
    except Exception:
        logging.getLogger("ocrlty").exception("auth_db_init_failed")

    # Artifact index (best-effort)
    try:
        artifact_index_init()
    except Exception:
        logging.getLogger("ocrlty").exception("artifact_index_init_failed")

    yield


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
