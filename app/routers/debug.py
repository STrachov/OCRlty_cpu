from functools import lru_cache
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from app.auth import require_scopes
from app.schemas.api_error import common_error_responses
from app.services.debug_service import DebugService
from app.settings import settings
from app import handlers

router = APIRouter(prefix="/v1/debug", tags=["debug"])
_ERR = common_error_responses(400, 401, 403, 404, 422, 500)


def _debug_enabled() -> None:
    if not settings.DEBUG_MODE:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not found")


@lru_cache(maxsize=1)
def get_debug_service() -> DebugService:
    return DebugService()


@router.get(
    "/artifacts",
    response_model=handlers.ArtifactListResponse,
    responses=_ERR,
    dependencies=[Depends(_debug_enabled), Depends(require_scopes(["debug:read_raw"]))],
)
async def list_artifacts(
    date: Optional[str] = Query(default=None, description="Filter by day (YYYY-MM-DD). If omitted, returns most recent across days."),
    limit: int = Query(default=50, ge=1, le=500),
    svc: DebugService = Depends(get_debug_service),
) -> handlers.ArtifactListResponse:
    return svc.list_artifacts(date=date, limit=limit)


@router.get(
    "/artifacts/backup.tar.gz",
    responses=_ERR,
    dependencies=[Depends(_debug_enabled), Depends(require_scopes(["debug:read_raw"]))],
)
async def get_artifacts_backup(
    background_tasks: BackgroundTasks,
    svc: DebugService = Depends(get_debug_service),
):
    return svc.backup_tar_gz(background_tasks)


@router.get(
    "/artifacts/{request_id}",
    response_model=Dict[str, Any],
    responses=_ERR,
    dependencies=[Depends(_debug_enabled), Depends(require_scopes(["debug:read_raw"]))],
)
async def read_artifact(
    request_id: str,
    date: Optional[str] = Query(default=None, description="Optional day (YYYY-MM-DD) to avoid search."),
    svc: DebugService = Depends(get_debug_service),
) -> Dict[str, Any]:
    return svc.read_artifact(request_id, date=date)


@router.get(
    "/artifacts/{request_id}/raw",
    response_model=Dict[str, Any],
    responses=_ERR,
    dependencies=[Depends(_debug_enabled), Depends(require_scopes(["debug:read_raw"]))],
)
async def read_artifact_raw_text(
    request_id: str,
    date: Optional[str] = Query(default=None, description="Optional day (YYYY-MM-DD) to avoid search."),
    svc: DebugService = Depends(get_debug_service),
) -> Dict[str, Any]:
    return svc.read_artifact_raw(request_id, date=date)


@router.post(
    "/eval/batch_vs_gt",
    response_model=handlers.EvalBatchVsGTResponse,
    responses=_ERR,
    dependencies=[Depends(_debug_enabled), Depends(require_scopes(["debug:run"]))],
)
async def eval_batch_vs_gt(
    req: handlers.EvalBatchVsGTRequest,
    svc: DebugService = Depends(get_debug_service),
) -> handlers.EvalBatchVsGTResponse:
    return await svc.eval_batch_vs_gt(req)
