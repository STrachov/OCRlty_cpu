from __future__ import annotations

from typing import Any, List

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from app.auth import require_scopes
from app.auth_store import ApiPrincipal
from app.schemas.api_error import common_error_responses
from app.services.ground_truths_service import (
    build_ground_truth_draft_for_principal,
    create_ground_truth_from_content,
    create_ground_truth_from_run,
    create_ground_truth_from_upload,
    delete_ground_truth_for_principal,
    get_ground_truth_for_principal,
    load_ground_truth_json,
    list_ground_truths_for_principal,
    update_ground_truth_content,
)
from app.services.ground_truths_store import GroundTruthRecord, GroundTruthsStore

router = APIRouter(prefix="/v1/ground_truths", tags=["ground_truths"])
_ERR = common_error_responses(400, 401, 403, 404, 422, 500, 501)


def _store(request: Request) -> GroundTruthsStore:
    s = getattr(request.app.state, "ground_truths_store", None)
    if s is None:
        raise HTTPException(status_code=500, detail="Ground truths store is not initialized.")
    return s


class GroundTruthView(BaseModel):
    gt_id: str
    name: str
    created_at: str
    task_id: str | None = None
    source_type: str | None = None
    source_run_id: str | None = None
    updated_at: str | None = None


class GroundTruthListResponse(BaseModel):
    items: List[GroundTruthView] = Field(default_factory=list)


def _to_view(rec: GroundTruthRecord) -> GroundTruthView:
    return GroundTruthView(
        gt_id=rec.gt_id,
        name=rec.name,
        created_at=rec.created_at,
        task_id=rec.task_id,
        source_type=rec.source_type,
        source_run_id=rec.source_run_id,
        updated_at=rec.updated_at,
    )


class GroundTruthFromRunRequest(BaseModel):
    run_id: str
    name: str | None = None


class GroundTruthUpdateRequest(BaseModel):
    name: str
    content: Any


class GroundTruthCreateRequest(BaseModel):
    name: str
    content: Any


class DeleteGroundTruthResponse(BaseModel):
    ok: bool
    gt_id: str


@router.post(
    "/upload",
    response_model=GroundTruthView,
    responses=_ERR,
    dependencies=[Depends(require_scopes(["debug:run"]))],
)
async def upload_ground_truth(
    request: Request,
    file: UploadFile = File(...),
    principal: ApiPrincipal = Depends(require_scopes(["debug:run"])),
) -> GroundTruthView:
    store = _store(request)
    data = await file.read()
    rec = create_ground_truth_from_upload(
        store=store,
        principal=principal,
        filename=file.filename,
        data=data,
    )
    return _to_view(rec)


@router.post(
    "",
    response_model=GroundTruthView,
    responses=_ERR,
    dependencies=[Depends(require_scopes(["debug:run"]))],
)
async def create_ground_truth(
    body: GroundTruthCreateRequest,
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes(["debug:run"])),
) -> GroundTruthView:
    store = _store(request)
    rec = create_ground_truth_from_content(
        store=store,
        principal=principal,
        name=body.name,
        payload=body.content,
    )
    return _to_view(rec)


@router.post(
    "/from_run/draft",
    response_model=Any,
    responses=_ERR,
    dependencies=[Depends(require_scopes(["debug:run"]))],
)
async def create_ground_truth_draft_from_run_endpoint(
    body: GroundTruthFromRunRequest,
    principal: ApiPrincipal = Depends(require_scopes(["debug:run"])),
) -> Any:
    return build_ground_truth_draft_for_principal(
        principal=principal,
        run_id=body.run_id,
        name=body.name,
    )


@router.post(
    "/from_run",
    response_model=GroundTruthView,
    responses=_ERR,
    dependencies=[Depends(require_scopes(["debug:run"]))],
)
async def create_ground_truth_from_run_endpoint(
    body: GroundTruthFromRunRequest,
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes(["debug:run"])),
) -> GroundTruthView:
    store = _store(request)
    rec = create_ground_truth_from_run(
        store=store,
        principal=principal,
        run_id=body.run_id,
        name=body.name,
    )
    return _to_view(rec)


@router.get(
    "",
    response_model=GroundTruthListResponse,
    responses=_ERR,
    dependencies=[Depends(require_scopes(["debug:run"]))],
)
async def list_ground_truths(
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes(["debug:run"])),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> GroundTruthListResponse:
    store = _store(request)
    rows = list_ground_truths_for_principal(store=store, principal=principal, limit=limit, offset=offset)
    return GroundTruthListResponse(items=[_to_view(row) for row in rows])


@router.get(
    "/{gt_id}",
    response_model=GroundTruthView,
    responses=_ERR,
    dependencies=[Depends(require_scopes(["debug:run"]))],
)
async def get_ground_truth(
    gt_id: str,
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes(["debug:run"])),
) -> GroundTruthView:
    store = _store(request)
    rec = get_ground_truth_for_principal(store=store, principal=principal, gt_id=gt_id)
    return _to_view(rec)


@router.get(
    "/{gt_id}/content",
    response_model=Any,
    responses=_ERR,
    dependencies=[Depends(require_scopes(["debug:run"]))],
)
async def get_ground_truth_content(
    gt_id: str,
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes(["debug:run"])),
) -> Any:
    store = _store(request)
    _, obj = load_ground_truth_json(store=store, principal=principal, gt_id=gt_id)
    return obj


@router.put(
    "/{gt_id}",
    response_model=GroundTruthView,
    responses=_ERR,
    dependencies=[Depends(require_scopes(["debug:run"]))],
)
async def update_ground_truth(
    gt_id: str,
    body: GroundTruthUpdateRequest,
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes(["debug:run"])),
) -> GroundTruthView:
    store = _store(request)
    rec = update_ground_truth_content(
        store=store,
        principal=principal,
        gt_id=gt_id,
        name=body.name,
        payload=body.content,
    )
    return _to_view(rec)


@router.delete(
    "/{gt_id}",
    response_model=DeleteGroundTruthResponse,
    responses=_ERR,
    dependencies=[Depends(require_scopes(["debug:run"]))],
)
async def delete_ground_truth(
    gt_id: str,
    request: Request,
    principal: ApiPrincipal = Depends(require_scopes(["debug:run"])),
) -> DeleteGroundTruthResponse:
    store = _store(request)
    rec = delete_ground_truth_for_principal(store=store, principal=principal, gt_id=gt_id)
    return DeleteGroundTruthResponse(ok=True, gt_id=rec.gt_id)
