from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.auth import require_scopes
from app.auth_store import ApiPrincipal
from app.services.task_registry import list_task_summaries
from app.schemas.api_error import common_error_responses

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])
_ERR = common_error_responses(400, 401, 403, 404, 422, 500)


class TaskSummary(BaseModel):
    task_id: str
    description: str = ""


class TaskListResponse(BaseModel):
    items: List[TaskSummary] = Field(default_factory=list)


@router.get(
    "",
    response_model=TaskListResponse,
    responses=_ERR,
    dependencies=[Depends(require_scopes(["extract:run"]))],
)
async def list_tasks(
    _principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
) -> TaskListResponse:
    return TaskListResponse(items=[TaskSummary(**item) for item in list_task_summaries()])
