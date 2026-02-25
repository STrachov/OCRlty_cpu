from fastapi import APIRouter
from pydantic import BaseModel

from app.handlers import me, health
from app.schemas.api_error import common_error_responses


class MeResponse(BaseModel):
    key_id: str
    role: str
    scopes: list[str]


class HealthResponse(BaseModel):
    ok: bool
    auth_enabled: bool
    debug_mode: bool
    inference_backend: str
    model: str

router = APIRouter(prefix="/v1", tags=["core"])
router.get("/me", response_model=MeResponse, responses=common_error_responses(401, 403, 500))(me)
router.get("/health", response_model=HealthResponse, responses=common_error_responses(500))(health)
