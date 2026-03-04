from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.auth import require_scopes
from app.auth_store import ApiPrincipal
from app.schemas.api_error import common_error_responses
from app.services.s3_service import s3_key, s3_presign_get_url
from app.settings import settings

router = APIRouter(prefix="/v1/inputs", tags=["inputs"])
_ERR = common_error_responses(400, 401, 403, 404, 500, 501)


class InputPresignResponse(BaseModel):
    input_ref: str
    url: str
    expires_in: int


@router.get(
    "/presign",
    response_model=InputPresignResponse,
    responses=_ERR,
)
def presign_input(
    input_ref: str = Query(...),
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
) -> InputPresignResponse:
    rel = str(input_ref or "").replace("\\", "/").lstrip("/")
    if not rel or not rel.startswith("inputs/") or "/../" in f"/{rel}/":
        raise HTTPException(status_code=400, detail="Invalid input_ref.")

    parts = rel.split("/", 2)
    owner = parts[1] if len(parts) > 1 else ""
    if (principal.role or "") != "admin":
        if owner != str(principal.api_key_id):
            raise HTTPException(status_code=404, detail="Not found")

    key = s3_key(rel)
    expires_in = int(settings.S3_PRESIGN_TTL_S)
    url = s3_presign_get_url(key=key, expires_s=expires_in)
    return InputPresignResponse(input_ref=rel, url=url, expires_in=expires_in)
