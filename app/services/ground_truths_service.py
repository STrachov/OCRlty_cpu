from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, List, Tuple

from fastapi import HTTPException

from app.auth_store import ApiPrincipal
from app.services.ground_truths_store import GroundTruthRecord, GroundTruthsStore
from app.services.s3_service import s3_enabled, s3_get_json, s3_key, s3_put_bytes


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _gt_s3_rel(gt_id: str) -> str:
    return f"ground_truths/{gt_id}.json"


def _normalize_name(name: str | None) -> str:
    raw = str(name or "").strip()
    return raw or "ground_truth.json"


def _enforce_owner_or_admin(principal: ApiPrincipal, rec: GroundTruthRecord) -> None:
    if principal.role == "admin":
        return
    if rec.owner_key_id != principal.key_id:
        raise HTTPException(status_code=404, detail="Ground truth file not found.")


def _parse_gt_json(data: bytes) -> Any:
    try:
        text = (data or b"").decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Ground truth file must be valid UTF-8 JSON.")
    try:
        obj = json.loads(text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ground truth file must be valid JSON: {e}")
    if not isinstance(obj, (dict, list)):
        raise HTTPException(status_code=400, detail="Ground truth JSON must be an object or an array.")
    return obj


def create_ground_truth_from_upload(
    *,
    store: GroundTruthsStore,
    principal: ApiPrincipal,
    filename: str | None,
    data: bytes,
) -> GroundTruthRecord:
    if not s3_enabled():
        raise HTTPException(status_code=501, detail="S3 is not enabled.")
    _parse_gt_json(data)
    gt_id = uuid.uuid4().hex
    created_at = _utc_now_iso()
    name = _normalize_name(filename)
    s3_rel = _gt_s3_rel(gt_id)
    ok = s3_put_bytes(
        key=s3_key(s3_rel),
        data=data,
        content_type="application/json; charset=utf-8",
    )
    if not ok:
        raise HTTPException(status_code=409, detail="Ground truth object already exists.")
    return store.create_ground_truth(
        gt_id=gt_id,
        owner_key_id=str(principal.key_id),
        name=name,
        s3_rel=s3_rel,
        created_at=created_at,
    )


def get_ground_truth_for_principal(
    *,
    store: GroundTruthsStore,
    principal: ApiPrincipal,
    gt_id: str,
) -> GroundTruthRecord:
    rec = store.get_ground_truth(gt_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Ground truth file not found.")
    _enforce_owner_or_admin(principal, rec)
    return rec


def list_ground_truths_for_principal(
    *,
    store: GroundTruthsStore,
    principal: ApiPrincipal,
    limit: int,
    offset: int,
) -> List[GroundTruthRecord]:
    owner_key_id = None if principal.role == "admin" else principal.key_id
    return store.list_ground_truths(owner_key_id=owner_key_id, limit=limit, offset=offset)


def load_ground_truth_json(
    *,
    store: GroundTruthsStore,
    principal: ApiPrincipal,
    gt_id: str,
) -> Tuple[GroundTruthRecord, Any]:
    rec = get_ground_truth_for_principal(store=store, principal=principal, gt_id=gt_id)
    obj = s3_get_json(s3_key(rec.s3_rel))
    if not isinstance(obj, (dict, list)):
        raise HTTPException(status_code=500, detail="Ground truth JSON must be an object or an array.")
    return rec, obj
